"""Audit a reasoning dataset for surface shortcuts (validity check).

Language-agnostic tooling: reads generated data, holds no language data.

    python scripts/shortcut_audit.py --tree hindi

WHY THIS EXISTS
---------------
v1 shipped a pattern that was solvable without any comparison. P4 asked "who is
the second most X?" over a 3-entity chain -- but in a 3-chain the second-most
and the second-least are the SAME entity, and it is exactly the entity named in
both premises. "Say the repeated word" scored 100% on all 960 P4 test examples,
the model learned precisely that, and at 16% of the data it inflated the
headline. Nothing in the test suite caught it, because every label was correct;
the task itself was degenerate.

So shortcuts are now measured rather than reasoned about. Each rule below is a
strategy that ignores the comparison entirely. If one of them scores far above
chance on a pattern, that pattern is not testing reasoning, whatever its labels
say.

The rules are deliberately dumb -- that is the point. A dumb rule that wins is
a benchmark bug.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def mentions(row: dict) -> list[str]:
    """Entities in the PREMISES, in order of appearance (question excluded)."""
    cut = row["prompt"].rfind("।")
    body = row["prompt"][:cut + 1] if cut != -1 else row["prompt"]
    chain = set(row["chain"])
    return [w for w in body.split() if w in chain]


def rules(row: dict) -> dict[str, str]:
    """Each rule returns the entity it would answer, using no comparison."""
    m = mentions(row)
    counts = Counter(m)
    cands = row["candidates"]
    most = counts.most_common()
    return {
        "first_mentioned": m[0] if m else "",
        "last_mentioned": m[-1] if m else "",
        "most_repeated": most[0][0] if most else "",
        "least_repeated": most[-1][0] if most else "",
        "first_candidate": cands[0],
        "last_candidate": cands[-1],
        "alphabetically_first": sorted(cands)[0],
    }


def audit(rows: list[dict]) -> dict:
    names = list(rules(rows[0]))
    by_pattern: dict[str, dict] = {}
    for r in rows:
        p = r["pattern"]
        b = by_pattern.setdefault(p, {"n": 0, "chance": 0.0,
                                      **{k: 0 for k in names},
                                      "majority": Counter()})
        b["n"] += 1
        b["chance"] += 1 / len(r["candidates"])
        b["majority"][r["answer"]] += 1
        got = rules(r)
        for k in names:
            b[k] += int(got[k] == r["answer"])
    out = {}
    for p, b in sorted(by_pattern.items()):
        n = b["n"]
        row = {"n": n, "chance": b["chance"] / n,
               "majority_answer": max(b["majority"].values()) / n}
        for k in names:
            row[k] = b[k] / n
        # The comparator a result must actually beat. Raw 1/k understates the
        # floor whenever the task has intrinsic structure -- a superlative's
        # answer is never the middle entity, an ordinal's is never an extreme --
        # so the strongest trivial rule is the honest baseline to report.
        row["best_trivial_rule"] = max(names, key=lambda k: row[k])
        row["informed_baseline"] = max(row[k] for k in names)
        out[p] = row
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Audit a dataset for surface shortcuts.")
    ap.add_argument("--tree", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--splits", default="train,test,ood")
    ap.add_argument("--flag-margin", type=float, default=0.15,
                    help="flag a rule scoring this far above chance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ddir = Path(args.data_dir) if args.data_dir else REPO / args.tree / "reasoning_data"
    report, flagged = {}, []
    for split in args.splits.split(","):
        path = ddir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        a = audit(rows)
        report[split] = a
        print(f"\n=== {args.tree} / {split} ===")
        skip = {"n", "chance", "best_trivial_rule", "informed_baseline"}
        keys = [k for k in next(iter(a.values())) if k not in skip]
        print(f"{'pattern':8s} {'n':>6s} {'chance':>7s} " +
              " ".join(f"{k[:13]:>14s}" for k in keys))
        for p, row in a.items():
            cells = []
            for k in keys:
                v = row[k]
                mark = "*" if v - row["chance"] >= args.flag_margin else " "
                cells.append(f"{v:>13.3f}{mark}")
                if mark == "*":
                    flagged.append({"split": split, "pattern": p, "rule": k,
                                    "accuracy": v, "chance": row["chance"]})
            print(f"{p:8s} {row['n']:>6d} {row['chance']:>7.3f} " + " ".join(cells))

    print(f"\n{'-'*60}")
    if flagged:
        print(f"FLAGGED: {len(flagged)} rule/pattern combination(s) beat chance "
              f"by >= {args.flag_margin}")
        for f in flagged:
            print(f"  {f['split']}/{f['pattern']}: {f['rule']} = {f['accuracy']:.3f} "
                  f"vs chance {f['chance']:.3f}")
    else:
        print(f"CLEAN: no shortcut rule beats chance by >= {args.flag_margin}")

    out = Path(args.out) if args.out else (
        REPO / "report" / "phase3" / args.tree / "shortcut_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"flag_margin": args.flag_margin, "flagged": flagged,
                               "by_split": report}, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    raise SystemExit(1 if flagged else 0)


if __name__ == "__main__":
    main()
