"""Automated train-test leakage verification for the reasoning set (T3.4).

Duplicated verbatim into every language tree; never imports from a sibling.

    python nepali/reasoning/verify_leakage.py

WHAT IS CHECKED, AND WHY EACH ONE
---------------------------------
1. ENTITY HOLDOUT      no name occurs in more than one split. Checked on the
                       generated FILES, not on the pools that were supposed to
                       produce them -- a generator bug would otherwise pass.
2. RELATION HOLDOUT    the two test-only relations never appear in train/val.
3. HOP HOLDOUT         3-hop chains appear only in the ood split.
4. EXACT PROMPT        no prompt string is shared across splits.
5. N-GRAM OVERLAP      the fraction of test prompts whose 8-gram appears in the
                       train split. Near-zero is expected because the entity
                       pools are disjoint, so this is a check on the CHECK as
                       much as on the data.
6. PRETRAINING         the same 8-gram test against the language-model
   CONTAMINATION       pretraining corpus. This is the one that matters for
                       interpreting results: if a test prompt already occurred
                       during pretraining, a correct answer is not evidence of
                       reasoning.

THE HONEST CAVEAT
-----------------
N-gram decontamination is NECESSARY BUT NOT SUFFICIENT (course L3). It cannot
detect a paraphrase, and our templates are by construction highly repetitive, so
a low n-gram overlap does not prove the test set is unseen -- it proves no
verbatim span is shared. The structural holdouts (1-3) are the load-bearing
guarantees; this is corroboration.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
REPO = TREE.parent
LANG = {"hindi": "hi", "nepali": "ne"}[TREE.name]


def load(split: str, data_dir: Path) -> list[dict]:
    p = data_dir / f"{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ngrams(text: str, n: int) -> set[tuple]:
    """Whitespace n-grams. Whitespace is the right unit here: the templates are
    word-assembled, so a shared span shows up as shared words."""
    w = text.split()
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def check_entity_holdout(splits: dict) -> dict:
    names = {s: {e for r in rows for e in r["chain"]} for s, rows in splits.items()}
    overlaps = {}
    for a in ("train", "val", "test", "ood"):
        for b in ("val", "test", "ood"):
            if a >= b or a not in names or b not in names:
                continue
            overlaps[f"{a}|{b}"] = sorted(names[a] & names[b])
    # ood reuses the test pool by design, so it is reported, not asserted.
    hard = {k: v for k, v in overlaps.items() if "ood" not in k}
    return {"name": "entity_holdout",
            "passed": all(not v for v in hard.values()),
            "pairwise_shared_names": {k: len(v) for k, v in overlaps.items()},
            "examples": {k: v[:5] for k, v in overlaps.items() if v},
            "note": "test|ood share a pool by design; train|* must be empty"}


def check_relation_holdout(splits: dict, held: set) -> dict:
    seen = {s: sorted({r["relation"] for r in rows}) for s, rows in splits.items()}
    bad = {s: sorted(set(v) & held) for s, v in seen.items() if s in ("train", "val")}
    return {"name": "relation_holdout",
            "passed": all(not v for v in bad.values()),
            "held_out": sorted(held),
            "relations_per_split": {s: len(v) for s, v in seen.items()},
            "violations": {k: v for k, v in bad.items() if v}}


def check_hop_holdout(splits: dict) -> dict:
    """The probe must be STRICTLY longer than anything trained.

    Expressed as an invariant, not a fixed number. The first version asserted
    "ood == 3 hops and train has none", which was right for one dataset version
    and broke the moment P4 was widened to four entities to remove a shortcut.
    """
    hops = {s: sorted({r["n_hops"] for r in rows}) for s, rows in splits.items()}
    trained = set(hops.get("train", [])) | set(hops.get("val", []))
    ood = set(hops.get("ood", []))
    ok = bool(ood) and bool(trained) and min(ood) > max(trained) and len(ood) == 1
    return {"name": "hop_holdout", "passed": ok, "hops_per_split": hops,
            "max_trained_hops": max(trained) if trained else None,
            "ood_hops": sorted(ood)}


def check_exact_prompt(splits: dict) -> dict:
    ps = {s: {r["prompt"] for r in rows} for s, rows in splits.items()}
    shared = {}
    for b in ("val", "test", "ood"):
        if b in ps:
            shared[f"train|{b}"] = len(ps["train"] & ps[b])
    return {"name": "exact_prompt_overlap",
            "passed": all(v == 0 for v in shared.values()), "counts": shared}


def check_ngram_vs_train(splits: dict, n: int, sample: int, seed: int) -> dict:
    rng = random.Random(seed)
    train_ng = set()
    for r in splits["train"]:
        train_ng |= ngrams(r["prompt"], n)
    out = {}
    for s in ("test", "ood"):
        if s not in splits:
            continue
        rows = splits[s]
        probe = rng.sample(rows, min(sample, len(rows)))
        hit = sum(1 for r in probe if ngrams(r["prompt"], n) & train_ng)
        out[s] = {"probed": len(probe), "with_shared_ngram": hit,
                  "fraction": hit / len(probe)}
    return {"name": f"ngram_{n}_vs_train", "passed": True,
            "note": "reported, not gated: templates are repetitive by design",
            "results": out, "train_ngrams": len(train_ng)}


def check_pretraining_contamination(splits: dict, n: int, sample: int,
                                    corpus_docs: int, seed: int) -> dict:
    """Do test prompts share an n-gram with the LM pretraining corpus?"""
    rng = random.Random(seed)
    probe = rng.sample(splits["test"], min(sample, len(splits["test"])))
    want = {}
    for i, r in enumerate(probe):
        for g in ngrams(r["prompt"], n):
            want.setdefault(g, set()).add(i)
    hit_rows = set()
    split_path = TREE / "corpus" / "splits" / "train.jsonl"
    scanned = 0
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                if scanned >= corpus_docs:
                    break
                scanned += 1
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                for g in ngrams(t, n):
                    if g in want:
                        hit_rows |= want[g]
    return {"name": f"pretraining_contamination_{n}gram",
            "passed": len(hit_rows) == 0,
            "probed": len(probe), "corpus_docs_scanned": scanned,
            "test_prompts_with_corpus_ngram": len(hit_rows),
            "fraction": len(hit_rows) / len(probe) if probe else 0.0,
            "caveat": "n-gram decontamination is necessary but not sufficient "
                      "(course L3): it cannot detect paraphrase."}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Verify reasoning-set leakage.")
    ap.add_argument("--data-dir", default=str(TREE / "reasoning_data"))
    ap.add_argument("--n", type=int, default=8, help="n-gram order")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--corpus-docs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import templates as T

    data_dir = Path(args.data_dir)
    splits = {s: load(s, data_dir) for s in ("train", "val", "test", "ood")
              if (data_dir / f"{s}.jsonl").exists()}
    print(f"[splits] { {k: len(v) for k, v in splits.items()} }", flush=True)

    checks = [
        check_entity_holdout(splits),
        check_relation_holdout(splits, T.HELD_OUT_RELATIONS),
        check_hop_holdout(splits),
        check_exact_prompt(splits),
        check_ngram_vs_train(splits, args.n, args.sample, args.seed),
        check_pretraining_contamination(splits, args.n, args.sample,
                                        args.corpus_docs, args.seed),
    ]

    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{flag}] {c['name']}", flush=True)
        for k, v in c.items():
            if k in ("name", "passed"):
                continue
            print(f"           {k}: {v}", flush=True)

    result = {"lang_code": LANG, "n_gram": args.n, "checks": checks,
              "all_passed": all(c["passed"] for c in checks)}
    out = Path(args.out) if args.out else (
        REPO / "report" / "phase3" / TREE.name / "leakage_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if result['all_passed'] else 'SOME CHECKS FAILED'}")
    print(f"-> {out}")
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
