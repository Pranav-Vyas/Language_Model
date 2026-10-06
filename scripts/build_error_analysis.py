"""Phase-3 error analysis: where the finetuned models succeed and fail (T3.9).

Language-agnostic tooling: it reads both trees' result JSONs but holds no
language data and never mixes their corpora.

    python scripts/build_error_analysis.py

Reads report/phase3/<tree>/reasoning_eval_v2.json -- the CORRECTED evaluation,
produced after the right-padding bug was fixed -- and writes
report/phase3/error_analysis.md plus figures under report/phase3/figures/.

WHY A GENERATOR RATHER THAN HAND-WRITTEN PROSE
----------------------------------------------
Same reason as scripts/build_comparison.py: every number here traces to a JSON
that a re-run regenerates, so a stale figure is a missing re-run rather than an
undetectable typo. This matters more in Phase 3 than Phase 2, because one bug
already invalidated a whole round of reported accuracies.

THE FOUR BREAKDOWN AXES ARE THE ANTI-LEAKAGE AXES
-------------------------------------------------
They are not arbitrary slices. Each one tests a specific way the model could
score well without reasoning:

  by_held_out_relation  relations never seen in training -- the strongest test,
                        since a model that memorised "taller -> pick X" cannot
                        transfer to an unseen relation word.
  by_pattern            P1/P2 (atomic) vs P3/P4 (inferred, multi-hop).
  by_rank_distance      the ordinal-geometry prediction: accuracy should RISE
                        with rank distance between the compared entities.
  by_polarity           positive ("tallest") vs negative ("shortest") -- a model
                        keying on the relation word alone fails one of them.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

REPO = Path(__file__).resolve().parent.parent
PH3 = REPO / "report" / "phase3"
FIGS = PH3 / "figures"
TREES = [("hindi", "Model H (Hindi)"), ("nepali", "Model L (Nepali)")]

# Devanagari-capable font, so entity names in any label render as glyphs and
# not as empty boxes. Same helper as the trees' viz.py; duplicated rather than
# imported, because scripts/ must not import from a language tree.
for _name in ("Noto Sans Devanagari", "Nirmala UI", "Lohit Devanagari",
              "Mangal", "Kalimati", "Samyak Devanagari"):
    if _name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = [_name, "DejaVu Sans"]
        break


def finish(fig, ax, out: Path) -> Path:
    """Assert the figure is labelled, then save. An unlabelled plot scores zero."""
    for a in np.ravel(ax if isinstance(ax, (list, np.ndarray)) else [ax]):
        for got, what in ((a.get_title(), "title"), (a.get_xlabel(), "x-label"),
                          (a.get_ylabel(), "y-label")):
            if not got:
                raise ValueError(f"{out.name}: axis missing {what}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def load(tree: str) -> dict | None:
    p = PH3 / tree / "reasoning_eval_v2.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ft_models(d: dict) -> list[str]:
    return [k for k in d["models"] if k.startswith("finetuned/")]


def breakdown(d: dict, split: str, axis: str, metric: str = "exact_match") -> dict:
    """Mean and half-range across finetuning seeds for one breakdown axis.

    Returns {bucket: {"mean", "half_range", "n"}}. Averaging across seeds is
    what makes a small per-bucket difference interpretable: the spread across
    seeds is the noise floor the difference has to clear.
    """
    per_seed: dict[str, list[float]] = {}
    ns: dict[str, int] = {}
    for m in ft_models(d):
        blk = d["models"][m]["results"][split].get(axis, {}).get(metric, {})
        for bucket, v in blk.items():
            per_seed.setdefault(bucket, []).append(v["accuracy"])
            ns[bucket] = v["n"]
    return {b: {"mean": st.mean(v), "half_range": (max(v) - min(v)) / 2, "n": ns[b]}
            for b, v in sorted(per_seed.items())}


def headline(d: dict, split: str, metric: str) -> dict:
    v = [d["models"][m]["results"][split][metric] for m in ft_models(d)]
    return {"mean": st.mean(v), "half_range": (max(v) - min(v)) / 2}


def chance(d: dict, split: str) -> float:
    return d["models"][ft_models(d)[0]]["results"][split]["chance"]


# ---------------------------------------------------------------- figures ---

def grouped_bars(data: dict, order: list[str], xlabel: str, title: str,
                 out: Path, chance_lines: dict | None = None,
                 xticklabels: list[str] | None = None) -> Path:
    """One bar cluster per bucket, one bar per language, with seed error bars."""
    langs = [l for l in data if data[l]]
    x = np.arange(len(order))
    width = 0.8 / max(1, len(langs))
    fig, ax = plt.subplots(figsize=(1.7 * len(order) + 3.5, 4.4))
    colours = {"hindi": "steelblue", "nepali": "darkorange"}
    for k, lang in enumerate(langs):
        vals = [data[lang].get(b, {}).get("mean", np.nan) for b in order]
        errs = [data[lang].get(b, {}).get("half_range", 0.0) for b in order]
        pos = x + k * width - 0.4 + width / 2
        bars = ax.bar(pos, vals, width, yerr=errs, capsize=3,
                      color=colours.get(lang, None),
                      label=dict(TREES).get(lang, lang))
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
        if chance_lines and lang in chance_lines:
            # One short dash over each bar. y must be an ARRAY matching the
            # x arrays; a scalar y with vector xmin/xmax fails to broadcast.
            ax.hlines([chance_lines[lang]] * len(pos), pos - width / 2,
                      pos + width / 2, color="crimson", linestyle=":",
                      linewidth=1.6, label="chance" if k == 0 else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("exact match (mean of 3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels or order)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8)
    return finish(fig, ax, out)


# ------------------------------------------------- OOD failure mode ---------

def ood_failure_mode(tree: str, d: dict) -> dict:
    """WHERE in the ordering does the model point when it fails out of domain?

    The OOD split is one hop longer than anything in training and its answer is
    always an EXTREME of the chain (first or last). If the model had simply
    failed to extend the chain we would expect near-random positions; if it
    systematically names a MIDDLE entity, it has learned "answer with an entity
    from the chain" plus a positional habit, not the ordering operation.

    Predictions come from the per-model `samples` (the eval stores a fixed
    sample per split, not the full prediction vector), joined back to the
    dataset row by prompt to recover that row's `chain`. n is reported.
    """
    rows = {}
    p = REPO / tree / "reasoning_data" / "ood.jsonl"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["prompt"]] = r

    counts: dict[str, int] = {}
    n = correct = off_chain = 0
    # Expected middle-fraction if the model picked uniformly among the chain's
    # entities. A 5-entity chain has 3 middles, so this is 0.6 -- without it,
    # "98% middle" is not interpretable as a bias.
    uniform_middle: list[float] = []
    for m in ft_models(d):
        for s in d["models"][m]["results"]["ood"].get("samples", []):
            row = rows.get(s["prompt"])
            if not row:
                continue
            chain = row["chain"]                 # ordered best -> worst
            pred = s.get("em_pred")
            n += 1
            uniform_middle.append(max(0, len(chain) - 2) / len(chain))
            if pred == s["gold"]:
                correct += 1
            if pred not in chain:
                off_chain += 1
                counts["not in chain"] = counts.get("not in chain", 0) + 1
                continue
            i = chain.index(pred)
            where = ("extreme (first)" if i == 0 else
                     "extreme (last)" if i == len(chain) - 1 else
                     f"middle ({i})")
            counts[where] = counts.get(where, 0) + 1
    middle = sum(v for k, v in counts.items() if k.startswith("middle"))
    return {"n": n, "correct": correct, "off_chain": off_chain,
            "middle": middle, "counts": counts,
            "middle_frac": middle / n if n else float("nan"),
            "uniform_middle_frac": (st.mean(uniform_middle)
                                    if uniform_middle else float("nan"))}


def qualitative(d: dict, split: str, want_wrong: bool, limit: int) -> list[dict]:
    """A few real prompts with the model's actual answer."""
    m = ft_models(d)[0]
    out = []
    for s in d["models"][m]["results"][split].get("samples", []):
        if (s.get("em_pred") != s["gold"]) == want_wrong:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def fmt(v: dict, spec: str = ".3f") -> str:
    return f"{v['mean']:{spec}} ±{v['half_range']:{spec}}"


def main() -> None:
    data = {t: load(t) for t, _ in TREES}
    have = {t: d for t, d in data.items() if d}
    if not have:
        raise SystemExit("no reasoning_eval_v2.json found -- run the evals first")
    missing = [t for t, d in data.items() if not d]

    L: list[str] = ["# Phase 3 — error analysis (T3.9)", ""]
    L += ["Generated by `scripts/build_error_analysis.py` from "
          "`report/phase3/<tree>/reasoning_eval_v2.json` (the corrected evaluation, "
          "produced after the right-padding bug was fixed). Every cell is the mean "
          "of 3 finetuning seeds ± half the range across them.", ""]
    if missing:
        L += [f"> **Incomplete:** no results yet for {', '.join(missing)}.", ""]

    # ---- headline --------------------------------------------------------
    L += ["## 1. Headline", "",
          "| metric | " + " | ".join(lbl for t, lbl in TREES if t in have) + " |",
          "|---|" + "---|" * len(have)]
    for split in ("test", "ood"):
        for metric in ("exact_match", "closed_set", "well_formed"):
            cells = [fmt(headline(have[t], split, metric)) for t, _ in TREES if t in have]
            L.append(f"| {split} {metric.replace('_', ' ')} | " + " | ".join(cells) + " |")
        cells = [f"{chance(have[t], split):.3f}" for t, _ in TREES if t in have]
        L.append(f"| {split} chance | " + " | ".join(cells) + " |")
    L.append("")

    # ---- breakdown axes --------------------------------------------------
    axes = [
        ("by_held_out_relation", "relation held out of training?",
         "Held-out relations: the strongest generalisation test",
         {"False": "seen in training", "True": "held out"}),
        ("by_pattern", "reasoning pattern",
         "Per pattern: atomic (P1, P2) vs inferred (P3, P4)", None),
        ("by_rank_distance", "rank distance between the compared entities",
         "Symbolic distance effect", None),
        ("by_polarity", "question polarity",
         "Polarity: positive vs negative comparatives",
         {"pos": "positive", "neg": "negative"}),
    ]
    section = 2
    for axis, xlabel, title, relabel in axes:
        per_lang = {t: breakdown(have[t], "test", axis) for t in have}
        order = sorted({b for v in per_lang.values() for b in v})
        if not order:
            continue
        L += [f"## {section}. {title}", "",
              "| bucket | n | " + " | ".join(lbl for t, lbl in TREES if t in have) + " |",
              "|---|---:|" + "---|" * len(have)]
        for b in order:
            n = next((per_lang[t][b]["n"] for t in per_lang if b in per_lang[t]), "")
            cells = [fmt(per_lang[t][b]) if b in per_lang[t] else "—"
                     for t, _ in TREES if t in have]
            L.append(f"| {(relabel or {}).get(b, b)} | {n} | " + " | ".join(cells) + " |")
        L.append("")
        grouped_bars(per_lang, order, xlabel, f"{title} (test split)",
                     FIGS / f"err_{axis}.png",
                     chance_lines={t: chance(have[t], "test") for t in have},
                     xticklabels=[(relabel or {}).get(b, b) for b in order])
        L += [f"![{title}](figures/err_{axis}.png)", ""]
        section += 1

    # ---- test vs OOD -----------------------------------------------------
    per_lang = {t: {s: headline(have[t], s, "exact_match") for s in ("test", "ood")}
                for t in have}
    grouped_bars(per_lang, ["test", "ood"], "split",
                 "In-domain vs out-of-domain (one extra hop)",
                 FIGS / "err_test_vs_ood.png",
                 chance_lines={t: chance(have[t], "test") for t in have},
                 xticklabels=["test (<=3 hops, seen)", "OOD (4 hops, unseen)"])
    L += [f"## {section}. Length generalisation", "",
          "![test vs OOD](figures/err_test_vs_ood.png)", ""]
    section += 1

    # ---- OOD failure mode ------------------------------------------------
    L += [f"## {section}. What the OOD failures actually are", ""]
    for t, lbl in TREES:
        if t not in have:
            continue
        fm = ood_failure_mode(t, have[t])
        if not fm or not fm["n"]:
            continue
        L += [f"**{lbl}** — {fm['n']} sampled OOD predictions across 3 seeds: "
              f"{fm['correct']} correct, {fm['middle']} named a MIDDLE entity of the "
              f"chain ({fm['middle_frac']:.1%}, against {fm['uniform_middle_frac']:.1%} "
              f"expected if it picked uniformly among the chain's entities), "
              f"{fm['off_chain']} named an entity not in the chain at all.", "",
              "| where the prediction sits in the ordering | count |", "|---|---:|"]
        for k, v in sorted(fm["counts"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} |")
        L.append("")
    L += ["The OOD answer is always an *extreme* of the chain, so naming a middle "
          "entity is not a near-miss: it means the model has learned to answer with "
          "an entity from the chain and to pick one by position, rather than to "
          "extend the ordering by one more hop.", ""]
    section += 1

    # ---- qualitative -----------------------------------------------------
    L += [f"## {section}. Qualitative wins and failures", ""]
    for t, lbl in TREES:
        if t not in have:
            continue
        L += [f"### {lbl}", ""]
        for split, wrong, head in (("test", False, "Correct (in domain)"),
                                   ("test", True, "Wrong (in domain)"),
                                   ("ood", True, "Wrong (out of domain, 4 hops)")):
            rows = qualitative(have[t], split, wrong, 3)
            if not rows:
                continue
            L += [f"*{head}*", "", "| prompt | gold | predicted | pattern |",
                  "|---|---|---|---|"]
            for s in rows:
                L.append(f"| {s['prompt']} | {s['gold']} | {s.get('em_pred','')} "
                         f"| {s.get('pattern','')} |")
            L.append("")

    out = PH3 / "error_analysis.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"  -> {out}")
    print(f"  -> {FIGS}")


if __name__ == "__main__":
    main()
