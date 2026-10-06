"""Generate the bonus no-positional-embeddings comparison tables (BONUS).

Language-agnostic tooling: it reads one tree's outputs but holds no language
data. Deliberately SEPARATE from `build_comparison.py` rather than a mode of
it. That script produces a graded Phase-2 deliverable whose format the Phase-2
report already refers to; generalising it to N columns would put a closed
phase's artifact at risk to save a little duplication. The ablation also asks a
different question and therefore wants different rows -- attention distance and
content-dependence are the headline here, not the H-vs-L BPB gap.

    python scripts/build_ablation_tables.py     # -> report/phase2/bonus_ablation_tables.md

Every figure traces to a JSON under report/phase2/, so a stale number is a
missing re-run rather than an undetectable typo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PH2 = REPO / "report" / "phase2"

# The controlled comparison. Each ablated column differs from the standard
# column to its left in exactly one respect: the positional embedding table.
# Two horizons are reported because the standard model was already evaluated at
# both, so the ablation inherits two comparison points for one training run.
COLUMNS = [
    ("nepali",              "standard, 13,547 steps"),
    ("nepali_nope",         "no-PE, 13,547 steps"),
    ("nepali_matched",      "standard, 8,732 steps"),
    ("nepali_nope_matched", "no-PE, 8,732 steps"),
]
# Which ablated column each standard column is the control for, used for the
# deltas. Keys and values are indices into COLUMNS.
PAIRS = [(0, 1), (2, 3)]


def load(tag: str, name: str) -> dict | None:
    p = PH2 / tag / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row(label: str, values: list[str]) -> str:
    return f"| {label} | " + " | ".join(values) + " |"


def head(labels: list[str]) -> list[str]:
    return [row("", labels), "|" + "---|" * (len(labels) + 1)]


def fmt(v, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def delta(a, b, spec: str = "+.4f") -> str:
    """Ablated minus standard. Positive = the ablation made the metric larger."""
    return "—" if (a is None or b is None) else format(b - a, spec)


LABELS = [lab for _, lab in COLUMNS]


def setup_table() -> list[str]:
    """What actually differs between the runs -- the controlled-variable claim."""
    return [
        "### Ablation setup", "",
        "| | standard | no-PE |", "|---|---|---|",
        "| positional embeddings | learned absolute, `nn.Embedding(512, 384)` | **none — table not constructed** |",
        "| trainable parameters | 24,070,656 | **23,874,048** (−196,608 = 512 × 384) |",
        "| layers / d_model / heads / d_ff | 10 / 384 / 6 / 1536 | identical |",
        "| context | 512 | identical (the causal-mask buffer still bounds it) |",
        "| peak LR / warmup / dropout | 6e-4 / 500 / 0.1 | identical |",
        "| batch | 8 × 16 = 65,536 tok/step | identical |",
        "| LR horizon / seed | 13,547 / 1337 | identical |",
        "| precision | bfloat16 | **float16** — see note |",
        "",
        "> **The one deviation.** The standard run used bf16 and took 23 h. The "
        "Phase-2 precision benchmark measured fp16 and bf16 producing identical "
        "losses at 3.11× the throughput on a T4 (Turing tensor cores implement "
        "fp16, not bf16), so the ablation used fp16 to fit one session. Recorded "
        "here rather than left for a reader to find.", "",
    ]


def lm_table() -> list[str]:
    out = ["### Intrinsic language-modelling metrics", ""] + head(LABELS)
    data = {t: load(t, "lm_metrics.json") for t, _ in COLUMNS}

    def get(tag, split, key):
        d = data.get(tag)
        if not d:
            return None
        r = next((r for r in d["results"] if r["split"] == split), None)
        return r[key] if r else None

    for split in ("val", "test"):
        for key, label, spec in [
                ("mean_nats_per_token", f"{split} cross-entropy (nats/token)", ".4f"),
                ("perplexity", f"{split} perplexity", ".2f"),
                ("bits_per_byte", f"**{split} bits-per-byte (BPB)**", ".4f")]:
            vals = [fmt(get(t, split, key), spec) for t, _ in COLUMNS]
            if key == "bits_per_byte":
                vals = [f"**{v}**" for v in vals]
            out.append(row(label, vals))

    out.append("")
    out += ["**Cost of removing positions** (ablated − standard, test split):", "",
            "| | Δ cross-entropy (nats) | Δ perplexity | Δ BPB | relative PPL |",
            "|---|---|---|---|---|"]
    for i, j in PAIRS:
        s, a = COLUMNS[i][0], COLUMNS[j][0]
        ce_s, ce_a = get(s, "test", "mean_nats_per_token"), get(a, "test", "mean_nats_per_token")
        pp_s, pp_a = get(s, "test", "perplexity"), get(a, "test", "perplexity")
        bp_s, bp_a = get(s, "test", "bits_per_byte"), get(a, "test", "bits_per_byte")
        rel = "—" if not (pp_s and pp_a) else f"×{pp_a / pp_s:.2f}"
        out.append(row(COLUMNS[j][1],
                       [delta(ce_s, ce_a), delta(pp_s, pp_a, "+.2f"),
                        delta(bp_s, bp_a), rel]))
    out += ["", "> BPB and perplexity move together here because both columns "
            "use the **same tokenizer and the same language** — the usual "
            "caveat about PPL not being portable applies across languages, not "
            "within this ablation.", ""]
    return out


def generation_table(metric: str, label: str, spec: str = ".3f") -> list[str]:
    out = [f"**{label}**", ""] + head(LABELS)
    data = {t: load(t, "generation.json") for t, _ in COLUMNS}
    ref = next((d for d in data.values() if d), None)
    if ref is None:
        return out + ["_generation.json missing_", ""]
    for s in [r["setting"] for r in ref["results"]]:
        vals = []
        for t, _ in COLUMNS:
            d = data.get(t)
            r = next((r for r in d["results"] if r["setting"] == s), None) if d else None
            vals.append(fmt(r[metric] if r else None, spec))
        out.append(row(s, vals))
    return out + [""]


def attention_tables() -> list[str]:
    """Per-layer profile and head specialisation, standard vs ablated.

    This is where the ablation's mechanism should be visible. Two predictions
    worth checking against the numbers rather than asserting:

      * mean attention distance -- a model with no positional table cannot
        implement a fixed "look back k tokens" head directly, so purely
        positional short-range heads have no cheap way to exist.
      * content-dependence -- with position unavailable, heads that survive
        must key on token identity, so the distribution should shift upward.
    """
    data = {t: load(t, "attention.json") for t, _ in COLUMNS}
    out: list[str] = []

    for key, label, spec in [("entropy_normalised", "normalised attention entropy", ".3f"),
                             ("mean_distance", "mean attention distance (tokens)", ".2f")]:
        out += [f"### Attention: {label} by layer", ""] + head(LABELS)
        layers = sorted({int(k) for d in data.values() if d for k in d["per_layer"]})
        for layer in layers:
            vals = []
            for t, _ in COLUMNS:
                d = data.get(t)
                v = d["per_layer"].get(str(layer), {}).get(key) if d else None
                vals.append(fmt(v, spec))
            out.append(row(f"layer {layer}", vals))
        # Whole-model mean, so the headline direction is one number not ten.
        means = []
        for t, _ in COLUMNS:
            d = data.get(t)
            if not d:
                means.append(None); continue
            vs = [v[key] for v in d["per_layer"].values() if v.get(key) is not None]
            means.append(sum(vs) / len(vs) if vs else None)
        out.append(row("**all layers (mean)**", [f"**{fmt(m, spec)}**" for m in means]))
        out.append("")

    out += ["### Attention: head specialisation", ""] + head(LABELS)
    for label, key, rev, spec in [
            ("most local (min mean distance)", "mean_distance", False, ".2f"),
            ("most distant (max mean distance)", "mean_distance", True, ".2f"),
            ("sharpest (min normalised entropy)", "entropy_normalised", False, ".3f"),
            ("most positional (min content-dependence)", "content_dependence", False, ".5f"),
            ("most content-based (max content-dependence)", "content_dependence", True, ".5f")]:
        cells = []
        for t, _ in COLUMNS:
            d = data.get(t)
            if not d or not d.get("per_head") or d["per_head"][0].get(key) is None:
                cells.append("—"); continue
            h = sorted(d["per_head"], key=lambda x: x[key], reverse=rev)[0]
            cells.append(f"L{h['layer']}H{h['head']} ({fmt(h[key], spec)})")
        out.append(row(label, cells))
    out.append("")

    # Content-dependence distribution: the sharpest single diagnostic.
    out += ["### Content-dependence across all heads", "",
            "Attention change when token identities are permuted within the "
            "sequence. A head near zero is reading *position*; a high value "
            "means it is reading *what the tokens are*.", ""] + head(LABELS)
    for stat, spec in [("mean", ".5f"), ("median", ".5f"), ("min", ".5f"), ("max", ".5f")]:
        vals = []
        for t, _ in COLUMNS:
            d = data.get(t)
            if not d or not d.get("per_head"):
                vals.append("—"); continue
            xs = sorted(h["content_dependence"] for h in d["per_head"]
                        if h.get("content_dependence") is not None)
            if not xs:
                vals.append("—"); continue
            v = {"mean": sum(xs) / len(xs), "median": xs[len(xs) // 2],
                 "min": xs[0], "max": xs[-1]}[stat]
            vals.append(fmt(v, spec))
        out.append(row(stat, vals))
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the bonus ablation tables.")
    ap.add_argument("--out", default=str(PH2 / "bonus_ablation_tables.md"))
    args = ap.parse_args()

    missing = [t for t, _ in COLUMNS if not (PH2 / t / "lm_metrics.json").exists()]
    doc = ["# Bonus ablation — Model L without positional embeddings", "",
           "Generated by `scripts/build_ablation_tables.py` from the JSON "
           "results in this directory. Do not edit by hand: re-run the script, "
           "so the tables cannot drift from their evidence.", ""]
    if missing:
        doc += [f"> **Incomplete run.** No `lm_metrics.json` for: "
                f"{', '.join(missing)}. Columns show `—`.", ""]
    doc += setup_table()
    doc += lm_table()
    doc += ["### Generation quality", "",
            "Held-out test prompts (64 tokens), up to 128 new tokens, scored "
            "against the true continuation of the same document.", ""]
    for metric, label, spec in [
            ("bleu4", "BLEU-4 (corpus-level, `intl` tokenizer)", ".2f"),
            ("chrf", "chrF", ".2f"),
            ("chrf_pp", "chrF++", ".2f"),
            ("rouge_l_f", "ROUGE-L (F, LCS over whitespace tokens)", ".4f"),
            ("repetition_4gram", "4-gram repetition rate", ".3f"),
            ("distinct_1", "Distinct-1", ".3f"),
            ("distinct_2", "Distinct-2", ".3f"),
            ("prompt_copy_4gram", "prompt-copy rate (4-gram)", ".3f"),
            ("devanagari_ratio", "Devanagari character ratio", ".3f"),
            ("danda_per_100w", "danda per 100 words", ".2f")]:
        doc += generation_table(metric, label, spec)
    doc += attention_tables()

    out = Path(args.out)
    out.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(doc)} lines)"
          + (f"  [INCOMPLETE: missing {missing}]" if missing else ""))


if __name__ == "__main__":
    main()
