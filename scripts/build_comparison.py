"""Generate the side-by-side Model H / Model L tables from the result JSONs (T2.15).

Language-agnostic tooling: it reads both trees' outputs but holds no language
data and never mixes their corpora.

WHY THIS IS A GENERATOR AND NOT HAND-WRITTEN MARKDOWN
-----------------------------------------------------
Every figure in the tables below traces to a file under report/phase2/. Typing
them into the report by hand works exactly once: the moment anything is
re-evaluated the prose silently disagrees with its own evidence. That already
happened on this project -- the Phase-2 README quoted validation losses sampled
during training (40 batches) while the committed lm_metrics.json held the
full-split values, and the two differed by 0.012 nats on Nepali.

Running this regenerates the tables from the JSONs, so a stale number is a
missing re-run rather than an undetectable typo.

    python scripts/build_comparison.py            # writes report/phase2/comparison_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PH2 = REPO / "report" / "phase2"

# (directory, column label). nepali_matched is Nepali stopped at Hindi's exact
# token budget, so it belongs beside the other two rather than in its own table.
COLUMNS = [("hindi", "Model H (Hindi)"),
           ("hindi_baseline", "Model H (pre-tuning)"),
           ("nepali", "Model L (Nepali)"),
           ("nepali_matched", "Model L @ matched tokens")]


def load(tag: str, name: str) -> dict | None:
    p = PH2 / tag / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row(label: str, values: list[str]) -> str:
    return f"| {label} | " + " | ".join(values) + " |"


def header(cols: list[str]) -> list[str]:
    return [row("", cols), "|" + "---|" * (len(cols) + 1)]


def fmt(v, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def lm_table() -> list[str]:
    """Intrinsic metrics. BPB is the only cross-language-valid row."""
    labels = [lab for _, lab in COLUMNS]
    out = ["### Intrinsic language-modelling metrics", "",
           row("", labels), "|" + "---|" * (len(labels) + 1)]
    data = {t: load(t, "lm_metrics.json") for t, _ in COLUMNS}

    def get(tag, split, key):
        d = data.get(tag)
        if not d:
            return None
        r = next((r for r in d["results"] if r["split"] == split), None)
        return r[key] if r else None

    for split in ("val", "test"):
        out.append(row(f"{split} cross-entropy (nats/token)",
                       [fmt(get(t, split, "mean_nats_per_token")) for t, _ in COLUMNS]))
        out.append(row(f"{split} perplexity",
                       [fmt(get(t, split, "perplexity"), ".2f") for t, _ in COLUMNS]))
        out.append(row(f"**{split} bits-per-byte (BPB)**",
                       [f"**{fmt(get(t, split, 'bits_per_byte'))}**" for t, _ in COLUMNS]))
    out.append(row("tokens scored (test)",
                   [fmt(get(t, "test", "tokens_scored"), ",") for t, _ in COLUMNS]))
    out.append(row("UTF-8 bytes scored (test)",
                   [fmt(get(t, "test", "bytes_utf8"), ",") for t, _ in COLUMNS]))
    out += ["",
            "> Perplexity is **per token** and the two models use different "
            "tokenizers, so the PPL column is comparable *within* a model only. "
            "BPB normalises by bytes of text and is the cross-language "
            "comparator; note it **reverses** the PPL ordering."]
    return out


def generation_table(metric: str, label: str, spec: str = ".3f") -> list[str]:
    labels = [lab for _, lab in COLUMNS]
    out = [f"**{label}**", "",
           row("decoding", labels), "|" + "---|" * (len(labels) + 1)]
    data = {t: load(t, "generation.json") for t, _ in COLUMNS}
    settings = [r["setting"] for r in (data["hindi"] or {"results": []})["results"]]
    for s in settings:
        vals = []
        for t, _ in COLUMNS:
            d = data.get(t)
            r = next((r for r in d["results"] if r["setting"] == s), None) if d else None
            vals.append(fmt(r[metric] if r else None, spec))
        out.append(row(s, vals))
    out.append("")
    return out


def attention_table() -> list[str]:
    out = ["### Attention: layer profile", "",
           "| layer | H entropy (norm) | H distance | L entropy (norm) | L distance |",
           "|---|---|---|---|---|"]
    hi, ne = load("hindi", "attention.json"), load("nepali", "attention.json")
    if not (hi and ne):
        return out + ["_attention.json missing_", ""]
    hl = {int(k): v for k, v in hi["per_layer"].items()}
    nl = {int(k): v for k, v in ne["per_layer"].items()}
    for layer in sorted(hl):
        out.append(row(str(layer), [
            fmt(hl[layer]["entropy_normalised"], ".3f"),
            fmt(hl[layer]["mean_distance"], ".2f"),
            fmt(nl.get(layer, {}).get("entropy_normalised"), ".3f"),
            fmt(nl.get(layer, {}).get("mean_distance"), ".2f")]))
    out.append("")

    out += ["### Attention: most specialised heads", "",
            "| | Model H (Hindi) | Model L (Nepali) |", "|---|---|---|"]
    def extreme(d, key, reverse=False):
        h = sorted(d["per_head"], key=lambda x: x[key], reverse=reverse)[0]
        return h
    for label, key, rev, spec in [
            ("most local (min mean distance)", "mean_distance", False, ".2f"),
            ("most distant (max mean distance)", "mean_distance", True, ".2f"),
            ("sharpest (min normalised entropy)", "entropy_normalised", False, ".3f"),
            ("most positional (min content-dependence)", "content_dependence", False, ".5f"),
            ("most content-based (max content-dependence)", "content_dependence", True, ".5f")]:
        cells = []
        for d in (hi, ne):
            if not d["per_head"] or d["per_head"][0].get(key) is None:
                cells.append("—"); continue
            h = extreme(d, key, rev)
            cells.append(f"L{h['layer']}H{h['head']} ({fmt(h[key], spec)})")
        out.append(row(label, cells))
    out.append("")
    return out


def vocab_table() -> list[str]:
    """Perplexity of the FINAL Hindi model under other vocabulary sizes.

    Derived, not typed: perplexity is per-token, so it moves with the
    tokenizer. Holding BPB fixed (the tokenizer-invariant quantity),

        loss = BPB x ln(2) x bytes_per_token

    and bytes_per_token scales with the chars/token that Phase 1 measured for
    every vocabulary in its sweep. Regenerated here because the model's BPB
    changes whenever it is retrained, and a hand-written version of this table
    would silently keep the old model's numbers.
    """
    import math
    lm = load("hindi", "lm_metrics.json")
    if not lm:
        return []
    t = next(r for r in lm["results"] if r["split"] == "test")
    bpb = t["bits_per_byte"]
    bpt = t["bytes_utf8"] / t["tokens_scored"]
    sweep_p = REPO / "hindi" / "corpus" / "stats" / "tokenizer_sweep.json"
    if not sweep_p.exists():
        return []
    sw = json.loads(sweep_p.read_text())
    rows_ = sw if isinstance(sw, list) else sw.get("sweep", sw.get("results", []))
    cpt = {r.get("vocab_size") or r.get("vocab"): r["chars_per_token"] for r in rows_}
    base = cpt[16000]

    out = ["### Perplexity of Model H under other vocabulary sizes", "",
           f"Measured: BPB **{bpb:.4f}**, bytes/token **{bpt:.4f}**, and "
           f"`BPB x ln2 x bytes/token = {bpb*math.log(2)*bpt:.4f}` reproduces the "
           f"measured test loss exactly.", "",
           "| vocab | chars/token | bytes/token | loss | perplexity |",
           "|---|---|---|---|---|"]
    for v in sorted(cpt):
        scaled = bpt * cpt[v] / base
        l = bpb * math.log(2) * scaled
        mark = " **(ours)**" if v == 16000 else ""
        out.append(f"| {v:,}{mark} | {cpt[v]:.3f} | {scaled:.3f} | {l:.4f} | "
                   f"**{math.exp(l):.1f}** |")
    for v, c in ((7000, cpt[8000]*0.975), (6000, cpt[8000]*0.95), (5000, cpt[8000]*0.92)):
        l = bpb * math.log(2) * bpt * (c / base)
        out.append(f"| {v:,} *(extrapolated)* | ~{c:.3f} | — | {l:.4f} | ~{math.exp(l):.1f} |")
    out += ["", "> The model is identical in every row; only the unit of the "
            "denominator changes. A larger vocabulary packs more text into each "
            "token, making each prediction harder and perplexity higher, while "
            "the cost per byte is unchanged.", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase-2 H/L comparison tables.")
    ap.add_argument("--out", default=str(PH2 / "comparison_tables.md"))
    args = ap.parse_args()

    doc = ["# Phase 2 — Model H vs Model L, side by side", "",
           "Generated by `scripts/build_comparison.py` from the JSON results in "
           "this directory. Do not edit by hand: re-run the script instead, so "
           "the tables cannot drift from their evidence.", ""]
    doc += lm_table() + [""]
    doc += vocab_table()
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
    doc += attention_table()

    out = Path(args.out)
    out.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(doc)} lines)")


if __name__ == "__main__":
    main()
