"""Plotting for Phase-2 evaluation. Presentation only -- no metric is computed here.

Duplicated verbatim into every language tree; never imports from a sibling.

Every function takes already-computed data and draws it. That separation is
deliberate: a number buried inside a plotting call cannot be unit-tested or
reused, and the spec grades the numbers as well as the figures.

`_finish` REFUSES to save a figure missing a title, x-label or y-label,
because the spec scores an unlabelled plot as zero. Making that structural
means it cannot be forgotten under deadline pressure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: these run on Kaggle and in CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _use_devanagari_font() -> str | None:
    """Prefer a Devanagari-capable font for tick labels.

    matplotlib's default (DejaVu Sans) has no Devanagari glyphs, so every
    token label on an attention heatmap renders as an empty box -- a figure
    that looks broken in the report and tells a grader nothing. Falls back
    silently if no such font is installed, since the numeric plots do not
    need one.
    """
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans Devanagari", "Nirmala UI", "Lohit Devanagari",
                 "Mangal", "Kalimati", "Samyak Devanagari"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name
    return None


DEVANAGARI_FONT = _use_devanagari_font()


def _finish(fig, ax, out_path: str | Path, legend: bool = False) -> Path:
    """Assert the figure is labelled, then save it."""
    axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
    for a in np.ravel(axes):
        if not a.get_title():
            raise ValueError(f"figure for {out_path} has an axis with no title")
        if not a.get_xlabel():
            raise ValueError(f"figure for {out_path} has an axis with no x-label")
        if not a.get_ylabel():
            raise ValueError(f"figure for {out_path} has an axis with no y-label")
        if legend and a.get_legend() is None:
            a.legend()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_loss_curves(history: list[dict], val_curve: list[dict], lang: str,
                     out_path) -> Path:
    """Training loss and validation loss against optimizer step."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([h["step"] for h in history], [h["loss"] for h in history],
            lw=0.8, alpha=0.6, label="training loss (per logged step)")
    ax.plot([v["step"] for v in val_curve], [v["val_loss"] for v in val_curve],
            "o-", ms=3, lw=1.5, color="crimson", label="validation loss")
    ax.set_title(f"{lang}: pretraining loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.grid(alpha=0.3)
    return _finish(fig, ax, out_path, legend=True)


def plot_attention_heatmap(attn: np.ndarray, layer: int, heads: list[int],
                           tokens: list[str], lang: str, out_path) -> Path:
    """Heatmaps of several heads from one layer.

    attn : (heads, T, T) for a single sequence, already on CPU as numpy.
    """
    n = len(heads)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    T = attn.shape[-1]
    step = max(1, T // 12)
    for k, head in enumerate(heads):
        a = axes[0][k]
        im = a.imshow(attn[head], cmap="viridis", aspect="auto",
                      vmin=0.0, vmax=float(attn[head].max()))
        a.set_title(f"{lang}: layer {layer}, head {head}")
        a.set_xlabel("key position (attended to)")
        a.set_ylabel("query position (attending from)")
        if tokens:
            a.set_xticks(range(0, T, step))
            a.set_xticklabels([tokens[i] for i in range(0, T, step)],
                              rotation=90, fontsize=6)
            a.set_yticks(range(0, T, step))
            a.set_yticklabels([tokens[i] for i in range(0, T, step)], fontsize=6)
        fig.colorbar(im, ax=a, fraction=0.046, label="attention weight")
    return _finish(fig, axes, out_path)


def plot_head_grid(stats: list[dict], metric: str, ylabel: str, title: str,
                   out_path) -> Path:
    """Per-head values of one metric, as a layer x head grid."""
    layers = sorted({s["layer"] for s in stats})
    heads = sorted({s["head"] for s in stats})
    grid = np.full((len(layers), len(heads)), np.nan)
    for s in stats:
        grid[layers.index(s["layer"]), heads.index(s["head"])] = s[metric]

    fig, ax = plt.subplots(figsize=(1.1 * len(heads) + 3, 0.45 * len(layers) + 2.5))
    im = ax.imshow(grid, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("head index")
    ax.set_ylabel("layer index (0 = earliest)")
    ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    for i in range(len(layers)):
        for j in range(len(heads)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white")
    fig.colorbar(im, ax=ax, label=ylabel)
    return _finish(fig, ax, out_path)


def plot_layer_profile(summary: dict, lang: str, out_path) -> Path:
    """Entropy and attention distance against depth -- the early-vs-late view."""
    layers = sorted(summary.keys(), key=int)
    ent = [summary[l]["entropy_normalised"] for l in layers]
    dist = [summary[l]["mean_distance"] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(layers, ent, "o-", color="steelblue", label="normalised entropy")
    ax.set_title(f"{lang}: attention entropy and distance by depth")
    ax.set_xlabel("layer index (0 = earliest)")
    ax.set_ylabel("normalised entropy (1.0 = uniform)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(layers, dist, "s--", color="darkorange", label="mean attention distance")
    ax2.set_ylabel("mean attention distance (tokens)")
    # twinx axes need no title/xlabel of their own; merge the legends.
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="best")
    return _finish(fig, ax, out_path)


def plot_decoding_comparison(rows: list[dict], metric: str, ylabel: str,
                             title: str, out_path) -> Path:
    """One metric across decoding settings (greedy, temperatures, top-p)."""
    settings = [r["setting"] for r in rows]
    vals = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(settings, vals, color="steelblue")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("decoding setting")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _finish(fig, ax, out_path)


# --------------------------------------------------------------------------
# Phase 3 (T3.10): pretrained-vs-finetuned attention comparison.
# Additive -- none of the Phase-2 functions above change, so the Phase-2
# figures still regenerate byte-for-byte from the same calls.
# --------------------------------------------------------------------------


def plot_layer_profile_compare(summaries: dict, metric: str, ylabel: str,
                               title: str, out_path) -> Path:
    """One layer-level metric against depth, one line per model.

    summaries : {model_label: {layer: {metric: value}}}, as produced by
                attention_analysis.summarise_by_layer.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = [("o-", "steelblue"), ("s--", "darkorange"), ("^:", "seagreen")]
    for (label, summary), (mk, col) in zip(summaries.items(), styles):
        layers = sorted(summary.keys(), key=int)
        ax.plot(layers, [summary[l][metric] for l in layers], mk,
                color=col, label=label)
    ax.set_title(title)
    ax.set_xlabel("layer index (0 = earliest)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return _finish(fig, ax, out_path, legend=True)


def plot_head_delta_grid(deltas: list[dict], metric: str, ylabel: str,
                         title: str, out_path) -> Path:
    """Per-head CHANGE in one metric, as a layer x head grid.

    Uses a diverging colormap centred on zero so the sign is readable at a
    glance: `magma` (used for the Phase-2 absolute grids) would render "no
    change" as an arbitrary mid-tone.
    """
    layers = sorted({d["layer"] for d in deltas})
    heads = sorted({d["head"] for d in deltas})
    grid = np.full((len(layers), len(heads)), np.nan)
    for d in deltas:
        grid[layers.index(d["layer"]), heads.index(d["head"])] = d[metric]

    lim = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(1.1 * len(heads) + 3, 0.45 * len(layers) + 2.5))
    im = ax.imshow(grid, cmap="coolwarm", aspect="auto", vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xlabel("head index")
    ax.set_ylabel("layer index (0 = earliest)")
    ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    for i in range(len(layers)):
        for j in range(len(heads)):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label=ylabel)
    return _finish(fig, ax, out_path)


def plot_answer_attention(rows: list[dict], chance: float, title: str,
                          out_path) -> Path:
    """Share of candidate-entity attention landing on the GOLD entity.

    rows : [{"model": str, "layer_label": str, "share": float, "sem": float}]
           grouped into one bar cluster per layer_label.
    """
    labels = list(dict.fromkeys(r["layer_label"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4.2))
    for k, model in enumerate(models):
        vals = [next((r["share"] for r in rows
                      if r["model"] == model and r["layer_label"] == lab), np.nan)
                for lab in labels]
        errs = [next((r.get("sem", 0.0) for r in rows
                      if r["model"] == model and r["layer_label"] == lab), 0.0)
                for lab in labels]
        bars = ax.bar(x + k * width - 0.4 + width / 2, vals, width,
                      yerr=errs, capsize=3, label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.axhline(chance, color="crimson", linestyle=":", linewidth=1.5,
               label=f"chance ({chance:.2f})")
    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel("share of candidate attention on gold entity")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=8)
    return _finish(fig, ax, out_path, legend=True)
