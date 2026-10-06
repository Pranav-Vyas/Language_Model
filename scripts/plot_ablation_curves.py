"""Overlay the standard and no-PE training curves (BONUS ablation).

Language-agnostic tooling: reads one tree's logs, holds no language data.

`<lang>/eval/run_eval.py` plots a loss curve only into a directory that IS the
tree's own run (`out.name == TREE.name`), which deliberately excludes the
ablation's output directory. That guard exists for a good reason -- it stopped
a milestone checkpoint being given its parent run's full curve -- and the
figure worth having here is different anyway: BOTH runs on one axis, so the
cost of removing positions is read off directly rather than by flipping
between two plots.

    python scripts/plot_ablation_curves.py \
        --nope-history  kaggle_nope_out/checkpoints/nepali_nope/history.jsonl \
        --nope-log      kaggle_nope_out/lma-bonus-nope.log

Computation and presentation stay separate: `parse_val_curve` returns data,
`plot` draws it. Every axis is labelled and the legend is mandatory -- an
unlabelled figure scores zero under the spec.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "report" / "phase2" / "training"

# "  [val] step 1,250 loss 4.6229 ppl 101.8" -- the thousands separator is why
# this is a regex and not a split().
VAL_RE = re.compile(r"\[val\]\s+step\s+([\d,]+)\s+loss\s+([\d.]+)")
FINAL_RE = re.compile(r"\[final\]\s+step\s+([\d,]+)\s+val_loss\s+([\d.]+)")


def parse_val_curve(log_text: str) -> list[dict]:
    """Extract [(step, val_loss)] from a training log. Pure function, no I/O."""
    out = [{"step": int(s.replace(",", "")), "val_loss": float(v)}
           for s, v in VAL_RE.findall(log_text)]
    out += [{"step": int(s.replace(",", "")), "val_loss": float(v)}
            for s, v in FINAL_RE.findall(log_text)]
    # De-duplicate on step (the final line often repeats the last periodic one)
    # keeping the first, then order by step.
    seen: dict[int, float] = {}
    for r in out:
        seen.setdefault(r["step"], r["val_loss"])
    return [{"step": k, "val_loss": seen[k]} for k in sorted(seen)]


def load_history(path: Path) -> list[dict]:
    """Training-loss records written by loop.py as one JSON object per line."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def plot(std_hist, std_val, abl_hist, abl_val, out_path: Path,
         title: str, abl_label: str) -> Path:
    """Draw both runs on shared axes. Takes data, returns a path -- no compute."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot([h["step"] for h in std_hist], [h["loss"] for h in std_hist],
            lw=0.8, alpha=0.75, color="tab:blue", label="standard (with positions)")
    if abl_hist:
        ax.plot([h["step"] for h in abl_hist], [h["loss"] for h in abl_hist],
                lw=0.8, alpha=0.75, color="tab:red", label=abl_label)
    ax.set_title("Training loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([v["step"] for v in std_val], [v["val_loss"] for v in std_val],
            marker="o", ms=2.5, lw=1.2, color="tab:blue",
            label="standard (with positions)")
    if abl_val:
        ax.plot([v["step"] for v in abl_val], [v["val_loss"] for v in abl_val],
                marker="o", ms=2.5, lw=1.2, color="tab:red", label=abl_label)
        # Annotate the final gap: the single number the figure exists to show.
        if std_val and abl_val:
            gap = abl_val[-1]["val_loss"] - std_val[-1]["val_loss"]
            ax.annotate(f"final gap {gap:+.4f} nats",
                        xy=(abl_val[-1]["step"], abl_val[-1]["val_loss"]),
                        xytext=(0.42, 0.80), textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    ax.set_title("Validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Structural guard, same rule the per-tree viz helper enforces: an
    # unlabelled figure can score zero, so refuse to claim success without them.
    for a in axes:
        assert a.get_title() and a.get_xlabel() and a.get_ylabel() and a.get_legend()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay standard vs no-PE curves.")
    ap.add_argument("--tree", default="nepali")
    ap.add_argument("--nope-history", required=True,
                    help="history.jsonl from the ablation's checkpoint dir")
    ap.add_argument("--nope-log", required=True,
                    help="kernel log containing the ablation's [val] lines")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    std_hist = load_history(TRAINING / f"{args.tree}_train_history.jsonl")
    std_val = json.loads((TRAINING / f"{args.tree}_val_curve.json").read_text())

    abl_hist = load_history(Path(args.nope_history))
    abl_val = parse_val_curve(Path(args.nope_log).read_text(encoding="utf-8",
                                                            errors="replace"))
    print(f"[std] {len(std_hist)} train points, {len(std_val)} val points")
    print(f"[abl] {len(abl_hist)} train points, {len(abl_val)} val points")
    if not abl_val:
        raise SystemExit("no [val] lines found in the ablation log — wrong file?")

    # Persist the parsed curve so the figure is reproducible without re-parsing
    # a log, matching how the standard run's curve is stored.
    vc = TRAINING / f"{args.tree}_nope_val_curve.json"
    vc.write_text(json.dumps(abl_val, indent=2) + "\n", encoding="utf-8")
    hc = TRAINING / f"{args.tree}_nope_train_history.jsonl"
    hc.write_text("\n".join(json.dumps(r) for r in abl_hist) + "\n", encoding="utf-8")
    print(f"[write] {vc}\n[write] {hc}")

    out = Path(args.out) if args.out else (
        REPO / "report" / "phase2" / f"{args.tree}_nope" / "loss_curves_ablation.png")
    p = plot(std_hist, std_val, abl_hist, abl_val, out,
             f"Model L ({args.tree}): with vs without positional embeddings",
             "no positional embeddings")
    print(f"[write] {p}")
    if std_val and abl_val:
        print(f"[gap] final val: standard {std_val[-1]['val_loss']:.4f} "
              f"vs no-PE {abl_val[-1]['val_loss']:.4f} "
              f"({abl_val[-1]['val_loss'] - std_val[-1]['val_loss']:+.4f} nats)")


if __name__ == "__main__":
    main()
