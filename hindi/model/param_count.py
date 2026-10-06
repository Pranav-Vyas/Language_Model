"""Parameter accounting and the depth/width justification (T2.6).

Duplicated verbatim into every language tree; never imports from a sibling.

The spec asks for the exact trainable parameter count per model and a brief
justification of the depth/width trade-off chosen against the compute budget.
This module computes both, so the numbers in the report are generated rather
than transcribed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformer import ModelConfig, TransformerLM  # noqa: E402


def analytic_total(cfg: ModelConfig) -> int:
    """Closed-form parameter count, independent of the built module.

    Kept separate from `model.num_parameters()` on purpose: two derivations
    that agree are evidence, one number is an assertion. A mismatch means the
    architecture is not what the arithmetic assumes.
    """
    emb = cfg.vocab_size * cfg.d_model
    if cfg.use_positional_embeddings:
        emb += cfg.max_seq_len * cfg.d_model
    attn = 4 * cfg.d_model ** 2                       # W_q, W_k, W_v, W_o (no bias)
    ffn = 2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model   # with biases
    lns = 2 * 2 * cfg.d_model                          # 2 LayerNorms, weight+bias
    blocks = (attn + ffn + lns) * cfg.n_layers
    final_ln = 2 * cfg.d_model
    head = 0 if cfg.tie_weights else cfg.vocab_size * cfg.d_model
    return emb + blocks + final_ln + head


def sweep(vocab: int, budget: int = 25_000_000) -> list[dict]:
    """Candidate shapes at roughly the target budget, for the trade-off table."""
    out = []
    for d_model, n_layers, n_heads in [
        (256, 12, 4), (320, 10, 5), (384, 8, 6), (384, 10, 6),
        (384, 12, 6), (448, 8, 7), (512, 5, 8), (512, 6, 8),
    ]:
        cfg = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                          n_heads=n_heads, d_ff=4 * d_model)
        total = analytic_total(cfg)
        emb = vocab * d_model
        out.append({
            "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
            "d_k": d_model // n_heads, "total": total,
            "embedding_share": emb / total,
            "delta_from_budget": total - budget,
        })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Parameter counts and shape sweep.")
    ap.add_argument("--vocab-size", type=int, default=16_000)
    ap.add_argument("--sweep", action="store_true", help="print the shape trade-off table")
    args = ap.parse_args(argv)

    if args.sweep:
        print(f"{'d_model':>8s} {'layers':>7s} {'heads':>6s} {'d_k':>4s} "
              f"{'total':>12s} {'emb share':>10s} {'vs 25M':>10s}")
        for r in sweep(args.vocab_size):
            print(f"{r['d_model']:>8d} {r['n_layers']:>7d} {r['n_heads']:>6d} "
                  f"{r['d_k']:>4d} {r['total']:>12,} {r['embedding_share']*100:>9.1f}% "
                  f"{r['delta_from_budget']:>+10,}")
        print()

    cfg = ModelConfig(vocab_size=args.vocab_size)
    model = TransformerLM(cfg)
    counted = model.num_parameters()
    analytic = analytic_total(cfg)

    print("LOCKED CONFIGURATION")
    for k, v in cfg.to_dict().items():
        if k != "lang_code":
            print(f"  {k:16s} {v}")

    print("\nPARAMETER BREAKDOWN")
    for k, v in model.parameter_breakdown().items():
        print(f"  {k:20s} {v:>12,}")

    print(f"\n  analytic total   {analytic:>12,}")
    print(f"  module total     {counted:>12,}")
    print(f"  agreement        {'MATCH' if analytic == counted else 'MISMATCH'}")
    if analytic != counted:
        raise SystemExit("analytic and module parameter counts disagree")
    print(f"  vs 25M budget    {counted - 25_000_000:>+12,} "
          f"({counted / 25_000_000 * 100:.1f}% of target)")


if __name__ == "__main__":
    main()
