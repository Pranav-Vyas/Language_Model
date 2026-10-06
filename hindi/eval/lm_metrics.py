"""Intrinsic LM evaluation: cross-entropy, perplexity, bits-per-byte (T2.12).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY BPB IS THE ONLY HONEST CROSS-LANGUAGE NUMBER
------------------------------------------------
Perplexity is `exp(mean cross-entropy per TOKEN)`, so its unit is "per token
of this tokenizer". Hindi and Nepali have separate 16,000-piece vocabularies
with different fertility (1.386 vs 1.360 tokens/word) and different characters
per token (3.892 vs 4.644). A Nepali token therefore carries more text than a
Hindi one on average, so the two perplexities are denominated in different
units and comparing them directly is meaningless -- a model could "win" purely
by having a coarser tokenizer.

Bits-per-byte removes the tokenizer from the denominator:

    BPB = (total cross-entropy in nats / ln 2) / (UTF-8 bytes of that text)

Bytes are a property of the text, not of how we chose to segment it, so BPB is
comparable across tokenizers and therefore across our two models. Every
cross-language claim in the report rests on it; PPL is reported per model only.

HOW THE CORPUS IS SCORED
------------------------
Non-overlapping windows of `ctx+1` tokens with stride `ctx`:

    window i = tokens[i*ctx : i*ctx + ctx + 1]
    x = window[:-1]   y = window[1:]

Consecutive windows score targets [1..ctx], [ctx+1..2*ctx], ... -- contiguous,
no gaps and no double counting. Token 0 is never a target (nothing precedes
it), which is a 1-in-`n_tokens` effect and is reported rather than hidden.

Early tokens in each window are predicted from little context, which makes
this a slightly pessimistic estimate compared with a stride-1 sliding window.
Stride-1 would cost `ctx` times more compute for a fraction of a nat; the
choice is stated so the number is reproducible, and it is applied identically
to both languages so the comparison stays fair.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class LMResult:
    """Intrinsic metrics for one split of one model."""

    split: str
    lang_code: str
    checkpoint: str
    tokens_scored: int
    windows: int
    total_nats: float
    mean_nats_per_token: float
    perplexity: float
    bytes_utf8: int
    bits_per_byte: float
    tokens_not_scored: int
    ctx: int

    def to_dict(self) -> dict:
        return asdict(self)


def _decode_bytes(sp, ids: np.ndarray, chunk: int = 100_000) -> int:
    """UTF-8 byte count of the text these token ids represent.

    Decoded in chunks because SentencePiece builds the whole string in memory
    and a validation split is millions of tokens. Chunk boundaries can only
    affect a leading space marker, which is well under the precision at which
    BPB is reported.
    """
    total = 0
    for start in range(0, len(ids), chunk):
        piece = ids[start:start + chunk].tolist()
        total += len(sp.decode(piece).encode("utf-8"))
    return total


@torch.no_grad()
def evaluate_split(model, sp, shard_path: str | Path, ctx: int, device,
                   autocast_ctx, batch_size: int = 8, max_windows: int | None = None,
                   split: str = "val", checkpoint: str = "") -> LMResult:
    """Score an entire packed shard and return loss / PPL / BPB.

    Sums cross-entropy in NATS over every scored token rather than averaging
    per batch: a mean of per-batch means is only correct when every batch has
    the same token count, and the final batch does not.
    """
    model.eval()
    tokens = np.memmap(Path(shard_path), dtype=np.uint16, mode="r")
    n = int(tokens.shape[0])

    starts = list(range(0, n - ctx - 1, ctx))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError(f"shard {shard_path} too short for ctx={ctx}")

    total_nats, tokens_scored = 0.0, 0
    for b0 in range(0, len(starts), batch_size):
        batch_starts = starts[b0:b0 + batch_size]
        win = np.stack([np.asarray(tokens[s:s + ctx + 1], dtype=np.int64)
                        for s in batch_starts])
        x = torch.from_numpy(win[:, :-1]).to(device)
        y = torch.from_numpy(win[:, 1:]).to(device)
        with autocast_ctx():
            logits, _ = model(x)
        # reduction="sum" in fp32: summing millions of bf16 terms would lose
        # precision, and the total is what BPB divides.
        loss_sum = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum")
        total_nats += float(loss_sum)
        tokens_scored += y.numel()

    # Bytes of exactly the tokens that were scored: targets span
    # tokens[1 .. last_start+ctx].
    last = starts[-1] + ctx
    scored_ids = np.asarray(tokens[1:last + 1], dtype=np.int64)
    n_bytes = _decode_bytes(sp, scored_ids)

    mean_nats = total_nats / tokens_scored
    return LMResult(
        split=split,
        lang_code="",
        checkpoint=checkpoint,
        tokens_scored=tokens_scored,
        windows=len(starts),
        total_nats=total_nats,
        mean_nats_per_token=mean_nats,
        perplexity=math.exp(mean_nats),
        bytes_utf8=n_bytes,
        bits_per_byte=(total_nats / math.log(2)) / n_bytes,
        tokens_not_scored=n - tokens_scored,
        ctx=ctx,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: score one checkpoint on val and test, write JSON next to the report."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (autocast_for, load_model_from_checkpoint, load_tokenizer,
                        pick_device, seed_everything, shard_path, write_json, TREE)

    ap = argparse.ArgumentParser(description="LM metrics: loss, PPL, BPB.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap windows (smoke testing); omit to score the full split")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seed_everything()
    device = pick_device(prefer_cuda=not args.cpu)
    model, cfg, meta = load_model_from_checkpoint(args.checkpoint, device)
    sp = load_tokenizer(cfg.vocab_size)
    actx = autocast_for(device)

    print(f"[{meta['lang_code']}] {Path(args.checkpoint).name}  step {meta['step']:,}  "
          f"params {model.num_parameters():,}  device {device}")

    results = []
    for split in args.splits:
        p = shard_path(split)
        if not p.exists():
            print(f"  ! {p} missing, skipping")
            continue
        r = evaluate_split(model, sp, p, cfg.max_seq_len, device, actx,
                           batch_size=args.batch_size, max_windows=args.max_windows,
                           split=split, checkpoint=Path(args.checkpoint).name)
        r.lang_code = meta["lang_code"]
        results.append(r)
        print(f"  {split:5s} tokens {r.tokens_scored:>11,}  "
              f"loss {r.mean_nats_per_token:.4f}  ppl {r.perplexity:>8.2f}  "
              f"BPB {r.bits_per_byte:.4f}  ({r.bytes_utf8:,} bytes)")

    out = Path(args.out) if args.out else \
        Path("report/phase2") / TREE.name / "lm_metrics.json"
    payload = {"lang_code": meta["lang_code"], "checkpoint": meta,
               "note": ("PPL is per-token and therefore NOT comparable across "
                        "languages; BPB is per-byte and is the cross-language "
                        "comparator."),
               "results": [r.to_dict() for r in results]}
    print(f"  -> {write_json(payload, out)}")


if __name__ == "__main__":
    main()
