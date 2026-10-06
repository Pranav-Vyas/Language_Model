"""Batching for supervised finetuning on the reasoning set (T3.6).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY THIS IS NOT `TokenShardLoader`
----------------------------------
Pretraining reads a contiguous uint16 memmap and slices fixed 512-token windows;
every position is a training target. Finetuning has none of those properties:
examples are short (15-40 tokens measured), variable length, and only the ANSWER
span should contribute to the loss. So this is a separate loader rather than a
flag on the old one.

ANSWER-SPAN MASKING
-------------------
`labels` is the input shifted by one, with every prompt position replaced by
-100. `Transformer.forward` already passes `ignore_index=-100` to
`cross_entropy`, so no model change is needed -- the masking is entirely a
property of the data.

Why mask at all: the prompt is template-generated and trivially predictable, so
letting it into the loss spends gradient on modelling our own templates rather
than on producing the answer. That is the standard SFT argument, and because it
is an argument rather than a proof, `mask_prompt=False` is supported and the
choice is settled by an A/B rather than asserted.

DYNAMIC PADDING
---------------
Each batch is padded to its own longest example, not to a global maximum. With
lengths spanning 15-40 tokens a global pad would spend roughly half the compute
on padding. Padding positions are masked out of the loss too, so they cannot
contribute gradient regardless.

DETERMINISM
-----------
Example order is a pure function of (seed, epoch): `_order(epoch)` reshuffles
with a seed derived from both. No RNG state has to be checkpointed for a resume
to see the same batches, matching the pretraining loader's contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

IGNORE_INDEX = -100      # what F.cross_entropy skips; matches model/transformer.py


class SFTDataset:
    """Tokenised (prompt, target) pairs with answer-span label masking.

    Shapes
    ------
    batch(i) -> x (B, T) int64, y (B, T) int64, where y[b, t] is the token the
    model must predict at position t, or IGNORE_INDEX if that position is
    prompt or padding.
    """

    def __init__(self, path: str | Path, sp, *, max_len: int = 128,
                 target_field: str = "target_answer_only",
                 mask_prompt: bool = True, pad_id: int | None = None):
        self.path = Path(path)
        self.sp = sp
        self.max_len = max_len
        self.target_field = target_field
        self.mask_prompt = mask_prompt
        self.pad_id = sp.pad_id() if pad_id is None else pad_id
        self.eos_id = sp.eos_id()

        self.rows: list[dict] = []
        self.examples: list[tuple[list[int], int]] = []   # (ids, n_prompt_tokens)
        n_truncated = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                p_ids = sp.encode(r["prompt"])
                t_ids = sp.encode(r[target_field]) + [self.eos_id]
                ids = p_ids + t_ids
                if len(ids) > max_len:
                    # Truncating would cut the ANSWER, silently turning a
                    # correct example into a mislabelled one. Count and drop.
                    n_truncated += 1
                    continue
                self.rows.append(r)
                self.examples.append((ids, len(p_ids)))
        self.n_truncated = n_truncated
        if not self.examples:
            raise ValueError(f"no usable examples in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def token_stats(self) -> dict:
        """Corpus-level token accounting. Respects `mask_prompt`.

        The supervised count MUST depend on the masking mode -- an earlier
        version reported the masked figure in both modes, which would have made
        the mask-vs-no-mask A/B look like it changed nothing.
        """
        lens = [len(ids) for ids, _ in self.examples]
        if self.mask_prompt:
            sup = [len(ids) - n for ids, n in self.examples]
        else:
            # Every position except the final one is a prediction target.
            sup = [len(ids) - 1 for ids, _ in self.examples]
        return {"examples": len(lens), "truncated_dropped": self.n_truncated,
                "tokens_total": sum(lens), "mask_prompt": self.mask_prompt,
                "len_mean": sum(lens) / len(lens), "len_max": max(lens),
                "supervised_tokens": sum(sup),
                "supervised_frac": sum(sup) / sum(lens)}

    def _order(self, epoch: int, seed: int) -> list[int]:
        """Deterministic example order for one epoch. Pure in (seed, epoch)."""
        g = torch.Generator().manual_seed(hash((seed, epoch)) % (2**31))
        return torch.randperm(len(self.examples), generator=g).tolist()

    def batches(self, batch_size: int, *, epoch: int = 0, seed: int = 1337,
                device="cpu", shuffle: bool = True):
        """Yield (x, y) batches, each padded to its own longest example."""
        idx = self._order(epoch, seed) if shuffle else list(range(len(self.examples)))
        for start in range(0, len(idx), batch_size):
            chunk = [self.examples[i] for i in idx[start:start + batch_size]]
            yield self._collate(chunk, device)

    def _collate(self, chunk, device):
        """Pad to the batch's own max length and build the masked labels."""
        width = max(len(ids) for ids, _ in chunk)
        B = len(chunk)
        # Inputs are the sequence minus its last token; targets are shifted by
        # one. Building both at width-1 keeps the standard next-token alignment.
        T = width - 1
        x = torch.full((B, T), self.pad_id, dtype=torch.long)
        y = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
        for b, (ids, n_prompt) in enumerate(chunk):
            n = len(ids)
            x[b, :n - 1] = torch.tensor(ids[:-1], dtype=torch.long)
            tgt = torch.tensor(ids[1:], dtype=torch.long)
            if self.mask_prompt:
                # Position t predicts ids[t+1]. That prediction is supervised
                # only once ids[t+1] is inside the target span, i.e. t+1 >=
                # n_prompt, i.e. t >= n_prompt - 1.
                keep = torch.zeros(n - 1, dtype=torch.bool)
                keep[max(n_prompt - 1, 0):] = True
                tgt = torch.where(keep, tgt, torch.full_like(tgt, IGNORE_INDEX))
            y[b, :n - 1] = tgt
        return x.to(device), y.to(device)
