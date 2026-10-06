"""Batch sampling from packed uint16 token shards.

Duplicated verbatim into every language tree; never imports from a sibling.

WHY memmap RATHER THAN LOADING THE SHARD
----------------------------------------
The Hindi train shard is 1.07 GiB and the Nepali one 1.68 GiB. Reading either
fully into RAM is possible but pointless: a training step touches only
`batch x (ctx+1)` tokens, so the OS page cache serves the hot pages and never
faults in the rest. `np.memmap` gives that for free and keeps startup instant,
which matters when a Kaggle session dies and has to be restarted.

WHY BATCHES ARE A PURE FUNCTION OF (seed, step)
-----------------------------------------------
This is the property that makes resume *exact* instead of approximate.

A conventional shuffling loader holds RNG state that must be serialised into
the checkpoint and restored perfectly; if it isn't, training silently resumes
on a different data order. Nothing raises, the loss curve looks fine, and the
run is no longer the one described in the report.

Here `batch(step)` derives its own generator from `seed + step`, so the batch
at step 5,000 is identical whether it was reached by training straight through
or by resuming from a checkpoint at step 4,999. The kill-and-resume test
(T2.8) asserts exactly this.

The cost is that sampling is with replacement rather than a true epoch
permutation. At 572M (hi) / 880M (ne) tokens against ~65k tokens per step, a
run of ~10k steps sees well under two passes; the difference between sampling
with and without replacement at that ratio is negligible, and the exact
resume property is worth far more than the small variance saving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenShardLoader:
    """Serves `(x, y)` batches of next-token training pairs from a .bin shard.

    Shapes
    ------
    x : (batch, ctx)   int64 -- input tokens
    y : (batch, ctx)   int64 -- targets, y[:, t] == x[:, t+1]

    The two are offset views of one window of length `ctx+1`, so the caller
    never has to do the shift and cannot get it backwards.
    """

    def __init__(self, shard_path: str | Path, ctx: int, batch_size: int, seed: int = 1337):
        self.path = Path(shard_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"token shard not found: {self.path}. Run data_prep/tokenize_corpus.py "
                "(or restore the shard from Drive) before training."
            )
        self.ctx = ctx
        self.batch_size = batch_size
        self.seed = seed

        # mode="r" is read-only: training must never be able to mutate the corpus.
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self.tokens.shape[0])

        # Need ctx+1 tokens to form one (input, target) pair.
        self.max_start = self.n_tokens - (ctx + 1)
        if self.max_start <= 0:
            raise ValueError(
                f"shard holds {self.n_tokens:,} tokens, too few for ctx={ctx}."
            )

    def __len__(self) -> int:
        return self.n_tokens

    def batch(self, step: int, device: torch.device | str = "cpu"):
        """Sample the batch for a given step. Deterministic in (seed, step)."""
        rng = np.random.default_rng(self.seed + step)
        starts = rng.integers(0, self.max_start, size=self.batch_size, endpoint=False)

        # uint16 -> int64 because nn.Embedding requires a long index tensor.
        # np.stack materialises only batch x (ctx+1) tokens.
        window = np.stack([
            np.asarray(self.tokens[s: s + self.ctx + 1], dtype=np.int64)
            for s in starts
        ])
        x = torch.from_numpy(window[:, :-1])
        y = torch.from_numpy(window[:, 1:])

        if str(device).startswith("cuda"):
            # pin_memory + non_blocking overlaps the host->device copy with
            # compute; on a T4 the copy is otherwise a visible stall.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def eval_batches(self, n_batches: int, device: torch.device | str = "cpu"):
        """Yield a FIXED, evenly-spaced set of batches for validation.

        Deliberately not random: a validation number that moves because the
        sample moved is not a measurement. Evenly spaced windows across the
        whole shard, identical at every evaluation, so successive val losses
        differ only because the model changed.
        """
        stride = max(1, self.max_start // max(1, n_batches * self.batch_size))
        idx = 0
        for _ in range(n_batches):
            starts = [(idx + i * stride) % self.max_start for i in range(self.batch_size)]
            idx = (starts[-1] + stride) % self.max_start
            window = np.stack([
                np.asarray(self.tokens[s: s + self.ctx + 1], dtype=np.int64)
                for s in starts
            ])
            x = torch.from_numpy(window[:, :-1]).to(device)
            y = torch.from_numpy(window[:, 1:]).to(device)
            yield x, y
