"""Training hyperparameters, loaded from YAML and logged beside checkpoints.

Duplicated verbatim into every language tree; never imports from a sibling.

Kept separate from the tree's `config.py` (which describes the *corpus*) and
from `model/transformer.py`'s `ModelConfig` (which describes the
*architecture*). Three configs because they change for three different
reasons: the corpus is frozen after Phase 1, the architecture is frozen once
parameter arithmetic is done, and the training knobs are the only ones tuned
against whatever GPU the session happens to hand us.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Every training knob. Defaults are sized for a single Tesla T4 (16GB).

    T4 SIZING NOTES (see `micro_batch_size` for the binding constraint)
    -------------------------------------------------------------------
    The memory ceiling on this model is NOT the parameters. At 24.07M params,
    weights + gradients + AdamW's two moment buffers come to roughly 385 MB in
    fp32 -- under 3% of a T4's 16GB.

    What actually fills the card is activations, and specifically two tensors:

      * attention scores/weights, shape (B, h, T, T). Our attention is written
        from primitives with no fused/flash kernel (the spec forbids one), so
        the full T x T matrix is materialised and kept for backward. At T=512,
        h=6 that is ~3.1 MB per sample per layer in bf16, times 10 layers.
      * output logits, shape (B, T, vocab) = B x 512 x 16000. At bf16 that is
        16.4 MB per sample, and cross-entropy upcasts internally.

    Both scale linearly in batch size, so batch is the knob that matters.
    `micro_batch_size=8` is set deliberately conservative: T2.9's smoke run
    reports peak memory, and the value should be raised only on that evidence,
    not on an estimate. Effective batch is restored by accumulation, which
    costs time but no memory.
    """

    # --- data ---------------------------------------------------------------
    train_shard: str = "nepali/data/tokenized/train.bin"
    val_shard: str = "nepali/data/tokenized/val.bin"
    ctx: int = 512                     # must equal model max_seq_len

    # --- batch ---------------------------------------------------------------
    micro_batch_size: int = 8          # per forward pass; the VRAM knob
    grad_accum_steps: int = 16         # 8 x 16 = 128 sequences per optimizer step
                                       # = 128 x 512 = 65,536 tokens per step

    # --- optimizer (AdamW) ---------------------------------------------------
    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1          # applied to 2D matmul weights only
    beta1: float = 0.9
    beta2: float = 0.95                # 0.95 not 0.999: shorter memory suits
                                       # the noisier gradients of a small model
    eps: float = 1e-8
    grad_clip: float = 1.0

    # --- schedule ------------------------------------------------------------
    max_steps: int = 20_000
    warmup_steps: int = 500            # insurance, not necessity -- see schedule.py

    # --- precision -----------------------------------------------------------
    # bf16 on T4: see trainer.py's autocast note. No GradScaler is used, and
    # that is a property of bf16, not an oversight.
    dtype: str = "bfloat16"

    # --- cadence -------------------------------------------------------------
    eval_every: int = 500
    eval_batches: int = 40             # fixed windows; see data.eval_batches
    ckpt_every: int = 500
    log_every: int = 10
    keep_last_n: int = 3               # Kaggle working dir is size-capped

    # Steps at which to force a checkpoint regardless of `ckpt_every`, and to
    # keep it exempt from pruning. Used for the matched-token comparison:
    # Nepali saves at step 8,732, the point where it has consumed the same
    # 572,264,559 tokens as Hindi's entire training set, so Model L can be
    # reported both at its true data scale and at Hindi's token budget.
    extra_ckpt_steps: tuple[int, ...] = ()

    # --- bookkeeping ---------------------------------------------------------
    ckpt_dir: str = "nepali/checkpoints"
    seed: int = 1337

    # Off by default: hashing the batch forces a device->host sync every step.
    # Enabled by the T2.8 kill-and-resume proof.
    record_batch_hash: bool = False

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.ctx

    def to_dict(self) -> dict:
        return asdict(self)


def load_train_config(path: str | Path, **overrides) -> TrainConfig:
    """Read a train YAML, apply CLI overrides, and reject unknown keys.

    Unknown keys are an error rather than being ignored: a typo'd
    `micro_batch` that silently leaves `micro_batch_size` at its default is
    the kind of failure that only shows up as an OOM two hours in.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {sorted(unknown)}. Known keys: {sorted(known)}"
        )
    return TrainConfig(**raw)


def load_model_config(path: str | Path, model_config_cls):
    """Read a model YAML into the tree's own `ModelConfig`, rejecting typos."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(model_config_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) in {path}: {sorted(unknown)}")
    return model_config_cls(**raw)
