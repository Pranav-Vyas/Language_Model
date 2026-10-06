"""Resume-capable checkpointing, with the language-independence gate.

Duplicated verbatim into every language tree; never imports from a sibling.

WHAT A CHECKPOINT MUST CONTAIN
------------------------------
The spec makes resume-capability a hard constraint: a run that cannot resume
after a Colab/Kaggle kill loses marks. "Resume" means the run continues as if
it had never stopped, which needs all five of:

    model weights      -- obvious
    optimizer state    -- AdamW's per-parameter m and v. Dropping these
                          restarts the moment estimates at zero, so the first
                          steps after resume take badly-scaled updates. The
                          loss visibly bumps; nothing raises.
    scheduler state   -- the spec lists this explicitly. Our schedule holds no
                          mutable state: lr(step) is a pure function, so the
                          step number is *sufficient* to reconstruct it. That
                          is a stronger guarantee than serialising a mutable
                          scheduler, not a weaker one -- there is no state to
                          restore incorrectly. A `scheduler_state` block is
                          written anyway, carrying the parameters that define
                          the curve plus the LR in force at this step, so the
                          schedule can be reconstructed or audited from the
                          checkpoint alone without consulting a config file.
    step               -- also selects the data batch (see data.py), so the
                          resumed run sees the same tokens in the same order.
    config             -- the architecture the weights belong to, so a
                          checkpoint can be rebuilt without guessing.

THE lang_code GATE
------------------
Locked project rule (§0.2, level 4): no checkpoint-loading path may accept a
checkpoint whose stored `lang_code` differs from the model being built. This
is a **hard failure, not a warning**.

The reason it is worth a dedicated guard: Hindi and Nepali share a script,
share a vocab size (16,000), and therefore share an identical tensor shape for
every parameter. A Nepali checkpoint loads into a Hindi model with no shape
error at all -- `load_state_dict` succeeds, training continues, and the run is
an independence violation that no exception would ever reveal. Shape checking
cannot catch this; only an explicit identity check can.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    """Snapshot every RNG the training step consumes."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict | None) -> bool:
    """Put the RNGs back. Returns False if the checkpoint predates RNG saving."""
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = state["torch"]
    torch.set_rng_state(cpu_state.cpu().to(torch.uint8)
                        if hasattr(cpu_state, "cpu") else cpu_state)
    if "cuda" in state and torch.cuda.is_available():
        # Only restore if the device count matches; a checkpoint written on a
        # 2-GPU box and resumed on 1 GPU must not crash here.
        saved = state["cuda"]
        if len(saved) == torch.cuda.device_count():
            # .cpu() is REQUIRED, not defensive. Loading with
            # map_location="cuda" moves every tensor in the payload onto the
            # GPU, RNG states included, but set_rng_state_all() accepts only
            # CPU ByteTensors and raises "RNG state must be a torch.ByteTensor"
            # otherwise. Found by the first chained GPU resume; a CPU-only test
            # cannot reach this branch because there is no CUDA RNG to restore.
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in saved])
    return True


class LanguageMismatchError(RuntimeError):
    """Raised when a checkpoint's language does not match the model's.

    Its own type, not a bare RuntimeError, so a caller can never swallow it
    with a broad `except RuntimeError` meant for something else.
    """


def scheduler_state(train_cfg: dict | None, step: int) -> dict:
    """The LR schedule, materialised so a checkpoint is self-describing.

    Derived rather than stored during training, because a value that is only
    ever computed cannot drift from the thing it describes.
    """
    tc = train_cfg or {}
    peak = tc.get("peak_lr")
    warm = tc.get("warmup_steps")
    total = tc.get("max_steps")
    ratio = tc.get("min_lr_ratio")
    lr_now = None
    if None not in (peak, warm, total, ratio):
        import math
        min_lr = peak * ratio
        if step < warm:
            lr_now = peak * (step + 1) / warm
        elif step >= total:
            lr_now = min_lr
        else:
            prog = (step - warm) / (total - warm)
            lr_now = min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return {"type": "warmup_cosine", "stateless": True,
            "peak_lr": peak, "warmup_steps": warm, "total_steps": total,
            "min_lr_ratio": ratio,
            "min_lr": (peak * ratio) if None not in (peak, ratio) else None,
            "step": step, "lr_at_step": lr_now}


def save_checkpoint(path: str | Path, *, model, optimizer, step: int,
                    model_cfg, train_cfg: dict | None = None,
                    val_loss: float | None = None, scaler=None) -> Path:
    """Write a resume-capable checkpoint atomically.

    Atomic because the failure this project has already hit twice is a process
    killed mid-write, leaving a truncated file. Here the payload is written to
    a temporary name and then `os.replace`d, which is atomic on POSIX: the
    destination either holds the old checkpoint or the complete new one, never
    a half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "model_cfg": asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__") else dict(model_cfg),
        "train_cfg": train_cfg or {},
        "val_loss": val_loss,
        # Stored at the top level too, so the gate can be checked without
        # reconstructing the config dataclass.
        "lang_code": model_cfg.lang_code,
        "scheduler_state": scheduler_state(train_cfg, step),
        # fp16 runs carry the GradScaler's scale factor: resuming with a reset
        # scale would re-run its warmup and skip a handful of updates.
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "format_version": 3,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)          # atomic rename
    return path


def load_checkpoint(path: str | Path, *, expected_lang_code: str, map_location="cpu") -> dict:
    """Load a checkpoint, hard-failing on a language mismatch.

    `expected_lang_code` is required, not optional with a default. A gate that
    can be skipped by forgetting an argument is not a gate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    # weights_only=False: our payload holds a config dict, not just tensors.
    # Safe here because we only ever load checkpoints this project wrote.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    stored = ckpt.get("lang_code")
    if stored != expected_lang_code:
        raise LanguageMismatchError(
            f"checkpoint {path} was trained on lang_code={stored!r} but is being "
            f"loaded into a lang_code={expected_lang_code!r} model.\n"
            "Hindi and Nepali share a vocab size (16,000), so every tensor shape "
            "matches and this would load silently -- which is exactly why it is "
            "blocked here. Loading it would violate the model-independence "
            "constraint (separate data, tokenizer, vocabulary, and weights)."
        )
    return ckpt


def restore_into(ckpt: dict, *, model, optimizer=None, restore_rng: bool = True,
                 scaler=None) -> int:
    """Restore weights, optimizer state and RNG in place; return the step.

    Returns `step` as stored -- the step count already COMPLETED -- so the
    caller resumes at that number and does not repeat or skip one.

    `restore_rng` defaults to True because that is what makes the resumed run
    bit-identical to an uninterrupted one (see the module docstring). It is
    exposed only so a caller can deliberately branch a run onto a different
    random path.
    """
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    if restore_rng and not _restore_rng(ckpt.get("rng_state")):
        print("[warn] checkpoint carries no RNG state (format_version < 2); "
              "resume will be statistically equivalent but NOT bit-identical.")
    return int(ckpt["step"])


def find_latest(ckpt_dir: str | Path) -> Path | None:
    """Newest `step_*.pt` in a directory, or None. Used for automatic resume.

    Sorted by the integer step parsed from the filename, not by mtime or
    lexicographic order -- `step_9000.pt` must not sort after `step_10000.pt`.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    found = []
    for p in ckpt_dir.glob("step_*.pt"):
        try:
            found.append((int(p.stem.split("_")[1]), p))
        except (IndexError, ValueError):
            continue
    return max(found)[1] if found else None
