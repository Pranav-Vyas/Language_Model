"""Entry point: pretrain this tree's model. One process, one GPU.

Duplicated verbatim into every language tree; never imports from a sibling.

    python hindi/train/loop.py --steps 200 --smoke

Parallel training of both languages is a *launcher* concern, not this file's:

    CUDA_VISIBLE_DEVICES=0 python hindi/train/loop.py  &
    CUDA_VISIBLE_DEVICES=1 python nepali/train/loop.py &

Each process then sees exactly one device (its own `cuda:0`), reads only its
own tree's config, corpus, and tokenizer, and writes only its own checkpoint
directory. The independence constraint is enforced by the OS.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent.parent
# Own tree only. Importing the sibling's model/ would be an independence violation.
sys.path.insert(0, str(TREE / "model"))
sys.path.insert(0, str(TREE / "train"))

from train_config import load_model_config, load_train_config   # noqa: E402
from trainer import Trainer, assert_cuda_available, evaluate     # noqa: E402
from transformer import ModelConfig, TransformerLM               # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed every RNG the run touches, so a rerun reproduces the numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=f"Pretrain the {TREE.name} model.")
    ap.add_argument("--model-config", default=str(TREE / "configs" / "model.yaml"))
    ap.add_argument("--train-config", default=str(TREE / "configs" / "train.yaml"))
    ap.add_argument("--steps", type=int, default=None,
                    help="override max_steps -- the LR SCHEDULE HORIZON. Must be "
                         "the total for the whole run, identical on every link.")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="stop this process here WITHOUT changing the schedule "
                         "horizon; this is how a chain splits one cosine across "
                         "several Kaggle sessions.")
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--train-shard", default=None,
                    help="override the shard path (Kaggle mounts data read-only "
                         "under /kaggle/input, not at the repo-relative path)")
    ap.add_argument("--val-shard", default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--keep-last-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="explicit warmup; otherwise derived from --steps")
    ap.add_argument("--dtype", default=None,
                    choices=["float16", "bfloat16", "float32"],
                    help="autocast precision. fp16 is 3.1x faster than bf16 on "
                         "a T4 (Turing tensor cores are fp16); bf16 suits Ampere+")
    ap.add_argument("--peak-lr", type=float, default=None,
                    help="override peak learning rate (hyperparameter sweeps)")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None,
                    help="override emb/attn/resid dropout together. Dropout is a "
                         "MODEL-config field, so this is applied after the model "
                         "config is loaded rather than through the train config.")
    ap.add_argument("--record-batch-hash", action="store_true",
                    help="fingerprint each step's batch (kill-resume proof)")
    ap.add_argument("--resume", default=None, help="explicit checkpoint path")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints and start at step 0")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU (tests and tiny smoke runs only)")
    ap.add_argument("--smoke", action="store_true",
                    help="short run: frequent logging, report peak VRAM")
    args = ap.parse_args(argv)

    # GPU check FIRST -- before the corpus is opened or the model is built, so
    # a forgotten Kaggle Accelerator fails in seconds rather than after setup.
    device = assert_cuda_available(allow_cpu=args.allow_cpu)

    train_cfg = load_train_config(
        args.train_config,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        ckpt_dir=args.ckpt_dir,
        train_shard=args.train_shard,
        val_shard=args.val_shard,
        dtype=args.dtype,
        peak_lr=args.peak_lr,
        weight_decay=args.weight_decay,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        keep_last_n=args.keep_last_n,
        seed=args.seed,
        record_batch_hash=args.record_batch_hash or None,
    )
    model_cfg = load_model_config(args.model_config, ModelConfig)
    if args.dropout is not None:
        # All three dropout sites move together: they regularise the same model
        # against the same (absent) overfitting, so sweeping them independently
        # would spend runs on a distinction the evidence does not support.
        model_cfg.emb_dropout = args.dropout
        model_cfg.attn_dropout = args.dropout
        model_cfg.resid_dropout = args.dropout
        print(f"[sweep] dropout overridden to {args.dropout}")
    if args.steps is not None:
        train_cfg.max_steps = args.steps
        # Shrink warmup so a short run still exercises both legs of the
        # schedule. Explicit --warmup-steps wins: a DERIVED value here silently
        # changes the LR curve, which makes two runs that look identically
        # configured produce different weights (found by the T2.8 proof).
        train_cfg.warmup_steps = (
            args.warmup_steps if args.warmup_steps is not None
            else min(train_cfg.warmup_steps, max(1, args.steps // 10)))
    elif args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps
    print(f"[sched] warmup {train_cfg.warmup_steps} -> cosine to "
          f"{train_cfg.max_steps} steps, peak_lr {train_cfg.peak_lr:g}, "
          f"weight_decay {train_cfg.weight_decay:g}")
    if args.smoke:
        train_cfg.log_every = max(1, train_cfg.log_every // 5)

    seed_everything(train_cfg.seed)

    # Autocast on CUDA only: on CPU both half precisions are slower than fp32.
    _DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if device.type == "cuda" and train_cfg.dtype in _DTYPES:
        _dt = _DTYPES[train_cfg.dtype]
        autocast_ctx = lambda: torch.autocast("cuda", dtype=_dt)
        print(f"[amp] autocast {train_cfg.dtype}")
    else:
        autocast_ctx = contextlib.nullcontext

    model = TransformerLM(model_cfg).to(device)
    print(f"[model] lang={model_cfg.lang_code} params={model.num_parameters():,} "
          f"d={model_cfg.d_model} L={model_cfg.n_layers} h={model_cfg.n_heads} "
          f"ctx={model_cfg.max_seq_len}")
    print(f"[batch] micro={train_cfg.micro_batch_size} x accum={train_cfg.grad_accum_steps}"
          f" = {train_cfg.micro_batch_size * train_cfg.grad_accum_steps} seqs"
          f" = {train_cfg.tokens_per_step:,} tokens/step")

    trainer = Trainer(model, model_cfg, train_cfg, device, autocast_ctx)
    if not args.no_resume:
        trainer.maybe_resume(args.resume)

    print(f"[data] train {len(trainer.train_loader):,} tokens | "
          f"val {len(trainer.val_loader):,} tokens")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Stop point and schedule horizon are DIFFERENT quantities, and conflating
    # them silently corrupts a chained run. An earlier version gave each link
    # its own --steps, so link 1 planned a cosine over 100 steps and annealed to
    # the floor, then link 2 re-planned over 200 and jumped the LR back up ~6x
    # (6.02e-05 -> 3.63e-04; grad-norm 0.21 -> 0.98 across the boundary). Over a
    # real multi-link run that is a sawtooth, not the single cosine decay the
    # run is meant to follow.
    stop_at = args.stop_at if args.stop_at is not None else train_cfg.max_steps
    if stop_at > train_cfg.max_steps:
        raise SystemExit(f"--stop-at {stop_at} exceeds the schedule horizon "
                         f"{train_cfg.max_steps}; the cosine ends there.")
    if args.stop_at is not None:
        print(f"[chain] stop at step {stop_at:,} of a {train_cfg.max_steps:,}-step "
              f"schedule (LR horizon unchanged)", flush=True)
    history = trainer.train(max_steps=stop_at)

    # Final validation, so the run always ends with a comparable number.
    vloss = evaluate(model, trainer.val_loader, train_cfg.eval_batches,
                     device, autocast_ctx)
    print(f"\n[final] step {trainer.step:,}  val_loss {vloss:.4f}")
    ckpt = trainer.save(val_loss=vloss)
    print(f"[final] checkpoint {ckpt}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[vram] peak allocated {peak:.2f} GiB of {total:.1f} GiB "
              f"({100*peak/total:.0f}%) at micro_batch={train_cfg.micro_batch_size}")

    # Loss history next to the checkpoints: the Phase-2 report needs curves,
    # and a number that only ever existed in a terminal is not reproducible.
    hist_path = Path(train_cfg.ckpt_dir) / "history.jsonl"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("a", encoding="utf-8") as fh:
        for rec in history:
            fh.write(json.dumps(rec) + "\n")
    print(f"[log] appended {len(history)} records -> {hist_path}")


if __name__ == "__main__":
    main()
