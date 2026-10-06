"""Supervised finetuning on the reasoning set (T3.6/T3.8).

Duplicated verbatim into every language tree; never imports from a sibling.

    python hindi/train/finetune.py --epochs 100 \
        --init-from kaggle_tuned_out/link1/checkpoints/hindi/step_8732.pt

WHY 100 EPOCHS AND NOT THE USUAL 3
----------------------------------
On comparison/transitive tasks, generalisation to held-out entities arrives
LONG after training accuracy saturates -- the grokking result. A 3-epoch run
(the HuggingFace tutorial default, written for a 0.6B instruction-tuned base)
would show high train accuracy, near-chance held-out accuracy, and invite the
conclusion "a 24M model cannot do transitive reasoning" when it simply had not
grokked yet. So this loop trains long and, more importantly, MEASURES held-out
accuracy throughout rather than only at the end. The train-vs-held-out curve is
a reported figure, not a diagnostic.

WHAT IS EVALUATED, AND WHY THREE SPLITS
---------------------------------------
  val   - held-out ENTITY NAMES, same relations and hop counts as training.
  test  - held-out entities AND two relations never seen in training.
  ood   - 3-hop chains, a depth never seen in training.
Tracking all three separately is what distinguishes "learned the template" from
"learned the relation" from "learned to chain".

INITIALISE-FROM versus RESUME
-----------------------------
Two different operations that must not be confused:
  --init-from  starts a NEW finetune from a pretrained checkpoint. Weights are
               loaded; the optimizer is fresh and the step counter starts at 0.
  --resume     continues an interrupted finetune, restoring optimizer state and
               step. Default behaviour if a finetune checkpoint already exists.
Loading a pretrain checkpoint's optimizer state into a finetune would carry
Adam moments fitted to a different objective and a different LR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
REPO = TREE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TREE / "model"))

from checkpoint import find_latest, load_checkpoint, save_checkpoint   # noqa: E402
from schedule import WarmupCosineSchedule                              # noqa: E402
from sft_data import SFTDataset                                        # noqa: E402
from trainer import build_optimizer                                    # noqa: E402
from transformer import ModelConfig, TransformerLM                     # noqa: E402

# The held-out accuracy printed during training comes from the SAME module
# as the final reported numbers. Two copies of "exact match" would be two
# chances to disagree, and the in-training curve would then not be
# evidence for the final table.
sys.path.insert(0, str(TREE / "eval"))
from reasoning_metrics import exact_match                              # noqa: E402

LANG = {"hindi": "hi", "nepali": "ne"}[TREE.name]


def load_tokenizer():
    import sentencepiece as spm
    return spm.SentencePieceProcessor(
        model_file=str(TREE / "tokenizer" / f"{LANG}_bpe_16000.model"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=f"Finetune {TREE.name} on reasoning.")
    ap.add_argument("--init-from", default=None,
                    help="pretrained checkpoint to start a NEW finetune from")
    ap.add_argument("--resume", default=None, help="continue a finetune checkpoint")
    ap.add_argument("--data-dir", default=str(TREE / "reasoning_data"))
    ap.add_argument("--ckpt-dir", default=str(TREE / "checkpoints_sft"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--target-field", default="target_answer_only",
                    choices=["target_answer_only", "target_with_trace"],
                    help="the chain-of-thought A/B")
    ap.add_argument("--no-mask-prompt", action="store_true",
                    help="supervise the prompt too (the masking A/B)")
    ap.add_argument("--eval-every", type=int, default=200,
                    help="steps between held-out accuracy evaluations")
    ap.add_argument("--eval-n", type=int, default=500,
                    help="examples per split per evaluation")
    ap.add_argument("--eval-max-new", type=int, default=None,
                    help="tokens to generate per eval example. Default is "
                         "derived from --target-field: an answer-only target "
                         "needs ~4 tokens, a reasoning trace ~26. Capping a "
                         "trace at the answer-only budget truncates it before "
                         "'उत्तर:' is emitted, which scores every CoT run 0.000 "
                         "for a reason that has nothing to do with the model.")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-n", type=int, default=None,
                    help="cap training examples (smoke runs)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise SystemExit("no GPU; pass --allow-cpu for a smoke run")

    use_amp = device.type == "cuda" and args.dtype != "float32"
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype)
    autocast_ctx = ((lambda: torch.autocast(device.type, dtype=amp_dtype))
                    if use_amp else None)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.dtype == "float16"
                                                        and device.type == "cuda"))

    sp = load_tokenizer()
    data_dir = Path(args.data_dir)
    train_ds = SFTDataset(data_dir / "train.jsonl", sp, max_len=args.max_len,
                          target_field=args.target_field,
                          mask_prompt=not args.no_mask_prompt)
    if args.train_n:
        train_ds.examples = train_ds.examples[:args.train_n]
        train_ds.rows = train_ds.rows[:args.train_n]

    eval_rows = {}
    for split in ("val", "test", "ood"):
        p = data_dir / f"{split}.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            eval_rows[split] = rows[:args.eval_n]
    # A fixed slice of TRAIN, so train-vs-held-out is measured the same way.
    eval_rows["train"] = train_ds.rows[:args.eval_n]

    # --- model: initialise-from vs resume ---------------------------------
    ckpt_dir = Path(args.ckpt_dir + (f"_{args.tag}" if args.tag else ""))
    resume_path = args.resume or find_latest(ckpt_dir)
    start_step, init_from = 0, None
    if resume_path and Path(resume_path).exists():
        ck = load_checkpoint(resume_path, expected_lang_code=LANG, map_location=device)
        if ck.get("train_cfg", {}).get("stage") != "finetune":
            raise SystemExit(
                f"{resume_path} is not a finetune checkpoint. Use --init-from to "
                "start a new finetune from a pretrained one.")
        cfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                             if k in ModelConfig.__dataclass_fields__})
        model = TransformerLM(cfg).to(device)
        model.load_state_dict(ck["model_state"])
        start_step = int(ck["step"])
        init_from = ck.get("train_cfg", {}).get("init_from")
        print(f"[resume] finetune from {resume_path} @ step {start_step}", flush=True)
    else:
        if not args.init_from:
            raise SystemExit("--init-from is required to start a new finetune")
        ck = load_checkpoint(args.init_from, expected_lang_code=LANG,
                             map_location=device)
        cfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                             if k in ModelConfig.__dataclass_fields__})
        model = TransformerLM(cfg).to(device)
        model.load_state_dict(ck["model_state"])
        init_from = str(args.init_from)
        # Optimizer state deliberately NOT restored: it was fitted to the
        # pretraining objective at a different learning rate.
        print(f"[init] weights from {args.init_from} (pretrain step "
              f"{ck['step']}); fresh optimizer, step 0", flush=True)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    # Shim carrying exactly the fields build_optimizer reads, so the finetune
    # reuses the pretraining optimizer construction (decay on 2D weights only)
    # rather than duplicating that logic and risking it drifting.
    class _C:
        peak_lr = args.peak_lr
        weight_decay = args.weight_decay
        beta1, beta2, eps = 0.9, 0.95, 1e-8
    optimizer = build_optimizer(model, _C())
    if resume_path and Path(resume_path).exists():
        optimizer.load_state_dict(ck["optimizer_state"])
    sched = WarmupCosineSchedule(args.peak_lr, warmup, total_steps, args.min_lr_ratio)

    st = train_ds.token_stats()
    print(f"[data] {st['examples']:,} train examples, {st['tokens_total']:,} tokens, "
          f"supervised {st['supervised_frac']:.3f} "
          f"(mask_prompt={st['mask_prompt']}, target={args.target_field})", flush=True)
    print(f"[sched] {steps_per_epoch:,} steps/epoch x {args.epochs} = "
          f"{total_steps:,} steps, warmup {warmup}, peak_lr {args.peak_lr:g}", flush=True)
    print(f"[eval] every {args.eval_every} steps on "
          f"{ {k: len(v) for k, v in eval_rows.items()} }", flush=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_path = ckpt_dir / "history.jsonl"
    train_cfg = {"stage": "finetune", "init_from": init_from,
                 "target_field": args.target_field,
                 "mask_prompt": not args.no_mask_prompt,
                 "peak_lr": args.peak_lr, "warmup_steps": warmup,
                 "max_steps": total_steps, "min_lr_ratio": args.min_lr_ratio,
                 "batch_size": args.batch_size, "epochs": args.epochs,
                 "max_len": args.max_len, "seed": args.seed,
                 "dtype": args.dtype, "weight_decay": args.weight_decay}

    # A trace target needs far more room than a bare answer. Derived rather
    # than hardcoded so the CoT A/B is not silently handicapped.
    eval_max_new = args.eval_max_new
    if eval_max_new is None:
        eval_max_new = 48 if "trace" in args.target_field else 12
    print(f"[eval] max_new={eval_max_new} tokens (target={args.target_field})",
          flush=True)

    def evaluate_all(step):
        rec = {"step": step}
        for split, rows in eval_rows.items():
            m = exact_match(model, sp, rows, device, autocast_ctx=autocast_ctx,
                            max_new=eval_max_new)
            rec[split] = m
            print(f"  [{split:5s}] EM {m['exact_match']:.3f} "
                  f"(chance {m['chance']:.3f}, last-mention {m['last_mention']:.3f}, "
                  f"well-formed {m['well_formed']:.3f})", flush=True)
        return rec

    step = start_step
    t0 = time.time()
    model.train()
    with open(hist_path, "a", encoding="utf-8") as hf:
        for epoch in range(args.epochs):
            if step >= total_steps:
                break
            for x, y in train_ds.batches(args.batch_size, epoch=epoch,
                                         seed=args.seed, device=device):
                lr = sched.apply(optimizer, step)
                optimizer.zero_grad(set_to_none=True)
                if autocast_ctx is not None:
                    with autocast_ctx():
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                step += 1

                if step % args.log_every == 0:
                    # .detach() before the scalar read: float() on a grad-tracking
                    # tensor warns, and silently holding a reference to the graph
                    # in a logging dict is a real leak in a long run.
                    rec = {"step": step, "epoch": epoch,
                           "loss": float(loss.detach()),
                           "lr": lr, "grad_norm": float(gnorm),
                           "elapsed_s": time.time() - t0}
                    hf.write(json.dumps(rec) + "\n"); hf.flush()
                    print(f"  step {step:>6,}/{total_steps:,} ep {epoch:>3} | "
                          f"loss {float(loss.detach()):.4f} | lr {lr:.2e} | "
                          f"gnorm {float(gnorm):.2f} | {time.time()-t0:.0f}s", flush=True)

                if step % args.eval_every == 0 or step == total_steps:
                    print(f"[eval @ step {step:,}, epoch {epoch}]", flush=True)
                    rec = evaluate_all(step)
                    rec["kind"] = "eval"
                    hf.write(json.dumps(rec) + "\n"); hf.flush()

                if step % args.ckpt_every == 0 or step == total_steps:
                    p = save_checkpoint(ckpt_dir / f"step_{step}.pt", model=model,
                                        optimizer=optimizer, step=step,
                                        model_cfg=cfg, train_cfg=train_cfg,
                                        scaler=scaler if scaler.is_enabled() else None)
                    print(f"  [ckpt] {p}", flush=True)
                if step >= total_steps:
                    break

    print(f"\n[final] step {step:,} after {(time.time()-t0)/60:.1f} min", flush=True)
    rec = evaluate_all(step); rec["kind"] = "final"
    with open(hist_path, "a", encoding="utf-8") as hf:
        hf.write(json.dumps(rec) + "\n")
    p = save_checkpoint(ckpt_dir / f"step_{step}.pt", model=model,
                        optimizer=optimizer, step=step, model_cfg=cfg,
                        train_cfg=train_cfg,
                        scaler=scaler if scaler.is_enabled() else None)
    print(f"[final] checkpoint {p}", flush=True)


if __name__ == "__main__":
    main()
