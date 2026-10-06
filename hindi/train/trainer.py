"""The pretraining loop: AdamW, warmup-cosine LR, clipping, bf16, accumulation.

Duplicated verbatim into every language tree; never imports from a sibling.

ONE PROCESS, ONE GPU -- ALWAYS
------------------------------
Kaggle gives two T4s, and Hindi and Nepali are trained in parallel, one per
card. That parallelism lives entirely in the *launcher*: each language runs as
its own process with `CUDA_VISIBLE_DEVICES` pinned, so this file only ever
sees a single device and contains no distributed code at all.

That is a deliberate design choice, not a shortcut. The two models are
required to share no data, tokenizer, vocabulary, or weights; running them as
two isolated OS processes enforces that at the strongest boundary available,
whereas a DistributedDataParallel setup would put both languages inside one
process group and make independence a property of careful coding rather than
of the operating system. Process isolation is both simpler and safer here.

PRECISION: MEASURED, NOT ASSUMED
--------------------------------
Benchmarked on the actual T4 (see scripts/kaggle/bench_kernel), 20 measured
steps per variant on the real model and shards:

    bf16 micro=8   11,049 tok/s   loss 7.5459   <- the original choice
    fp16 micro=8   34,315 tok/s   loss 7.5459   <- 3.11x faster, same loss
    fp32 micro=8   13,149 tok/s   loss 7.5462

bf16 is SLOWER THAN fp32 on this card. The T4 is Turing (sm_75), whose tensor
cores implement fp16; bf16 tensor cores arrive with Ampere (sm_80). So bf16
here costs fp32 arithmetic *plus* conversion overhead.
`torch.cuda.is_bf16_supported()` returns True, but that means "works", not
"is accelerated" -- which is precisely how this went unexamined for a 14-hour
training run.

So the default is now fp16 + GradScaler on this hardware. The reasoning below
for why bf16 is attractive IN GENERAL still stands, and `dtype: bfloat16`
remains selectable for a GPU where it is native (any Ampere or later).

WHY bf16 IS ATTRACTIVE IN GENERAL
---------------------------------
T4 (Turing, sm_75) has fp16 tensor cores; bf16 is emulated rather than native,
so it is not always faster. It is chosen anyway for numerical reasons: bf16
carries the same 8-bit exponent as fp32, so it has fp32's dynamic range and
cannot overflow to inf where fp32 would not. fp16's 5-bit exponent overflows
around 65,504 and needs a GradScaler to keep gradients representable.

That extra failure mode is the price of the 3x, and it is managed rather than
avoided: the scaler is created for every dtype (it is a documented no-op when
`enabled=False`, so there is ONE code path rather than a precision branch that
could diverge), skipped steps are counted and logged, and the scale factor is
saved into every checkpoint so a resume continues with the same scaling.

Autocast covers matmuls; the loss is computed in fp32 because
`F.cross_entropy` is on autocast's fp32 list, and the master weights stay
fp32 throughout.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch

from checkpoint import find_latest, load_checkpoint, restore_into, save_checkpoint
from data import TokenShardLoader
from schedule import WarmupCosineSchedule


def assert_cuda_available(allow_cpu: bool = False) -> torch.device:
    """Fail loudly at startup if there is no GPU.

    Kaggle resets a notebook's Accelerator to *None* every time a new session
    starts, so the single most likely way to waste an afternoon is to launch
    training that quietly runs on CPU and discover it an hour later. A ~25M
    model at 65k tokens/step is roughly two orders of magnitude slower on CPU:
    it does not crash, it just never finishes.

    Checked before the corpus is opened or the model is built, so the failure
    arrives in seconds.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[device] {name}  ({total:.1f} GiB)  bf16={torch.cuda.is_bf16_supported()}")
        return dev
    if allow_cpu:
        print("[device] CPU (--allow-cpu given; for tests and smoke runs only)")
        return torch.device("cpu")
    raise RuntimeError(
        "torch.cuda.is_available() is False -- no GPU visible.\n"
        "\n"
        "On Kaggle the Accelerator resets to 'None' for every NEW session, so this\n"
        "is almost always the cause. Fix: notebook sidebar -> Session options ->\n"
        "Accelerator -> 'GPU T4 x2', then restart the session and re-run.\n"
        "\n"
        "Refusing to continue: this model trains ~100x slower on CPU, so a CPU run\n"
        "would not crash, it would simply never finish. Pass --allow-cpu only for\n"
        "unit tests and tiny smoke runs."
    )


def build_optimizer(model, cfg):
    """AdamW with weight decay on matmul weights only.

    Splitting the parameters is standard practice and worth being able to
    justify: weight decay is an L2 pull toward zero, which is meaningful for
    the 2D weight matrices that define linear maps. Applying it to LayerNorm
    gains and biases (1D) instead fights the normalisation itself -- shrinking
    a LayerNorm gain toward zero suppresses that layer's output for no
    regularisation benefit.

    Note the tied embedding/LM-head matrix is 2D and therefore *is* decayed,
    once, since it is a single shared tensor.
    """
    decay, no_decay, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.peak_lr,
                            betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    n_decay = sum(p.numel() for p in decay)
    n_plain = sum(p.numel() for p in no_decay)
    print(f"[optim] AdamW  decayed {n_decay:,} params in {len(decay)} tensors | "
          f"undecayed {n_plain:,} in {len(no_decay)}")
    return opt


@torch.no_grad()
def evaluate(model, loader: TokenShardLoader, n_batches: int, device, autocast_ctx) -> float:
    """Mean cross-entropy over a FIXED set of validation windows.

    `model.eval()` matters: dropout stays on in train mode and would inflate
    the reported loss by a noise term that has nothing to do with model
    quality. Restored to train mode by the caller.
    """
    model.eval()
    total, count = 0.0, 0
    for x, y in loader.eval_batches(n_batches, device=device):
        with autocast_ctx():
            _, loss = model(x, y)
        total += loss.float().item()
        count += 1
    model.train()
    return total / max(1, count)


class Trainer:
    """Owns the training loop for ONE language on ONE device."""

    def __init__(self, model, model_cfg, train_cfg, device, autocast_ctx):
        self.model = model
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = device
        self.autocast_ctx = autocast_ctx

        if train_cfg.ctx != model_cfg.max_seq_len:
            raise ValueError(
                f"train ctx={train_cfg.ctx} != model max_seq_len="
                f"{model_cfg.max_seq_len}. Learned positional embeddings have no "
                "row beyond max_seq_len, so these must agree exactly."
            )

        # One code path for every precision: GradScaler is a documented no-op
        # when disabled, so bf16/fp32 runs go through the same calls as fp16
        # instead of a branch that could drift out of sync.
        # CUDA only: autocast is a no-op on CPU (see loop.py), so a scaler
        # there would rescale gradients that were never in fp16.
        self.use_fp16 = (str(getattr(train_cfg, "dtype", "")) == "float16"
                         and device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_fp16)
        self.skipped_steps = 0
        if self.use_fp16:
            print("[amp] fp16 + GradScaler (T4 tensor cores are fp16, not bf16)")
        self.optimizer = build_optimizer(model, train_cfg)
        self.schedule = WarmupCosineSchedule(
            peak_lr=train_cfg.peak_lr,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.max_steps,
            min_lr_ratio=train_cfg.min_lr_ratio,
        )
        self.train_loader = TokenShardLoader(
            train_cfg.train_shard, train_cfg.ctx, train_cfg.micro_batch_size, train_cfg.seed)
        self.val_loader = TokenShardLoader(
            train_cfg.val_shard, train_cfg.ctx, train_cfg.micro_batch_size, train_cfg.seed)
        self.step = 0
        self.history: list[dict] = []

    # -- resume --------------------------------------------------------------

    def maybe_resume(self, explicit: str | None = None) -> bool:
        """Restore from an explicit checkpoint, else the newest in ckpt_dir.

        Resume is the DEFAULT, not an opt-in flag: on a preemptible machine
        the common case is "the session died, start it again", and a launcher
        that silently restarts from step 0 would throw away GPU hours while
        looking like it was working.
        """
        path = Path(explicit) if explicit else find_latest(self.cfg.ckpt_dir)
        if path is None or not Path(path).exists():
            print(f"[resume] no checkpoint found in {self.cfg.ckpt_dir} -- starting at step 0")
            return False
        ckpt = load_checkpoint(path, expected_lang_code=self.model_cfg.lang_code,
                               map_location=self.device)
        self.step = restore_into(ckpt, model=self.model, optimizer=self.optimizer,
                                 scaler=self.scaler)
        print(f"[resume] {path} -> step {self.step:,} "
              f"(val_loss {ckpt.get('val_loss')})")
        return True

    def save(self, val_loss: float | None = None) -> Path:
        path = Path(self.cfg.ckpt_dir) / f"step_{self.step}.pt"
        save_checkpoint(path, model=self.model, optimizer=self.optimizer,
                        step=self.step, model_cfg=self.model_cfg,
                        train_cfg=self.cfg.to_dict(), val_loss=val_loss,
                        scaler=self.scaler)
        self._prune()
        return path

    def _prune(self) -> None:
        """Keep only the newest N checkpoints; Kaggle's working dir is capped.

        Steps listed in `extra_ckpt_steps` are never pruned -- they exist for a
        specific comparison in the report, so losing one to a size cap would
        silently delete a result rather than a backup.
        """
        keep = self.cfg.keep_last_n
        if keep <= 0:
            return
        protected = {int(s) for s in self.cfg.extra_ckpt_steps}
        ckpts = sorted(Path(self.cfg.ckpt_dir).glob("step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        prunable = [c for c in ckpts if int(c.stem.split("_")[1]) not in protected]
        for old in prunable[:-keep] if keep else prunable:
            old.unlink(missing_ok=True)

    # -- the loop -------------------------------------------------------------

    def train(self, max_steps: int | None = None) -> list[dict]:
        """Run until `max_steps`, checkpointing and validating on cadence."""
        cfg = self.cfg
        target = max_steps if max_steps is not None else cfg.max_steps
        self.model.train()
        t0 = time.time()

        while self.step < target:
            lr = self.schedule.apply(self.optimizer, self.step)

            # --- gradient accumulation -------------------------------------
            # set_to_none=True frees the gradient buffers rather than zeroing
            # them, which is both faster and lower-peak-memory.
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            batch_hash = None
            for micro in range(cfg.grad_accum_steps):
                # Distinct batch per micro-step, still a pure function of step.
                x, y = self.train_loader.batch(
                    self.step * cfg.grad_accum_steps + micro, device=self.device)
                if cfg.record_batch_hash and micro == 0:
                    # Fingerprint of the first micro-batch. Off by default: it
                    # forces a device->host sync, which would stall the GPU
                    # every step for no training benefit. Turned on by the
                    # kill-and-resume proof, where it is the evidence that a
                    # resumed run sees the same data as an uninterrupted one.
                    batch_hash = hashlib.blake2b(
                        x.detach().cpu().numpy().tobytes(), digest_size=8).hexdigest()
                with self.autocast_ctx():
                    _, loss = self.model(x, y)
                # Divide so the accumulated gradient is the MEAN over the whole
                # effective batch, not the sum -- otherwise the effective LR
                # would scale with grad_accum_steps.
                self.scaler.scale(loss / cfg.grad_accum_steps).backward()
                accum_loss += loss.float().item() / cfg.grad_accum_steps

            # --- clip, then step --------------------------------------------
            # Clipping AFTER the full accumulation, so the norm is that of the
            # true effective-batch gradient. Clipping per micro-step would clip
            # a different quantity.
            # unscale_ BEFORE clipping: the gradients are still multiplied by
            # the scale factor at this point, so clipping them un-unscaled would
            # clip a scaled quantity and the effective clip threshold would move
            # with the scaler.
            self.scaler.unscale_(self.optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # A drop in the scale factor means the step contained inf/NaN and
            # was skipped. Counted so a run that is quietly skipping most of its
            # updates is visible rather than merely slow to converge.
            if self.scaler.get_scale() < scale_before:
                self.skipped_steps += 1
            self.step += 1

            if self.step % cfg.log_every == 0:
                elapsed = time.time() - t0
                tok = self.step * cfg.tokens_per_step
                rec = {"step": self.step, "loss": accum_loss, "lr": lr,
                       "grad_norm": float(gnorm), "tokens": tok,
                       "ppl": math.exp(min(accum_loss, 20)), "elapsed_s": elapsed,
                       "skipped_steps": self.skipped_steps}
                if batch_hash is not None:
                    rec["batch_hash"] = batch_hash
                self.history.append(rec)
                print(f"  step {self.step:>6,} | loss {accum_loss:.4f} | "
                      f"ppl {rec['ppl']:>9,.1f} | lr {lr:.2e} | "
                      f"gnorm {float(gnorm):.2f} | {tok/1e6:.1f}M tok | "
                      f"{elapsed:.0f}s", flush=True)

            # A milestone step always checkpoints, and always validates first
            # so the saved file carries a val_loss for the report.
            if self.step in {int(x) for x in cfg.extra_ckpt_steps}:
                vloss = evaluate(self.model, self.val_loader, cfg.eval_batches,
                                 self.device, self.autocast_ctx)
                print(f"  [milestone] step {self.step:,} val_loss {vloss:.4f} "
                      f"({self.step * cfg.tokens_per_step:,} tokens seen)", flush=True)
                print(f"  [ckpt] {self.save(val_loss=vloss)}", flush=True)
            elif cfg.eval_every and self.step % cfg.eval_every == 0:
                vloss = evaluate(self.model, self.val_loader, cfg.eval_batches,
                                 self.device, self.autocast_ctx)
                print(f"  [val] step {self.step:,} loss {vloss:.4f} "
                      f"ppl {math.exp(min(vloss, 20)):,.1f}", flush=True)
                if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                    print(f"  [ckpt] {self.save(val_loss=vloss)}", flush=True)
            elif cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                print(f"  [ckpt] {self.save()}", flush=True)

        return self.history
