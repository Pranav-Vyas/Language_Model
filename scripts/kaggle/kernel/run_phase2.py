"""Kaggle kernel entry point: run both models, one per GPU, in parallel.

Pushed by scripts/kaggle/run.py. Reads code and data from ATTACHED DATASETS
(no internet, no git credentials) and writes everything to /kaggle/working,
which is what `kaggle kernels output` later pulls back.

This script is deliberately thin. All the real logic lives in the repo's own
`<lang>/train/loop.py`; a kernel that reimplemented training would be a second
copy to keep correct.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")
# Code is unpacked OUTSIDE /kaggle/working on purpose. Everything under
# /kaggle/working becomes the kernel's output, which `kaggle kernels output`
# then downloads -- so a repo copy there means re-downloading ~85 code files
# on every fetch and burying the checkpoints we actually want.
REPO = Path("/tmp/lma_repo")

# LMA_STEPS is the LR schedule horizon for the WHOLE run, identical on every
# link. LMA_STOP_AT is where THIS link stops. Separate values are what keep a
# chained run on one continuous cosine.
STEPS = int(os.environ.get("LMA_STEPS", "8732"))
STOP_AT = int(os.environ.get("LMA_STOP_AT", "8732")) or STEPS

# Which trees this run trains. A tuned-Hindi retrain must NOT touch Nepali:
# its checkpoint is final and re-running it would burn ~23h for nothing.
TREES = [t for t in os.environ.get("LMA_TREES", "hindi").split(",") if t]
# Sweep-selected hyperparameters. Empty/0 means "use the tree's own config".
PEAK_LR = float(os.environ.get("LMA_PEAK_LR", "0.003"))
DROPOUT = os.environ.get("LMA_DROPOUT", "0.0")
# fp16 measured 3.11x faster than bf16 on the T4 (Turing tensor cores are fp16,
# not bf16). Empty means "use the tree's config".
DTYPE = os.environ.get("LMA_DTYPE", "float16")

# Per-language stop points. The two models share one kernel session but have
# DIFFERENT horizons (Hindi 8,732 steps for one epoch, Nepali 13,547) and
# different throughput, so one shared stop point would either cut Hindi short
# or overrun the session. 0 means "use the tree's own configured max_steps".
STOP_HI = int(os.environ.get("LMA_STOP_HI", "8732"))
STOP_NE = int(os.environ.get("LMA_STOP_NE", "0"))
# When set, each tree's own configs/train.yaml supplies the LR horizon, so the
# schedule is per-language and --steps is not passed at all.
PER_LANG = STOP_HI > 0 or STOP_NE > 0
MICRO = int(os.environ.get("LMA_MICRO_BATCH", "8"))
ACCUM = int(os.environ.get("LMA_GRAD_ACCUM", "16"))
DATA_SLUG = os.environ.get("LMA_DATA_SLUG", "lma-tokens")
# Chain links keep ONE checkpoint. Only the newest crosses to the next link,
# and `kaggle kernels output` downloads everything under /kaggle/working --
# so keeping 3 would triple the transfer per link to move the same state.
# Milestone checkpoints (extra_ckpt_steps) are exempt from pruning.
KEEP_LAST = int(os.environ.get("LMA_KEEP_LAST", "1"))
# Checkpoint/eval cadence. 0 = derive a smoke-run value from STEPS. For the
# real run these are set explicitly: at ~5.9 s/step, 250 steps is ~25 min, so a
# session kill costs at most that much work, and a 40-batch validation every
# 250 steps is well under 1% of compute.
CKPT_EVERY = int(os.environ.get("LMA_CKPT_EVERY", "250")) or max(50, STEPS // 2)
EVAL_EVERY = int(os.environ.get("LMA_EVAL_EVERY", "250")) or max(50, STEPS // 2)
SMOKE = os.environ.get("LMA_SMOKE", "0") == "1"

print("=" * 70, flush=True)
print("LMA Phase 2 — Kaggle run", flush=True)
print("=" * 70, flush=True)

# --- 1. hard GPU gate, before anything expensive -------------------------
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print(f"[env] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"device_count={n_gpu}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: no GPU. Kaggle resets Accelerator to 'None' on every new session "
        "and on kernel pushes that omit enable_gpu. Set Accelerator = GPU T4 x2."
    )
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}",
          flush=True)
# We ASK for T4 x2; Kaggle may hand back a single P100 instead. Detect rather
# than assume -- a silent fallback would make the parallel plan invalid and the
# VRAM numbers meaningless.
if n_gpu < 2:
    print(f"[warn] expected 2 GPUs, got {n_gpu}. Falling back to SEQUENTIAL runs.", flush=True)
print(f"[env] bf16 supported: {torch.cuda.is_bf16_supported()}", flush=True)

# --- 2. unpack the code dataset ------------------------------------------
# Kaggle's mount path for a dataset is NOT reliably its slug, and an attached
# dataset that is still processing mounts as nothing at all. Both were guessed
# wrong on the first attempts, so nothing is hardcoded now: list what is really
# there, then FIND the roots by looking for files we know must exist.
print(f"[mount] /kaggle/input contains: "
      f"{sorted(p.name for p in IN.glob('*')) if IN.exists() else 'MISSING'}", flush=True)
for d in sorted(IN.glob("*")):
    kids = sorted(x.name for x in d.glob("*"))[:8]
    print(f"[mount]   {d.name}/ -> {kids}", flush=True)


def find_root(marker: str):
    """Locate the directory containing `marker`, at ANY depth under /kaggle/input.

    Measured layout (29 Aug): datasets mount at
        /kaggle/input/datasets/<owner>/<slug>/...
    not at /kaggle/input/<slug>/ as the docs imply. Rather than encode that --
    it is Kaggle's to change -- search for a file we know the tree must
    contain. Costs one directory walk over a handful of files.
    """
    leaf = Path(marker).name
    for hit in IN.rglob(leaf):
        root = Path(str(hit)[: -len(marker)]) if str(hit).endswith(marker) else None
        if root is not None and (root / marker).exists():
            return root
    return None


REPO.mkdir(parents=True, exist_ok=True)
code_src = find_root("hindi/train/loop.py")
if code_src is None:
    tarball = next(IN.rglob("lma_code.tar.gz"), None)
    if tarball is None:
        raise SystemExit("code not found in any attached dataset — is lma-code attached?")
    with tarfile.open(tarball) as tar:
        tar.extractall(REPO)
    print(f"[code] unpacked {tarball} -> {REPO}", flush=True)
else:
    # Copied, not used in place: /kaggle/input is READ-ONLY.
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
    print(f"[code] copied {code_src} -> {REPO}", flush=True)

data_root = find_root("hindi/train.bin")
if data_root is None:
    raise SystemExit(f"token shards not found in any attached dataset "
                     f"(looked for hindi/train.bin under {IN})")
print(f"[data] shard root: {data_root}", flush=True)
for tree in ("hindi", "nepali"):
    sh = data_root / tree / "train.bin"
    print(f"[data]   {tree}/train.bin "
          f"{'OK ' + format(sh.stat().st_size/2**20, '.1f') + ' MB' if sh.exists() else 'MISSING'}",
          flush=True)

# --- 2b. seed checkpoints from the previous link in the chain ------------
# A Kaggle run is capped at 12 hours, so a multi-day pretrain is a CHAIN of
# runs, each resuming where the last stopped. The previous link's checkpoints
# arrive as an attached input (a dataset version, or another kernel's output).
# They are COPIED into the writable working dir because /kaggle/input is
# read-only and the trainer must be able to write the next checkpoint beside
# them. loop.py then resumes by default -- find_latest() picks the newest
# step_*.pt, and the lang_code gate refuses a checkpoint from the wrong tree.
resume_root = find_root("hindi/step_meta.json") or find_root("checkpoints/hindi")
seeded = {}
for tree in TREES:
    dst = WORK / "checkpoints" / tree
    dst.mkdir(parents=True, exist_ok=True)
    found = sorted(IN.rglob(f"*/{tree}/step_*.pt"))
    if not found:
        continue
    # Newest by step number, not by name: step_9000 must not beat step_10000.
    newest = max(found, key=lambda q: int(q.stem.split("_")[1]))
    shutil.copy2(newest, dst / newest.name)
    seeded[tree] = newest.name
    print(f"[chain] seeded {tree} <- {newest} ({newest.stat().st_size/2**20:.0f} MB)",
          flush=True)
print(f"[chain] resume seeds: {seeded or 'none (fresh start)'}", flush=True)

# --- 3. launch one process per language, pinned to its own GPU -----------
# CUDA_VISIBLE_DEVICES is what makes each process see exactly ONE device (its
# own cuda:0). Two isolated processes also enforce the project's model
# independence constraint at the OS level rather than by convention.
procs = []
print(f"[cfg] trees={TREES} peak_lr={PEAK_LR or 'config'} "
      f"dropout={DROPOUT or 'config'} dtype={DTYPE or 'config'}", flush=True)
for idx, tree in enumerate(TREES):
    ckpt_dir = WORK / "checkpoints" / tree
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES=str(idx % max(1, n_gpu)))
    cmd = [sys.executable, str(REPO / tree / "train" / "loop.py")]
    if PER_LANG:
        # Horizon comes from <tree>/configs/train.yaml (8,732 / 13,547).
        stop = STOP_HI if tree == "hindi" else STOP_NE
        if stop:
            cmd += ["--stop-at", str(stop)]
    else:
        cmd += ["--steps", str(STEPS), "--stop-at", str(STOP_AT)]
    if PEAK_LR:
        cmd += ["--peak-lr", str(PEAK_LR)]
    if DROPOUT != "":
        cmd += ["--dropout", str(DROPOUT)]
    if DTYPE:
        cmd += ["--dtype", DTYPE]
    cmd += ["--micro-batch-size", str(MICRO),
           "--grad-accum-steps", str(ACCUM),
           "--train-shard", str(data_root / tree / "train.bin"),
           "--val-shard", str(data_root / tree / "val.bin"),
           "--ckpt-dir", str(ckpt_dir),
           "--ckpt-every", str(CKPT_EVERY),
           "--eval-every", str(EVAL_EVERY),
           "--keep-last-n", str(KEEP_LAST),
           "--log-every", "20"]
    if SMOKE:
        cmd.append("--smoke")
    if os.environ.get("LMA_FRESH", "1") == "1":
        cmd.append("--no-resume")
    log = open(WORK / f"{tree}_train.log", "w")
    print(f"[launch] {tree} on CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    procs.append((tree, subprocess.Popen(cmd, cwd=REPO, env=env,
                                         stdout=log, stderr=subprocess.STDOUT), log))
    if n_gpu < 2:
        # Only one card: run to completion before starting the next language.
        procs[-1][1].wait()

t0 = time.time()
results = {}
for tree, proc, log in procs:
    rc = proc.wait()
    log.close()
    results[tree] = rc
    print(f"[done] {tree} rc={rc}  ({time.time()-t0:.0f}s elapsed)", flush=True)

# --- 4. surface the logs in the kernel's own output ----------------------
for tree in TREES:
    p = WORK / f"{tree}_train.log"
    if not p.exists():
        continue
    print(f"\n{'='*70}\n{tree} log\n{'='*70}", flush=True)
    print(p.read_text()[-4000:], flush=True)

print(f"\n[summary] {results}", flush=True)
if any(rc != 0 for rc in results.values()):
    raise SystemExit(f"a training process failed: {results}")
print("[summary] ALL OK", flush=True)
