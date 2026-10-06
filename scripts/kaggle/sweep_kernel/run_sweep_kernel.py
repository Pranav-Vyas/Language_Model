"""Kaggle kernel: Hindi training-hyperparameter sweep.

WHAT THIS TESTS AND WHY
-----------------------
The completed Hindi run shows validation loss BELOW training loss at all 33
checkpoints (mean -0.12 nats). That is not overfitting, it is the opposite:
dropout is active during training and off during evaluation, so the model is
being regularised against overfitting that never occurs. With 572M tokens and
24M parameters seeing each token once, dropout 0.1 is plausibly a net cost.

The second suspicion is the learning rate: 6e-4 is the value nanoGPT uses for a
124M model, and smaller models generally tolerate more.

So the sweep fixes the ARCHITECTURE (d=384, L=10, h=6, ctx=512 -- changing it
would confound the Model H vs Model L comparison, which is the point of the
project) and varies only training hyperparameters.

COMPARABILITY
-------------
Every run uses `--steps 8732 --stop-at 1500`: the LR schedule horizon is the
full run, and the run merely stops early. The first 1,500 steps therefore
follow exactly the same schedule shape as the completed baseline, so val@1500
is directly comparable to the baseline's known 4.1865 -- which means the
baseline costs no GPU time to include.

Checkpoints go to /tmp, not /kaggle/working: they are not the product here and
would add ~1.1 GB to every output download.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/lma_repo")
OUT = WORK / "sweep"
STOP_AT = int(os.environ.get("LMA_SWEEP_STOP", "1500"))
HORIZON = int(os.environ.get("LMA_SWEEP_HORIZON", "8732"))

# (name, dropout, peak_lr). Baseline (0.1, 6e-4) is omitted deliberately: its
# curve already exists from the completed run.
CONFIGS = [
    ("A_drop0.0_lr6e-4",  0.0, 6e-4),   # isolates the dropout effect alone
    ("B_drop0.0_lr1.5e-3", 0.0, 1.5e-3),
    ("C_drop0.0_lr3e-3",  0.0, 3e-3),
    ("D_drop0.0_lr6e-3",  0.0, 6e-3),   # may diverge; that is itself an answer
]

print("=" * 70, flush=True)
print("LMA Phase 2 — Hindi hyperparameter sweep", flush=True)
print("=" * 70, flush=True)

import torch  # noqa: E402
if not torch.cuda.is_available():
    raise SystemExit("FATAL: no GPU. Set Accelerator = GPU T4 x2 and re-run.")
n_gpu = torch.cuda.device_count()
print(f"[env] torch {torch.__version__}  device_count={n_gpu}", flush=True)
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} {p.name} {p.total_memory/2**30:.1f} GiB sm_{p.major}{p.minor}",
          flush=True)


def find_root(marker: str):
    leaf = Path(marker).name
    for hit in IN.rglob(leaf):
        s = str(hit)
        if s.endswith(marker):
            root = Path(s[: -len(marker)])
            if (root / marker).exists():
                return root
    return None


REPO.mkdir(parents=True, exist_ok=True)
code_src = find_root("hindi/train/loop.py")
if code_src is not None:
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
    print(f"[code] copied {code_src}", flush=True)
else:
    tb = next(IN.rglob("lma_code.tar.gz"), None)
    if tb is None:
        raise SystemExit("code dataset not found")
    with tarfile.open(tb) as tar:
        tar.extractall(REPO)

data_root = find_root("hindi/train.bin")
if data_root is None:
    raise SystemExit("token shards not found")
print(f"[data] {data_root}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
results = {}

# Two at a time: one config per GPU, pinned with CUDA_VISIBLE_DEVICES so each
# process sees exactly one device.
for i in range(0, len(CONFIGS), max(1, n_gpu)):
    batch = CONFIGS[i:i + max(1, n_gpu)]
    procs = []
    for slot, (name, drop, lr) in enumerate(batch):
        rundir = OUT / name
        rundir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES=str(slot % max(1, n_gpu)))
        cmd = [sys.executable, str(REPO / "hindi" / "train" / "loop.py"),
               "--steps", str(HORIZON), "--stop-at", str(STOP_AT),
               "--peak-lr", str(lr), "--dropout", str(drop),
               "--micro-batch-size", "8", "--grad-accum-steps", "16",
               "--train-shard", str(data_root / "hindi" / "train.bin"),
               "--val-shard", str(data_root / "hindi" / "val.bin"),
               "--ckpt-dir", f"/tmp/sweep_ckpt/{name}",
               "--ckpt-every", "0", "--eval-every", "250",
               "--keep-last-n", "1", "--log-every", "50", "--no-resume"]
        log = open(rundir / "train.log", "w")
        print(f"[launch] {name} on cuda:{env['CUDA_VISIBLE_DEVICES']} "
              f"(dropout={drop}, peak_lr={lr:g})", flush=True)
        procs.append((name, subprocess.Popen(cmd, cwd=REPO, env=env,
                                             stdout=log, stderr=subprocess.STDOUT), log))
    for name, proc, log in procs:
        rc = proc.wait()
        log.close()
        results[name] = rc
        print(f"[done] {name} rc={rc}  ({time.time()-t0:.0f}s elapsed)", flush=True)

# --- collect val curves -------------------------------------------------
import re
print(f"\n{'='*70}\nSWEEP RESULTS (val loss at step {STOP_AT})\n{'='*70}", flush=True)
print("  baseline (dropout 0.1, lr 6e-4) val@1500 = 4.1865   [from the completed run]",
      flush=True)
summary = {"baseline_val_at_1500": 4.1865, "stop_at": STOP_AT,
           "horizon": HORIZON, "runs": {}}
for name, drop, lr in CONFIGS:
    f = OUT / name / "train.log"
    vals = []
    if f.exists():
        for line in f.read_text(errors="ignore").splitlines():
            m = re.search(r"\[(?:val|final)\] step ([\d,]+)\s+(?:loss|val_loss) ([\d.]+)", line)
            if m:
                vals.append((int(m.group(1).replace(",", "")), float(m.group(2))))
    final = vals[-1][1] if vals else None
    summary["runs"][name] = {"dropout": drop, "peak_lr": lr, "rc": results.get(name),
                             "val_curve": vals, "final_val": final}
    delta = (final - 4.1865) if final is not None else None
    print(f"  {name:22s} dropout {drop:<4} lr {lr:<8g} val {final if final is None else f'{final:.4f}'}"
          f"   {'' if delta is None else f'vs baseline {delta:+.4f}'}", flush=True)

(WORK / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n[summary] {results}", flush=True)
if any(v != 0 for v in results.values()):
    raise SystemExit(f"a sweep run failed: {results}")
print("[summary] ALL OK", flush=True)
