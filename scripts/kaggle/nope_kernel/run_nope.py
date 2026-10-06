"""Kaggle kernel: the optional BONUS ablation — Model L with no positional embeddings.

Pushed by scripts/kaggle/run.py --kernel-dir scripts/kaggle/nope_kernel.

Trains Nepali from scratch with the positional embedding table removed, then
runs the FULL Phase-2 evaluation suite on the result, in ONE session. Combined
because the alternative is a 276 MB checkpoint round-trip between two kernels
for no analytical benefit; ~6.6 h training + ~1 h evaluation sits well inside
the 12 h batch cap.

WHY NEPALI AND NOT HINDI
------------------------
Either language answers the spec's question. Nepali is chosen because its
standard model has already been evaluated at BOTH its full 13,547-step horizon
and at Hindi's matched 8,732-token budget, so the ablation inherits two
comparison points instead of one for the same GPU spend.

THE CONTROLLED VARIABLE
-----------------------
Everything below is copied from the Nepali baseline's own stored checkpoint
config, so the ablation differs from its comparator in exactly one
architectural respect: `use_positional_embeddings`.

  peak_lr 6e-4 · dropout 0.1 · warmup 500 · micro 8 x accum 16 · seed 1337
  LR horizon 13,547 steps · ckpt/eval every 250

ONE DELIBERATE DEVIATION: precision. The baseline ran bfloat16 and took 23 h.
The Phase-2 precision benchmark measured fp16 and bf16 producing identical
losses at 3.11x the throughput on a T4 (Turing tensor cores implement fp16,
not bf16), so this run uses fp16 to fit the day. The deviation is recorded
here and in the report rather than left for a reader to discover.
"""

import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/lma_repo")

TREE = "nepali"
MODEL_CONFIG = "configs/model_nope.yaml"      # the ablation, versioned in the repo
STEPS = 13547        # LR schedule horizon == the baseline's
PEAK_LR = "0.0006"   # baseline value, NOT Hindi's tuned 3e-3
DROPOUT = "0.1"      # baseline value
DTYPE = "float16"    # the one documented deviation; see module docstring
MICRO, ACCUM = 8, 16
CKPT_EVERY = EVAL_EVERY = 250
# step_8732 is Hindi's matched token budget. nepali/configs/train.yaml lists it
# in extra_ckpt_steps and milestones are exempt from pruning, so it survives
# keep_last_n and gives the second comparison point for free.
MILESTONE = 8732

print("=" * 70, flush=True)
print("LMA BONUS — no-positional-embeddings ablation (Model L / nepali)", flush=True)
print("=" * 70, flush=True)

# --- 1. hard GPU gate, before anything expensive -------------------------
import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print(f"[env] torch {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={n_gpu}",
      flush=True)
if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: no GPU. Kaggle resets Accelerator to 'None' on new sessions and "
        "on pushes that omit enable_gpu. Set Accelerator = GPU T4 x2."
    )
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} {p.name} {p.total_memory/2**30:.1f} GiB sm_{p.major}{p.minor}",
          flush=True)

# --- 2. dependencies BEFORE any compute ----------------------------------
# Learned the hard way: a missing metric library that surfaces only at the
# scoring step throws away everything computed before it.
for pkg, mod in (("sacrebleu", "sacrebleu"),):
    try:
        __import__(mod)
        print(f"[deps] {mod} already present", flush=True)
    except ImportError:
        print(f"[deps] installing {pkg}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"pip install {pkg} failed:\n{r.stdout}\n{r.stderr}")
        __import__(mod)
        print(f"[deps] {mod} OK", flush=True)


def find_root(marker: str):
    """Locate the directory containing `marker`, at ANY depth under /kaggle/input.

    Datasets do not mount at their slug: the measured layout is
    /kaggle/input/datasets/<owner>/<slug>/. Rather than encode a path Kaggle
    owns, search for a file the tree must contain.
    """
    leaf = Path(marker).name
    for hit in IN.rglob(leaf):
        root = Path(str(hit)[: -len(marker)]) if str(hit).endswith(marker) else None
        if root is not None and (root / marker).exists():
            return root
    return None


# --- 3. unpack code and locate shards ------------------------------------
print(f"[mount] /kaggle/input: {sorted(p.name for p in IN.glob('*'))}", flush=True)
REPO.mkdir(parents=True, exist_ok=True)
code_src = find_root(f"{TREE}/train/loop.py")
if code_src is None:
    tarball = next(IN.rglob("lma_code.tar.gz"), None)
    if tarball is None:
        raise SystemExit("code not found — is lma-code attached?")
    with tarfile.open(tarball) as tar:
        tar.extractall(REPO)
    print(f"[code] unpacked {tarball}", flush=True)
else:
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)   # /kaggle/input is read-only
    print(f"[code] copied {code_src} -> {REPO}", flush=True)

cfg_path = REPO / TREE / MODEL_CONFIG
if not cfg_path.exists():
    raise SystemExit(f"FATAL: ablation config missing: {cfg_path}. "
                     "The lma-code dataset predates the ablation — re-sync it.")
print(f"[cfg] {cfg_path}:", flush=True)
print("      " + cfg_path.read_text().replace("\n", "\n      "), flush=True)

data_root = find_root(f"{TREE}/train.bin")
if data_root is None:
    raise SystemExit(f"token shards not found under {IN}")
print(f"[data] shard root {data_root}", flush=True)

# The eval code reads shards from <tree>/data/tokenized/. Symlink rather than
# copy: these are gigabytes and /kaggle/input is read-only but readable.
dst = REPO / TREE / "data" / "tokenized"
dst.mkdir(parents=True, exist_ok=True)
for name in ("train.bin", "val.bin", "test.bin", "meta.json"):
    src = data_root / TREE / name
    if src.exists() and not (dst / name).exists():
        (dst / name).symlink_to(src)
    print(f"[data]   {name}: {'OK' if src.exists() else 'MISSING'}", flush=True)

# --- 4. train ------------------------------------------------------------
ckpt_dir = WORK / "checkpoints" / f"{TREE}_nope"
ckpt_dir.mkdir(parents=True, exist_ok=True)

train_cmd = [
    sys.executable, str(REPO / TREE / "train" / "loop.py"),
    "--model-config", str(cfg_path),
    "--steps", str(STEPS), "--stop-at", str(STEPS),
    "--peak-lr", PEAK_LR, "--dropout", DROPOUT, "--dtype", DTYPE,
    "--micro-batch-size", str(MICRO), "--grad-accum-steps", str(ACCUM),
    "--train-shard", str(data_root / TREE / "train.bin"),
    "--val-shard", str(data_root / TREE / "val.bin"),
    "--ckpt-dir", str(ckpt_dir),
    "--ckpt-every", str(CKPT_EVERY), "--eval-every", str(EVAL_EVERY),
    "--keep-last-n", "1", "--log-every", "20",
    "--no-resume",            # a fresh ablation, never a chain link
]
print(f"\n[train] {' '.join(train_cmd)}\n", flush=True)
t0 = time.time()
# Streamed to the console, not redirected to a file: the Phase-2 training
# kernel wrote its log to disk and an 11-hour run was completely opaque while
# it ran. Visible progress is worth more than tidy output here.
rc = subprocess.run(train_cmd, cwd=REPO).returncode
print(f"\n[train] rc={rc} after {(time.time()-t0)/3600:.2f} h", flush=True)
if rc != 0:
    raise SystemExit(f"training failed rc={rc}")

# --- 5. evaluate: final horizon AND the matched-token milestone ----------
saved = sorted(ckpt_dir.glob("step_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
print(f"[ckpt] saved: {[p.name for p in saved]}", flush=True)

targets = []
final = ckpt_dir / f"step_{STEPS}.pt"
if final.exists():
    targets.append((final, f"{TREE}_nope"))
milestone = ckpt_dir / f"step_{MILESTONE}.pt"
if milestone.exists():
    targets.append((milestone, f"{TREE}_nope_matched"))
if not targets:
    raise SystemExit(f"no checkpoint to evaluate in {ckpt_dir}")

rc_all = {}
for ck, tag in targets:
    outdir = WORK / "report" / "phase2" / tag
    cmd = [sys.executable, str(REPO / TREE / "eval" / "run_eval.py"),
           "--checkpoint", str(ck), "--out-dir", str(outdir),
           "--batch-size", "16", "--n-prompts", "200", "--new-tokens", "128"]
    print(f"\n{'='*70}\n[eval] {ck.name} -> report/phase2/{tag}\n{'='*70}", flush=True)
    rc_all[tag] = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[eval] {tag} rc={rc_all[tag]}  ({(time.time()-t0)/3600:.2f} h total)", flush=True)

print(f"\n[summary] eval {rc_all}", flush=True)
for p in sorted(WORK.rglob("*")):
    if p.is_file() and p.suffix in (".json", ".png", ".md", ".jsonl", ".pt"):
        print(f"  {p.relative_to(WORK)}  {p.stat().st_size:,} B", flush=True)
if any(v != 0 for v in rc_all.values()):
    raise SystemExit(f"an evaluation failed: {rc_all}")
print("[summary] ALL OK", flush=True)
