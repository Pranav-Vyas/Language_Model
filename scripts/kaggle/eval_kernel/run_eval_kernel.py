"""Kaggle kernel: run the Phase-2 evaluation suite for BOTH languages.

Pushed by scripts/kaggle/run_eval.py. Reads code, token shards and the final
checkpoints from attached datasets; writes every artifact to /kaggle/working
so `kaggle kernels output` brings them home.

Evaluation is far cheaper than training, so both languages run SEQUENTIALLY on
one GPU here rather than one per card. That keeps the output logs readable in
the kernel console -- the training kernel's file redirection made an 11-hour
run completely opaque, which was the wrong trade.
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
REPO = Path("/tmp/lma_repo")

N_PROMPTS = int(os.environ.get("LMA_N_PROMPTS", "200"))
NEW_TOKENS = int(os.environ.get("LMA_NEW_TOKENS", "128"))
MAX_WINDOWS = int(os.environ.get("LMA_MAX_WINDOWS", "0"))   # 0 = score fully
BATCH = int(os.environ.get("LMA_BATCH", "16"))

print("=" * 70, flush=True)
print("LMA Phase 2 — evaluation (T2.12 / T2.13 / T2.14)", flush=True)
print("=" * 70, flush=True)

# Kaggle's image ships neither sacrebleu nor rouge-score. Installed here,
# BEFORE any model work, so a missing metric library fails in seconds rather
# than after 7 minutes of generation compute has already been done and thrown
# away (which is exactly what happened on the first attempt).
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

import torch  # noqa: E402
if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: no GPU. Kaggle resets Accelerator to 'None' for every new "
        "session. Set Accelerator = GPU T4 x2 and re-run.")
print(f"[env] torch {torch.__version__}  device_count={torch.cuda.device_count()}",
      flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} {p.name} {p.total_memory/2**30:.1f} GiB sm_{p.major}{p.minor}",
          flush=True)

print(f"[mount] {sorted(x.name for x in IN.glob('*'))}", flush=True)


def find_root(marker: str):
    """Locate the mount containing `marker`, at any depth (see the training kernel)."""
    leaf = Path(marker).name
    for hit in IN.rglob(leaf):
        s = str(hit)
        if s.endswith(marker):
            root = Path(s[: -len(marker)])
            if (root / marker).exists():
                return root
    return None


REPO.mkdir(parents=True, exist_ok=True)
code_src = find_root("hindi/eval/run_eval.py")
if code_src is not None:
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
    print(f"[code] copied {code_src}", flush=True)
else:
    tb = next(IN.rglob("lma_code.tar.gz"), None)
    if tb is None:
        raise SystemExit("code dataset not found")
    with tarfile.open(tb) as tar:
        tar.extractall(REPO)
    print(f"[code] unpacked {tb}", flush=True)

data_root = find_root("hindi/train.bin")
if data_root is None:
    raise SystemExit(f"token shards not found under {IN}")
print(f"[data] shards {data_root}", flush=True)


def resolve_ckpt(tree: str, name: str):
    """Locate one checkpoint, whatever shape the dataset ended up in.

    Kaggle's layout depends on how many top-level directories were staged: a
    dataset with several keeps `<tree>/<file>`, but one staged with a SINGLE
    directory came back flattened to `<file>` at the root. Depending on either
    shape fails on the other, so try the nested path first and then fall back
    to the bare filename.
    """
    root = find_root(f"{tree}/{name}")
    if root is not None:
        return root / tree / name
    hits = [h for h in IN.rglob(name) if h.is_file()]
    if hits:
        print(f"[ckpt] flat layout: {hits[0]}", flush=True)
        return hits[0]
    return None

# The eval code reads shards from <tree>/data/tokenized/ and writes under
# report/phase2/<tree>/. /kaggle/input is read-only, so symlink the shards into
# the writable repo copy rather than copying 2.8 GB.
for tree in ("hindi", "nepali"):
    dst = REPO / tree / "data" / "tokenized"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("train.bin", "val.bin", "test.bin", "meta.json"):
        src = data_root / tree / name
        if src.exists() and not (dst / name).exists():
            (dst / name).symlink_to(src)

# Which checkpoints to evaluate, as "tree:file,tree:file". Default covers the
# original Phase-2 set; the tuned-Hindi retrain evaluates only its own model.
_spec = os.environ.get("LMA_EVAL_CKPTS",
                       "hindi:step_8732.pt")
CKPTS = {}
for item in [x for x in _spec.split(",") if x]:
    tree, name = item.split(":")
    CKPTS.setdefault(tree, []).append(name)
# Output directory suffix, so a tuned run does not overwrite the original's
# results in report/phase2/hindi/.
OUT_TAG = os.environ.get("LMA_OUT_TAG", "_tuned")

t0 = time.time()
rc_all = {}
for tree, names in CKPTS.items():
    for name in names:
        ck = resolve_ckpt(tree, name)
        if ck is None or not ck.exists():
            print(f"[skip] {tree}/{name} not found in any attached dataset",
                  flush=True)
            continue
        # The matched-token checkpoint gets its own output dir so it does not
        # overwrite the full-corpus results for the same language.
        tag = "" if name == names[0] else "_matched"
        outdir = WORK / "report" / "phase2" / f"{tree}{tag}{OUT_TAG}"
        cmd = [sys.executable, str(REPO / tree / "eval" / "run_eval.py"),
               "--checkpoint", str(ck), "--out-dir", str(outdir),
               "--batch-size", str(BATCH),
               "--n-prompts", str(N_PROMPTS), "--new-tokens", str(NEW_TOKENS)]
        if MAX_WINDOWS:
            cmd += ["--max-windows", str(MAX_WINDOWS)]
        print(f"\n{'='*70}\n[run] {tree} / {name}\n{'='*70}", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        rc_all[f"{tree}/{name}"] = r.returncode
        print(f"[run] rc={r.returncode}  ({time.time()-t0:.0f}s elapsed)", flush=True)

print(f"\n[summary] {rc_all}", flush=True)
for p in sorted((WORK / "report").rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(WORK)}  {p.stat().st_size:,} B", flush=True)
if any(v != 0 for v in rc_all.values()):
    raise SystemExit(f"an evaluation failed: {rc_all}")
print("[summary] ALL OK", flush=True)
