"""Kaggle kernel: measure training throughput by precision and batch size.

WHY THIS EXISTS
---------------
The completed Hindi run reached 12,227 tokens/s on a T4. Against the T4's fp32
peak (8.1 TFLOP/s) that is ~22% MFU -- healthy. Against its fp16 peak
(65 TFLOP/s) it is ~2.7% -- meaning we are getting NO tensor-core acceleration.

The suspected cause: the T4 is Turing (sm_75), whose tensor cores implement
fp16 but NOT bf16 -- bf16 tensor cores arrive with Ampere (sm_80). We train
under `autocast(bfloat16)`, so the maths likely falls back to fp32 plus
conversion overhead. `torch.cuda.is_bf16_supported()` returns True on this
card, but that means "works", not "is fast", which is exactly how the choice
went unexamined.

That was documented as a risk in trainer.py and never measured. This measures
it, before committing ~13h of GPU to a retrain.

WHAT IS COMPARED
----------------
Variants differ in precision and micro-batch, so seconds-per-step is NOT
comparable between them -- the metric reported is TOKENS PER SECOND.
Loss is printed for every variant as a sanity check: fp16 can overflow, and a
NaN would make a fast number worthless.
"""

import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/lma_repo")
CTX = 512
WARMUP, MEASURE = 5, 20

print("=" * 70, flush=True)
print("LMA — precision / batch throughput benchmark", flush=True)
print("=" * 70, flush=True)

import numpy as np      # noqa: E402
import torch            # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("FATAL: no GPU. Set Accelerator = GPU T4 x2.")
prop = torch.cuda.get_device_properties(0)
print(f"[env] torch {torch.__version__}  {prop.name}  sm_{prop.major}{prop.minor}  "
      f"{prop.total_memory/2**30:.1f} GiB", flush=True)
print(f"[env] is_bf16_supported() = {torch.cuda.is_bf16_supported()}  "
      f"(True means 'works', not 'accelerated')", flush=True)
# Tensor cores for fp32 matmuls; harmless for the mixed-precision variants.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
code_src = find_root("hindi/model/transformer.py")
if code_src is not None:
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
else:
    tb = next(IN.rglob("lma_code.tar.gz"), None)
    if tb is None:
        raise SystemExit("code dataset not found")
    with tarfile.open(tb) as tar:
        tar.extractall(REPO)
print(f"[code] {REPO}", flush=True)

sys.path.insert(0, str(REPO / "hindi" / "model"))
sys.path.insert(0, str(REPO / "hindi" / "train"))
from transformer import ModelConfig, TransformerLM   # noqa: E402
from trainer import build_optimizer                  # noqa: E402
from train_config import TrainConfig                 # noqa: E402

data_root = find_root("hindi/train.bin")
if data_root is None:
    raise SystemExit("shards not found")
tokens = np.memmap(data_root / "hindi" / "train.bin", dtype=np.uint16, mode="r")
print(f"[data] {len(tokens):,} tokens", flush=True)

dev = torch.device("cuda")
N_PARAMS = 24_070_656


def batch(micro, step):
    rng = np.random.default_rng(1337 + step)
    starts = rng.integers(0, len(tokens) - CTX - 1, size=micro)
    w = np.stack([np.asarray(tokens[s:s + CTX + 1], dtype=np.int64) for s in starts])
    x = torch.from_numpy(w[:, :-1]).to(dev, non_blocking=True)
    y = torch.from_numpy(w[:, 1:]).to(dev, non_blocking=True)
    return x, y


def run(name, dtype, micro, accum, use_scaler):
    """One variant. Returns tokens/s, peak VRAM and final loss."""
    torch.manual_seed(1337)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = ModelConfig(vocab_size=16000, lang_code="hi",
                      emb_dropout=0.0, attn_dropout=0.0, resid_dropout=0.0)
    model = TransformerLM(cfg).to(dev)
    opt = build_optimizer(model, TrainConfig())
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def one_step(step):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for m in range(accum):
            x, y = batch(micro, step * accum + m)
            if dtype is None:
                _, loss = model(x, y)
            else:
                with torch.autocast("cuda", dtype=dtype):
                    _, loss = model(x, y)
            scaler.scale(loss / accum).backward()
            total += float(loss) / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return total

    for s in range(WARMUP):
        one_step(s)
    torch.cuda.synchronize()
    t0 = time.time()
    last = 0.0
    for s in range(WARMUP, WARMUP + MEASURE):
        last = one_step(s)
    torch.cuda.synchronize()
    el = time.time() - t0

    tps = micro * accum * CTX
    tok_s = MEASURE * tps / el
    peak = torch.cuda.max_memory_allocated() / 2**30
    tflops = tok_s * 6 * N_PARAMS / 1e12
    # NOT `del model, opt` here: `one_step` closes over them, and deleting the
    # binding would leave that closure referencing dead names. Scope exit plus
    # the empty_cache() at the top of the next variant frees the memory.
    return {"name": name, "dtype": str(dtype), "micro": micro, "accum": accum,
            "scaler": use_scaler, "s_per_step": el / MEASURE, "tokens_per_s": tok_s,
            "peak_gib": peak, "tflops": tflops, "final_loss": last,
            # Normalised to the real run's 65,536 tokens/step so variants with
            # different micro-batches stay comparable.
            "hours_for_8732_steps": 8732 * 65536 / tok_s / 3600}


VARIANTS = [
    ("bf16  micro=8  (current)", torch.bfloat16, 8, 16, False),
    ("fp16  micro=8",            torch.float16,  8, 16, True),
    ("fp32  micro=8",            None,           8, 16, False),
    ("fp16  micro=16",           torch.float16, 16,  8, True),
    ("fp16  micro=32",           torch.float16, 32,  4, True),
    ("bf16  micro=32",           torch.bfloat16, 32,  4, False),
]

results = []
for name, dt, micro, accum, sc in VARIANTS:
    try:
        r = run(name, dt, micro, accum, sc)
        results.append(r)
        print(f"  {name:26} {r['tokens_per_s']:>9,.0f} tok/s  "
              f"{r['tflops']:5.2f} TFLOP/s  peak {r['peak_gib']:5.2f} GiB  "
              f"loss {r['final_loss']:.4f}", flush=True)
    except RuntimeError as e:
        print(f"  {name:26} FAILED: {str(e)[:80]}", flush=True)
        results.append({"name": name, "error": str(e)[:200]})

ok = [r for r in results if "tokens_per_s" in r]
if ok:
    base = next((r for r in ok if r["name"].startswith("bf16  micro=8")), ok[0])
    best = max(ok, key=lambda r: r["tokens_per_s"])
    print(f"\n{'='*70}")
    print(f"  baseline : {base['name']:26} {base['tokens_per_s']:>9,.0f} tok/s")
    print(f"  fastest  : {best['name']:26} {best['tokens_per_s']:>9,.0f} tok/s "
          f"({best['tokens_per_s']/base['tokens_per_s']:.2f}x)")
    hrs = 8732 * 65536 / best["tokens_per_s"] / 3600
    print(f"  -> 8,732 steps at 65,536 tok/step would take {hrs:.2f}h "
          f"(vs {8732*65536/base['tokens_per_s']/3600:.2f}h now)")
(WORK / "bench_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n[done]", flush=True)
