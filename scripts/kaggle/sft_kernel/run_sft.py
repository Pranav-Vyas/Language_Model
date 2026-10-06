"""Kaggle kernel: reasoning-finetune probe / production runs (T3.7-T3.8).

Pushed by scripts/kaggle/run.py --kernel-dir scripts/kaggle/sft_kernel.

THE DATA IS GENERATED HERE, NOT UPLOADED
----------------------------------------
`<lang>/reasoning/generate.py` is deterministic in (split, n, seed), so shipping
130 MB of JSONL to Kaggle would be uploading something the kernel can rebuild in
under a minute. Generating in-kernel is cheaper AND is a live proof that the
committed generator really does reproduce the dataset -- the claim the report
makes when it ships scripts instead of data.

FOUR MODES
----------
LMA_SFT_MODE=probe       run a grid of short configs to choose LR / CoT / masking
LMA_SFT_MODE=production  run the chosen config for both languages x seeds
LMA_SFT_MODE=lrprobe     re-select LR with the FIXED eval harness (T3.13)
LMA_SFT_MODE=abprobe     re-verify the CoT / masking A/Bs on it too (T3.14)

Both write per-run history.jsonl and a summary JSON to /kaggle/working so
`kaggle kernels output` brings the metrics home.
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

# The default names the run most recently EXECUTED, since Kaggle cannot
# receive environment variables at push time -- a parameter has to be a
# literal in the uploaded file. The reported production configuration is
# MODE=probe (see the run list below); MODE=lrprobe is the T3.13 LR
# re-selection on the corrected eval harness.
MODE = os.environ.get("LMA_SFT_MODE", "abprobe")
# Curriculum arm: train on 1-3 hops and probe 4-hop, instead of the default
# 1-2 hops probing 3-hop. Writes and reads a SEPARATE data directory so the
# committed baseline dataset is untouched and both results stay comparable.
CURRICULUM = os.environ.get("LMA_SFT_CURRICULUM", "0") == "1"
DATA_DIRNAME = "reasoning_data_curriculum" if CURRICULUM else "reasoning_data"
# Probe round 2: val/test peak by epoch ~9 and are FLAT from ~15 to 29 with
# training loss already 0.0000 by epoch 10, while the 3-hop OOD score actually
# DECAYS after its epoch-9 peak (0.287 -> 0.233). There is no grokking here to
# wait for, so the planned 100 epochs would burn GPU hours to make one metric
# worse. 15 sits just past the peak.
EPOCHS = int(os.environ.get("LMA_SFT_EPOCHS", "15"))
BATCH = int(os.environ.get("LMA_SFT_BATCH", "256"))
EVAL_EVERY = int(os.environ.get("LMA_SFT_EVAL_EVERY", "250"))
EVAL_N = int(os.environ.get("LMA_SFT_EVAL_N", "300"))
TRAIN_N = int(os.environ.get("LMA_SFT_TRAIN_N", "60000"))
SEEDS = [int(s) for s in os.environ.get("LMA_SFT_SEEDS", "1337,42,7").split(",")]

print("=" * 70, flush=True)
print(f"LMA Phase 3 — reasoning finetune ({MODE})", flush=True)
print("=" * 70, flush=True)

import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={n_gpu}",
      flush=True)
if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: no GPU. Kaggle resets Accelerator to 'None' on new sessions and "
        "on pushes that omit enable_gpu. Set Accelerator = GPU T4 x2.")
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} {p.name} {p.total_memory/2**30:.1f} GiB "
          f"sm_{p.major}{p.minor}", flush=True)


def find_root(marker: str):
    """Locate the directory containing `marker` at ANY depth under /kaggle/input.

    Datasets do not mount at their slug; the measured layout is
    /kaggle/input/datasets/<owner>/<slug>/. Search rather than hardcode.
    """
    leaf = Path(marker).name
    for hit in IN.rglob(leaf):
        root = Path(str(hit)[: -len(marker)]) if str(hit).endswith(marker) else None
        if root is not None and (root / marker).exists():
            return root
    return None


print(f"[mount] {sorted(p.name for p in IN.glob('*'))}", flush=True)
REPO.mkdir(parents=True, exist_ok=True)
code_src = find_root("hindi/train/finetune.py")
if code_src is None:
    tarball = next(IN.rglob("lma_code.tar.gz"), None)
    if tarball is None:
        raise SystemExit("code not found — is lma-code attached AND re-synced "
                         "since finetune.py was written?")
    with tarfile.open(tarball) as tar:
        tar.extractall(REPO)
    print(f"[code] unpacked {tarball}", flush=True)
else:
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
    print(f"[code] copied {code_src}", flush=True)

for need in ("hindi/train/finetune.py", "hindi/train/sft_data.py",
             "hindi/reasoning/generate.py", "nepali/reasoning/generate.py"):
    if not (REPO / need).exists():
        raise SystemExit(f"FATAL: {need} missing from the code dataset — re-sync it.")

# --- regenerate the datasets in-kernel ------------------------------------
# Probe mode targets whichever tree is being tuned; production does both.
PROBE_TREE = os.environ.get("LMA_SFT_PROBE_TREE", "nepali")
TREES = ["hindi", "nepali"]   # both trees needed in either mode now
SPLIT_N = {"train": TRAIN_N, "val": 6000, "test": 6000, "ood": 2000}
for tree in TREES:
    for split, n in SPLIT_N.items():
        cmd = [sys.executable, str(REPO / tree / "reasoning" / "generate.py"),
               "--split", split, "--n", str(n)]
        if CURRICULUM:
            cmd.append("--curriculum")
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"generate {tree}/{split} failed:\n{r.stdout}\n{r.stderr}")
    got = {s: sum(1 for _ in open(REPO / tree / DATA_DIRNAME / f"{s}.jsonl"))
           for s in SPLIT_N}
    # Hop counts are the point of this arm, so they are printed, not assumed.
    import json as _json
    hops = {}
    for s in SPLIT_N:
        with open(REPO / tree / DATA_DIRNAME / f"{s}.jsonl", encoding="utf-8") as fh:
            hops[s] = sorted({_json.loads(l)["n_hops"] for l in fh})
    print(f"[data] {tree}: {got}  curriculum={CURRICULUM}  hops={hops}", flush=True)

# --- locate the pretrained checkpoints ------------------------------------
# TWO datasets contain a file called step_8732.pt and they are DIFFERENT models:
#   lma-ckpt-tuned/step_8732.pt        Model H after the LR/dropout sweep, val 3.3785
#   lma-ckpt-final/hindi/step_8732.pt  Model H BEFORE tuning,               val 3.5574
# Picking by filename, or by "does the path mention hindi", selects the wrong
# one -- silently finetuning from the weaker model and confounding every result.
# So each checkpoint is identified by its RECORDED val_loss, which is unique to
# the run that produced it, and the kernel refuses to start if it cannot find a
# match. Identity is verified, not inferred from a path.
EXPECT = {
    "hindi":  {"step": 8732,  "val_loss": 3.3785, "what": "Model H, tuned"},
    "nepali": {"step": 13547, "val_loss": 3.7040, "what": "Model L, full epoch"},
}
TOL = 0.01

CKPT = {}
for tree in TREES:
    want = EXPECT[tree]
    cands, seen = [], []
    for h in IN.rglob("step_*.pt"):
        try:
            meta = torch.load(h, map_location="cpu", weights_only=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"[ckpt] unreadable {h}: {exc}", flush=True)
            continue
        seen.append((str(h), meta.get("step"), meta.get("lang_code"),
                     meta.get("val_loss")))
        if (meta.get("lang_code") == {"hindi": "hi", "nepali": "ne"}[tree]
                and int(meta.get("step", -1)) == want["step"]
                and meta.get("val_loss") is not None
                and abs(float(meta["val_loss"]) - want["val_loss"]) < TOL):
            cands.append(h)
    if not cands:
        print("[ckpt] candidates seen:", flush=True)
        for row in seen:
            print(f"        {row}", flush=True)
        raise SystemExit(
            f"FATAL: no checkpoint matching {tree} {want} within {TOL}. "
            "Attach the right dataset, or update EXPECT if the intended "
            "checkpoint genuinely changed.")
    CKPT[tree] = cands[0]
    print(f"[ckpt] {tree} <- {cands[0]}  ({want['what']}, "
          f"step {want['step']}, val {want['val_loss']}) "
          f"[{len(cands)} match(es)]", flush=True)

# --- build the run list ----------------------------------------------------
runs = []
if MODE == "abprobe":
    # T3.14 -- RE-VERIFY the CoT and prompt-masking A/Bs on the FIXED harness.
    #
    # Both were originally decided on numbers from the buggy left-padding
    # decoder, exactly like the LR choice that turned out to be wrong (see
    # MODE=lrprobe). Their margins were much wider than the LR's -- answer-only
    # .813 vs CoT .263, masked .747 vs unmasked .033 -- so neither is likely to
    # flip, but "likely" is not a measurement, and the whole point of the
    # correction pass is that decisions rest on numbers that were computed
    # correctly. Hindi only: it is the model that is not compute-limited here,
    # and these are recipe questions, not language questions.
    for seed in SEEDS[:2]:
        runs.append(dict(tree="hindi", lr=3e-5,
                         target="target_answer_only", mask=True, seed=seed))
        runs.append(dict(tree="hindi", lr=3e-5,
                         target="target_with_trace", mask=True, seed=seed))
        runs.append(dict(tree="hindi", lr=3e-5,
                         target="target_answer_only", mask=False, seed=seed))
elif MODE == "lrprobe":
    # T3.13 -- RE-SELECT the finetune LR with the FIXED evaluation harness.
    #
    # The per-language LR choice recorded in BEST_LR below was made on numbers
    # produced by the batched-decoding path BEFORE the right-padding bug was
    # found. These models have learned ABSOLUTE positional embeddings and
    # `forward()` takes no attention mask, so left-padding shifted every
    # position index and corrupted every generated answer. Nepali's margin was
    # .190 (3e-6) vs .133 (3e-5) -- narrow enough for the bug alone to flip it,
    # and it selected an LR 10x below Hindi's, against the locked decision to
    # hold the finetune LR EQUAL across languages so that the H-vs-L reasoning
    # comparison is not confounded by optimisation.
    #
    # Independent evidence that 3e-6 UNDER-TRAINS Model L, neither piece of
    # which depends on the buggy decoder: its SFT loss plateaus at 0.130 with
    # grad-norm still 3-9 while the schedule decays to 3e-7 (Hindi reaches
    # 0.0000 with grad-norm 0.00), and the finetuned model's attention shows no
    # answer-retrieval circuit at all (report/phase3/nepali/attention_compare).
    #
    # So the sweep is re-run with the corrected harness. Selection is on
    # VALIDATION only, exactly as the original sweep was.
    LRS = [float(x) for x in
           os.environ.get("LMA_SFT_LRS", "3e-6,1e-5,3e-5").split(",")]
    for tree in [t for t in
                 os.environ.get("LMA_SFT_TREES", "nepali").split(",") if t]:
        for lr in LRS:
            for seed in SEEDS:
                runs.append(dict(tree=tree, lr=lr,
                                 target="target_answer_only", mask=True,
                                 seed=seed))
elif MODE == "probe":
    # THE REPORTED CONFIGURATION -- reproduces both committed result sets:
    #   LMA_SFT_CURRICULUM=0  (default)  -> report/phase3/sft_baseline/
    #   LMA_SFT_CURRICULUM=1             -> report/phase3/sft_curriculum/
    # The two arms are identical in every respect except the hop distribution,
    # which is what makes the comparison between them a controlled one.
    #
    # How these values were chosen (all on validation, never on test):
    #   * LR is selected PER LANGUAGE, not shared. The two models were pretrained
    #     at different scales (peak 3e-3 vs 6e-4), so one raw finetune LR is a
    #     different effective rate for each. Full sweeps:
    #       Hindi   3e-6 .663 | 3e-5 .713                          -> 3e-5
    #       Nepali  3e-6 .190 | 3e-5 .133 | 1e-4 .003 | 1e-3 .000 -> 3e-6
    #   * answer-only targets: reasoning traces lost decisively at 24M (.263
    #     vs .813), once the eval harness stopped truncating them.
    #   * prompt masked: unmasked lost 22x (.033 vs .747).
    #   * 15 epochs: held-out accuracy peaks by ~9 and is flat to 29, with the
    #     OOD score DECAYING after its peak; 100 epochs would only cost GPU.
    BEST_LR = {"hindi": 3e-5, "nepali": 3e-6}
    for tree in ("hindi", "nepali"):
        for seed in (1337, 42, 7):
            runs.append(dict(tree=tree, lr=BEST_LR[tree],
                             target="target_answer_only", mask=True, seed=seed))
else:
    # 3e-5 beat 1e-4 on every held-out metric (val .813 vs .747, test .847 vs
    # .740); 3e-4 and 1e-3 were destructive. Lower is better in this regime.
    lr = float(os.environ.get("LMA_SFT_LR", "3e-5"))
    # Reasoning traces LOSE decisively at 24M: val .263 vs .813 answer-only.
    tf = os.environ.get("LMA_SFT_TARGET", "target_answer_only")
    mask = os.environ.get("LMA_SFT_MASK", "1") == "1"
    for tree in TREES:
        for seed in SEEDS:
            runs.append(dict(tree=tree, lr=lr, target=tf, mask=mask, seed=seed))

print(f"\n[plan] {len(runs)} run(s), {EPOCHS} epochs each, batch {BATCH}", flush=True)
for i, r in enumerate(runs):
    print(f"  {i}: {r['tree']} lr={r['lr']:g} {r['target']} "
          f"mask={r['mask']} seed={r['seed']}", flush=True)


def tag_of(r, i):
    t = f"{r['tree']}_lr{r['lr']:g}_{'cot' if 'trace' in r['target'] else 'ans'}"
    t += "_mask" if r["mask"] else "_nomask"
    return f"{t}_s{r['seed']}_{i}"


def launch(r, i, gpu):
    tag = tag_of(r, i)
    out = WORK / "sft" / tag
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / r["tree"] / "train" / "finetune.py"),
           "--init-from", str(CKPT[r["tree"]]),
           "--data-dir", str(REPO / r["tree"] / DATA_DIRNAME),
           "--ckpt-dir", str(out),
           "--epochs", str(EPOCHS), "--batch-size", str(BATCH),
           "--peak-lr", str(r["lr"]), "--target-field", r["target"],
           "--eval-every", str(EVAL_EVERY), "--eval-n", str(EVAL_N),
           "--ckpt-every", str(10**9),          # only the final checkpoint
           "--log-every", "100", "--seed", str(r["seed"]),
           "--dtype", "float16"]
    if not r["mask"]:
        cmd.append("--no-mask-prompt")
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
    log = open(WORK / "sft" / f"{tag}.log", "w")
    print(f"[launch] {tag} on gpu{gpu}", flush=True)
    return tag, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT), log, out


# --- execute, n_gpu at a time ---------------------------------------------
t0 = time.time()
results = {}
lanes = max(1, n_gpu)
for start in range(0, len(runs), lanes):
    batch = runs[start:start + lanes]
    procs = [launch(r, start + j, j % lanes) for j, r in enumerate(batch)]
    for (tag, proc, log, out), r in zip(procs, batch):
        rc = proc.wait()
        log.close()
        hist = out / "history.jsonl"
        finals = []
        if hist.exists():
            for line in hist.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") in ("eval", "final"):
                    finals.append(rec)
        results[tag] = {"config": {k: (str(v) if k == "target" else v)
                                   for k, v in r.items()},
                        "rc": rc, "evals": finals}
        last = finals[-1] if finals else {}
        summary = "  ".join(
            f"{s}={last[s]['exact_match']:.3f}" for s in ("train", "val", "test", "ood")
            if s in last)
        print(f"[done] {tag} rc={rc}  {summary}  "
              f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
        if rc != 0:
            print((WORK / "sft" / f"{tag}.log").read_text()[-2500:], flush=True)

(WORK / "sft_summary.json").write_text(
    json.dumps({"mode": MODE, "epochs": EPOCHS, "batch": BATCH,
                "results": results}, indent=2), encoding="utf-8")

print(f"\n{'='*70}\n[summary] {MODE}, {(time.time()-t0)/60:.0f} min\n{'='*70}", flush=True)
hdr = f"{'run':44s} {'train':>7s} {'val':>7s} {'test':>7s} {'ood':>7s}"
print(hdr, flush=True)
for tag, r in results.items():
    last = r["evals"][-1] if r["evals"] else {}
    cells = "".join(f"{last[s]['exact_match']:>8.3f}" if s in last else f"{'-':>8s}"
                    for s in ("train", "val", "test", "ood"))
    print(f"{tag:44s}{cells}", flush=True)

bad = [t for t, r in results.items() if r["rc"] != 0]
if bad:
    raise SystemExit(f"failed runs: {bad}")
print("[summary] ALL OK", flush=True)
