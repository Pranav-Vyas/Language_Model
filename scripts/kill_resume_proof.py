"""T2.8 — prove a killed training run resumes bit-identically.

Language-agnostic harness (takes `--tree hindi|nepali`); holds no language
data and imports nothing from either tree at module level.

WHY THIS EXISTS
---------------
"Resume-capable" is a hard spec constraint, and the usual demonstration --
restart the script, watch it print a loss -- proves almost nothing. A run that
restores weights but drops the optimizer state, or the RNG, or silently
restarts the data order, does exactly that: it resumes, it prints a plausible
loss, and it is a different run from the one that was interrupted. Nothing
raises.

So this proves four things instead, against an uninterrupted REFERENCE run:

  1. REAL KILL          SIGKILL (signal 9) to a live subprocess mid-step.
                        Not a graceful exit, not a KeyboardInterrupt -- the
                        process gets no chance to flush, close, or clean up.
                        This is what proves the on-disk checkpoint is
                        self-consistent rather than finished by an exit hook.
  2. SAME DATA          batch fingerprints at N+1, N+2 equal the reference's.
  3. SAME NUMBERS       loss and grad-norm at N+1, N+2 equal the reference's
                        to within float noise -- a direct numeric comparison,
                        not "looks reasonable".
  4. EVERY COMPONENT    weights, AdamW m and v, LR, step, config, lang_code
                        each asserted individually on the loaded checkpoint.

Run:  python scripts/kill_resume_proof.py --tree hindi
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

KILL_AT = 3          # checkpoint written here, then the process is killed
TOTAL = 6            # reference runs this far; resume must reproduce 4..6


def _import_tree(tree: str):
    """Import one tree's train+model modules. One tree per process, never both."""
    for sub in ("model", "train"):
        path = str(REPO / tree / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import checkpoint as ckpt_mod
    import trainer as trainer_mod
    from train_config import load_model_config, load_train_config
    from transformer import ModelConfig, TransformerLM
    return ckpt_mod, trainer_mod, load_model_config, load_train_config, ModelConfig, TransformerLM


def _cfgs(tree: str, ckpt_dir: Path, load_model_config, load_train_config, ModelConfig):
    """Small, fast, but the REAL architecture and the REAL corpus shards."""
    tcfg = load_train_config(REPO / tree / "configs" / "train.yaml")
    tcfg.micro_batch_size = 2
    tcfg.grad_accum_steps = 1
    tcfg.max_steps = TOTAL
    tcfg.warmup_steps = 2          # exercise BOTH legs of the schedule
    tcfg.eval_every = 0
    tcfg.ckpt_every = KILL_AT
    tcfg.log_every = 1
    tcfg.keep_last_n = 10
    tcfg.record_batch_hash = True
    tcfg.ckpt_dir = str(ckpt_dir)
    tcfg.train_shard = str(REPO / tree / "data" / "tokenized" / "train.bin")
    tcfg.val_shard = str(REPO / tree / "data" / "tokenized" / "val.bin")
    mcfg = load_model_config(REPO / tree / "configs" / "model.yaml", ModelConfig)
    return mcfg, tcfg


def _seed(n: int) -> None:
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)


def reference_run(tree, ckpt_dir, snapshot_at):
    """Uninterrupted run. Returns per-step records + a snapshot at `snapshot_at`."""
    mods = _import_tree(tree)
    ckpt_mod, trainer_mod, lmc, ltc, ModelConfig, TransformerLM = mods
    mcfg, tcfg = _cfgs(tree, ckpt_dir, lmc, ltc, ModelConfig)
    tcfg.ckpt_every = 0                      # reference writes nothing

    _seed(tcfg.seed)
    model = TransformerLM(mcfg)
    tr = trainer_mod.Trainer(model, mcfg, tcfg, torch.device("cpu"), contextlib.nullcontext)

    snap = {}
    tr.train(max_steps=snapshot_at)
    snap["weights"] = {k: v.clone() for k, v in tr.model.state_dict().items()}
    # deepcopy, NOT state_dict() directly: state_dict returns live references
    # into the optimizer's buffers, so without this the "step 3 snapshot" keeps
    # mutating until step 6 and the comparison is against the wrong state.
    snap["opt"] = copy.deepcopy(tr.optimizer.state_dict())
    snap["lr_next"] = tr.schedule.lr_at(snapshot_at)
    tr.train(max_steps=TOTAL)
    return tr.history, snap


def killed_run(tree: str, ckpt_dir: Path) -> dict:
    """Launch a REAL subprocess, wait for the checkpoint, then SIGKILL it."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, str(REPO / tree / "train" / "loop.py"),
           "--steps", str(TOTAL), "--micro-batch-size", "2", "--grad-accum-steps", "1",
           "--ckpt-every", str(KILL_AT), "--eval-every", "0", "--log-every", "1",
           "--warmup-steps", "2",
           "--record-batch-hash", "--allow-cpu", "--no-resume",
           "--ckpt-dir", str(ckpt_dir)]
    proc = subprocess.Popen(cmd, cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    target = ckpt_dir / f"step_{KILL_AT}.pt"
    deadline = time.time() + 600
    while time.time() < deadline:
        if target.exists() and target.stat().st_size > 0:
            # Give it a moment to be mid-way through the NEXT step, so the kill
            # genuinely lands during computation rather than between steps.
            time.sleep(1.5)
            break
        if proc.poll() is not None:
            raise RuntimeError(f"subprocess exited early:\n{proc.stdout.read()}")
        time.sleep(0.05)
    else:
        proc.kill()
        raise TimeoutError("checkpoint never appeared")

    os.kill(proc.pid, signal.SIGKILL)        # <-- the real kill, signal 9
    rc = proc.wait(timeout=30)
    out = proc.stdout.read()
    return {"returncode": rc, "pid": proc.pid,
            "signalled": rc == -signal.SIGKILL,
            "last_lines": [l for l in out.strip().splitlines() if l.strip()][-3:]}


def resumed_run(tree: str, ckpt_dir: Path, ref_snap: dict) -> tuple[list, dict]:
    """Resume from the killed process's checkpoint; verify every component."""
    mods = _import_tree(tree)
    ckpt_mod, trainer_mod, lmc, ltc, ModelConfig, TransformerLM = mods
    mcfg, tcfg = _cfgs(tree, ckpt_dir, lmc, ltc, ModelConfig)

    # Deliberately seed DIFFERENTLY from the reference. If resume works only
    # because the process happened to start from the same seed, this exposes
    # it: every reproduced number must come from the checkpoint, not the seed.
    _seed(tcfg.seed + 999)
    model = TransformerLM(mcfg)
    tr = trainer_mod.Trainer(model, mcfg, tcfg, torch.device("cpu"), contextlib.nullcontext)

    path = ckpt_dir / f"step_{KILL_AT}.pt"
    raw = ckpt_mod.load_checkpoint(path, expected_lang_code=mcfg.lang_code)

    checks: dict[str, object] = {}
    checks["step"] = (int(raw["step"]), int(raw["step"]) == KILL_AT)
    checks["lang_code"] = (raw["lang_code"], raw["lang_code"] == mcfg.lang_code)
    checks["config_matches"] = all(
        raw["model_cfg"][k] == getattr(mcfg, k)
        for k in ("d_model", "n_layers", "n_heads", "d_ff", "max_seq_len",
                  "vocab_size", "tie_weights"))
    checks["has_rng_state"] = bool(raw.get("rng_state"))
    checks["format_version"] = raw.get("format_version")

    # weights, tensor by tensor, against the reference at the same step
    wdiff = max((raw["model_state"][k].float() - v.float()).abs().max().item()
                for k, v in ref_snap["weights"].items())
    checks["weights_max_abs_diff"] = wdiff

    # AdamW's m (exp_avg) and v (exp_avg_sq), individually
    ref_state, got_state = ref_snap["opt"]["state"], raw["optimizer_state"]["state"]
    checks["opt_param_count"] = (len(got_state), len(got_state) == len(ref_state))
    m_diff = max((got_state[i]["exp_avg"] - ref_state[i]["exp_avg"]).abs().max().item()
                 for i in ref_state)
    v_diff = max((got_state[i]["exp_avg_sq"] - ref_state[i]["exp_avg_sq"]).abs().max().item()
                 for i in ref_state)
    checks["adam_m_max_abs_diff"] = m_diff
    checks["adam_v_max_abs_diff"] = v_diff
    checks["adam_step_counter"] = int(next(iter(got_state.values()))["step"])

    # restore, then check the schedule lands on the right LR for the next step
    step = ckpt_mod.restore_into(raw, model=tr.model, optimizer=tr.optimizer)
    tr.step = step
    checks["lr_at_resume"] = (tr.schedule.lr_at(step), ref_snap["lr_next"],
                              abs(tr.schedule.lr_at(step) - ref_snap["lr_next"]) < 1e-12)

    tr.train(max_steps=TOTAL)
    return tr.history, checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Kill-and-resume proof (T2.8).")
    ap.add_argument("--tree", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    work = Path(args.workdir or f"/tmp/kr_proof_{args.tree}")
    shutil.rmtree(work, ignore_errors=True)
    ref_dir, kill_dir = work / "ref", work / "killed"

    print(f"\n{'='*72}\nKILL-AND-RESUME PROOF — {args.tree}\n{'='*72}")

    print(f"\n[1/4] reference run (uninterrupted, {TOTAL} steps)")
    ref_hist, snap = reference_run(args.tree, ref_dir, KILL_AT)

    print(f"\n[2/4] killed run — subprocess, SIGKILL after step {KILL_AT}")
    kill_info = killed_run(args.tree, kill_dir)
    print(f"      pid {kill_info['pid']}  returncode {kill_info['returncode']} "
          f"(-9 = SIGKILL)  killed_by_signal={kill_info['signalled']}")

    print(f"\n[3/4] resume from step_{KILL_AT}.pt (fresh process state, different seed)")
    res_hist, checks = resumed_run(args.tree, kill_dir, snap)

    print(f"\n[4/4] comparison\n")
    ref_by = {r["step"]: r for r in ref_hist}
    res_by = {r["step"]: r for r in res_hist}
    compared = sorted(set(ref_by) & set(res_by))

    print(f"  {'step':>4} | {'batch_hash ref':>16} {'resumed':>16} {'ok':>3} | "
          f"{'loss ref':>10} {'resumed':>10} {'delta':>9} | {'gnorm ref':>9} "
          f"{'resumed':>9} {'delta':>9}")
    print(f"  {'-'*4}-+-{'-'*16}-{'-'*16}-{'-'*3}-+-{'-'*10}-{'-'*10}-{'-'*9}-+-"
          f"{'-'*9}-{'-'*9}-{'-'*9}")
    all_ok = True
    for s in compared:
        a, b = ref_by[s], res_by[s]
        hok = a.get("batch_hash") == b.get("batch_hash")
        dl = abs(a["loss"] - b["loss"]); dg = abs(a["grad_norm"] - b["grad_norm"])
        ok = hok and dl < 1e-6 and dg < 1e-4
        all_ok &= ok
        print(f"  {s:>4} | {a.get('batch_hash',''):>16} {b.get('batch_hash',''):>16} "
              f"{'OK' if hok else 'X':>3} | {a['loss']:>10.6f} {b['loss']:>10.6f} "
              f"{dl:>9.2e} | {a['grad_norm']:>9.4f} {b['grad_norm']:>9.4f} {dg:>9.2e}")

    print(f"\n  checkpoint contents:")
    for k, v in checks.items():
        print(f"    {k:<24} {v}")

    verdicts = {
        "1. real SIGKILL (rc == -9)": kill_info["signalled"],
        "2. batch hashes identical": all(ref_by[s].get("batch_hash") == res_by[s].get("batch_hash")
                                         for s in compared),
        "3. loss & grad-norm match reference": all_ok,
        "4a. weights restored": checks["weights_max_abs_diff"] == 0.0,
        "4b. AdamW m restored": checks["adam_m_max_abs_diff"] == 0.0,
        "4c. AdamW v restored": checks["adam_v_max_abs_diff"] == 0.0,
        "4d. LR at correct schedule point": checks["lr_at_resume"][2],
        "4e. step count restored": checks["step"][1],
        "4f. config restored": checks["config_matches"],
        "4g. lang_code restored": checks["lang_code"][1],
        "4h. RNG state present": checks["has_rng_state"],
    }
    print(f"\n  {'-'*60}")
    for name, ok in verdicts.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(verdicts.values())
    print(f"  {'-'*60}\n  VERDICT: {'PROVEN' if passed else 'FAILED'} — {args.tree}\n")

    (work / "proof.json").write_text(json.dumps(
        {"tree": args.tree, "kill": kill_info,
         "checks": {k: str(v) for k, v in checks.items()},
         "verdicts": verdicts, "reference": ref_hist, "resumed": res_hist},
        indent=2), encoding="utf-8")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
