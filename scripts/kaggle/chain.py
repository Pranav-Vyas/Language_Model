"""Run a multi-link training chain on Kaggle, unattended.

Language-agnostic tooling.

    python scripts/kaggle/chain.py --links 6 --steps 4000

WHY A CHAIN AND NOT ONE LONG RUN
--------------------------------
A Kaggle batch kernel is killed at 12 hours (9 on some tiers). Pretraining
both models to convergence does not fit in one run, so the run is split into
links: each link trains for a bounded number of steps, writes checkpoints to
/kaggle/working, and the NEXT link resumes from them.

This is only safe because T2.8 proved resume is bit-identical: batches are a
pure function of (seed, step), and the checkpoint carries weights, optimizer
moments, step, config, lang_code AND RNG state. A chain built on an
approximate resume would silently produce a model trained on a different data
order than the one the report describes.

HOW THE CHECKPOINT CROSSES THE LINK BOUNDARY
--------------------------------------------
Link N's output is downloaded, then re-uploaded as a new version of the
checkpoint dataset, which link N+1 attaches as an input. That round trip is
the reliable mechanism: `kaggle datasets version` uploads from local disk, so
the bytes have to come home first.

The cost is real and worth stating: a checkpoint is ~290 MB per language
(24.07M params x 4 bytes for weights, plus AdamW's m and v), so ~580 MB down
and the same back up per link. At a 12-hour link that overhead is a rounding
error; at a 30-minute link it would dominate, which is an argument for making
links long rather than many.

FAILURE POLICY
--------------
A link that ends in `error` stops the chain rather than pushing the next one.
Resuming from a checkpoint written by a failed run is exactly the situation
where silently continuing produces a model nobody can account for.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# Per-language / cadence overrides for the real run; 0 keeps smoke behaviour.
_P = {"stop_hi": 0, "stop_ne": 0, "ckpt_every": 0, "eval_every": 0, "smoke": 1,
      "trees": "hindi,nepali", "peak_lr": 0, "dropout": "", "dtype": ""}
KERNEL_DIR = Path(__file__).resolve().parent / "kernel"

_STATUS_RE = re.compile(r"KernelWorkerStatus\.(\w+)", re.I)


def parse_status(stdout: str) -> str:
    """Extract the kernel status from `kaggle kernels status` output.

    Parsed from the explicit `KernelWorkerStatus.X` token rather than by
    substring-searching the text. The earlier version scanned for the words
    "complete"/"error"/"running" in priority order over stdout+stderr, so any
    transient error message on stderr matched "error" before "running" and a
    perfectly healthy run was declared dead -- which is exactly what happened
    to a live evaluation kernel. Unrecognised output returns "unknown", which
    the caller retries rather than treating as failure.
    """
    m = _STATUS_RE.search(stdout or "")
    return m.group(1).lower() if m else "unknown"

CKPT_SLUG = "lma-ckpt"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, cwd=REPO)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def set_params(steps: int, stop_at: int, micro: int, accum: int,
               data_slug: str, fresh: bool) -> None:
    """Bake this link's parameters into the kernel script (see run.py).

    `steps` is the LR schedule horizon and is the SAME on every link;
    `stop_at` is where this link stops. See loop.py for why these must not be
    the same knob.
    """
    import re
    p = KERNEL_DIR / "run_phase2.py"
    s = p.read_text(encoding="utf-8")
    for key, val in [("LMA_STEPS", steps), ("LMA_STOP_AT", stop_at),
                     ("LMA_MICRO_BATCH", micro), ("LMA_GRAD_ACCUM", accum),
                     ("LMA_FRESH", int(fresh)),
                     ("LMA_STOP_HI", _P["stop_hi"]), ("LMA_STOP_NE", _P["stop_ne"]),
                     ("LMA_CKPT_EVERY", _P["ckpt_every"]),
                     ("LMA_PEAK_LR", _P["peak_lr"]),
                     ("LMA_EVAL_EVERY", _P["eval_every"]),
                     ("LMA_SMOKE", _P["smoke"])]:
        s = re.sub(rf'os\.environ\.get\("{key}", "\d+"\)',
                   f'os.environ.get("{key}", "{val}")', s)
    s = re.sub(r'os\.environ\.get\("LMA_DATA_SLUG", "[^"]*"\)',
               f'os.environ.get("LMA_DATA_SLUG", "{data_slug}")', s)
    s = re.sub(r'os\.environ\.get\("LMA_TREES", "[^"]*"\)',
               f'os.environ.get("LMA_TREES", "{_P["trees"]}")', s)
    s = re.sub(r'os\.environ\.get\("LMA_DROPOUT", "[^"]*"\)',
               f'os.environ.get("LMA_DROPOUT", "{_P["dropout"]}")', s)
    s = re.sub(r'os\.environ\.get\("LMA_DTYPE", "[^"]*"\)',
               f'os.environ.get("LMA_DTYPE", "{_P["dtype"]}")', s)
    p.write_text(s, encoding="utf-8")


def set_sources(user: str, data_slug: str, attach_ckpt: bool) -> str:
    """Point the kernel at this link's inputs; return the kernel id."""
    p = KERNEL_DIR / "kernel-metadata.json"
    meta = json.loads(p.read_text())
    sources = [f"{user}/lma-code", f"{user}/{data_slug}"]
    if attach_ckpt:
        sources.append(f"{user}/{CKPT_SLUG}")
    meta["dataset_sources"] = sources
    if not meta.get("machine_shape"):
        raise SystemExit("kernel-metadata.json lost machine_shape; Kaggle would "
                         "fall back to a P100 whose sm_60 its torch cannot run.")
    p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  sources: {sources}")
    return meta["id"]


def push_and_wait(kid: str, interval: int, timeout_s: int) -> str:
    kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        status = parse_status(kaggle("kernels", "status", kid, check=False).stdout)
        if status != last:
            print(f"  [{status}] +{time.time()-t0:.0f}s", flush=True)
            last = status
        if status in {"complete", "error", "cancelacknowledged"}:
            time.sleep(5)
            if parse_status(kaggle("kernels", "status", kid,
                                   check=False).stdout) == status:
                return status
            print(f"  ['{status}' not confirmed; continuing]", flush=True)
            last = None
        time.sleep(interval)
    raise TimeoutError(f"link exceeded {timeout_s}s")


def fetch_output(kid: str, out_dir: Path) -> Path:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", str(out_dir), check=False)
    return out_dir


def publish_checkpoints(user: str, out_dir: Path, stage: Path, first: bool,
                        message: str) -> dict[str, str]:
    """Re-upload the newest checkpoint per language as a dataset version."""
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    published = {}
    for tree in ("hindi", "nepali"):
        cks = sorted(out_dir.rglob(f"*{tree}/step_*.pt"),
                     key=lambda q: int(q.stem.split("_")[1]))
        if not cks:
            continue
        newest = cks[-1]
        dst = stage / tree
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(newest, dst / newest.name)
        published[tree] = newest.name
        print(f"  publish {tree}/{newest.name} ({newest.stat().st_size/2**20:.0f} MB)")
    if not published:
        raise SystemExit("no checkpoints in the link output — nothing to chain from")

    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "LMA checkpoints", "id": f"{user}/{CKPT_SLUG}",
        "licenses": [{"name": "unknown"}]}, indent=2), encoding="utf-8")
    cmd = (["datasets", "create", "-p", str(stage), "--dir-mode", "zip"] if first
           else ["datasets", "version", "-p", str(stage), "-m", message,
                 "--dir-mode", "zip"])
    kaggle(*cmd)
    return published


def wait_ready(user: str, expect: dict[str, str] | None = None,
               tries: int = 60, interval: int = 15) -> bool:
    """Wait until the checkpoint dataset actually SERVES the expected files.

    Checking `datasets status` alone is not enough, and this was measured, not
    guessed: after publishing link 1 the status returned "ready" while the file
    listing still showed the PREVIOUS version's step_200.pt. Had the next link
    been pushed on that signal it would have mounted the stale checkpoint and
    silently restarted from step 200, discarding 11.5 hours of training with no
    error anywhere.

    So the ready condition is the filenames themselves, not the status string.
    """
    for _ in range(tries):
        out = kaggle("datasets", "status", f"{user}/{CKPT_SLUG}", check=False).stdout or ""
        if "ready" in out.lower():
            listing = kaggle("datasets", "files", f"{user}/{CKPT_SLUG}",
                             check=False).stdout or ""
            if not expect:
                return True
            if all(name in listing for name in expect.values()):
                print(f"  checkpoint dataset serves {list(expect.values())}")
                return True
        time.sleep(interval)
    return False


def sync_code_first(user: str) -> None:
    """Upload the current working tree as a new code-dataset version.

    The kernel script is pushed fresh on every link, but the CODE it drives
    lives in a dataset that only changes when explicitly re-uploaded. Editing
    one without the other produces a version skew that fails at argparse --
    "unrecognized arguments: --keep-last-n 1" -- after the checkpoints have
    already been seeded, which looks like a chain bug and is not one.

    Syncing here makes the two move together, so a link always runs the code
    that is on disk right now.
    """
    print("[sync] uploading current code as a new lma-code version")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "sync_code.py"),
                        "--user", user, "--message", "chain link sync"],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"code sync failed:\n{r.stdout}\n{r.stderr}")
    for _ in range(60):
        if "ready" in (kaggle("datasets", "status", f"{user}/lma-code",
                              check=False).stdout or "").lower():
            print("[sync] lma-code ready")
            return
        time.sleep(10)
    raise SystemExit("lma-code never became ready; a link would run stale code")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unattended Kaggle training chain.")
    ap.add_argument("--user", default="anuragkaushal183")
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200,
                    help="TOTAL steps for the whole run = the LR schedule horizon")
    ap.add_argument("--stop-every", type=int, default=None,
                    help="steps to advance per link; defaults to the full horizon")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--data-slug", default="lma-tokens-smoke")
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--link-timeout", type=int, default=13 * 3600)
    ap.add_argument("--out", default=str(REPO / "kaggle_out"))
    ap.add_argument("--stop-hi", type=int, nargs="*", default=None,
                    help="per-link Hindi stop steps, e.g. --stop-hi 7089 8732")
    ap.add_argument("--stop-ne", type=int, nargs="*", default=None,
                    help="per-link Nepali stop steps, e.g. --stop-ne 6764 13547")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--trees", default="hindi,nepali",
                    help="which language trees this run trains")
    ap.add_argument("--peak-lr", type=float, default=0)
    ap.add_argument("--dropout", default="")
    ap.add_argument("--dtype", default="", choices=["", "float16", "bfloat16", "float32"])
    ap.add_argument("--ckpt-slug", default=None,
                    help="checkpoint dataset slug; use a NEW one so an existing "
                         "final checkpoint is never overwritten")
    ap.add_argument("--real", action="store_true",
                    help="real run: full logging cadence, no --smoke")
    ap.add_argument("--ckpt-exists", action="store_true",
                    help="the checkpoint dataset already exists (version it, don't create)")
    args = ap.parse_args()

    global CKPT_SLUG
    if args.ckpt_slug:
        CKPT_SLUG = args.ckpt_slug
        print(f"[chain] checkpoint dataset: {args.user}/{CKPT_SLUG}")
    sync_code_first(args.user)
    out_root = Path(args.out)
    ckpt_exists = args.ckpt_exists
    for link in range(1, args.links + 1):
        # The horizon is the FULL run on every link; only the stop point moves.
        horizon = args.steps
        stop_at = min(args.stop_every * link, horizon) if args.stop_every else horizon
        print(f"\n{'='*66}\nLINK {link}/{args.links} — stop at step {stop_at:,} "
              f"of a {horizon:,}-step schedule\n{'='*66}")
        if args.stop_hi:
            _P["stop_hi"] = args.stop_hi[min(link - 1, len(args.stop_hi) - 1)]
        if args.stop_ne:
            _P["stop_ne"] = args.stop_ne[min(link - 1, len(args.stop_ne) - 1)]
        _P["ckpt_every"], _P["eval_every"] = args.ckpt_every, args.eval_every
        _P["smoke"] = 0 if args.real else 1
        _P["trees"], _P["peak_lr"] = args.trees, args.peak_lr
        _P["dropout"], _P["dtype"] = args.dropout, args.dtype
        if args.stop_hi or args.stop_ne:
            print(f"  per-language stop: hindi={_P['stop_hi']} nepali={_P['stop_ne']}")
        set_params(horizon, stop_at, args.micro_batch, args.grad_accum,
                   args.data_slug, fresh=(link == 1 and not ckpt_exists))
        kid = set_sources(args.user, args.data_slug, attach_ckpt=ckpt_exists)

        status = push_and_wait(kid, args.interval, args.link_timeout)
        print(f"  status: {status}")
        out_dir = fetch_output(kid, out_root / f"link{link}")
        if status != "complete":
            raise SystemExit(f"link {link} ended in '{status}' — chain stopped so a "
                             "later link cannot resume from a failed run")

        if stop_at >= horizon:
            # out_dir was already fetched above; do not download it twice.
            print(f"  reached the schedule horizon at link {link}; run finished")
            break
        published = publish_checkpoints(args.user, out_dir,
                                        Path(f"/tmp/lma_kaggle/{CKPT_SLUG}"),
                                        first=not ckpt_exists,
                                        message=f"after link {link} (stop {stop_at}, hi={_P['stop_hi']}, ne={_P['stop_ne']})")
        ckpt_exists = True
        if not wait_ready(args.user, expect=published):
            raise SystemExit("checkpoint dataset never became ready; next link "
                             "would mount it empty and silently restart training")
        print(f"  link {link} OK -> {published}")

    print(f"\nchain complete: {args.links} links")


if __name__ == "__main__":
    main()
