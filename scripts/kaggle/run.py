"""Push a kernel, poll until it finishes, fetch its output. One command.

Language-agnostic tooling.

    python scripts/kaggle/run.py --steps 200

WHAT THIS AUTOMATES, AND THE ONE THING IT CANNOT
------------------------------------------------
push -> poll status -> download output is the whole Kaggle batch API surface,
and it is genuinely reliable. Two properties are worth knowing up front rather
than discovering during a multi-day run:

  * A kernel run is CAPPED at 12 hours (9 on some tiers). Training longer than
    that is not one long run; it is a CHAIN of runs, each resuming from the
    previous one's output. That is why checkpointing had to be proven (T2.8)
    before any of this mattered.
  * A pushed kernel is queued, not started instantly. Queue time is usually
    under a minute but is not guaranteed, so the poller reports elapsed time
    rather than assuming.

`enable_gpu` must be explicit in kernel-metadata.json: Kaggle defaults a pushed
kernel to CPU, and a CPU run of this model does not fail, it merely never
finishes. The kernel script asserts the GPU again on the far side, because
metadata says what we ASKED for and only torch says what we GOT.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KERNEL_DIR = Path(__file__).resolve().parent / "kernel"   # overridable via --kernel-dir

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

TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, cwd=REPO)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def set_params(steps: int, micro: int, accum: int, data_slug: str) -> None:
    """Bake run parameters into the kernel script before pushing.

    `kaggle kernels push` uploads only the code file named by
    kernel-metadata.json -- there is no way to pass arguments, and the local
    environment does NOT cross into the kernel. So a parameter has to be a
    literal in the file that gets uploaded.
    """
    p = KERNEL_DIR / "run_phase2.py"
    s = p.read_text(encoding="utf-8")
    for key, val in [("LMA_STEPS", steps), ("LMA_MICRO_BATCH", micro),
                     ("LMA_GRAD_ACCUM", accum)]:
        s = re.sub(rf'os\.environ\.get\("{key}", "\d+"\)',
                   f'os.environ.get("{key}", "{val}")', s)
    s = re.sub(r'os\.environ\.get\("LMA_DATA_SLUG", "[^"]*"\)',
               f'os.environ.get("LMA_DATA_SLUG", "{data_slug}")', s)
    p.write_text(s, encoding="utf-8")
    print(f"[params] steps={steps} micro_batch={micro} grad_accum={accum} "
          f"data={data_slug}")


def push(accelerator: str | None) -> str:
    meta = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    kid = meta["id"]
    if not meta.get("enable_gpu"):
        raise SystemExit("kernel-metadata.json must set enable_gpu: true — "
                         "Kaggle silently defaults a pushed kernel to CPU.")
    shape = accelerator or meta.get("machine_shape")
    print(f"[push] {kid}  machine_shape={shape!r}"
          f"{' (from metadata)' if not accelerator else ' (CLI override)'}")
    print(f"[push] datasets: {meta['dataset_sources']}")
    args = ["kernels", "push", "-p", str(KERNEL_DIR)]
    if accelerator:
        args += ["--accelerator", accelerator]
    r = kaggle(*args)
    print("  " + (r.stdout.strip() or r.stderr.strip()))
    return kid


def poll(kid: str, interval: int, timeout_s: int) -> str:
    """Block until the kernel reaches a terminal state."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        # stdout only: stderr carries warnings and transient API errors that
        # must not be mistaken for the kernel's state.
        status = parse_status(kaggle("kernels", "status", kid, check=False).stdout)
        if status != last:
            print(f"[poll] {status:<20} (+{time.time()-t0:.0f}s)")
            last = status
        if status in TERMINAL:
            # Confirm a terminal verdict with a second reading; a single
            # blip should never end a multi-hour run.
            time.sleep(5)
            again = parse_status(kaggle("kernels", "status", kid, check=False).stdout)
            if again == status:
                return status
            print(f"[poll] '{status}' not confirmed (second read '{again}'); continuing")
            last = None
        time.sleep(interval)
    raise TimeoutError(f"kernel {kid} still running after {timeout_s}s")


def fetch(kid: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] -> {out_dir}")
    r = kaggle("kernels", "output", kid, "-p", str(out_dir), check=False)
    print("  " + (r.stdout.strip() or r.stderr.strip())[:400])
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description="push -> poll -> fetch a Kaggle kernel")
    # Default None on purpose. The CLI does `machine_shape = acc or metadata`,
    # and Kaggle silently NORMALISES an unrecognised accelerator string to the
    # generic "Gpu" (= P100) instead of erroring -- which is how five runs
    # landed on a P100 whose sm_60 Kaggle's own torch cannot even execute.
    # Passing nothing lets kernel-metadata.json's value stand.
    # The canonical string, read back from Kaggle, is "NvidiaTeslaT4".
    ap.add_argument("--accelerator", default=None)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--out", default=str(REPO / "kaggle_out"))
    ap.add_argument("--no-wait", action="store_true", help="push and exit")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--data-slug", default="lma-tokens-smoke")
    ap.add_argument("--kernel-dir", default=None,
                    help="push a different kernel folder (e.g. eval_kernel)")
    ap.add_argument("--no-params", action="store_true",
                    help="skip parameter baking (kernels with their own knobs)")
    args = ap.parse_args()

    if args.kernel_dir:
        globals()["KERNEL_DIR"] = Path(args.kernel_dir).resolve()

    if not args.no_params:
        set_params(args.steps, args.micro_batch, args.grad_accum, args.data_slug)
    kid = push(args.accelerator)
    if args.no_wait:
        print(f"[run] pushed; poll later with: kaggle kernels status {kid}")
        return

    status = poll(kid, args.interval, args.timeout)
    print(f"[run] terminal status: {status}")
    files = fetch(kid, Path(args.out))
    print(f"[fetch] {len(files)} files")
    for f in files[:40]:
        print(f"    {f.relative_to(Path(args.out))}  {f.stat().st_size/2**20:.2f} MB")
    sys.exit(0 if status == "complete" else 1)


if __name__ == "__main__":
    main()
