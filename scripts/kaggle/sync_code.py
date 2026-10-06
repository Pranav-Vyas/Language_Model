"""Package the repo's code as a small Kaggle Dataset (create or new version).

Language-agnostic tooling; holds no language data.

WHY A DATASET AND NOT `git clone` IN THE KERNEL
-----------------------------------------------
`kaggle kernels push` uploads exactly one script/notebook plus its metadata --
it cannot carry a directory tree. So the `hindi/` and `nepali/` trees have to
reach the kernel some other way. The two candidates:

  git clone   The GitHub Classroom repo is PRIVATE (anonymous API GET returns
              404), so this needs a Personal Access Token stored as a Kaggle
              Secret. That is a second credential to manage, an extra thing to
              expire mid-run, and it puts a token into an environment whose
              logs are easy to make public by accident.

  dataset     No credentials at all beyond the Kaggle token already in use.
              Versioned, so every training run records exactly which code
              version produced it. Works identically whether the repo is
              public or private.

The dataset wins on both simplicity and secret-hygiene, so that is the route.

A SINGLE .tar.gz, NOT LOOSE FILES
---------------------------------
The code tree is ~200 small files. Kaggle uploads per-file, so loose files
make every re-version slow. A single .tar.gz uploads as ONE file, so a
re-version costs 4 MB regardless of how many files changed.

MEASURED CAVEAT: Kaggle extracts uploaded archives server-side, .tar.gz
included -- the dataset ends up holding the unpacked tree, not the archive.
(Found the hard way: the first kernel run failed with "code archive not
found ... []" because it globbed for the .tar.gz.) The kernel therefore
handles BOTH layouts rather than trusting either.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SLUG = "lma-code"

# Code only. Corpora, checkpoints, venvs and caches are excluded by construction
# rather than by .gitignore, because this upload is not a git operation.
INCLUDE = ["hindi", "nepali", "scripts", "tests"]
# Directory names are matched EXACTLY, not by prefix, so every new artifact
# directory needs its own entry. `reasoning_data` was missed on the first
# Phase-3 sync and shipped 130 MB of regenerable data inside a 4 MB code
# dataset; `checkpoints_sft` would have slipped past "checkpoints" for the same
# reason. The reasoning data is deliberately NOT shipped: the kernel regenerates
# it from the committed generator, which is both cheaper and a live proof that
# the dataset really is reproducible from its seed.
EXCLUDE_DIRS = {"corpus", "data", "__pycache__", ".venv", "checkpoints", ".git",
                ".pytest_cache", ".ipynb_checkpoints",
                "reasoning_data", "checkpoints_sft", "checkpoints_nope"}


def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return None
    if info.name.endswith((".pyc", ".bin", ".pt", ".ckpt")):
        return None
    return info


def build_archive(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name in INCLUDE:
            tar.add(REPO / name, arcname=name, filter=_filter)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync repo code to a Kaggle Dataset.")
    ap.add_argument("--user", required=True)
    ap.add_argument("--staging", default="/tmp/lma_kaggle/code")
    ap.add_argument("--message", default="code sync")
    ap.add_argument("--create", action="store_true", help="first upload (else: new version)")
    args = ap.parse_args()

    stage = Path(args.staging)
    stage.mkdir(parents=True, exist_ok=True)
    archive = build_archive(stage / "lma_code.tar.gz")
    size_mb = archive.stat().st_size / 2**20

    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "LMA code",
        "id": f"{args.user}/{SLUG}",
        "licenses": [{"name": "unknown"}],
    }, indent=2), encoding="utf-8")

    # Count what actually went in, so a silently-empty archive is visible.
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
    print(f"[code] {archive}  {size_mb:.2f} MB  {len(members)} files")
    for tree in INCLUDE:
        n = sum(1 for m in members if m.name.startswith(tree + "/"))
        print(f"       {tree:<10} {n:>4} files")
    if any(m.name.endswith(".bin") for m in members):
        sys.exit("refusing to upload: corpus .bin leaked into the code archive")

    cmd = (["datasets", "create", "-p", str(stage), "--dir-mode", "tar"] if args.create
           else ["datasets", "version", "-p", str(stage), "-m", args.message, "--dir-mode", "tar"])
    print(f"[code] kaggle {' '.join(cmd)}")
    r = subprocess.run([sys.executable, "-m", "kaggle", *cmd], cwd=stage,
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
