"""Upload packed token shards to a private Kaggle Dataset.

Language-agnostic tooling; it copies shards but never inspects or mixes them,
and each language's files keep their own directory inside the dataset.

WHY A DATASET RATHER THAN PUSHING DATA WITH THE KERNEL
------------------------------------------------------
The full shards are 2.8 GB (hi 1.1 GB, ne 1.7 GB). A kernel push carries one
script; data has to be attached as a data source. Uploading once and attaching
by slug means a code change re-uploads 4 MB, not 2.8 GB -- which is the whole
reason the code and the data are two SEPARATE datasets rather than one.

SMOKE vs FULL
-------------
`--max-train-tokens` truncates the train shard for a smoke dataset. The cut is
at an even uint16 boundary, so the result is still a valid shard: the loader
memmaps it identically and simply sees fewer tokens. Peak VRAM and step time
do not depend on shard length at all (they depend on batch, context and model
size), so a slice validates the pipeline and the memory numbers honestly,
while keeping the first upload small enough to iterate on.

val/test are copied whole -- they are small, and a truncated validation set
would make the smoke run's loss non-comparable to the real run's.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TREES = ("hindi", "nepali")


def stage_tree(tree: str, stage: Path, max_train_tokens: int | None) -> list[tuple[str, int]]:
    """Copy one tree's shards into the staging dir. Returns (name, bytes)."""
    src = REPO / tree / "data" / "tokenized"
    dst = stage / tree
    dst.mkdir(parents=True, exist_ok=True)
    written = []

    for name in ("train.bin", "val.bin", "test.bin", "meta.json"):
        s = src / name
        if not s.exists():
            continue
        d = dst / name
        if name == "train.bin" and max_train_tokens:
            # uint16 => 2 bytes per token. Truncating on an even byte boundary
            # keeps the file a valid array of whole tokens.
            # Read ONLY the bytes we keep. copyfileobj-then-truncate would
            # stream the whole 1.07 GB / 1.65 GB shard through disk first --
            # slow, and needless pressure on a machine already tight on space.
            nbytes = min(max_train_tokens * 2, s.stat().st_size)
            with s.open("rb") as fi, d.open("wb") as fo:
                remaining = nbytes
                while remaining > 0:
                    chunk = fi.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fo.write(chunk)
                    remaining -= len(chunk)
        else:
            shutil.copy2(s, d)
        written.append((f"{tree}/{name}", d.stat().st_size))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync token shards to a Kaggle Dataset.")
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", required=True, help="e.g. lma-tokens-smoke or lma-tokens")
    ap.add_argument("--staging", default=None)
    ap.add_argument("--max-train-tokens", type=int, default=None)
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--message", default="shard sync")
    args = ap.parse_args()

    stage = Path(args.staging or f"/tmp/lma_kaggle/{args.slug}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)

    total = 0
    for tree in TREES:
        for name, nbytes in stage_tree(tree, stage, args.max_train_tokens):
            total += nbytes
            print(f"  {name:<28} {nbytes/2**20:>9.1f} MB")
    print(f"  {'TOTAL':<28} {total/2**20:>9.1f} MB")

    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": f"LMA tokens ({args.slug})",
        "id": f"{args.user}/{args.slug}",
        "licenses": [{"name": "unknown"}],
    }, indent=2), encoding="utf-8")

    cmd = (["datasets", "create", "-p", str(stage), "--dir-mode", "zip"] if args.create
           else ["datasets", "version", "-p", str(stage), "-m", args.message, "--dir-mode", "zip"])
    print(f"[data] kaggle {' '.join(cmd)}")
    r = subprocess.run([sys.executable, "-m", "kaggle", *cmd], cwd=stage,
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
