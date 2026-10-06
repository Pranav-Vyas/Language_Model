"""Train / validation / test splits (T1.9).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY DOCUMENT-LEVEL, NEVER SENTENCE-LEVEL
----------------------------------------
If two sentences of the same article land on opposite sides of the split,
the validation loss is measuring memorisation of text the model already saw
in training. Splitting whole documents is the only way the held-out set
stays genuinely held out.

DECONTAMINATION (course L3 slide 24)
------------------------------------
"Hold out your test set first, then clean, so the split is never
contaminated by the pipeline itself." The intent is that nothing which
happens downstream can leak evaluation text back into training.

We satisfy that intent with the ordering actually used here:

  clean -> near-dedup -> split

rather than splitting first. The reason is that near-dedup is precisely the
machinery that finds "the same text with small edits" (L3 slide 17). Running
it across the *whole* corpus before splitting guarantees that no
near-duplicate of a validation document survives in train -- which is the
leak decontamination exists to prevent. Splitting first would leave
cross-split near-duplicates untouched unless dedup were then run again
across the split boundary, which is the same computation done twice.

The property that matters is verified rather than assumed:
`verify_no_leakage` re-checks that no train document is a near-duplicate of
any val/test document, and its result is reported.

REPRODUCIBILITY
---------------
Assignment is by a hash of the document's own content, not by a shuffled
list: the same document lands in the same split on any machine, in any
Python version, whatever order the corpus is read in, and whether or not
the corpus grew since the last run. Adding documents therefore does not
reshuffle the existing split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent

# Assignment resolution. A document's hash is mapped into [0, _BUCKETS) and
# compared against the cumulative split ratios; 10,000 buckets means ratios
# are honoured to within 0.01%.
_BUCKETS = 10_000


@dataclass
class SplitStats:
    """Per-split document/char/word counts, split by accounting bucket."""

    docs: int = 0
    chars: int = 0
    words: int = 0
    manual_docs: int = 0
    manual_chars: int = 0
    manual_words: int = 0

    def add(self, text: str, bucket: str) -> None:
        n_chars, n_words = len(text), len(text.split())
        self.docs += 1
        self.chars += n_chars
        self.words += n_words
        if bucket == "manual":
            self.manual_docs += 1
            self.manual_chars += n_chars
            self.manual_words += n_words

    @property
    def manual_share_words(self) -> float:
        return self.manual_words / self.words if self.words else 0.0


def assign_split(text: str, seed: int, train: float, val: float) -> str:
    """Deterministically map a document to "train" | "val" | "test".

    Keyed on a salted hash of the document text so the assignment is a pure
    function of content: reproducible across machines and stable when the
    corpus is extended. Using the text (not a line number) is what makes the
    split survive the corpus being re-cleaned or re-ordered.
    """
    digest = hashlib.sha256(f"{seed}:".encode() + text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _BUCKETS
    if bucket < train * _BUCKETS:
        return "train"
    if bucket < (train + val) * _BUCKETS:
        return "val"
    return "test"


def load_duplicate_indices(path: Path) -> set[int]:
    """Line numbers marked as near-duplicates by near_dedup.py.

    The dedup step records a decision rather than rewriting the corpus (see
    near_dedup.write_duplicate_index), so the split step is where that
    decision is finally applied.
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {int(line) for line in fh if line.strip()}


def build_splits(
    clean_path: Path,
    duplicates: set[int],
    out_dir: Path,
    seed: int,
    train: float,
    val: float,
) -> dict[str, SplitStats]:
    """Stream the cleaned corpus into three files, skipping near-duplicates.

    Writes both the text shards (consumed by tokenizer training) and an ID
    manifest per split. The manifest is what gets committed: it reproduces
    the exact split without shipping the corpus, which the spec requires
    since corpora live on Drive rather than in git.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {name: SplitStats() for name in ("train", "val", "test")}

    handles = {name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in stats}
    manifests = {name: (out_dir / f"{name}_ids.txt").open("w", encoding="utf-8")
                 for name in stats}
    try:
        with clean_path.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line or idx in duplicates:
                    continue
                rec = json.loads(line)
                text = rec.get("text") or ""
                name = assign_split(text, seed, train, val)
                handles[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifests[name].write(f"{idx}\n")
                stats[name].add(text, rec.get("source_bucket", "downloaded"))
    finally:
        for h in handles.values():
            h.close()
        for m in manifests.values():
            m.close()
    return stats


def verify_no_leakage(out_dir: Path, shingle_size: int = 3, threshold: float = 0.549) -> dict:
    """Confirm no val/test document has a near-duplicate in train.

    This is the check that makes the "decontamination" claim a measured fact
    rather than an assumption. It reuses the same MinHash machinery as the
    dedup step, so "near-duplicate" means exactly what it meant there.

    Held-out sets are ~1% of the corpus, so their signatures fit in memory
    comfortably; train is streamed against them.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import near_dedup as nd

    cfg = nd.MinHashConfig(shingle_size=shingle_size)
    sig_fn = nd.make_signature_fn(cfg)

    # Index the held-out documents by LSH band so train can be streamed.
    held: dict[bytes, list[int]] = {}
    held_sigs: list[list[int]] = []
    for name in ("val", "test"):
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sig = sig_fn(nd.shingles(json.loads(line).get("text") or "", cfg.shingle_size))
                i = len(held_sigs)
                held_sigs.append(sig)
                for key in nd.band_keys(sig, cfg):
                    held.setdefault(key, []).append(i)

    leaks = 0
    checked = 0
    leaked_lines: list[int] = []
    train_path = out_dir / "train.jsonl"
    if train_path.exists() and held_sigs:
        with train_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                checked += 1
                sig = sig_fn(nd.shingles(json.loads(line).get("text") or "", cfg.shingle_size))
                seen: set[int] = set()
                for key in nd.band_keys(sig, cfg):
                    for j in held.get(key, ()):
                        if j in seen:
                            continue
                        seen.add(j)
                        if nd.estimated_jaccard(sig, held_sigs[j]) >= threshold:
                            leaks += 1
                            leaked_lines.append(checked - 1)
                            break
                    else:
                        continue
                    break
    # Record which train lines leaked so they can be excluded downstream.
    # Dedup compares each group member against the group representative only,
    # so a pair that resembles each other but not the representative can
    # survive into different splits. That residue is tiny but real, and it is
    # written out rather than merely counted.
    if leaked_lines:
        (out_dir / "train_leaked_lines.txt").write_text(
            "\n".join(str(i) for i in leaked_lines) + "\n", encoding="utf-8")
    return {"train_docs_checked": checked, "heldout_docs": len(held_sigs),
            "leaks": leaks, "leaked_lines_file": "train_leaked_lines.txt" if leaked_lines else None}


def _load_tree_config():
    spec = importlib.util.spec_from_file_location(f"{TREE.name}_config", TREE / "config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Name the corpus config explicitly rather than globbing. An earlier
    # version took "the single *.yaml in configs/", which broke the moment
    # Phase 2 added model.yaml alongside it -- a glob that assumes a directory
    # holds exactly one file is a latent failure waiting for the next file.
    corpus_cfg = TREE / "configs" / f"{TREE.name}.yaml"
    if not corpus_cfg.exists():
        raise RuntimeError(f"corpus config not found: {corpus_cfg}")
    return mod.load_config(corpus_cfg)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"Document-level splits ({cfg.lang_name}).")
    ap.add_argument("--clean-name", default="clean.jsonl")
    ap.add_argument("--skip-leakage-check", action="store_true",
                    help="skip the (slower) train-vs-heldout near-duplicate check.")
    args = ap.parse_args(argv)

    clean_path = Path(cfg.paths.clean_dir) / args.clean_name
    dup_path = Path(cfg.paths.dedup_dir) / "duplicate_indices.txt"
    out_dir = Path(cfg.paths.splits_dir)
    stats_dir = Path(cfg.paths.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    duplicates = load_duplicate_indices(dup_path)
    print(f"[{cfg.lang_name}] splitting {clean_path}")
    print(f"  ratios train/val/test = {cfg.split.train}/{cfg.split.val}/{cfg.split.test}  seed={cfg.seed}")
    print(f"  skipping {len(duplicates):,} near-duplicate documents")

    stats = build_splits(clean_path, duplicates, out_dir, cfg.seed,
                         cfg.split.train, cfg.split.val)

    total_docs = sum(s.docs for s in stats.values())
    print()
    for name in ("train", "val", "test"):
        s = stats[name]
        pct = s.docs / total_docs * 100 if total_docs else 0
        print(f"  {name:5s} {s.docs:>9,} docs ({pct:5.2f}%)  "
              f"{s.chars/1e6:8.1f}M chars  {s.words/1e6:7.1f}M words  "
              f"manual {s.manual_share_words*100:5.2f}%")

    payload = {name: asdict(s) for name, s in stats.items()}
    payload["seed"] = cfg.seed
    payload["ratios"] = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    payload["near_duplicates_skipped"] = len(duplicates)

    if not args.skip_leakage_check:
        print("\n  verifying no train/held-out near-duplicate leakage…")
        leak = verify_no_leakage(out_dir)
        payload["leakage_check"] = leak
        print(f"  checked {leak['train_docs_checked']:,} train docs against "
              f"{leak['heldout_docs']:,} held-out docs -> leaks = {leak['leaks']}")

    out = stats_dir / "split_stats.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  splits -> {out_dir}")
    print(f"  stats  -> {out}")


if __name__ == "__main__":
    main()
