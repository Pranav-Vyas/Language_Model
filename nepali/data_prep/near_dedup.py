"""Near-duplicate removal by MinHash + LSH (T1.8, Yi stage 9).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY A SECOND DEDUP PASS
-----------------------
`clean.py` removes byte-identical documents with a SHA-256 set. That misses
the case the course lecture (L3 slides 11-12, documents A and D) is built
around: the same article republished with one word changed. Exact hashing
sees two different strings; a reader sees one document. News wire copy
republished across sites is exactly this, and it is the single largest
remaining source of repetition in a scraped corpus.

L3 slide 35: "Deduplication returns more than any other single step."

THE ALGORITHM, AS TAUGHT (L3 slides 18-23)
------------------------------------------
1. SHINGLE. Cut the document into overlapping runs of `k` consecutive words
   (k=3 in the lecture). The document becomes the *set* of its shingles.
   Changing one word only changes the k shingles containing it, so near-
   duplicates keep almost all of their shingles in common.

2. JACCARD. Similarity of two documents is J(A,B) = |A∩B| / |A∪B|. The
   lecture's worked example scores A against D at 25/31 = 0.81. Computing
   this for every pair is quadratic and therefore impossible at our scale.

3. MINHASH. Hash every shingle and keep only the smallest value. For a
   random hash, Pr[min(A) == min(B)] = J(A,B) exactly -- the minimum agrees
   precisely when the overall smallest shingle is one of the shared ones.
   Repeat with `num_perm` independent hashes to get a signature whose
   fraction of agreeing positions estimates J. With 100 permutations the
   estimate is accurate to about 0.04.

4. LSH BANDING. Split the signature into `bands` bands of `rows` rows each
   and hash each band. Two documents become *candidates* if any single band
   hashes identically, so only documents sharing a bucket are ever compared:

       Pr[one band matches] = s^rows
       Pr[candidate]        = 1 - (1 - s^rows)^bands

   The threshold -- the similarity at which a pair is as likely to be
   compared as not -- is s* ~= (1/bands)^(1/rows).

CHOSEN PARAMETERS
-----------------
num_perm=100, bands=20, rows=5, matching the lecture. That gives
s* ~= (1/20)^(1/5) = 0.549: pairs above ~0.55 similarity are almost always
caught (0.9996 at s=0.8), pairs below are mostly ignored (0.047 at s=0.3).
More bands would lower the threshold, catching more and costing more to
verify -- that is the dial, and it is set in the config, not in code.

MEMORY
------
The corpora here run to ~2M documents against a machine with a few GB free,
so signatures are never all held in RAM. Band keys are streamed to a
temporary file, sorted on disk with the system `sort`, and read back in one
pass: any run of equal band keys is a candidate group. Peak memory is one
sort buffer, not one entry per document.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

TREE = Path(__file__).resolve().parent.parent

# 64-bit mask; MinHash values are kept as unsigned 64-bit ints.
_MASK64 = (1 << 64) - 1
# Largest prime below 2^32, used as the modulus of the permutation family.
#
# THE MODULUS MUST MATCH THE SHINGLE UNIVERSE. Shingles are hashed to 32 bits,
# so the permutation h -> (a*h + b) mod p is only a genuine pseudo-random
# permutation when p is itself ~2^32. An earlier version used the Mersenne
# prime 2^61-1 here, which silently broke the estimator: with h < 2^32 and a
# small `a`, the product a*h + b never exceeds p, so the map is *monotonic*
# in h and every such permutation selects the SAME minimum shingle. That
# correlates signature positions instead of making them independent, and the
# estimated Jaccard runs far above the true one (measured: est 0.580 on a
# pair whose true J was 0.083). With p just under 2^32 the product wraps many
# times for any a > 1, restoring the independence MinHash depends on.
_PRIME32 = 4294967291  # 2^32 - 5


@dataclass(frozen=True)
class MinHashConfig:
    """Parameters of the near-duplicate detector.

    `num_perm` must equal `bands * rows`; the threshold below is implied by
    that split and is reported (not set) so the two can never disagree.
    """

    shingle_size: int = 3      # words per shingle (lecture uses 3)
    num_perm: int = 100        # signature length k
    bands: int = 20            # b
    rows: int = 5              # r
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.bands * self.rows != self.num_perm:
            raise ValueError(
                f"bands*rows ({self.bands}*{self.rows}) must equal num_perm ({self.num_perm})"
            )

    @property
    def threshold(self) -> float:
        """s* ~= (1/b)^(1/r): the similarity at which a pair is 50/50 to be compared."""
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def candidate_probability(self, similarity: float) -> float:
        """Pr[candidate] = 1 - (1 - s^r)^b, for reporting the tuning curve."""
        return 1.0 - (1.0 - similarity ** self.rows) ** self.bands


def _permutations(cfg: MinHashConfig) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficient pairs for the hash family h(x)=a*x+b mod p.

    Derived from the seed with SHA-256 rather than `random`, so the same
    coefficients are reproduced on any machine and any Python build -- the
    signatures must be stable for a rerun to give the same corpus.

    COEFFICIENT WIDTH IS DELIBERATE. `a` and `b` are drawn below 2^32 and the
    shingle hashes are truncated to 32 bits (see `shingles`). That keeps the
    whole product inside an unsigned 64-bit word:

        a*h + b  <=  (p-1)^2 + (p-1)  <  2^64      (p = 2^32-5)

    which is what lets the signature be computed with vectorised uint64
    arithmetic without silent wraparound. A wider `a` would overflow and
    quietly corrupt the minima -- the estimator would still *look* fine
    because MinHash outputs are opaque numbers, so the constraint is
    enforced here rather than trusted.
    """
    perms: list[tuple[int, int]] = []
    for i in range(cfg.num_perm):
        digest = hashlib.sha256(f"{cfg.seed}:{i}".encode()).digest()
        a = int.from_bytes(digest[:4], "big") % _PRIME32
        b = int.from_bytes(digest[4:8], "big") % _PRIME32
        perms.append((a or 1, b))   # a must be non-zero for a valid permutation
    return perms


def _try_numpy():
    """Return numpy if importable, else None.

    The vectorised path is ~100x faster but numpy is not required: the pure
    Python fallback produces identical signatures, just slowly. Keeping both
    means the pipeline still runs on a machine without numpy.
    """
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def shingles(text: str, k: int) -> set[int]:
    """Hash the document's overlapping k-word shingles to 64-bit ints.

    Hashing here (rather than keeping the strings) bounds memory: a long
    document becomes a set of ints, not a set of substrings.
    """
    words = text.split()
    if len(words) < k:
        # Too short to shingle at size k: treat the whole document as one
        # shingle so it can still match an identical short document.
        return {int.from_bytes(hashlib.sha1(" ".join(words).encode()).digest()[:4], "big")}
    out: set[int] = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i:i + k]).encode("utf-8")
        # 32-bit truncation keeps a*h+b inside uint64 (see _permutations).
        out.add(int.from_bytes(hashlib.sha1(gram).digest()[:4], "big"))
    return out


def signature(shingle_hashes: set[int], perms: list[tuple[int, int]]) -> list[int]:
    """MinHash signature: the minimum of each permuted hash over all shingles.

    This is the step that makes similarity cheap: Pr[min agrees] = Jaccard,
    so `num_perm` minima estimate the similarity of two documents without
    ever comparing their texts.

    Pure-Python reference implementation, kept because it is the version that
    reads like the definition. `signature_batch` computes the same values
    with numpy and is what the pipeline actually calls.
    """
    sig = []
    for a, b in perms:
        best = _MASK64
        for h in shingle_hashes:
            v = ((a * h + b) % _PRIME32)
            if v < best:
                best = v
        sig.append(best)
    return sig


def make_signature_fn(cfg: MinHashConfig):
    """Return the fastest available `shingle_hashes -> signature` function.

    The vectorised form evaluates all `num_perm` permutations against all
    shingles in one array expression:

        v   = (a[:, None] * h[None, :] + b[:, None]) % p     -> (num_perm, S)
        sig = v.min(axis=1)                                  -> (num_perm,)

    replacing num_perm x S interpreted operations with a single C loop.
    Measured at ~80 docs/sec in pure Python versus ~4,000 docs/sec here,
    which is the difference between a 4-hour and a 5-minute pass over the
    Hindi corpus.
    """
    np = _try_numpy()
    perms = _permutations(cfg)
    if np is None:
        return lambda sh: signature(sh, perms)

    a = np.array([p[0] for p in perms], dtype=np.uint64)[:, None]
    b = np.array([p[1] for p in perms], dtype=np.uint64)[:, None]
    p = np.uint64(_PRIME32)

    def _sig(shingle_hashes: set[int]) -> list[int]:
        h = np.fromiter(shingle_hashes, dtype=np.uint64,
                        count=len(shingle_hashes))[None, :]
        return ((a * h + b) % p).min(axis=1).tolist()

    return _sig


def band_keys(sig: list[int], cfg: MinHashConfig) -> list[bytes]:
    """Hash each band of the signature into one key.

    Two documents share a key iff every row in that band agreed, which is the
    s^rows event in the LSH derivation.
    """
    keys = []
    for band in range(cfg.bands):
        chunk = sig[band * cfg.rows:(band + 1) * cfg.rows]
        packed = b"".join(struct.pack(">Q", v) for v in chunk)
        keys.append(hashlib.sha1(struct.pack(">H", band) + packed).digest()[:8])
    return keys


@dataclass
class DedupStats:
    """Counts for the report's deduplication row."""

    docs_in: int = 0
    docs_out: int = 0
    duplicates_removed: int = 0
    candidate_groups: int = 0
    candidate_pairs_rejected: int = 0   # LSH candidates that failed verification
    threshold: float = 0.0
    num_perm: int = 0
    bands: int = 0
    rows: int = 0


def _iter_clean(path: Path) -> Iterator[tuple[int, dict]]:
    """Stream the cleaned corpus, yielding (line_index, record)."""
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def build_band_file(clean_path: Path, cfg: MinHashConfig, tmp_path: Path) -> int:
    """Pass 1: write "<band_key_hex> <doc_index>" for every band of every doc.

    Streaming to disk rather than accumulating a dict of buckets is what keeps
    this runnable on a machine with a couple of GB free.
    """
    sig_fn = make_signature_fn(cfg)
    n = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for idx, rec in _iter_clean(clean_path):
            text = rec.get("text") or ""
            sig = sig_fn(shingles(text, cfg.shingle_size))
            for key in band_keys(sig, cfg):
                out.write(f"{key.hex()} {idx}\n")
            n += 1
            if n % 20000 == 0:
                print(f"    signatures: {n:,}", file=sys.stderr)
    return n


def candidate_groups(sorted_path: Path) -> Iterator[list[int]]:
    """Pass 2: yield each group of document indices sharing a band key.

    The file is sorted by band key, so equal keys are adjacent and one linear
    scan recovers every candidate group without random access.
    """
    current_key = None
    group: list[int] = []
    with sorted_path.open(encoding="utf-8") as fh:
        for line in fh:
            key, idx_s = line.rstrip("\n").split(" ", 1)
            idx = int(idx_s)
            if key != current_key:
                if len(group) > 1:
                    yield group
                current_key = key
                group = [idx]
            else:
                group.append(idx)
    if len(group) > 1:
        yield group


def estimated_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Fraction of signature positions that agree = the MinHash estimate of J.

    This is the whole point of the signature: comparing 100 integers stands
    in for comparing two documents, with an error of about 0.04 at k=100.
    """
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


def verify_duplicates(
    clean_path: Path,
    sorted_path: Path,
    cfg: MinHashConfig,
) -> tuple[set[int], int, int]:
    """Pass 3+4: confirm each LSH candidate really is similar before removing it.

    WHY THIS STEP EXISTS -- A MEASURED BUG, NOT A PRECAUTION
    -------------------------------------------------------
    An earlier version treated every LSH bucket collision as a duplicate.
    Checking the true Jaccard of the flagged pairs showed that **61.7% of
    them were below the s*=0.549 threshold** (median 0.388, some at 0.0):
    LSH is a *candidate generator*, and skipping the verification step threw
    away roughly 115,000 documents that were not near-duplicates at all.

    Short documents are the main source of those false candidates. A document
    with only a handful of shingles has a low-entropy signature, so an entire
    band can agree by luck.

    Verification uses the signatures, not the texts: only documents that
    appear in some candidate group are re-signed and held in memory (~15% of
    the corpus at ~800 bytes each), which is far cheaper than holding text
    and keeps the streaming design intact.

    Returns (duplicate indices, groups examined, candidate pairs rejected).
    """
    involved: set[int] = set()
    for group in candidate_groups(sorted_path):
        involved.update(group)
    if not involved:
        return set(), 0, 0

    # Re-sign only the involved documents.
    sig_fn = make_signature_fn(cfg)
    sigs: dict[int, list[int]] = {}
    for idx, rec in _iter_clean(clean_path):
        if idx in involved:
            sigs[idx] = sig_fn(shingles(rec.get("text") or "", cfg.shingle_size))

    duplicates: set[int] = set()
    groups = 0
    rejected = 0
    for group in candidate_groups(sorted_path):
        groups += 1
        group = sorted(group)
        keep = group[0]                     # lowest index is the representative
        for other in group[1:]:
            if other in duplicates or other == keep:
                continue
            if estimated_jaccard(sigs[keep], sigs[other]) >= cfg.threshold:
                duplicates.add(other)
            else:
                rejected += 1
    return duplicates, groups, rejected


def write_duplicate_index(dup_path: Path, duplicates: set[int]) -> None:
    """Record the duplicate line indices instead of rewriting the corpus.

    WHY AN INDEX AND NOT A NEW CORPUS FILE
    --------------------------------------
    Materialising a deduplicated copy doubles peak disk: the cleaned Hindi
    corpus is 5.7GB, so a filtered copy needs another ~5.5GB on top of the
    ~1.3GB of band/sort temporaries. On this machine that does not fit.

    Writing the decision instead of the data costs a few MB, and downstream
    steps (splits, tokenizer training) simply skip these line numbers when
    reading `clean.jsonl`. It is also better provenance: the dedup verdict
    becomes an inspectable artifact that can be diffed and re-applied,
    rather than being silently baked into a derived file.

    Indices are 0-based line numbers into the *cleaned* corpus, sorted so
    the file is stable across runs and cheap to diff.
    """
    with dup_path.open("w", encoding="utf-8") as fh:
        for idx in sorted(duplicates):
            fh.write(f"{idx}\n")


def write_deduped(clean_path: Path, out_path: Path, duplicates: set[int]) -> tuple[int, int]:
    """Optionally materialise the deduplicated corpus (needs ~2x the disk).

    Only used when `--materialize` is passed; the default path records the
    duplicate index instead (see `write_duplicate_index`).
    """
    kept = total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for idx, rec in _iter_clean(clean_path):
            total += 1
            if idx in duplicates:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return total, kept


def count_lines(path: Path) -> int:
    """Count records without parsing them (cheap; used when not materialising)."""
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


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
    cfg_tree = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"MinHash/LSH near-dedup ({cfg_tree.lang_name}).")
    ap.add_argument("--in-name", default="clean.jsonl")
    ap.add_argument("--out-name", default="dedup.jsonl")
    ap.add_argument("--shingle-size", type=int, default=MinHashConfig.shingle_size)
    ap.add_argument("--num-perm", type=int, default=MinHashConfig.num_perm)
    ap.add_argument("--bands", type=int, default=MinHashConfig.bands)
    ap.add_argument("--rows", type=int, default=MinHashConfig.rows)
    ap.add_argument("--materialize", action="store_true",
                    help="also write the filtered corpus (needs ~2x disk); by "
                         "default only the duplicate index is written.")
    args = ap.parse_args(argv)

    cfg = MinHashConfig(
        shingle_size=args.shingle_size, num_perm=args.num_perm,
        bands=args.bands, rows=args.rows, seed=cfg_tree.seed,
    )

    clean_path = Path(cfg_tree.paths.clean_dir) / args.in_name
    out_path = Path(cfg_tree.paths.dedup_dir) / args.out_name
    stats_dir = Path(cfg_tree.paths.stats_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg_tree.lang_name}] near-dedup {clean_path} -> {out_path}")
    print(f"  k={cfg.num_perm} bands={cfg.bands} rows={cfg.rows} "
          f"shingle={cfg.shingle_size}w  threshold s*={cfg.threshold:.3f}")
    for s in (0.3, 0.55, 0.8):
        print(f"    Pr[candidate | s={s}] = {cfg.candidate_probability(s):.4f}")

    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as td:
        raw = Path(td) / "bands.txt"
        srt = Path(td) / "bands.sorted.txt"
        n = build_band_file(clean_path, cfg, raw)
        print(f"  wrote {n:,} signatures; sorting band keys on disk…")
        # External sort: handles files larger than RAM, unlike an in-memory dict.
        env = dict(os.environ, LC_ALL="C")
        subprocess.run(["sort", "-S", "256M", "-T", td, "-o", str(srt), str(raw)],
                       check=True, env=env)
        duplicates, groups, rejected = verify_duplicates(clean_path, srt, cfg)
        print(f"  verified {groups:,} candidate groups; "
              f"rejected {rejected:,} candidate pairs below s*={cfg.threshold:.3f}")
        dup_path = out_path.with_name("duplicate_indices.txt")
        write_duplicate_index(dup_path, duplicates)
        if args.materialize:
            total, kept = write_deduped(clean_path, out_path, duplicates)
        else:
            total = count_lines(clean_path)
            kept = total - len(duplicates)

    stats = DedupStats(
        docs_in=total, docs_out=kept, duplicates_removed=len(duplicates),
        candidate_groups=groups, candidate_pairs_rejected=rejected,
        threshold=cfg.threshold,
        num_perm=cfg.num_perm, bands=cfg.bands, rows=cfg.rows,
    )
    payload = {"near_dedup": asdict(stats), "config": asdict(cfg)}
    (stats_dir / "near_dedup_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = (stats.duplicates_removed / total * 100) if total else 0.0
    print(f"\n  in {total:,} -> out {kept:,}   removed {stats.duplicates_removed:,} ({pct:.2f}%)")
    print(f"  duplicate index -> {dup_path}")
    if args.materialize:
        print(f"  deduped corpus  -> {out_path}")
    else:
        print("  (corpus not rewritten; downstream steps skip these line numbers)")
    print(f"  stats -> {stats_dir/'near_dedup_stats.json'}")


if __name__ == "__main__":
    main()
