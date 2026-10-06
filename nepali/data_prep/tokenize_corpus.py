"""Encode a split corpus into packed uint16 token shards (T2.1).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY A PACKED BINARY FORMAT AT ALL
---------------------------------
Training reads the corpus many times. Re-tokenising JSON on every epoch would
spend most of the GPU budget in SentencePiece and json.loads rather than in
matrix multiplies. Encoding once into a flat array of token ids means the
training loader is a memory-mapped slice — effectively free.

WHY uint16
----------
The vocabulary is 16,000, and 16,000 < 65,536 = 2^16, so every token id fits
in an unsigned 16-bit integer. That halves the file versus uint32:

    Hindi   582,711,232 tokens  ->  1.09 GiB   (2.18 GiB as uint32)
    Nepali  903,937,802 tokens  ->  1.68 GiB   (3.37 GiB as uint32)

The dtype is written into the sidecar metadata rather than assumed by the
reader, and `assert vocab_size <= 65535` runs before any encoding starts:
silently wrapping ids modulo 65536 would produce a corpus that trains without
error and is quietly wrong, which is the failure mode this project keeps
running into.

DOCUMENT BOUNDARIES
-------------------
Documents are concatenated into one contiguous stream with an EOS token
between them. The model therefore learns that documents end, and a training
window that straddles a boundary sees the EOS rather than an invisible seam.
Sequence packing (rather than one-document-per-row with padding) means no
compute is wasted on pad tokens, which matters at a 25M-parameter budget.

A document can be read back out of the packed stream for the qualitative
generation samples in Phase 2 — but "exactly" needs a caveat, measured rather
than assumed: SentencePiece's normaliser rewrites newlines as spaces, so a
decoded document matches the original in every non-whitespace character and
differs only in line breaks. The corpus therefore carries no paragraph
structure, and the model cannot learn to emit any.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent

# uint16 holds 0..65535. The guard below is a hard failure, not a warning.
_UINT16_MAX = 65535


@dataclass
class ShardStats:
    """What one encoded split contains, for verification and the report."""

    split: str
    documents: int
    tokens: int
    eos_tokens: int
    bytes_on_disk: int
    dtype: str
    min_id: int
    max_id: int


def encode_split(
    split_path: Path,
    out_path: Path,
    sp_model,
    eos_id: int,
    batch_docs: int = 2000,
) -> ShardStats:
    """Encode one split file into a flat uint16 array on disk.

    Streams and appends in batches so peak memory is one batch, not the whole
    corpus: the Nepali train split is 11GB of text and ~880M tokens, which
    would be 1.7GB as a single in-memory array plus the Python overhead of
    building it.

    SentencePiece is called with a *list* of texts rather than one at a time —
    batching moves the loop into the C++ extension and is several times faster
    than a Python-level per-document call.
    """
    import numpy as np

    documents = tokens = eos_count = 0
    min_id, max_id = _UINT16_MAX, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out_f:
        batch: list[str] = []

        def flush(batch: list[str]) -> None:
            nonlocal tokens, eos_count, min_id, max_id
            if not batch:
                return
            # One EOS appended per document, so windows that cross a document
            # boundary see an explicit end marker.
            ids: list[int] = []
            for piece in sp_model.encode(batch, out_type=int):
                ids.extend(piece)
                ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(out_f)
            tokens += arr.size
            eos_count += len(batch)
            if arr.size:
                min_id = min(min_id, int(arr.min()))
                max_id = max(max_id, int(arr.max()))

        with split_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text") or ""
                except json.JSONDecodeError:
                    continue
                if not text:
                    continue
                batch.append(text)
                documents += 1
                if len(batch) >= batch_docs:
                    flush(batch)
                    batch = []
                    if documents % 200_000 == 0:
                        print(f"    {split_path.stem}: {documents:,} docs, "
                              f"{tokens:,} tokens", file=sys.stderr)
            flush(batch)

    return ShardStats(
        split=split_path.stem,
        documents=documents,
        tokens=tokens,
        eos_tokens=eos_count,
        bytes_on_disk=out_path.stat().st_size,
        dtype="uint16",
        min_id=min_id if documents else 0,
        max_id=max_id,
    )


def verify_shard(out_path: Path, sp_model, split_path: Path, stats: ShardStats,
                 n_docs: int = 20) -> dict:
    """Re-encode the first N documents and check they match the shard's head.

    This is the test for the failure that would otherwise be invisible: an
    off-by-one in the packing, a dtype that silently wrapped, or a shard
    written from a different tokenizer than the one recorded. Comparing ids
    rather than trusting the token count is the point — a wrong-but-plausible
    count still matches.
    """
    import numpy as np

    expected: list[int] = []
    with split_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n_docs:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text") or ""
            if not text:
                continue
            expected.extend(sp_model.encode(text, out_type=int))
            expected.append(sp_model.eos_id())

    head = np.fromfile(out_path, dtype=np.uint16, count=len(expected))
    matches = bool(len(head) == len(expected) and (head == np.asarray(expected, dtype=np.uint16)).all())

    # File size must equal tokens x 2 bytes exactly; a mismatch means a
    # truncated write, which this project has already been bitten by twice.
    size_ok = stats.bytes_on_disk == stats.tokens * 2

    return {
        "head_ids_match": matches,
        "compared_tokens": len(expected),
        "size_consistent": size_ok,
        "expected_bytes": stats.tokens * 2,
        "actual_bytes": stats.bytes_on_disk,
    }


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
    import sentencepiece as spm

    cfg = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"Pack {cfg.lang_name} splits into uint16 shards.")
    ap.add_argument("--tokenizer", default=None,
                    help="path to .model (default: <tokenizer_dir>/<code>_bpe_16000.model)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write .bin shards (default: <root>/data/tokenized)")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--batch-docs", type=int, default=2000)
    args = ap.parse_args(argv)

    tok_path = Path(args.tokenizer) if args.tokenizer else \
        Path(cfg.paths.tokenizer_dir) / f"{cfg.lang_code}_bpe_16000.model"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.root) / "data" / "tokenized"

    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    vocab_size = sp.get_piece_size()

    # Hard guard: uint16 silently wraps above 65535, producing a corpus that
    # trains without error and is wrong.
    if vocab_size > _UINT16_MAX:
        raise ValueError(
            f"vocab_size={vocab_size} exceeds uint16 range ({_UINT16_MAX}); "
            "use uint32 and update the metadata dtype."
        )

    print(f"[{cfg.lang_name}] tokenizer={tok_path.name} vocab={vocab_size:,} "
          f"eos_id={sp.eos_id()}  -> {out_dir}")

    splits_dir = Path(cfg.paths.splits_dir)
    all_stats: list[ShardStats] = []
    verifications: dict[str, dict] = {}

    for split in args.splits:
        src = splits_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  ! {src} missing, skipping")
            continue
        dst = out_dir / f"{split}.bin"
        print(f"  encoding {split} …")
        st = encode_split(src, dst, sp, sp.eos_id(), args.batch_docs)
        all_stats.append(st)
        verifications[split] = verify_shard(dst, sp, src, st)
        print(f"    {st.documents:,} docs  {st.tokens:,} tokens  "
              f"{st.bytes_on_disk/2**30:.2f} GiB  ids[{st.min_id}..{st.max_id}]")

    total_tokens = sum(s.tokens for s in all_stats)
    payload = {
        "lang_code": cfg.lang_code,
        "tokenizer": tok_path.name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "eos_id": sp.eos_id(),
        "bos_id": sp.bos_id(),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "documents_separated_by": "eos",
        "shards": [asdict(s) for s in all_stats],
        "verification": verifications,
        "total_tokens": total_tokens,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"\n  TOTAL {total_tokens:,} tokens across {len(all_stats)} shards")
    for split, v in verifications.items():
        ok = v["head_ids_match"] and v["size_consistent"]
        print(f"    {split:5s} verify: {'PASS' if ok else 'FAIL'}  "
              f"(head {v['compared_tokens']:,} ids, size {v['actual_bytes']:,}B)")
    print(f"  meta -> {out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
