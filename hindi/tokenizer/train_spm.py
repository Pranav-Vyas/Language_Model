"""SentencePiece BPE tokenizer training and vocab-size sweep (T1.11/T1.12).

Duplicated verbatim into every language tree; never imports from a sibling.
The two tokenizers are trained separately and their vocabularies are never
shared -- that is a hard constraint of the project, not a preference.

WHAT THE SPEC ASKS FOR
----------------------
"Train a separate tokenizer from scratch for each language (BPE,
SentencePiece, or Unigram). Recommended vocabulary size is in the tens of
thousands per model; choose using fertility / unknown-token rate on held-out
text." So the vocab size must be *chosen on evidence*, which is why this
module sweeps several sizes and scores each one rather than picking a
round number.

WHY BPE
-------
Locked project decision. Devanagari is alphasyllabic with heavy inflection;
BPE's merge behaviour on it is straightforward to explain, and byte-fallback
guarantees any unseen byte still round-trips instead of becoming UNK.

TRAINING ON A SAMPLE, AND WHY THAT IS CORRECT
--------------------------------------------
SentencePiece holds its sampled training sentences in memory, so
`--input_sentence_size` is a hard RAM bound rather than a quality dial.
Measured here: passing all 2,536,394 sentence chunks (~2.3KB each) needed
~5.8GB and the kernel OOM-killed the run on a 7GB machine. The default is
therefore 400,000 chunks (~0.9GB), sampled uniformly with
`--shuffle_input_sentence`. That is ample: BPE merges are driven by
frequency statistics that converge long before the whole corpus is seen.
The sample size is recorded in the run config so the choice is visible.

The sweep trains on TRAIN ONLY. Fitting the vocabulary on held-out text
would leak evaluation data into the tokenizer, and every fertility/UNK
number afterwards would be measured on text the tokenizer had already seen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent

_WS_RE = re.compile(r"\s+")


@dataclass
class VocabScore:
    """Evidence for one candidate vocabulary size.

    `fertility` is tokens-per-word: how many pieces the tokenizer spends on
    an average word. Lower means a more efficient encoding, but pushing it
    down costs embedding parameters, and at ~25M total parameters the
    embedding matrix is a large share of the budget -- so this is a genuine
    trade-off rather than a metric to minimise.

    `unk_rate` is the fraction of tokens that decode to UNK. With byte
    fallback enabled this should be ~0; a non-zero value means something is
    escaping the byte fallback and needs investigating.
    """

    vocab_size: int
    fertility: float          # tokens / word on held-out text
    unk_rate: float           # UNK tokens / total tokens
    chars_per_token: float    # inverse compression: higher = denser encoding
    tokens_total: int
    words_total: int
    roundtrip_ok: bool        # no CONTENT lost (see score_model for the basis)
    roundtrip_exact: bool     # byte-identical round trip (stricter)
    docs_zwnj_altered: int    # docs whose ZWJ/ZWNJ the normaliser removed


def read_sample(path: Path, limit: int | None = None) -> list[str]:
    """Load documents from a split file as a list of strings.

    Used for *evaluation* (held-out scoring), which needs the text in
    memory. Training reads from a plain-text file instead, so the large
    corpus is never materialised as Python objects.
    """
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text") or ""
            if text:
                out.append(text)
    return out


# Devanagari sentence terminators (danda / double danda) plus ASCII
# terminators, used to cut documents at natural boundaries.
_SENT_END_RE = re.compile(r"(?<=[।॥.!?])\s+")

# SentencePiece silently SKIPS input lines longer than --max_sentence_length
# (default 4192 bytes). Devanagari costs 3 bytes per character in UTF-8, so a
# ~1,900-character article is already ~5,700 bytes: measured on this corpus,
# 34.9% of documents exceeded the default and they carried 81.8% of all the
# text. Feeding whole articles would therefore have trained the tokenizer on
# 18.2% of the corpus, and specifically on the *short* documents -- a biased
# sample of stubs and briefs, with no long-form prose at all.
#
# The fix is to feed sentence-sized units, which is what SentencePiece
# expects in the first place. Documents are cut at sentence boundaries and
# packed into chunks below the byte budget.
_CHUNK_BYTES = 3000


def _chunks(text: str):
    """Yield sentence-aligned chunks of `text` under the byte budget.

    Splitting at sentence ends (rather than mid-word every N bytes) keeps the
    token statistics honest: a cut through the middle of a word would invent
    word-boundary evidence that does not exist in the language.
    """
    buf, buf_bytes = [], 0
    for piece in _SENT_END_RE.split(text):
        if not piece:
            continue
        pb = len(piece.encode("utf-8"))
        if pb >= _CHUNK_BYTES:
            # A single sentence over budget (rare: tables, run-on OCR text).
            # Emit what we have, then hard-cut this one on whitespace.
            if buf:
                yield " ".join(buf); buf, buf_bytes = [], 0
            words, cur, cur_b = piece.split(), [], 0
            for w in words:
                wb = len(w.encode("utf-8")) + 1
                if cur and cur_b + wb > _CHUNK_BYTES:
                    yield " ".join(cur); cur, cur_b = [], 0
                cur.append(w); cur_b += wb
            if cur:
                yield " ".join(cur)
            continue
        if buf and buf_bytes + pb > _CHUNK_BYTES:
            yield " ".join(buf); buf, buf_bytes = [], 0
        buf.append(piece); buf_bytes += pb
    if buf:
        yield " ".join(buf)


def iter_training_sentences(split_path: Path, max_docs: int | None = None):
    """Yield one flattened line of text per training document.

    STREAMS INSTEAD OF WRITING A TEMP FILE. An earlier version flattened the
    train split to a ~4GB plain-text file for SentencePiece to read. On this
    machine that was a third full copy of the corpus (alongside clean.jsonl
    and the split files) and it filled the disk mid-run, truncating output.
    SentencePiece accepts a `sentence_iterator`, so the flattened text never
    has to exist on disk at all.

    Newlines inside a document become spaces so that one yielded item = one
    training sentence; keeping them would make SentencePiece treat every
    paragraph as an independent sentence and skew its length statistics.
    """
    n = 0
    with split_path.open(encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            text = (json.loads(line).get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            for chunk in _chunks(text):
                yield chunk
            n += 1
            if max_docs is not None and n >= max_docs:
                return


def train_one(
    sentences,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
    byte_fallback: bool,
    input_sentence_size: int,
    seed: int,
) -> None:
    """Train a single SentencePiece BPE model.

    `character_coverage` below 1.0 lets rare characters fall out of the
    vocabulary; for Devanagari 0.9995 keeps the script's real inventory while
    discarding one-off junk codepoints that survived cleaning. Byte fallback
    then guarantees those still encode losslessly.
    """
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=8,
        # Fixed ids so both languages' models agree on the special tokens,
        # which keeps the Phase 2 model code identical across trees.
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        train_extremely_large_corpus=False,
        seed_sentencepiece_size=1_000_000,
        minloglevel=1,
    )


def score_model(model_path: Path, held_out: list[str]) -> VocabScore:
    """Measure fertility, UNK rate and round-trip fidelity on held-out text.

    Round-trip is checked because a tokenizer that silently loses characters
    would still produce excellent-looking fertility numbers. It is the cheap
    test for the failure that would otherwise be invisible.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    tokens_total = words_total = unk_total = chars_total = 0
    roundtrip_ok = True
    roundtrip_exact = True
    zwnj_altered = 0

    for text in held_out:
        ids = sp.encode(text, out_type=int)
        tokens_total += len(ids)
        words_total += len(text.split())
        chars_total += len(text)
        unk_total += sum(1 for i in ids if i == unk_id)

        decoded = sp.decode(ids)
        if decoded != text:
            roundtrip_exact = False

        # SentencePiece applies NFKC-style normalisation of its own, so a
        # decoded string can differ from the input WITHOUT any content being
        # lost. Measured on this corpus, the differences are exactly two
        # kinds, both from the normaliser, not from the vocabulary:
        #   1. whitespace class changes (newline -> space)
        #   2. NFKC compatibility mappings, e.g. U+00B3 SUPERSCRIPT THREE -> "3"
        # Comparing raw strings therefore reports "FAILED" on ~78% of
        # documents while nothing is actually lost. The meaningful test is
        # whether content survives once the same normalisation is applied to
        # both sides.
        # ZWJ/ZWNJ are excluded here and counted separately below, so this
        # flag answers one question only: does anything OTHER than the known
        # normaliser behaviour go missing? Folding both into a single boolean
        # made every vocab size report "FAILED" and hid the fact that the
        # only real loss is the joiners.
        # Compare the non-whitespace character sequence only. Justification,
        # from measuring the actual differences rather than assuming:
        #   * newline -> space          (whitespace class change)
        #   * U+00B3 -> "3"             (NFKC compatibility mapping)
        #   * ZWNJ   -> space           (the normaliser substitutes, not deletes)
        # A character-multiset diff over 3,000 held-out documents found ZERO
        # characters present in the input and absent from the output, so no
        # content is lost; every difference is whitespace or NFKC. Stripping
        # whitespace and joiners before comparing is therefore the test that
        # actually answers "did we lose anything?".
        norm_in = _WS_RE.sub("", unicodedata.normalize("NFKC", text)) \
            .replace("\u200c", "").replace("\u200d", "")
        norm_out = _WS_RE.sub("", unicodedata.normalize("NFKC", decoded)) \
            .replace("\u200c", "").replace("\u200d", "")
        if norm_in != norm_out:
            roundtrip_ok = False

        # ZWJ/ZWNJ are tracked separately because they are a REAL loss, not a
        # cosmetic one: clean.py deliberately preserves them (they control
        # Devanagari conjunct formation) but SentencePiece's normaliser strips
        # them, so the model can never emit one. Reported, not silently
        # accepted, so the trade-off is visible in the corpus statistics.
        if ("\u200c" in text or "\u200d" in text) and \
           (("\u200c" in text) != ("\u200c" in decoded) or
            ("\u200d" in text) != ("\u200d" in decoded)):
            zwnj_altered += 1

    return VocabScore(
        vocab_size=sp.get_piece_size(),
        fertility=tokens_total / words_total if words_total else 0.0,
        unk_rate=unk_total / tokens_total if tokens_total else 0.0,
        chars_per_token=chars_total / tokens_total if tokens_total else 0.0,
        tokens_total=tokens_total,
        words_total=words_total,
        roundtrip_ok=roundtrip_ok,
        roundtrip_exact=roundtrip_exact,
        docs_zwnj_altered=zwnj_altered,
    )


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
    ap = argparse.ArgumentParser(description=f"SentencePiece BPE sweep ({cfg.lang_name}).")
    ap.add_argument("--train-docs", type=int, default=None,
                    help="cap documents flattened for training (default: all).")
    ap.add_argument("--input-sentence-size", type=int, default=400_000,
                    help="SentencePiece random sample size. THIS IS A MEMORY "
                         "BOUND, not a quality knob: SentencePiece holds the "
                         "sampled sentences in RAM, so 2.5M chunks at ~2.3KB "
                         "each is ~5.8GB and OOM-kills a 7GB machine (it did). "
                         "400k chunks is ~0.9GB and is ample for BPE, whose "
                         "merge statistics converge long before that.")
    ap.add_argument("--eval-docs", type=int, default=5000,
                    help="held-out documents used for fertility/UNK scoring.")
    ap.add_argument("--vocab-sizes", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    splits_dir = Path(cfg.paths.splits_dir)
    tok_dir = Path(cfg.paths.tokenizer_dir)
    stats_dir = Path(cfg.paths.stats_dir)
    tok_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    vocab_sizes = args.vocab_sizes or list(cfg.tokenizer.vocab_sizes)

    # The iterator is re-created per vocab size (it is consumed each time);
    # nothing is materialised on disk.
    train_path = splits_dir / "train.jsonl"
    # Counts CHUNKS, not documents: iter_training_sentences splits each
    # document into sentence-aligned pieces, so this is the number of
    # training units SentencePiece will see. Naming it "documents" would
    # overstate the corpus size in the reported statistics.
    # ONE pass, not two: an earlier version counted chunks and documents
    # separately, which meant reading a multi-GB split file twice before
    # SentencePiece had even started (it then reads it a third time).
    n_docs = n_chunks = 0
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            text = (json.loads(line).get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            n_docs += 1
            n_chunks += sum(1 for _ in _chunks(text))
            if args.train_docs is not None and n_docs >= args.train_docs:
                break
    print(f"[{cfg.lang_name}] {n_docs:,} train documents -> {n_chunks:,} "
          f"sentence chunks (streamed, no temp file)")

    held_out = read_sample(splits_dir / "val.jsonl", args.eval_docs)
    print(f"  {len(held_out):,} held-out documents for scoring\n")

    results: list[VocabScore] = []
    for vs in vocab_sizes:
        prefix = tok_dir / f"{cfg.lang_code}_bpe_{vs}"
        print(f"  training vocab_size={vs:,} …")
        train_one(
            iter_training_sentences(train_path, args.train_docs), prefix, vs,
            cfg.tokenizer.character_coverage, cfg.tokenizer.byte_fallback,
            args.input_sentence_size, cfg.seed,
        )
        score = score_model(prefix.with_suffix(".model"), held_out)
        results.append(score)
        print(f"    fertility={score.fertility:.3f} tok/word   "
              f"chars/token={score.chars_per_token:.3f}   "
              f"UNK={score.unk_rate:.6f}   content={'ok' if score.roundtrip_ok else 'LOST'}")

    payload = {
        "lang_code": cfg.lang_code,
        "model_type": cfg.tokenizer.model_type,
        "character_coverage": cfg.tokenizer.character_coverage,
        "byte_fallback": cfg.tokenizer.byte_fallback,
        "input_sentence_size": args.input_sentence_size,
        "train_documents": n_docs,
        "train_chunks": n_chunks,
        "eval_documents": len(held_out),
        "sweep": [asdict(r) for r in results],
    }
    out = stats_dir / "tokenizer_sweep.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  {'vocab':>8s} {'fertility':>10s} {'chars/tok':>10s} {'UNK rate':>10s} "
          f"{'content':>9s} {'exact':>7s} {'zwnj-lost':>10s}")
    for r in results:
        print(f"  {r.vocab_size:>8,} {r.fertility:>10.3f} {r.chars_per_token:>10.3f} "
              f"{r.unk_rate:>10.6f} {'ok' if r.roundtrip_ok else 'LOST':>9s} "
              f"{'ok' if r.roundtrip_exact else 'no':>7s} {r.docs_zwnj_altered:>10,}")
    print(f"\n  sweep -> {out}")
    print("  (vocab choice is made in T1.12 from these numbers, not automatically)")


if __name__ == "__main__":
    main()
