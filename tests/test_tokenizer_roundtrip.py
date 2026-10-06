"""Tokenizer and packed-shard round-trip tests (Phase 2, T2.5).

The model tests prove the architecture is correct. These prove the *data
reaching it* is correct -- the other half of a silent failure. If the packed
uint16 shards decoded back to something other than the corpus, training would
still converge, still print a falling loss, and still report a perplexity;
the number would just be measuring a different corpus than the one described
in the report.

Each language is tested against its OWN tokenizer only. There is no shared
vocabulary and no cross-tree import: independence is a graded constraint, so
the test is written per-tree and parametrised over the two trees rather than
factored into anything shared.

NOTE ON EXACTNESS
-----------------
SentencePiece's default normaliser (nmt_nfkc) collapses runs of whitespace,
so `"a\t\tb"` decodes as `"a b"`. That is a normaliser property, not a
tokenizer defect, and it is harmless here because the cleaning pipeline
already collapsed whitespace before training. The tests below therefore
assert exact round-trip on realistic corpus text (which is already
normalised) and assert the whitespace behaviour explicitly rather than
quietly tolerating a mismatch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

spm = pytest.importorskip("sentencepiece")
np = pytest.importorskip("numpy")

# (tree, tokenizer file, a sentence genuinely in that language)
TREES = [
    ("hindi", "hindi/tokenizer/hi_bpe_16000.model",
     "भारत में हिन्दी सबसे अधिक बोली जाने वाली भाषा है।"),
    ("nepali", "nepali/tokenizer/ne_bpe_16000.model",
     "नेपाल र भारतबीचको सम्बन्ध ऐतिहासिक रूपमा पुरानो छ।"),
]


def load(model_rel: str):
    path = REPO / model_rel
    if not path.exists():
        pytest.skip(f"{model_rel} not present")
    return spm.SentencePieceProcessor(model_file=str(path))


@pytest.mark.parametrize("tree,model_rel,sentence", TREES)
def test_encode_decode_is_exact(tree, model_rel, sentence):
    """encode -> decode returns the original string, character for character."""
    sp = load(model_rel)
    ids = sp.encode(sentence, out_type=int)
    assert ids, "tokenizer produced no ids"
    assert sp.decode(ids) == sentence, (
        f"[{tree}] round-trip changed the text:\n  in : {sentence!r}\n"
        f"  out: {sp.decode(ids)!r}"
    )


@pytest.mark.parametrize("tree,model_rel,sentence", TREES)
def test_no_unk_on_mixed_script_text(tree, model_rel, sentence):
    """Byte fallback means UNK should be unreachable, even off-distribution.

    Phase 1 reported UNK rate 0.000000 for both languages. That claim is only
    credible if it also holds for text the tokenizer was not trained on --
    Latin script, emoji, Devanagari and ASCII digits, rare punctuation.
    """
    sp = load(model_rel)
    hostile = sentence + " Ünïcode ☃ 東京 ١٢٣ १२३ 123 ~`^ \\ %&*"
    ids = sp.encode(hostile, out_type=int)
    unk = [i for i in ids if i == sp.unk_id()]
    assert not unk, f"[{tree}] {len(unk)} UNK ids on mixed-script text"
    assert sp.decode(ids) == hostile, f"[{tree}] mixed-script text did not survive"


@pytest.mark.parametrize("tree,model_rel,sentence", TREES)
def test_whitespace_collapsing_is_the_only_known_lossy_case(tree, model_rel, sentence):
    """Pin the one documented non-exact case so it can never widen unnoticed.

    Asserted rather than ignored: if a future normaliser change made the
    tokenizer lossy in some *other* way, a test that merely tolerated
    inexact round-trips would stay green.
    """
    sp = load(model_rel)
    padded = "क\t\tख   ग"
    assert sp.decode(sp.encode(padded, out_type=int)) == "क ख ग"
    # Newlines collapse the same way -- this is why the packed corpus cannot
    # reproduce a document's line breaks, and why generation samples in the
    # Phase-2 report will be unparagraphed. Documented, not a defect.
    assert sp.decode(sp.encode("क\nख", out_type=int)) == "क ख"


@pytest.mark.parametrize("tree,model_rel,sentence", TREES)
def test_packed_shard_decodes_back_to_the_corpus(tree, model_rel, sentence):
    """Read real ids off the uint16 shard and decode them to real documents.

    This closes the loop the training loader depends on:
        split.jsonl -> encode -> uint16 shard -> mmap -> decode == original.

    Skipped when the corpus is absent (it is gitignored and lives on Drive),
    so the suite still passes on a fresh clone.
    """
    root = REPO / tree
    shard = root / "data" / "tokenized" / "val.bin"
    meta_p = root / "data" / "tokenized" / "meta.json"
    split = root / "corpus" / "splits" / "val.jsonl"
    if not (shard.exists() and meta_p.exists() and split.exists()):
        pytest.skip(f"{tree} corpus artifacts not on this machine")

    sp = load(model_rel)
    meta = json.loads(meta_p.read_text())

    # The shard must have been written by THIS tokenizer and this vocab.
    assert meta["vocab_size"] == sp.get_piece_size()
    assert meta["lang_code"] == ("hi" if tree == "hindi" else "ne")
    assert meta["dtype"] == "uint16"

    # Rebuild what the head of the shard must contain from the source text.
    expected: list[int] = []
    originals: list[str] = []
    with split.open(encoding="utf-8") as fh:
        for _ in range(5):
            line = fh.readline()
            if not line:
                break
            text = json.loads(line).get("text") or ""
            if not text:
                continue
            originals.append(text)
            expected.extend(sp.encode(text, out_type=int))
            expected.append(sp.eos_id())

    head = np.fromfile(shard, dtype=np.uint16, count=len(expected))
    assert head.tolist() == expected, f"[{tree}] shard head != re-encoded source"

    # And the ids on disk decode back to the document they came from.
    #
    # "Exactly", precisely stated: the normaliser rewrites newlines as spaces
    # (measured on a real val document: 4 newlines -> 4 spaces, length
    # unchanged, every non-whitespace character preserved in order). So the
    # assertion is that NO non-whitespace character is lost, added, or
    # reordered, and that the text is identical once whitespace runs are
    # collapsed -- which is strictly stronger than an == that would have to be
    # dropped, and it would fail the moment any real character changed.
    first_eos = expected.index(sp.eos_id())
    decoded = sp.decode([int(i) for i in head[:first_eos]])
    source = originals[0]
    assert [c for c in decoded if not c.isspace()] == \
           [c for c in source if not c.isspace()], \
        f"[{tree}] shard decode altered non-whitespace content"
    assert re.sub(r"\s+", " ", decoded).strip() == \
           re.sub(r"\s+", " ", source).strip(), \
        f"[{tree}] shard decode differs beyond whitespace normalisation"

    # Every id on disk is a real vocabulary id -- catches a uint16 wrap, which
    # would otherwise produce ids that train without ever raising.
    assert int(head.max()) < sp.get_piece_size()
