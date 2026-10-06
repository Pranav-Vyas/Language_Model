"""Cleaning pipeline for one language tree (T1.6).

This file is duplicated *verbatim* into every language directory. It never
imports from a sibling language tree -- that would be an independence
violation (see config.py's guard).

WHY THIS EXISTS
---------------
The spec grades two things this module produces:

  1. A cleaned corpus: "remove invalid/duplicate data; normalize Unicode".
  2. The *cleaning steps table*: how many documents and tokens each step
     dropped. That table is only reconstructable if we count as we go, so
     every step records its own before/after numbers.

THE 20% FLOOR IS THE REASON FOR PER-BUCKET ACCOUNTING
-----------------------------------------------------
The spec requires >=20% of the *final training tokens* to come from manual
collection -- not 20% of the raw scrape. Manual text (scraped news/wiki)
and downloaded text (Sangraha) do NOT survive cleaning at the same rate:
scraped pages carry boilerplate, stubs and near-duplicates, while Sangraha
arrives already filtered. So a corpus that is 20% manual before cleaning can
land under 20% after it.

Every step therefore tracks docs/chars/words split by `source_bucket`, so
the final ratio is a measured number and the downloaded tier can be trimmed
afterwards to restore the floor if cleaning ate the margin.

WHY CHARS AND WORDS, NOT "TOKENS"
---------------------------------
Real token counts only exist once the tokenizer is trained (T1.11). Any
number reported before that is a fertility estimate, and mixing two
different fertility constants silently corrupts the manual percentage --
that bug actually happened on this project. So this module records raw
`chars` and `words` only; tokens are derived once, at report time, from a
single stated basis.

PIPELINE ORDER (and why)
------------------------
  normalize  -> NFC, strip control/zero-width chars, collapse whitespace
                (must run first: every later check compares normalized text,
                and dedup hashes would otherwise miss byte-level variants)
  filter     -> cheap structural rejects (empty/short/script-ratio/junk),
                ordered cheapest-first so expensive checks see fewer docs
  exact dedup-> SHA-256 over normalized text; catches byte-identical repeats
                (Sangraha and our own Wikipedia scrape genuinely overlap)

Near-duplicate (MinHash/LSH) detection is a separate pass (`near_dedup.py`)
because it needs memory proportional to the corpus and must not be run while
the scrapers still hold RAM.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

TREE = Path(__file__).resolve().parent.parent

# --- Unicode character classes we strip -------------------------------------
# C0/C1 control characters except tab/newline, plus the "format" characters
# (zero-width joiner/non-joiner, bidi marks, BOM). ZWJ/ZWNJ matter in
# Devanagari for ligature control, so they are handled explicitly below
# rather than being blanket-stripped.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH_RE = re.compile(r"[​‎‏﻿⁠]")
_MULTI_SPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Devanagari block (used by both Hindi and Nepali) plus the Vedic extensions.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
# Any character that carries linguistic content (letters only -- digits and
# punctuation are script-neutral and would inflate the ratio).
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# Normalizers: pure text -> text. Each is independently testable.
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply NFC composition.

    Devanagari can encode the same grapheme as either a precomposed codepoint
    or a base + combining mark. Without NFC the tokenizer would learn two
    different subword inventories for visually identical text, inflating
    vocabulary and UNK rate. NFC (not NFD) because we want the *composed*
    form, which is what Devanagari text on the web overwhelmingly uses.
    """
    return unicodedata.normalize("NFC", text)


def strip_invisible(text: str) -> str:
    """Remove control characters and zero-width/bidi formatting characters.

    Keeps `\\t` and `\\n` (real structure) and keeps ZWJ (U+200D) / ZWNJ
    (U+200C), which are meaningful in Devanagari conjunct formation --
    stripping those would change how words render and tokenize.
    """
    text = _CONTROL_RE.sub("", text)
    return _ZERO_WIDTH_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and cap blank runs at one empty line.

    Scraped HTML routinely yields long space runs and dozens of blank lines;
    left in, they waste tokenizer budget and distort avg-chars-per-token.
    """
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NL_RE.sub("\n\n", text).strip()


# Localized headings that begin a Wikipedia/Wikisource *navigation* tail:
# "See also", "References", "External links", "Notes", "Bibliography".
# Everything from the first such heading to the end of the document is
# apparatus, not prose. Nepali equivalents are included in this same list so
# the two trees' copies stay verbatim-identical (an unused entry is harmless);
# the language filter, not this list, is what separates the corpora.
_NAV_HEADINGS = (
    # Hindi
    "इन्हें भी देखें", "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "बाहरी कडियाँ",
    "टिप्पणी", "टिप्पणियाँ", "ग्रंथ सूची", "अग्रिम पठन", "इन्हे भी देखें",
    # Nepali
    "यो पनि हेर्नुहोस्", "सन्दर्भ सामग्री", "बाह्य लिङ्क", "बाह्य कडीहरू",
    "थप हेर्नुहोस्", "सन्दर्भहरू",
)


def strip_navigation_tail(text: str) -> str:
    """Remove the trailing See-also/References/External-links apparatus.

    RULE-BASED CORRECTION (Yi stage 5). This exists because of a measured
    false positive, not a guess: without it, `short_line_ratio` classified
    ~30% of Hindi Wikipedia as "mostly short lines" and discarded them. Those
    documents were real stub articles -- one genuine prose sentence followed
    by a see-also list of short link lines. The list dragged the short-line
    ratio above threshold and took the prose with it.

    Course L3 slide 30 is explicit that on the low-resource side you should
    "repair what you can rather than discarding it", so we cut the apparatus
    and re-judge the remaining prose on its own merits. A stub whose prose is
    genuinely too thin still fails `min_chars` afterwards -- but now it is
    dropped for the right reason, and articles with real bodies survive.

    Only a heading that sits on its own line counts, so a mid-sentence
    mention of the word "सन्दर्भ" cannot truncate a document.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() in _NAV_HEADINGS:
            return "\n".join(lines[:i]).strip()
    return text


def normalize(text: str) -> str:
    """Full normalization chain, in the only order that is correct.

    NFC first (so later regexes see canonical codepoints), then invisibles,
    then whitespace (which can only be judged once invisibles are gone),
    then the navigation-tail repair (which needs clean line boundaries to
    match a heading line exactly).
    """
    text = collapse_whitespace(strip_invisible(normalize_unicode(text)))
    return strip_navigation_tail(text)


# ---------------------------------------------------------------------------
# Metrics used by filters (separate from the filters so they are testable and
# so the stats module can reuse them without re-deciding policy).
# ---------------------------------------------------------------------------

def script_ratio(text: str) -> float:
    """Fraction of *letters* that are Devanagari.

    Digits/punctuation/whitespace are excluded from the denominator: a price
    table is not "less Hindi" for containing numerals. Returns 1.0 for text
    with no letters at all so that such docs are rejected by the length or
    junk filters instead of being double-penalised here.
    """
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 1.0
    deva = sum(1 for ch in letters if _DEVANAGARI_RE.match(ch))
    return deva / len(letters)


def repeated_line_ratio(text: str) -> float:
    """Fraction of lines that are exact repeats of an earlier line.

    Catches nav bars, "Advertisment" spacers, and scraper failures that
    duplicate a block -- the failure mode that leaked boilerplate into 73%
    of one source earlier in this project.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for ln in lines:
        if ln in seen:
            repeats += 1
        else:
            seen.add(ln)
    return repeats / len(lines)


def symbol_ratio(text: str) -> float:
    """Punctuation/symbol characters as a fraction of non-space characters.

    Yi heuristic (L3 slide 15, "Symbol ratio"): menus, link farms and broken
    tables carry far more brackets/pipes/bullets per letter than prose does.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    symbols = sum(1 for c in chars if not c.isalnum())
    return symbols / len(chars)


def short_line_ratio(text: str, threshold: int = 40) -> float:
    """Fraction of non-empty lines shorter than `threshold` characters.

    Yi heuristic (L3 slide 15, "Line structure"): a page of many short lines
    is a navigation menu or a link list, not running prose. Kept separate
    from `repeated_line_ratio` because a menu's entries are usually distinct.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 5:
        return 0.0
    return sum(1 for ln in lines if len(ln) < threshold) / len(lines)


def repeated_ngram_ratio(text: str, n: int = 10) -> float:
    """Fraction of word n-grams that repeat within the document.

    Yi heuristic (L3 slide 15, "Repetition": the same word, n-gram or
    paragraph repeated inside one document). Line-level repetition misses
    text that repeats mid-paragraph, e.g. a templated listing or a stuck
    generation loop, so this complements `repeated_line_ratio`.
    """
    words = text.split()
    if len(words) < n * 4:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


# ---------------------------------------------------------------------------
# Language identification (Yi stage 2 -- "the first filter in every pipeline")
# ---------------------------------------------------------------------------
# Hindi and Nepali are both Devanagari, so a script check cannot separate them.
# Course L3 slide 27 names exactly this as the headline Indic failure mode:
# "Hindi, Marathi, Nepali and Bhojpuri share Devanagari. Detectors confuse
# them, so your corpus is contaminated with a neighbouring language."
#
# Here it is also an *independence* risk: Model H and Model L must not share
# documents, and a Nepali page sitting in the Hindi corpus is both a language
# error and a contamination of the H-vs-L comparison.
#
# The marker lists below were derived empirically, not guessed: word
# frequencies were computed over 8,000 hi.wikipedia and 8,000 ne.wikipedia
# documents and ranked by log-odds. The selected markers are effectively
# disjoint between the two languages (most have a zero count on the other
# side). Measured on held-out slices, cross-language error is ~0.03% for
# Hindi and ~0.15% for Nepali; the residual is an "unknown" bucket of very
# short documents, which is kept rather than dropped.
#
# VALIDATED on held-out slices (4,000 docs each, disjoint from the slices the
# markers were derived from):
#   hi.wikipedia   98.45% hi, 1 doc (0.025%) mis-flagged as Nepali
#   ne.wikipedia   85.88% ne, 6 docs (0.150%) flagged Hindi
#   livehindustan  96.88% hi, 0 cross-language errors
#   nepalpress     98.95% ne, 0 cross-language errors
#   hi.wikisource  90.03% hi, 0 cross-language errors
# All 6 ne.wikipedia flags were read by hand: 2 are GENUINE Hindi pages
# sitting in ne.wikipedia (real contamination, correctly removed) and 4 are
# false positives on Sanskrit-heavy religious prose or articles quoting Hindi
# film/TV titles. The residual "unknown" bucket is short or atypical text and
# is deliberately kept, not dropped.
OWN_LANG = "ne"

_LID_PUNCT_RE = re.compile(r"[।,.;:!?\"'()\[\]{}—–-]")

# Every marker below was kept only if |log-odds| > 2 between 10,000
# hi.wikipedia and 10,000 ne.wikipedia documents. Two candidates were
# REMOVED after that check caught them as false friends:
#   "को"  log-odds -1.35 -- Hindi accusative/dative but ALSO the ordinary
#                           Nepali genitive, so it fires on clean Nepali.
#   "हो"  log-odds +0.88 -- common in both languages.
# Including "को" made the filter reject genuine Nepali articles (verified by
# reading them), which is why the list is data-derived rather than intuited.
HINDI_MARKERS = {
    "में", "हैं", "है", "उन्होंने", "करने", "था", "थे", "उन्हें", "करता", "लोगों",
    "जिसमें", "उसके", "इन्हें", "और", "से", "का", "की", "के", "कर",
    "गया", "किया", "रहा", "नहीं",
}
NEPALI_MARKERS = {
    "छ", "छन्", "थियो", "थिए", "रूपमा", "लागि", "भएको", "रहेको", "भने", "पनि",
    "उनले", "गरिएको", "गरेका", "गर्न", "गर्ने", "र", "भन्ने", "गरेको",
    "अनुसार", "तर", "भएका", "हुन्",
}


def classify_language(text: str, min_markers: int = 5, margin: float = 1.5) -> str:
    """Return "hi", "ne", "mixed", or "unknown" for a Devanagari document.

    Counts marker function words on each side and requires one side to
    exceed the other by `margin` before committing. Documents with fewer
    than `min_markers` total hits are "unknown" -- too short to judge.

    Deliberately conservative: only a *confident* other-language verdict
    causes a drop (see `check_document`). L3 slide 30 warns that filters
    tuned for a data surplus will empty a low-resource corpus, so ambiguous
    documents are kept.
    """
    tokens = [_LID_PUNCT_RE.sub("", w) for w in text.split()]
    hi = sum(1 for w in tokens if w in HINDI_MARKERS)
    ne = sum(1 for w in tokens if w in NEPALI_MARKERS)
    if hi + ne < min_markers:
        return "unknown"
    if ne > hi * margin:
        return "ne"
    if hi > ne * margin:
        return "hi"
    return "mixed"


# ---------------------------------------------------------------------------
# Filters: text -> drop-reason (str) or None to keep.
# Ordered cheapest-first by the caller.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterPolicy:
    """Thresholds for the structural filters.

    Config-driven rather than hard-coded so a run's policy is logged next to
    its outputs and can be reproduced. Defaults are deliberately permissive:
    at ~25M params the model benefits more from volume than from a pristine
    corpus, and over-filtering the manual tier directly threatens the 20%
    floor.
    """

    min_chars: int = 200          # below this a "document" is a headline stub
    min_words: int = 20
    max_chars: int = 2_000_000    # "absurdly long" guard (L3 slide 15)
    min_script_ratio: float = 0.50   # >=50% of letters must be Devanagari
    max_repeated_line_ratio: float = 0.40
    max_symbol_ratio: float = 0.45   # menus/link farms sit far above prose
    # LOW-RESOURCE TUNING (course L3 slide 30 / Activity 4). Model L has no
    # surplus: "every filter you add costs you data you cannot replace".
    # Measured on Hindi, short_line_ratio > 0.80 discarded ~16% of Wikipedia,
    # including list-shaped but genuine articles (film cast lists, year
    # entries). On the high-resource side that trade is affordable; here it
    # is not, so the threshold is loosened to 0.92 and only near-pure link
    # lists are removed. This is the one filter deliberately set differently
    # between the two languages, and it is reported as such.
    max_short_line_ratio: float = 0.92  # a page of stubs is a navigation list
    max_repeated_ngram_ratio: float = 0.30
    drop_other_language: bool = True    # drop confident other-language docs


def check_document(text: str, policy: FilterPolicy) -> str | None:
    """Return the name of the first failed check, or None if the doc is kept.

    Returning the *reason* (not just a bool) is what makes the per-step drop
    table in the report possible.

    Checks run cheapest-first: length is O(1)-ish, the ratio metrics are one
    pass each, and the n-gram check (which allocates) runs last so it only
    ever sees documents that already passed everything else.
    """
    if not text:
        return "empty"
    if len(text) < policy.min_chars:
        return "too_short_chars"
    if len(text) > policy.max_chars:
        return "too_long"
    if len(text.split()) < policy.min_words:
        return "too_short_words"
    if script_ratio(text) < policy.min_script_ratio:
        return "low_script_ratio"
    # Language filtering: only a *confident* verdict for the other language
    # drops the document. "unknown"/"mixed" are kept -- see classify_language.
    if policy.drop_other_language:
        verdict = classify_language(text)
        if verdict != OWN_LANG and verdict in ("hi", "ne"):
            return f"wrong_language_{verdict}"
    if symbol_ratio(text) > policy.max_symbol_ratio:
        return "high_symbol_ratio"
    if short_line_ratio(text) > policy.max_short_line_ratio:
        return "mostly_short_lines"
    if repeated_line_ratio(text) > policy.max_repeated_line_ratio:
        return "repetitive_lines"
    if repeated_ngram_ratio(text) > policy.max_repeated_ngram_ratio:
        return "repetitive_ngrams"
    return None


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------

class ExactDeduper:
    """Streaming exact-duplicate detector keyed on normalized text.

    Stores a truncated (8-byte) SHA-256 digest per unique document rather
    than the full 32-byte digest or the text itself: at ~2M documents the
    full corpus does not fit in the ~1GB of RAM available while scrapers are
    running, but 2M 64-bit ints do (~70MB).

    Truncating to 64 bits gives a birthday-collision probability of roughly
    n^2 / 2^65 -- about 1 in 10^7 at 2M docs. That is far below the rate at
    which the corpus itself contains genuine near-duplicates, so it is not
    the limiting source of error.
    """

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def is_duplicate(self, text: str) -> bool:
        """Record the document and report whether it was already present."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
        key = int.from_bytes(digest, "big")
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Statistics: computation only. Presentation lives in stats/report modules,
# per the spec's "keep visualization and computation logic separate" rule.
# ---------------------------------------------------------------------------

@dataclass
class BucketCounts:
    """Docs/chars/words for one accounting bucket (manual | downloaded)."""

    docs: int = 0
    chars: int = 0
    words: int = 0

    def add(self, text: str) -> None:
        self.docs += 1
        self.chars += len(text)
        self.words += len(text.split())


@dataclass
class SourceStats:
    """Per-source cleaning outcome, including why documents were dropped."""

    source_name: str
    source_bucket: str
    docs_in: int = 0
    docs_out: int = 0
    chars_in: int = 0
    chars_out: int = 0
    words_in: int = 0
    words_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream a .jsonl file one record at a time.

    Streaming (not json.load of the whole file) is mandatory here: the Hindi
    Sangraha extract alone is 2.9GB against ~1GB of free RAM.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truncated trailing line is expected if a scraper was
                # killed mid-write; skip it rather than aborting the run.
                continue


def iter_parquet(path: Path, text_column: str = "text") -> Iterator[dict]:
    """Stream a parquet file row-group by row-group.

    Row-group streaming keeps peak memory at one group (~64MB) instead of
    the whole 350MB+ shard.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    col = text_column if text_column in pf.schema_arrow.names else pf.schema_arrow.names[0]
    for batch in pf.iter_batches(batch_size=8192, columns=[col]):
        for value in batch.column(0).to_pylist():
            if value:
                yield {"text": value}


def clean_source(
    records: Iterator[dict],
    source_name: str,
    source_bucket: str,
    policy: FilterPolicy,
    deduper: ExactDeduper,
    out_handle,
) -> SourceStats:
    """Normalize, filter and dedup one source, writing kept docs to `out_handle`.

    The deduper is shared across sources on purpose: cross-source duplicates
    (our own Wikipedia scrape also appears inside Sangraha) must be removed,
    and doing it in one pass avoids a second read of the whole corpus.
    """
    stats = SourceStats(source_name=source_name, source_bucket=source_bucket)

    for rec in records:
        raw = rec.get("text") or ""
        stats.docs_in += 1
        stats.chars_in += len(raw)
        stats.words_in += len(raw.split())

        text = normalize(raw)

        reason = check_document(text, policy)
        if reason is not None:
            stats.record_drop(reason)
            continue

        if deduper.is_duplicate(text):
            stats.record_drop("exact_duplicate")
            continue

        out = {
            "text": text,
            "source_name": source_name,
            "source_bucket": source_bucket,
        }
        # Carry provenance through cleaning so the splits and the final stats
        # can still attribute every token to its source.
        for key in ("url", "title", "doc_id"):
            if rec.get(key):
                out[key] = rec[key]
        out_handle.write(json.dumps(out, ensure_ascii=False) + "\n")

        stats.docs_out += 1
        stats.chars_out += len(text)
        stats.words_out += len(text.split())

    return stats


def summarize(all_stats: list[SourceStats]) -> dict:
    """Aggregate per-source stats into the report-facing summary.

    Returns plain data (no printing, no plotting) so the same numbers feed
    the JSON artifact, the Markdown table, and the 20%-floor check.
    """
    buckets: dict[str, BucketCounts] = {
        "manual": BucketCounts(),
        "downloaded": BucketCounts(),
    }
    for s in all_stats:
        b = buckets[s.source_bucket]
        b.docs += s.docs_out
        b.chars += s.chars_out
        b.words += s.words_out

    total_chars = sum(b.chars for b in buckets.values())
    total_words = sum(b.words for b in buckets.values())
    manual_char_share = buckets["manual"].chars / total_chars if total_chars else 0.0
    manual_word_share = buckets["manual"].words / total_words if total_words else 0.0

    return {
        "sources": [asdict(s) for s in all_stats],
        "buckets": {k: asdict(v) for k, v in buckets.items()},
        "totals": {
            "docs": sum(b.docs for b in buckets.values()),
            "chars": total_chars,
            "words": total_words,
        },
        # Reported on both bases because the manual *token* share can differ
        # slightly from either, and only the trained tokenizer settles it.
        "manual_share_chars": manual_char_share,
        "manual_share_words": manual_word_share,
    }


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------
# Explicit map from config source name -> raw files on disk. This is
# deliberately explicit rather than a glob over corpus/raw/: that directory
# also holds intermediate and superseded extracts (e.g. an older
# downloaded_sangraha_hin.docs.jsonl covering only shards 0-2, which the
# parquet shards now supersede). Globbing would silently double-count them.
SOURCE_FILES: dict[str, dict] = {
    "wiki":     {"bucket": "manual",     "glob": "wiki_ne.docs.jsonl",   "fmt": "jsonl"},
    "news":     {"bucket": "manual",     "glob": "news_*.docs.jsonl",    "fmt": "jsonl"},
    "sangraha": {"bucket": "downloaded", "glob": "downloaded/*.parquet", "fmt": "parquet"},
}


def _load_tree_config():
    """Load this tree's single config, using the same loader the scrapers use."""
    import importlib.util

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
    """Run the cleaning pipeline over every configured source."""
    import argparse

    cfg = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"Cleaning pipeline ({cfg.lang_name}).")
    ap.add_argument("--limit-per-source", type=int, default=None,
                    help="stop after N input docs per source (smoke testing).")
    ap.add_argument("--min-chars", type=int, default=FilterPolicy.min_chars)
    ap.add_argument("--min-words", type=int, default=FilterPolicy.min_words)
    ap.add_argument("--min-script-ratio", type=float, default=FilterPolicy.min_script_ratio)
    ap.add_argument("--max-symbol-ratio", type=float, default=FilterPolicy.max_symbol_ratio)
    ap.add_argument("--keep-other-language", action="store_true",
                    help="disable the language filter (diagnostic only).")
    ap.add_argument("--out-name", default="clean.jsonl")
    args = ap.parse_args(argv)

    policy = FilterPolicy(
        min_chars=args.min_chars,
        min_words=args.min_words,
        min_script_ratio=args.min_script_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        drop_other_language=not args.keep_other_language,
    )

    raw_dir = Path(cfg.paths.raw_dir)
    clean_dir = Path(cfg.paths.clean_dir)
    stats_dir = Path(cfg.paths.stats_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / args.out_name

    # One shared deduper across all sources so cross-source duplicates die too.
    deduper = ExactDeduper()
    all_stats: list[SourceStats] = []

    print(f"[{cfg.lang_name}] cleaning -> {out_path}")
    print(f"  policy: {policy}")

    with out_path.open("w", encoding="utf-8") as out_handle:
        for src in cfg.data.sources:
            spec_entry = SOURCE_FILES.get(src.name)
            if spec_entry is None:
                print(f"  ! no file mapping for source '{src.name}', skipping")
                continue
            if spec_entry["bucket"] != src.bucket:
                raise ValueError(
                    f"bucket mismatch for '{src.name}': config says {src.bucket}, "
                    f"SOURCE_FILES says {spec_entry['bucket']}"
                )

            files = sorted(raw_dir.glob(spec_entry["glob"]))
            if not files:
                print(f"  ! no files for source '{src.name}' ({spec_entry['glob']}), skipping")
                continue

            for path in files:
                # A multi-file source (news_*, parquet shards) is reported
                # per file so the stats table shows each site/shard separately.
                label = f"{src.name}:{path.stem}" if len(files) > 1 else src.name
                if spec_entry["fmt"] == "jsonl":
                    records = iter_jsonl(path)
                else:
                    records = iter_parquet(path)
                if args.limit_per_source is not None:
                    records = _take(records, args.limit_per_source)

                st = clean_source(records, label, src.bucket, policy, deduper, out_handle)
                all_stats.append(st)
                kept = f"{st.docs_out:,}/{st.docs_in:,}"
                print(f"  {label:44s} kept {kept:>19s}  "
                      f"({st.chars_out/1e6:8.2f}M chars)")

    summary = summarize(all_stats)
    summary["policy"] = asdict(policy)
    summary["lang_code"] = cfg.lang_code
    stats_path = stats_dir / "clean_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    t = summary["totals"]
    print(f"\n  TOTAL kept {t['docs']:,} docs  {t['chars']/1e6:.1f}M chars  {t['words']/1e6:.1f}M words")
    print(f"  manual share: {summary['manual_share_words']*100:.2f}% of words, "
          f"{summary['manual_share_chars']*100:.2f}% of chars")
    if summary["manual_share_words"] < 0.20:
        print("  !! manual share is BELOW the 20% floor -- trim the downloaded tier "
              "before building splits.", file=sys.stderr)
    print(f"  stats -> {stats_path}")


def _take(it: Iterator[dict], n: int) -> Iterator[dict]:
    """Yield at most n records (smoke-test helper)."""
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


if __name__ == "__main__":
    main()
