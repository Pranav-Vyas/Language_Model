"""Measure how often each candidate template construction occurs in OUR corpus (T3.3).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY THIS EXISTS
---------------
The spec requires the finetuning text to be in "natural phrasing for that
language". That is easy to assert and hard to prove. Instead of asserting it,
every template this project ships is built from a construction that was first
counted in the model's OWN pretraining corpus, and the count is reported beside
the template. A construction the model saw tens of thousands of times during
pretraining is, by definition, natural for this corpus -- and it is also the
phrasing the model is best equipped to parse.

This converts an unverifiable claim into a measurement, which is the substitute
for a native-speaker audit.

WORD BOUNDARIES MATTER HERE
---------------------------
Naive substring counting is wrong in Devanagari. Scanning for "भन्दा" inside a longer word also
matches "से अधिकतर" ("mostly"), which is a different word and not a comparative
at all -- found in the first exploratory pass. Every pattern is therefore
matched with an explicit "not followed/preceded by another Devanagari letter"
guard.

    python nepali/reasoning/mine_constructions.py --docs 40000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
REPO = TREE.parent

# A Devanagari letter on either side means we matched inside a longer word.
DEV = r"ऀ-ॿ"
LEFT = rf"(?<![{DEV}])"
RIGHT = rf"(?![{DEV}])"

# Candidate adjective pairs: (masculine, feminine, gloss, polarity).
# Polarity +1 = "more of the attribute", -1 = "less of it". Recorded because the
# generator needs to know which entity the answer is, and getting a polarity
# backwards would silently produce a dataset whose labels are all inverted.
ADJECTIVES = [
    ("अग्लो",   "अग्ली",   "tall",       +1),
    ("होचो",    "होची",    "short",      -1),
    ("ठूलो",    "ठूली",    "big",        +1),
    ("सानो",    "सानी",    "small",      -1),
    ("भारी",    "भारी",    "heavy",      +1),
    ("हलुका",   "हलुकी",   "light",      -1),
    ("महँगो",   "महँगी",   "expensive",  +1),
    ("सस्तो",   "सस्ती",   "cheap",      -1),
    ("छिटो",    "छिटी",    "fast",       +1),
    ("ढिलो",    "ढिली",    "slow",       -1),
    ("पुरानो",  "पुरानी",  "old",        +1),
    ("नयाँ",    "नयाँ",    "new",        -1),
    ("धनी",     "धनी",     "rich",       +1),
    ("गरिब",    "गरिब",    "poor",       -1),
    ("बाक्लो",  "बाक्ली",  "thick",      +1),
    ("पातलो",   "पातली",   "thin",       -1),
    ("लामो",    "लामी",    "long",       +1),
    ("चौडा",    "चौडी",    "wide",       +1),
]

# Frame templates whose frequency we care about. {A} is the adjective slot.
FRAMES = {
    "comparative_se":    r"भन्दा\s+{A}",              # "X भन्दा अग्लो" — than-comparative
    "superlative_sabse": r"सबैभन्दा\s+{A}",           # "सबैभन्दा अग्लो" — superlative
    "ordinal_second":    r"दोस्रो\s+{A}",                # "दोस्रो ठूलो" — 2nd most
    "bare":              r"{A}",                     # the adjective on its own
}

# Question words and copulas the templates need, counted so the whole surface
# form is evidenced, not just the adjective.
FUNCTION_WORDS = ["को", "कुन", "हो", "छ", "छन्", "उत्तर", "सबैभन्दा", "भन्दा",
                  "बढी", "कम", "दोस्रो", "तेस्रो", "को तुलनामा"]


def count_patterns(texts, patterns: dict[str, str]) -> Counter:
    """Count each named regex over an iterable of documents. Pure function."""
    compiled = {name: re.compile(LEFT + pat + RIGHT) for name, pat in patterns.items()}
    out = Counter()
    for t in texts:
        for name, rx in compiled.items():
            n = len(rx.findall(t))
            if n:
                out[name] += n
    return out


def iter_docs(path: Path, limit: int):
    """Stream documents from a split file; the train split is gigabytes."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            try:
                yield json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Count template constructions in the corpus.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--docs", type=int, default=40_000,
                    help="documents to scan (streamed; the train split is GBs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    split_path = TREE / "corpus" / "splits" / f"{args.split}.jsonl"
    if not split_path.exists():
        raise SystemExit(f"corpus split not found: {split_path}")

    # Build every (frame, adjective) pattern plus the bare function words.
    patterns: dict[str, str] = {}
    for masc, fem, gloss, pol in ADJECTIVES:
        for form in {masc, fem}:
            for frame, tpl in FRAMES.items():
                patterns[f"{frame}|{form}"] = tpl.format(A=re.escape(form))
    for w in FUNCTION_WORDS:
        patterns[f"word|{w}"] = re.escape(w)

    print(f"[scan] {split_path.name}, up to {args.docs:,} docs, "
          f"{len(patterns):,} patterns", flush=True)
    counts = count_patterns(iter_docs(split_path, args.docs), patterns)

    # Reshape into per-adjective evidence.
    adj_rows = []
    for masc, fem, gloss, pol in ADJECTIVES:
        row = {"masculine": masc, "feminine": fem, "gloss": gloss, "polarity": pol}
        for frame in FRAMES:
            row[frame] = sum(counts.get(f"{frame}|{f}", 0) for f in {masc, fem})
        adj_rows.append(row)
    adj_rows.sort(key=lambda r: -r["comparative_se"])

    words = {w: counts.get(f"word|{w}", 0) for w in FUNCTION_WORDS}

    result = {"lang_code": "ne", "split": args.split, "docs_scanned": args.docs,
              "adjectives": adj_rows, "function_words": words}

    out = Path(args.out) if args.out else (
        REPO / "report" / "phase3" / "nepali" / "construction_frequencies.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'adjective':10s} {'gloss':14s} {'pol':>4s} {'X भन्दा ADJ':>12s} "
          f"{'सबैभन्दा ADJ':>12s} {'2nd-most':>9s} {'bare':>9s}")
    for r in adj_rows:
        print(f"{r['masculine']:10s} {r['gloss']:14s} {r['polarity']:>+4d} "
              f"{r['comparative_se']:>9,} {r['superlative_sabse']:>9,} "
              f"{r['ordinal_second']:>9,} {r['bare']:>9,}")
    print("\nfunction words: " + "  ".join(f"{w}={c:,}" for w, c in words.items()))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
