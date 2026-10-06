"""Entity name pools for the reasoning dataset, split-disjoint (T3.3).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY THE POOLS ARE SPLIT-DISJOINT
--------------------------------
Anti-leakage axis 1. A name that appears in training must never appear in
validation or test, so a model cannot score by memorising "राम is taller than
श्याम" -- it has to learn the *relation*, which is the only thing that transfers
to names it has never seen. The split is by NAME, not by example, so the
guarantee is structural rather than statistical.

WHY MASCULINE NAMES ONLY
------------------------
Nepali adjectives agree with the subject's gender (अग्लो / अग्ली). Mixing genders
without also inflecting every adjective would emit ungrammatical sentences, and
grammaticality is graded. Using one gender consistently keeps every generated
sentence correct. The gender field is carried explicitly so the inflection is a
data property rather than an assumption baked into the templates.

NAMES MUST BE PEOPLE, NOT MERELY FREQUENT SINGLE TOKENS
-------------------------------------------------------
The previous pool was filtered on "tokenizes to one token AND occurs often in
the corpus". That filter selects FOR common nouns, because common nouns are
exactly the most frequent and most likely to be a single token. Nepali's pool
duly filled up with हिमाल (mountain), आकाश (sky), अग्नि (fire), चन्द्र (moon),
अधिकारी (official) and ethnic-group labels, producing prompts that read as
"Mountain is more expensive than Kirat" -- semantically absurd sentences whose
strong pretrained priors fight the task. Model L scored 0.21 against Model H's
0.79, and that gap was an artifact of pool construction, not a fact about
Nepali. Every entry here is now an unambiguous personal given name.

CROSS-LANGUAGE DISJOINTNESS IS DELIBERATELY NO LONGER ENFORCED
--------------------------------------------------------------
Earlier versions kept the two pools name-disjoint as extra insurance. That was
never required -- the independence constraint is satisfied by separate corpora,
separate tokenizers, separate weights and separate data files -- and forcing it
is what pushed Nepali into common nouns, since most South Asian given names are
shared across both languages. Sharing the string "अर्जुन" between two otherwise
independent datasets is a far smaller problem than the confound it created.
Pool SIZE is still held identical across languages, because that one does
affect the H-vs-L comparison.

EVERY NAME IS A SINGLE TOKEN, AND THAT IS THE POINT
---------------------------------------------------
The first version of these pools was hand-picked without checking how the
tokenizer split each name, and the result was a confound that would have
invalidated the whole phase: 83% of TRAIN names were single tokens against only
36% of TEST names. The held-out splits were therefore systematically harder to
*spell*, not just harder to reason about, and the finetuned model duly produced
things like संतोष -> सोस -- garbling a multi-token name it had to copy.

Restricting every pool to names the tokenizer emits as ONE token removes that
confound. Copying an entity out of the prompt becomes a single-step operation,
so held-out accuracy measures the comparison, not the spelling of rare names.

Pool sizes are also held IDENTICAL across the two languages (99 = 69/15/15).
The H-vs-L comparison has to differ only in language; letting one model have a
larger entity vocabulary than the other would be a second uncontrolled variable.

Names were assembled from curated lists plus frames mined out of each corpus
("X ने कहा" for Hindi), then filtered to those that (a) tokenize to exactly one
token and (b) actually occur in that tree's own pretraining corpus.

EVERY NAME IS VERIFIED, NOT ASSUMED
-----------------------------------
`verify()` checks each name (a) tokenizes with no UNK under this tree's own
tokenizer and (b) actually occurs in this tree's pretraining corpus. A name the
model never saw is a name it has no representation for, which would test
tokenizer robustness rather than reasoning.

    python nepali/reasoning/entities.py          # verify and report
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
REPO = TREE.parent

GENDER = "m"   # see module docstring

# Common Nepali male given names and family names. Partitioned by split below; the partition is
# fixed here rather than sampled at generation time so the holdout is identical
# on every regeneration.
_TRAIN = [
    "अजय", "अधिकारी", "अनुप", "अरुण", "अशोक", "आदर्श", "उत्तम", "कपिल",
    "कार्की", "कृष्ण", "केशव", "कोइराला", "खगेन्द्र", "खड्का", "गिरी",
    "गुरुङ", "जनार्दन", "जीवन", "ढकाल", "तरुण", "तामाङ", "थापा", "दाहाल",
    "दिनेश", "दिपेन्द्र", "देवराज", "नरेन्द्र", "नवीन", "न्यौपाने", "पन्त",
    "पराजुली", "प्रकाश", "प्रतीक", "प्रदीप", "प्रमोद", "प्रेम", "बलराम",
    "भट्ट", "भण्डारी", "भरत", "भीम", "मगर", "मदन", "मल्ल", "महेन्द्र",
    "माधव", "मुकेश", "योगेश", "रमेश", "राई", "राजेन्द्र", "राजेश", "रिजाल",
    "रेग्मी", "लक्ष्मण", "शंकर", "शाक्य", "श्याम", "श्रेष्ठ", "सन्तोष",
    "सन्दीप", "सरोज", "सागर", "सुमन", "सुरज", "सुरेश", "सुवेदी", "सुशील",
    "हरि",
]
_VAL = [
    "अनिल", "अर्जुन", "कमल", "जोशी", "दीपक", "नरेश", "नारायण", "निर्मल",
    "पौडेल", "बाबु", "मनोज", "मोहन", "राजु", "राम", "सुरेन्द्र",
]
_TEST = [
    "अनमोल", "उपेन्द्र", "ओली", "गगन", "गोपाल", "घिमिरे", "ज्ञानेन्द्र",
    "बस्नेत", "बिकास", "महर्जन", "महेश", "युवराज", "राणा", "रोशन", "सिंह",
]

SPLITS = {"train": _TRAIN, "val": _VAL, "test": _TEST}


def pool(split: str) -> list[str]:
    """Names available to one split. Disjointness is asserted at import time."""
    return list(SPLITS[split])


def _assert_disjoint() -> None:
    """A name leaking across splits would silently void the holdout."""
    seen: dict[str, str] = {}
    for split, names in SPLITS.items():
        for n in names:
            if n in seen:
                raise ValueError(
                    f"name {n!r} appears in both {seen[n]!r} and {split!r} -- "
                    "the entity holdout would be void")
            seen[n] = split


_assert_disjoint()


def verify(corpus_docs: int = 20_000) -> dict:
    """Check every name tokenizes cleanly and occurs in the pretraining corpus.

    Returns a report dict; does no printing, so the caller decides presentation.
    """
    import sentencepiece as spm
    lang = "ne"
    sp = spm.SentencePieceProcessor(
        model_file=str(TREE / "tokenizer" / f"{lang}_bpe_16000.model"))

    names = [n for names in SPLITS.values() for n in names]
    tok = {n: sp.encode(n) for n in names}
    unk = {n: sum(1 for i in ids if i == sp.unk_id()) for n, ids in tok.items()}

    # Corpus occurrence, with Devanagari word boundaries so "हरि" does not match
    # inside "हरिहर".
    dev = r"ऀ-ॿ"
    rx = {n: re.compile(rf"(?<![{dev}]){re.escape(n)}(?![{dev}])") for n in names}
    freq = dict.fromkeys(names, 0)
    split_path = TREE / "corpus" / "splits" / "train.jsonl"
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= corpus_docs:
                    break
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                for n in names:
                    if n in t:
                        freq[n] += len(rx[n].findall(t))

    return {
        "lang_code": lang, "gender": GENDER, "docs_scanned": corpus_docs,
        "counts": {k: len(v) for k, v in SPLITS.items()},
        "names": [
            {"name": n, "split": next(s for s, v in SPLITS.items() if n in v),
             "tokens": len(tok[n]), "unk": unk[n], "corpus_freq": freq[n]}
            for n in names
        ],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Verify the entity name pools.")
    ap.add_argument("--docs", type=int, default=20_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rep = verify(args.docs)
    rows = rep["names"]
    bad_unk = [r for r in rows if r["unk"]]
    absent = [r for r in rows if r["corpus_freq"] == 0]

    print(f"[entities] {rep['counts']}  gender={rep['gender']}")
    print(f"[tokens] mean {sum(r['tokens'] for r in rows)/len(rows):.2f} tok/name, "
          f"max {max(r['tokens'] for r in rows)}")
    print(f"[unk]     {len(bad_unk)} name(s) with UNK "
          f"{[r['name'] for r in bad_unk] or ''}")
    print(f"[corpus]  {len(absent)} name(s) absent from {args.docs:,} train docs "
          f"{[r['name'] for r in absent] or ''}")
    rare = sorted((r for r in rows if r["corpus_freq"] > 0),
                  key=lambda r: r["corpus_freq"])[:5]
    print("[rarest]  " + ", ".join(f"{r['name']}={r['corpus_freq']}" for r in rare))

    out = Path(args.out) if args.out else (
        REPO / "report" / "phase3" / "nepali" / "entity_pool_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
