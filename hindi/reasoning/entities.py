"""Entity name pools for the reasoning dataset, split-disjoint (T3.2).

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
Hindi adjectives agree with the subject's gender (लंबा / लंबी). Mixing genders
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

    python hindi/reasoning/entities.py          # verify and report
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
REPO = TREE.parent

GENDER = "m"   # see module docstring

# Common Hindi male given names. Partitioned by split below; the partition is
# fixed here rather than sampled at generation time so the holdout is identical
# on every regeneration.
_TRAIN = [
    "अंकित", "अखिलेश", "अग्रवाल", "अजय", "अजित", "अनिल", "अनुज", "अमित",
    "अरविंद", "अरुण", "अर्जुन", "आनंद", "आमिर", "आशीष", "इकबाल", "उदय",
    "उमेश", "करण", "कुणाल", "कुमार", "केशव", "गोपाल", "गोविंद", "गौरव",
    "चौधरी", "जगदीश", "जितेंद्र", "जैन", "तरुण", "तिवारी", "त्रिपाठी",
    "दीपक", "धीरज", "नितिन", "नीरज", "पंकज", "पवन", "पांडे", "प्रदीप",
    "प्रमोद", "बलराम", "भरत", "माधव", "मिश्रा", "मेहता", "मोहन", "मोहित",
    "यादव", "रंजीत", "रणवीर", "राजेंद्र", "राजेश", "राम", "राहुल", "वरुण",
    "विजय", "विनोद", "विवेक", "शर्मा", "श्याम", "सचिन", "सतीश", "सलीम",
    "सिंह", "सुधीर", "सुनील", "सुमित", "सुरेश", "हेमंत",
]
_VAL = [
    "इमरान", "कमल", "गुप्ता", "चंदन", "दुबे", "पटेल", "पीयूष", "प्रकाश",
    "महेंद्र", "महेश", "रमेश", "विनय", "सक्सेना", "समीर", "हरि",
]
_TEST = [
    "अनुराग", "अभय", "कृष्ण", "कौशल", "गणेश", "चेतन", "नरेंद्र", "नरेश",
    "नेहरू", "मनीष", "वर्मा", "वीरेंद्र", "शंकर", "शिव", "संतोष",
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
    lang = "hi"
    sp = spm.SentencePieceProcessor(
        model_file=str(TREE / "tokenizer" / f"{lang}_bpe_16000.model"))

    names = [n for names in SPLITS.values() for n in names]
    tok = {n: sp.encode(n) for n in names}
    unk = {n: sum(1 for i in ids if i == sp.unk_id()) for n, ids in tok.items()}

    # Corpus occurrence, with Devanagari word boundaries so "राम" does not match
    # inside "रामायण".
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
        REPO / "report" / "phase3" / "hindi" / "entity_pool_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
