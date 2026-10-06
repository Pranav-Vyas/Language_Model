"""Generation-quality metrics, and an argument about what they are worth (T2.13).

Duplicated verbatim into every language tree; never imports from a sibling.

Computation only. Nothing here draws a plot.

WHY ROUGE-L IS IMPLEMENTED HERE INSTEAD OF USING `rouge_score`
--------------------------------------------------------------
Measured, not assumed: `rouge_score.tokenize` lowercases and then splits on
`[^a-z0-9]+`, which deletes every Devanagari character. Scoring two IDENTICAL
Hindi strings with it returns **0.0**, because both tokenise to the empty list.

Used naively, every ROUGE-L number in this report would have been 0.0 and would
have looked like catastrophic model failure rather than a broken metric. So
ROUGE-L is computed here from an explicit LCS over whitespace-delimited tokens,
which is script-agnostic.

WHAT EACH METRIC IS AND IS NOT INFORMATIVE FOR
----------------------------------------------
The spec asks for BLEU, chrF/chrF++ and ROUGE-L *and* for an explanation of
whether each is informative. The honest answer is that all three are weak here,
for a reason that is structural rather than a property of our models:

  * They are **precision/recall against a single reference**. Open-ended
    continuation has an enormous space of valid outputs; a fluent, on-topic
    continuation that simply differs from the one the corpus happened to
    contain scores near zero. The metric cannot distinguish "wrong" from
    "different but fine".
  * **BLEU is the weakest of the three here.** It needs exact n-gram matches
    (we report BLEU-4), and matching 4 consecutive word forms by chance in a
    free continuation is vanishingly unlikely. Expect near-zero scores that
    carry almost no signal about quality.
  * **chrF/chrF++ are the most useful of the three for Devanagari**, because
    they match *character* n-grams. Hindi and Nepali are morphologically rich
    with heavy inflection and agglutinative postpositions, so two forms of the
    same lemma share most of their characters while being entirely different
    word tokens. chrF sees that partial credit; BLEU does not. chrF++ adds
    word unigrams/bigrams on top (`word_order=2`).
  * **ROUGE-L** (LCS-based) is recall-oriented and order-sensitive but not
    contiguity-requiring, so it sits between the two -- more forgiving than
    BLEU, less character-aware than chrF.

They are still reported because the spec requires them, and because they are
*comparable across our own decoding settings*: BLEU at temperature 0.5 versus
1.5 on the same prompts is a meaningful within-model contrast even when the
absolute value is uninformative.

The metrics that actually say something about open-ended generation here are
**repetition rate** and **Distinct-1/2**, which need no reference at all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict


# --- reference-free diversity -------------------------------------------

def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n (Li et al. 2016): unique n-grams / total n-grams, corpus-wide.

    Computed over the whole corpus of generations rather than averaged per
    sample: a per-sample average hides the case where every sample is
    internally varied but they are all the same as each other.
    """
    total, uniq = 0, set()
    for t in texts:
        g = ngrams(t.split(), n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / total if total else 0.0


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams in one text that are duplicates.

    `1 - unique/total`, so 0.0 means every n-gram is fresh and values near 1.0
    mean the text is looping. This is the measure the course notes use to
    contrast decoding strategies (greedy degenerates badly; nucleus does not).
    """
    g = ngrams(text.split(), n)
    if not g:
        return 0.0
    return 1.0 - len(set(g)) / len(g)


def longest_repeated_run(text: str) -> int:
    """Longest run of one token repeated consecutively -- the visible failure
    mode of greedy decoding, which a rate averaged over a document can mask."""
    toks = text.split()
    best = run = 1 if toks else 0
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, run)
    return best


# --- reference-based overlap --------------------------------------------

def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (not substring).

    O(len(a) * len(b)) time, O(min) space -- two rolling rows rather than the
    full table, since continuations can be a few hundred tokens.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, reference: str, beta: float = 1.2) -> dict:
    """LCS-based ROUGE-L over whitespace tokens. Script-agnostic by construction.

    beta=1.2 weights recall slightly above precision, matching the original
    ROUGE definition.
    """
    h, r = hypothesis.split(), reference.split()
    l = lcs_length(h, r)
    if l == 0:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    p, rc = l / len(h), l / len(r)
    f = ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)
    return {"precision": p, "recall": rc, "fmeasure": f}


# --- script / language-correctness diagnostics ---------------------------

DEVANAGARI = range(0x0900, 0x0980)


def script_stats(text: str) -> dict:
    """Measurable proxies for "is this the right language, written correctly?".

    The spec asks for qualitative notes on coherence and language correctness.
    Coherence needs a reader; these are the parts that can be MEASURED, so the
    written notes rest on numbers rather than on impressions of a few samples:

      devanagari_ratio  fraction of non-space characters in the Devanagari
                        block. Near 1.0 means the model stayed in script; a
                        drop means it is emitting Latin or symbol runs.
      latin_ratio       ASCII letters -- some is legitimate (both corpora
                        contain English technical terms), a lot is drift.
      danda_per_100w    sentence terminators per 100 words. Devanagari uses
                        danda; a model that never emits one is producing
                        unbounded run-ons.
      digit_ratio       digit characters, which degenerate models over-emit.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0,
                "danda_per_100w": 0.0, "digit_ratio": 0.0}
    dev = sum(1 for c in chars if ord(c) in DEVANAGARI)
    lat = sum(1 for c in chars if ("a" <= c.lower() <= "z"))
    dig = sum(1 for c in chars if c.isdigit())
    words = max(1, len(text.split()))
    return {
        "devanagari_ratio": dev / len(chars),
        "latin_ratio": lat / len(chars),
        "danda_per_100w": 100.0 * (text.count("।") + text.count("॥")) / words,
        "digit_ratio": dig / len(chars),
    }


def prompt_copy_rate(hypothesis: str, prompt: str, n: int = 4) -> float:
    """Fraction of the hypothesis's n-grams that already appear in the prompt.

    Greedy decoding often "continues" by paraphrasing the prompt back. That
    inflates BLEU against a reference drawn from the same document while being
    a failure of generation, so it is measured separately rather than left to
    inflate the overlap scores unnoticed.
    """
    h = ngrams(hypothesis.split(), n)
    if not h:
        return 0.0
    p = set(ngrams(prompt.split(), n))
    return sum(1 for g in h if g in p) / len(h)


@dataclass
class GenMetrics:
    """Scores for one decoding setting over a set of prompts."""

    setting: str
    n_samples: int
    bleu4: float
    bleu_tokenizer: str
    chrf: float
    chrf_pp: float
    rouge_l_f: float
    repetition_4gram: float
    longest_repeated_run: float
    distinct_1: float
    distinct_2: float
    mean_tokens: float
    devanagari_ratio: float = 0.0
    latin_ratio: float = 0.0
    danda_per_100w: float = 0.0
    digit_ratio: float = 0.0
    prompt_copy_4gram: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def score_generations(hyps: list[str], refs: list[str], setting: str,
                      tokenize: str = "intl",
                      prompts: list[str] | None = None) -> GenMetrics:
    """Corpus-level BLEU/chrF/chrF++ plus mean ROUGE-L and diversity.

    BLEU is corpus-level (the statistically meaningful form) at the default
    max n-gram order of 4 -- stated explicitly because "BLEU" without an order
    is ambiguous. `tokenize="intl"` handles Unicode punctuation; sacrebleu's
    default `13a` is built around Latin script.
    """
    from sacrebleu.metrics import BLEU, CHRF

    bleu = BLEU(tokenize=tokenize)          # max_ngram_order defaults to 4
    chrf = CHRF()                           # char n-grams, word_order=0
    chrf_pp = CHRF(word_order=2)            # chrF++

    b = bleu.corpus_score(hyps, [refs]).score
    c = chrf.corpus_score(hyps, [refs]).score
    cpp = chrf_pp.corpus_score(hyps, [refs]).score
    rl = sum(rouge_l(h, r)["fmeasure"] for h, r in zip(hyps, refs)) / max(1, len(hyps))

    return GenMetrics(
        setting=setting,
        n_samples=len(hyps),
        bleu4=b, bleu_tokenizer=tokenize,
        chrf=c, chrf_pp=cpp,
        rouge_l_f=rl,
        repetition_4gram=sum(repetition_rate(h) for h in hyps) / max(1, len(hyps)),
        longest_repeated_run=sum(longest_repeated_run(h) for h in hyps) / max(1, len(hyps)),
        distinct_1=distinct_n(hyps, 1),
        distinct_2=distinct_n(hyps, 2),
        mean_tokens=sum(len(h.split()) for h in hyps) / max(1, len(hyps)),
        devanagari_ratio=_mean(script_stats(h)["devanagari_ratio"] for h in hyps),
        latin_ratio=_mean(script_stats(h)["latin_ratio"] for h in hyps),
        danda_per_100w=_mean(script_stats(h)["danda_per_100w"] for h in hyps),
        digit_ratio=_mean(script_stats(h)["digit_ratio"] for h in hyps),
        prompt_copy_4gram=(_mean(prompt_copy_rate(h, p)
                                 for h, p in zip(hyps, prompts))
                           if prompts else 0.0),
    )


def _mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0
