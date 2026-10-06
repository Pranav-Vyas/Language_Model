# Model H (Hindi) — qualitative notes on coherence and language correctness

Applies to the **final (tuned)** Model H: dropout 0.0, peak LR 3e-3, fp16.
Basis: 5 decoding settings × 191 held-out prompts in `generation.json`, the full
hypothesis dumps in `hypotheses_*.json`, and `generation_samples.md`. Statistics
are corpus-wide over all 191 generations per setting. The pre-tuning run's notes
are kept in [`../hindi_baseline/qualitative_notes.md`](../hindi_baseline/qualitative_notes.md)
for comparison.

**Scope.** Script correctness, repetition, sentence segmentation and prompt
copying are measured. Native-speaker fluency judgements are flagged where
relevant, not asserted.

## Language correctness: strong

| setting | Devanagari | Latin | danda /100w | digits | prompt-copy (4-gram) |
|---|---|---|---|---|---|
| greedy | 0.961 | 0.009 | 7.34 | 0.006 | 0.086 |
| temp 0.5 | 0.961 | 0.010 | 6.49 | 0.006 | 0.038 |
| temp 1.0 | 0.968 | 0.008 | 5.06 | 0.008 | 0.005 |
| temp 1.5 | 0.952 | 0.013 | 1.65 | 0.014 | 0.000 |
| top-p 0.95 | 0.961 | 0.010 | 5.37 | 0.008 | 0.004 |

The model never leaves its script: **95–97% of non-space characters are
Devanagari** at every setting, Latin under 1.3%. The Latin present is
legitimate — the corpus writes English technical terms in parentheses
(`(Insider Trading)`, `(बॉण्ड)`), and generations reproduce that convention
rather than drifting into English.

## What tuning changed

Repetition improved at every conservative setting: greedy 4-gram repetition
**0.617 → 0.578**, temp-0.5 **0.277 → 0.244**, and prompt-copy at greedy
**0.114 → 0.086**. Distinct-2 rose across the board (greedy 0.148 → 0.176).

The improvement is visible on individual samples. Given the same insider-trading
prompt, the pre-tuning model produced a run-on that lost its clause boundary:

> …का व्यवहार करते हुए कारोबार करने को कहते हैं. हालांकि, कंपनी के शेयर या अन्य प्रतिभूतियों…

while the tuned model closes the parenthesis it opened and keeps the syntax
intact before repeating:

> …(जैसे कि बॉण्ड या स्टॉक बेचने के सम्भावित विकल्पों) का व्यवहार करते हुए, कंपनी के शेयर…

Better calibration shows up as better *local* structure. It does not fix the
looping.

## Coherence: two distinct failure modes remain

**Greedy still loops**, at 57.8% of 4-grams. The repeated span is grammatical in
isolation — greedy produces locally excellent Hindi that is globally
degenerate. This is Holtzman et al.'s finding reproduced on our own model, and
tuning reduced but did not remove it.

**High temperature loses sentence structure.** Danda density falls
**7.34 → 1.65 per 100 words** between greedy and temperature 1.5, while
Distinct-2 peaks at 0.993. The diversity at 1.5 is bought by producing
unbounded run-on text; no diversity metric shows this, only the segmentation
statistic does.

**Semantic drift at temperature ≥1.0.** Syntax stays well-formed while meaning
does not. From an insider-trading prompt the temperature-1.0 continuation moves
into invented financial entities and malformed figures (`$00,000`), and at 1.5
the model produces plausible-looking but non-existent proper nouns.

## Factual grounding is out of scope at this scale

Generations contain confident false statements — invented place names,
fabricated etymologies, impossible figures. This is **not a tuning shortfall**.
A 24M-parameter model has no retrieval mechanism and no knowledge store; it
models which tokens follow which, and facts survive only where they are
overwhelmingly frequent in the corpus. There is nowhere near enough capacity to
hold, say, Indian geography. GPT-2 small is five times larger and still
confabulates freely.

## Assessment

Temperature **0.5 is the best operating point**: repetition 0.244, sentence
structure retained (danda 6.49), and the best BLEU-4 (3.53) and ROUGE-L (0.1252)
of any setting. Top-p 0.95 is the best choice when diversity matters more than
overlap — near-zero repetition (0.002) with chrF 21.20.

The model has learned Hindi **orthography, morphology and local syntax well**,
and **discourse-level coherence partially** — tuning measurably improved the
latter without changing the former. That split is the honest summary.
