# Model H (Hindi, PRE-TUNING baseline) — qualitative notes

> This describes the **pre-tuning** run (dropout 0.1, peak LR 6e-4, bf16),
> retained for comparison. The final Model H notes are in
> [`../hindi/qualitative_notes.md`](../hindi/qualitative_notes.md).

Basis: the 5 decoding settings × 191 held-out prompts in `generation.json`, the
full hypothesis dumps in `hypotheses_*.json`, and the samples in
`generation_samples.md`. Statistics quoted below are corpus-wide over all 191
generations per setting, not impressions of a few examples.

**Scope of this assessment.** Script correctness, repetition, sentence
segmentation and template-following are measured or directly observable and are
reported as findings. Judgements of *native-speaker fluency* — whether a
well-formed sentence reads naturally to a Hindi speaker — are flagged where
relevant but not asserted; that check should be added by a speaker.

## Language correctness: strong

| setting | Devanagari | Latin | danda /100w | digits |
|---|---|---|---|---|
| greedy | 0.962 | 0.008 | 8.41 | 0.005 |
| temp 0.5 | 0.965 | 0.008 | 8.43 | 0.004 |
| temp 1.0 | 0.963 | 0.009 | 5.27 | 0.008 |
| temp 1.5 | 0.953 | 0.013 | 1.99 | 0.015 |
| top-p 0.95 | 0.960 | 0.016 | 5.63 | 0.008 |

The model never leaves its script: **95–97% of non-space characters are
Devanagari** at every setting, with Latin under 2%. The Latin that does appear
is legitimate — the corpus contains English technical terms in parentheses
(e.g. `(Insider Trading)`), and generations reproduce that convention rather
than drifting into English. Devanagari numerals (`६०`, `२५`) are used correctly
alongside ASCII digits, matching corpus usage.

**Morphology is well-formed but not flawless.** At temperature 1.5 the sample
`ने औरन्हें` shows an agglutination error — a postposition fused to a pronoun in
a way Hindi does not license. These appear only at high temperature.

## Coherence: degrades in two different ways

**Greedy loops verbatim.** 61.7% of 4-grams are repeats — the highest of any
setting — and the failure is visible at clause level:

> …उन्होंने अपने पहले ही प्रयास में एक महत्वपूर्ण भूमिका निभाई, जो कि एक महान युद्ध के दौरान हुआ था। उन्होंने अपने पहले ही प्रयास में एक महत्वपूर्ण भूमिका निभाई, जो कि एक महान युद्ध के दौरान हुआ था।

The repeated clause is grammatical and fluent in isolation. Greedy decoding
produces *locally* excellent Hindi that is globally degenerate — exactly the
behaviour Holtzman et al. describe, reproduced here at 61.7% against ~0% for
every sampled setting.

Greedy also has the highest **prompt-copy rate (11.4% of 4-grams already appear
in the prompt)**, versus 0.4% at temperature 1.0. It partly "continues" by
paraphrasing the prompt back — which inflates its overlap scores against a
reference drawn from the same document while being a generation failure.

**High temperature loses sentence structure.** Danda density falls from
**8.41 to 1.99 per 100 words** between greedy and temperature 1.5. Temperature
1.5 has the best Distinct-2 (0.983) and zero repetition, but it buys that
diversity by producing unbounded run-on text with few sentence boundaries. No
single diversity metric shows this; it is only visible in the segmentation
statistic.

**Semantic drift at temperature ≥1.0.** Syntax stays well-formed while meaning
does not. Given a prompt about insider trading, the temperature-1.0 continuation
moves through "parallel debt policy", "soya credit card" and "Sutlej" — real
Hindi words in grammatical sentences with no coherent topic. At 1.5 the model
invents plausible-looking but non-existent proper nouns (`मेरिल्वौसिड`,
`केरेसिमन्`).

## Assessment

Temperature **0.5 is the best operating point** for this model: repetition drops
from 0.617 to 0.277, sentence structure is fully retained (danda 8.43), and it
scores the best BLEU-4 and ROUGE-L. Its remaining 27.7% repetition still shows
the model has not learned to sustain a topic — expected for 24M parameters
trained on a single epoch of 572M tokens.

The model has learned Hindi **orthography, morphology and local syntax well**,
and **discourse-level coherence poorly**. That split is the honest summary.
