# Model L (Nepali, matched-token checkpoint) — qualitative notes

Basis: the 5 decoding settings × 197 held-out prompts in `generation.json`, the
full hypothesis dumps in `hypotheses_*.json`, and `generation_samples.md`.
Statistics are corpus-wide over all 197 generations per setting.

**Scope.** Script correctness, repetition, segmentation and template-following
are measured. Native-speaker fluency judgements are flagged, not asserted.

## Language correctness: slightly stronger than Model H

| setting | Devanagari | Latin | danda /100w | digits |
|---|---|---|---|---|
| greedy | 0.969 | 0.005 | 10.35 | 0.015 |
| temp 0.5 | 0.974 | 0.005 | 10.18 | 0.015 |
| temp 1.0 | 0.986 | 0.006 | 7.35 | 0.015 |
| temp 1.5 | 0.987 | 0.002 | 2.43 | 0.010 |
| top-p 0.95 | 0.984 | 0.006 | 6.89 | 0.015 |

Nepali stays in script slightly more tightly than Hindi (**0.969–0.987** vs
0.953–0.965) with less Latin intrusion, and — unlike Hindi — its Devanagari
ratio *rises* with temperature rather than falling. Its danda density is also
consistently higher (10.35 vs 8.41 at greedy), i.e. **shorter sentences**, which
matches a corpus whose manual tier is 85% news rather than encyclopedic prose.

The higher digit ratio (0.015 vs Hindi's 0.005) is explained by corpus content —
see below.

## The dominant finding: template memorisation

Nepali generations are strikingly often **census boilerplate**:

> …राष्ट्रिय जनगणना २०६८मा यहाँको जनसङ्ख्या १८९७ रहेको थियो जसमध्ये १२९७ पुरुष र १२९७ महिला रहेका थिए भने यहाँ १२९ घरधुरी रहेको थियो।

This is not a decoding artifact — it appears at **greedy and at temperature
1.0**, with the *numbers varying* while the frame stays fixed:

> …जनसङ्ख्या २०८५ रहेको थियो भने यहाँ ६७७ घरधुरी रहेको थियो। यसको साविकका (१) वडाहरू…

The model has learned a **template with slots**, not a fact. This is a
**corpus-composition finding**: Nepali Wikipedia contains thousands of
near-identical village stubs built from the 2068 census, and at 30,037 total
articles they are a large fraction of the encyclopedic tier. It explains the
elevated digit ratio, and it is the most likely single explanation for anything
that looks like fluent recall in this model.

It also produces a visible **factual inconsistency**: `१२९७ पुरुष र १२९७ महिला`
— identical counts for men and women, which the template's slot-filling does not
constrain. The model reproduces the *shape* of a fact without its arithmetic.

## Coherence

**Greedy loops**, at 58.6% 4-gram repetition — the same degenerate mode as
Hindi, though slightly less severe, and here it loops the census template rather
than a narrative clause.

**Prompt copying is much lower than Hindi's**: 3.9% at greedy versus 11.4%. The
model is less inclined to paraphrase the prompt back, partly because the
template attractor pulls it elsewhere.

**High temperature loses sentence structure** the same way: danda density falls
**10.35 → 2.43** between greedy and temperature 1.5, while Distinct-2 rises to
0.987. Same trade as Model H — diversity purchased by abandoning segmentation.

## Assessment

Model L has learned **Nepali orthography and morphology at least as well as
Model H learned Hindi's**, and its per-byte modelling is better (BPB 0.4314 vs
0.5205). But a meaningful share of its apparent competence is **template
reproduction from a repetitive encyclopedic tier**, which is a data-quality
limitation rather than a modelling one, and is the most important qualitative
difference between the two models.

Temperature **0.5 is again the best operating point** (repetition 0.235, full
sentence structure, best BLEU-4 and ROUGE-L of the sampled settings).
