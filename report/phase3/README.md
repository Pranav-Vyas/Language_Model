# Phase 3 — Reasoning Finetuning, Attention Analysis, and Final Report

Model H = Hindi, Model L = Nepali. Two independent decoder-only Transformers
(24,070,656 parameters each), finetuned from their own Phase-2 checkpoints with
their own frozen tokenizers, on synthetic comparative/transitive reasoning data
generated separately for each language.

## Results

The assignment names three reasoning task types. Each is a pattern in the dataset,
and each is reported separately below — pretrained vs finetuned, against its own
chance line, on the 6,000-example held-out test split (three seeds, mean).

Scoring is **closed-set** (rank the candidate entities by next-token log-probability)
because that is the only way to give the *pretrained* model a fair test: it produces
the answer format 0% of the time, so its free-generation exact match is 0.000 by
construction and would say nothing about what it knows.

**The three task types the assignment asks for**

| assignment task type | chance | Model H pretrained → finetuned | Model L pretrained → finetuned |
|---|---|---|---|
| Comparison over **two** entities — *"who is taller?"* | 0.500 | 0.494 → **0.979** | 0.501 → **0.876** |
| **Multi-hop chaining** — *A > B, B > C ⇒ A vs C*, the queried pair never stated | 0.500 | 0.517 → **0.963** | 0.518 → **0.892** |
| **Superlative over three** — *"which is smallest / largest?"* | 0.333 | 0.260 → **0.957** | 0.198 → **0.868** |

The attributes compared are **height, age, price, size, weight, strength and
thickness**, plus **speed and wealth** held out of training entirely — covering the
"heights, ages, prices, or quantities" the assignment lists.

**Two harder types we added, beyond what was asked**

| extra task type | chance | Model H pretrained → finetuned | Model L pretrained → finetuned |
|---|---|---|---|
| Ordinal over **four** entities — *"second tallest"* | 0.250 | 0.379 → **0.918** | 0.369 → **0.824** |
| **4-hop chain over five entities — never trained** | 0.200 | 0.090 → 0.024 | 0.097 → **0.152** |

**Overall, and by free generation**

| test split, 6,000 examples | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| pretrained, closed-set | 0.399 | 0.378 |
| **finetuned, exact match** (greedy, must produce the answer verbatim) | **0.954 ±0.004** | **0.835 ±0.004** |
| finetuned, closed-set | 0.956 ±0.003 | 0.869 ±0.009 |
| finetuned, well-formed output | 0.996 | 0.957 |
| *chance · first-mention · last-mention* | *0.398 · 0.314 · 0.384* | *0.398 · 0.314 · 0.388* |

### What the numbers say

1. **Every task type the assignment asks for goes from chance to 0.87–0.98.** Both
   models. The capability is genuinely taught by finetuning, not unlocked — the
   pretrained control is at its chance line on the two-entity and chaining tasks
   and *below* it on the superlative.
2. **Multi-hop chaining is not harder than direct comparison for either model**
   (0.963 vs 0.979 for H; 0.892 vs 0.876 for L). The chaining step itself is cheap;
   what costs accuracy is the number of candidates to choose between.
3. **Neither model generalises one hop beyond training.** On the never-trained 4-hop
   chain both fall below chance, and the failure is systematic rather than random —
   Model H names a *middle* entity of the ordering 98.3% of the time when the answer
   is always an extreme (§7.2).
4. **Model L is the weaker reasoner but the better language model.** It collected
   more tokens and has better bits-per-byte, so the ~10-point gap is not a
   data-volume story; it concentrates in transfer to unseen relation words
   (0.518 vs 0.848) and tracks a 2× deficit in in-context copying (§9).
5. **Both models build the same mechanism.** At the token that emits the answer,
   attention on the correct entity rises well above baseline in both, and the
   strongest head is **L9H4 in each** — two models sharing no weights, tokenizer or
   data (§8).

> Two headline numbers here are **corrections** of earlier wrong ones: a left-padding
> bug in batched decoding (§5) and a Nepali learning rate that had been chosen on
> its output (§6.1). Superseded results are kept, not deleted.

## Where everything is

Every detailed table is generated from the result JSONs, so no number is typed by
hand twice.

| file | contents |
|---|---|
| [dataset_stats.md](dataset_stats.md) | dataset size, task-type mix, tokenisation, heuristic baselines, leakage checks |
| [error_analysis.md](error_analysis.md) | per-task-type, per-relation, per-distance and per-polarity breakdowns; OOD failure mode; qualitative samples |
| [attention_analysis.md](attention_analysis.md) | pretrained-vs-finetuned attention on both probe sets |
| `<tree>/` | per-model result JSONs and figures |
| `sft_*/` | training histories: the reported run, the curriculum arm (§7.2), the LR re-selection (§6.1), the A/B re-verification (§6.2) |
| `induction_probe.json` | in-context copying — the mechanism behind the residual H-vs-L gap (§9.3) |

Superseded results are kept rather than overwritten — `*_v1*` from before the
decoder fix (§5) and `nepali/*_lr3e-06*` from before the LR fix (§6.1) — because a
correction is only checkable against what it replaced. See
[V1_BASELINE_NOTE.md](V1_BASELINE_NOTE.md); none of the v1 accuracies are valid.

---

## 1. What the task is, and why this one

The spec asks for *comparative and transitive* reasoning. Every example is built
around a hidden total ordering over a set of people on some attribute — height,
age, price, weight, and so on. The prompt states some pairwise facts; the question
can only be answered by composing them. The five task types and how many of each
are in §2.

The task *family* was chosen deliberately rather than by convenience. Wang et al.
(arXiv 2405.15071) found that transformers generalise systematically on
**comparison**, while **composition** stayed at 0% out-of-distribution even after
22M optimisation steps. A 24M-parameter model is small, so choosing the family that
provably works means a negative result here is informative about *this* model
rather than a restatement of a known limitation.

## 2. The finetuning dataset

Built by `<lang>/reasoning/generate.py`, which is deterministic in (split, n, seed),
so the data is regenerated rather than committed. Independently per language: the
two generators share no code path, and the templates were mined from each corpus
rather than translated from one another.

### Splits, and what each one holds out

| split | examples | entity names | relations | hops | what it withholds |
|---|---|---|---|---|---|
| train | 60,000 | 69 | 7 | 1–3 | — |
| val | 6,000 | 15 | 7 | 1–3 | entity names never seen in training |
| test | 6,000 | 15 | 9 | 1–3 | held-out names **and** 2 relations (*speed*, *wealth*) never trained |
| ood | 2,000 | 15 | 9 | **4** | a chain one hop longer than anything trained |

The three withholding axes are independent, so a model cannot pass by memorising
names, by memorising a relation word, or by memorising a chain length (§3).

Concretely: **training** uses 69 entity names across 7 relations — age, height,
price, size, strength, thickness, weight — at 1–3 hops. **Test** draws from 15
entity names with **zero overlap** with the 69 training names (verified directly
against the generated files, not assumed), and adds two relations that never
appear anywhere in training — *speed* and *wealth* — while staying at the same
1–3 hops, so length is not a variable there. **OOD** changes exactly one axis:
it is a different pattern (P6) built as a 4-hop chain, one hop deeper than
anything trained, so it is a length-generalisation probe rather than another
held-out sample of the training distribution.

### Composition by task type

| task type | pattern | entities | hops | train | val | test |
|---|---|---|---|---|---|---|
| comparison over two entities | P1 | 2 | 1 | 11,993 (20.0%) | 1,253 | 1,213 |
| multi-hop chaining | P2 | 3 | 2 | 16,745 (27.9%) | 1,688 | 1,658 |
| superlative over three | P3 | 3 | 2 | 21,070 (35.1%) | 2,087 | 2,060 |
| ordinal over four | P4 | 4 | 3 | 10,192 (17.0%) | 972 | 1,069 |
| 4-hop chain *(never trained)* | P6 | 5 | 4 | — | — | 2,000 (ood) |

**inferred : atomic = 4 : 1.** Only P1 is atomic — both facts are stated and the
answer needs no composition. The other three all require combining premises. The
mix is deliberate and is the knob that matters: on comparison tasks the *ratio* of
inferred to atomic examples drives how fast generalisation appears, more than the
absolute dataset size does.

### What one record contains

Each line of the JSONL is one example plus the metadata the evaluation needs to
slice it. A real training record from each language — same index, generated
independently:

```json
{"pattern": "P2", "relation": "weight", "polarity": "neg",
 "n_entities": 3, "n_hops": 2, "rank_distance": 2,
 "prompt": "राजेंद्र शर्मा से हल्का है। शर्मा सचिन से हल्का है। सचिन और राजेंद्र में हल्का कौन है?",
 "answer": "राजेंद्र",
 "target_answer_only": " उत्तर: राजेंद्र",
 "target_with_trace": " राजेंद्र शर्मा से हल्का है, और शर्मा सचिन से हल्का है, इसलिए राजेंद्र सचिन से हल्का है। उत्तर: राजेंद्र",
 "candidates": ["राजेंद्र", "सचिन"], "chain": ["सचिन", "शर्मा", "राजेंद्र"],
 "first_mention": "राजेंद्र", "last_mention": "राजेंद्र", "held_out_relation": false}
```

The Nepali record at the same index, same fields, generated independently:

```json
"prompt": "राजेन्द्र श्रेष्ठ भन्दा हलुका छ। श्रेष्ठ सन्दीप भन्दा हलुका छ। सन्दीप र राजेन्द्र मध्ये को हलुका छ?",
"answer": "राजेन्द्र",  "candidates": ["राजेन्द्र", "सन्दीप"]
```

The two are the same *pattern*, not the same sentence: different surnames, and
Nepali's `भन्दा … छ` / `मध्ये को` against Hindi's `से … है` / `में … कौन`. Neither
was translated from the other.

| field | why it exists |
|---|---|
| `prompt`, `answer` | the example itself |
| `target_answer_only`, `target_with_trace` | the two renderings the CoT A/B compares (§6.2) |
| `candidates` | the closed set the pretrained model is scored against |
| `chain` | the true ordering — used to classify *where* an OOD failure lands (§7.2) |
| `pattern`, `relation`, `polarity`, `rank_distance`, `held_out_relation` | the slices in the error analysis |
| `first_mention`, `last_mention` | the surface-heuristic baselines |

### What the model is trained on, and what it is scored on

The model sees `prompt + target` as one sequence and is **scored only on the answer
span** — prompt positions are set to `ignore_index=-100`, so the loss rewards
answering, not reciting the prompt back:

```
राजेंद्र शर्मा से हल्का है। … हल्का कौन है?   उत्तर: राजेंद्र
└────────────── masked, no loss ──────────────┘└─ scored ─┘
```

Measured over the 60,000 Hindi training examples: mean 23.3 tokens per example,
longest 35, **0 examples truncated**, and **17.2% of tokens carry loss**. Zero UNK
tokens in either language — an out-of-vocabulary token would make the example test
tokenizer robustness instead of reasoning. Finetuning context is therefore 128,
comfortably above the longest sequence, down from the pretraining 512.

### Why the format uses only in-language punctuation

The tokenizer is frozen after Phase 1, so the format may use only tokens it already
has — a new special token is not an option. **Newlines do not survive either
tokenizer:** `"क\nख"` encodes to `['▁क','▁ख']` in both, so a
`premises\nquestion\nanswer` layout would have carried no structural signal at
all, silently. Delimiters that do survive as single native tokens are `।` `?` `:`
`,`, which is what the format is built from:

```
<premises, each ending in ।> <question ending in ?> उत्तर: <entity>
```

### Task contract: what the reasoning score measures

The Phase-3 score measures a specific **task contract**, not general-purpose
reasoning. The model receives a Hindi or Nepali comparative prompt containing
pairwise ordering facts and must return an **entity name** in the form
`उत्तर: <entity>`. For example, given `राम श्याम से लंबा है। श्याम मोहन से लंबा
है। सबसे लंबा कौन है?`, the expected answer is `उत्तर: राम`. It is not asked to
output the relation word `लंबा`, a numerical value, or a full explanation.
Therefore, the high scores show that finetuning taught comparative, transitive,
superlative, and ordinal ordering within this prompt-and-answer contract. For
example, `राम और श्याम में कौन लंबा है?` asks for an entity and fits the
contract, while `राम, श्याम से कितना लंबा है?` asks for a numeric difference and
is a different task.

The regular test split evaluates held-out names and relation words at the
trained 1--3-hop depths. The separate **OOD split** evaluates length
generalisation with a 4-hop chain, one hop deeper than the training data, and
its results are reported separately from the main test score. Examples that
require another answer type or a substantially different format are treated as
contract-shift stress tests, not as failures of the original Phase-3 task.

## 3. Anti-leakage: three structural holdouts, not an n-gram filter

The spec asks for a documented anti-leakage strategy. N-gram decontamination is
necessary but not sufficient — it cannot detect paraphrase — so the guarantees
here are structural, and each closes a specific way of scoring well without
reasoning:

1. **Entity holdout.** Train, val and test draw from disjoint name pools
   (69 / 15 / 15 of 99). A model cannot memorise "राम is taller than श्याम"; it
   has to learn the relation, which is the only thing that transfers.
2. **Relation holdout.** Two of the nine relations appear *only* at test time.
   A model that memorised "taller → pick X" cannot transfer to a relation word
   it has never seen. This is the strongest of the three.
3. **Hop holdout.** Training tops out at 3 hops; the OOD split is 4 hops. This
   is a length-generalisation probe, not an in-distribution test.

Verified by `<lang>/reasoning/verify_leakage.py`; results in
[dataset_stats.md](dataset_stats.md).

The two entity pools are **not** name-disjoint across languages, and are not
required to be — the independence constraint is satisfied by separate corpora,
tokenizers, weights and data files. Forcing cross-language disjointness is what
previously pushed the Nepali pool into common nouns (see §5). Pool *size* is
held identical, because that would affect the H-vs-L comparison.

## 4. Surface shortcuts are measured, not assumed away

A high accuracy is uninterpretable on its own: it is equally consistent with the
model having learned the template plus a positional habit. Three defences:

- **Three reported numbers, not one.** Exact match (greedy), closed-set scoring
  (rank the candidates by next-token log-probability), and heuristic baselines
  (first-mention, last-mention, majority). Closed-set scoring is what gives the
  *pretrained* model a fair test — free generation scores ~0 EM regardless of
  what it knows, whereas closed-set chance is exactly 1/k.
- **`scripts/shortcut_audit.py`** scores seven trivial rules per pattern and
  reports the best one as the honest comparator.
- **Premise order shuffled, polarity balanced, adjective randomly surfaced as
  itself or its antonym**, so first/last-mention sit at or below chance.

This machinery earned its keep. It caught a **degenerate P4**: with three
entities the "second most" and "second least" are the same entity, so the rule
"name the entity that appears twice" scored **1.000**. P4 was widened to four
entities, which drops that rule to 0.501.

## 5. Two corrections, and what they invalidated

Recorded because several numbers in this report replace earlier ones, and a
correction is only checkable against what it replaced.

**Left-padding corrupted every generated answer.** The batched decoder left-padded
so one shared "last position" logit slice would serve the batch. That is invalid
here: these models use *learned absolute* positional embeddings and `forward()`
takes no attention mask, so left-padding shifts every real token to a position it
was never trained at. It never crashed — it showed up as teacher-forced P(gold)
= 1.0000 against greedy 0.517, and as accuracy depending on **batch size** (1.000
at batch 1, 0.517 at batch 60). Fixed with right-padding plus a per-row gather of
each sequence's own last index, in one shared `reasoning_metrics.py` that
`finetune.py` also imports, so the training curve and the final table cannot
diverge. `test_batched_decoding_matches_unbatched` locks it.

This invalidated **every exact-match and closed-set number produced before it**,
including the A/B decisions made on them (§6). Everything was re-run.

**The Nepali entity pool selected for common nouns.** Filtering names on "one token
AND frequent" selects *for* common nouns, so the pool filled with हिमाल (mountain)
and आकाश (sky), producing absurd prompts whose pretrained priors fought the task.
Model L scored 0.21 against Model H's 0.79 — an artifact of pool construction, not
a fact about Nepali. A second confound rode along: 83% of train names were single
tokens against 36% of test names, so held-out splits were harder to *spell*, not
just to reason about. Every pool entry is now a person name the tokenizer emits as
exactly one token.

Superseded results are kept — `<tree>/reasoning_eval_v1.json`, `sft_baseline_v1/`,
`nepali/*_lr3e-06.json` — with [V1_BASELINE_NOTE.md](V1_BASELINE_NOTE.md) warning
that none of the v1 accuracies are valid.

## 6. Finetuning setup

| | value | why |
|---|---|---|
| init | each language's own Phase-2 checkpoint | Hindi `step_8732` (val 3.3785, the LR/dropout-tuned run); Nepali `step_13547` (val 3.7040) |
| tokenizer | frozen, each tree's own 16k BPE | spec requirement; also why the format uses only native punctuation |
| objective | next-token CE, **loss masked to the answer span** | prompt positions get `ignore_index=-100`, so the model is scored on answering, not on reciting the prompt back |
| target | answer only (`उत्तर: X`), not a reasoning trace | decided by A/B, see below |
| epochs | 15 | training loss reaches 0.0000 by step ~1,900 of 3,525, so the model has fully fitted the training distribution well before the end — more epochs buy nothing. (An earlier probe also put the held-out peak at ~9 epochs, but that was scored pre-fix, so the loss curve is the load-bearing evidence here.) |
| batch | 256 | 3,525 steps total |
| LR | **3e-05 for both languages** | see §6.1 |
| seeds | 1337 / 42 / 7 | every reported number is mean ± half-range over the three |
| context | 512 → 128 | longest prompt is 31 tokens |

### 6.1 The learning rate had to be re-selected

The original sweep chose **3e-05 for Hindi but 3e-06 for Nepali**, on a .190-vs-.133
margin — measured with the broken decoder (§5), and in breach of the project's own
rule to hold the finetune LR equal across languages so the H-vs-L comparison is not
confounded by optimisation.

Two signals independent of that bug said 3e-06 was **under-training** Model L: its
SFT loss plateaued at 0.130 with gradient norm still 3–9 while the schedule decayed
to 3e-07 (Hindi reaches loss 0.0000, grad-norm 0.00), and its finetuned attention
showed almost no answer-retrieval circuit.

Re-swept on the corrected harness, 3 LRs × 3 seeds, selecting on **validation only**:

| Nepali finetune LR | val EM | test EM | OOD EM |
|---|---|---|---|
| 3e-06 (original) | 0.632 ±0.002 | 0.489 ±0.010 | 0.022 ±0.001 |
| 1e-05 | 0.867 ±0.054 | 0.765 ±0.072 | 0.152 ±0.037 |
| **3e-05** (selected) | **0.929 ±0.030** | **0.841 ±0.017** | 0.125 ±0.044 |

*(In-training eval, n = 300/split, all three rows on the same harness including a
re-run of 3e-06 as the control. Re-scored offline on the full 6,000-row test split,
3e-06 gives 0.525 ±0.003 and 3e-05 gives 0.835 ±0.004.)*

3e-05 wins on validation and is Hindi's value, so the corrected run also **restores
the equal-LR condition**. Model L goes 0.525 → 0.835 test EM and 0.015 → 0.140 OOD,
closing ~72% of what had looked like a 43-point language gap. Most of the apparent
H-vs-L reasoning difference was an optimisation artifact.

### 6.2 The CoT and masking A/Bs, re-verified

Hindi, LR 3e-05, 2 seeds each, evaluated with the corrected harness
(mean ± half-range; OOD chance is 0.200):

| arm | val EM | test EM | **4-hop OOD EM** | well-formed (test) |
|---|---|---|---|---|
| answer-only + masked (**production**) | **0.983 ±0.010** | **0.960 ±0.010** | 0.020 ±0.003 | 0.999 |
| CoT trace + masked | 0.914 ±0.007 | 0.865 ±0.002 | **0.097 ±0.014** | 0.977 |
| answer-only, unmasked | 0.467 ±0.020 | 0.485 ±0.002 | 0.020 ±0.003 | 0.870 |

**Masking.** Direction confirmed, magnitude wrong by an order of magnitude: the
real gap is 0.983 vs 0.467 (2.1×), not 22×. The unmasked model is a competent
reasoner, not the near-total failure the corrupted numbers implied. The reported
0.033 was the padding bug, not the recipe.

**Chain-of-thought.** Direction confirmed *in domain*, magnitude badly wrong, and
the original framing — "traces are useless at 24M" — does not survive. CoT
reaches 0.914 val against answer-only's 0.983: a 6.9-point cost, not a 3× rout.

**And the in-domain number was hiding the opposite conclusion out of domain.**
On the 4-hop split, CoT scores **0.097 against answer-only's 0.020 — roughly 5×
better**, and it is the only intervention in this phase that moved that number
at all (the curriculum arm did not, though that arm predates the fix; §7.2).
That is a genuine trade:

> answer-only buys ~9.5 points of in-domain exact match; the reasoning trace
> buys a 5× improvement in generalising one hop beyond anything trained.

Neither arm reaches the 0.200 OOD chance line, so this is a large relative
improvement on a task both models still fail. But it points at the mechanism:
§7.2 showed the answer-only model fails OOD by naming a middle entity 98.3% of
the time — a positional heuristic. Forcing it to emit the intermediate
comparisons appears to interfere with that shortcut, which is exactly what a
trace is supposed to do.

**This is left as a documented trade-off rather than a silent config change.**
The headline configuration remains answer-only, because that is what the
committed results, checkpoints and attention analysis were produced with. If
length generalisation is the objective rather than in-domain accuracy, the CoT
arm is the better recipe and the numbers above are the evidence.

## 7. Error analysis

The headline numbers are in **Results** above. This section is where they break
down, and where the two models differ. Full per-seed tables in
[error_analysis.md](error_analysis.md).

### 7.1 Which slices are hard

The breakdowns are more informative than the headline. Test split, exact match:

| | Model H | Model L | what it tests |
|---|---|---|---|
| relation seen in training | 0.983 ±0.006 | 0.923 ±0.007 | in-distribution |
| **relation held out** | **0.848 ±0.001** | **0.518 ±0.015** | transfer to a new relation word |
| P1 · direct comparison | 0.977 | 0.868 | atomic |
| P2 · 2-hop transitive | 0.956 | 0.857 | inferred |
| P3 · superlative of 3 | 0.956 | 0.835 | inferred |
| P4 · ordinal of 4 | 0.917 | 0.766 | inferred, hardest |
| rank distance 1 → 2 | 0.949 → 0.956 | 0.820 → 0.845 | symbolic distance |
| polarity neg / pos | 0.941 / 0.966 | 0.836 / 0.835 | wording sensitivity |

- **Held-out relations are the real generalisation test, and the two models
  differ most there.** Model H loses 13.5 points on a relation word it has never
  seen; Model L loses 40.5. Both still beat the 0.398 chance line, so both
  transfer *something* — but Model L transfers much less. This is where the
  residual H-vs-L gap concentrates (§9.4).
- **Difficulty is ordered as expected** for both models, P1 easiest through P4
  hardest, which is what a model doing the task should look like.
- **The symbolic-distance effect appears in the predicted direction for both.**
  The ordinal-geometry literature predicts accuracy rises with the rank distance
  between the compared entities; it does, by 0.7 points (H) and 2.5 points (L).
  A weak confirmation, reported as weak.
- **Polarity costs Model H 2.5 points and Model L nothing.** Negative
  comparatives ("who is shortest") are slightly harder for the stronger model
  and indistinguishable for the weaker one.

### 7.2 Length generalisation fails, and the failure has a shape

On the 4-hop OOD split — one hop deeper than anything trained — Model H scores
**0.023**, far *below* the 0.200 chance line, while still producing well-formed
answers 98.8% of the time. A model that had simply run out of capability would
score near chance; scoring below it means the errors are systematic.

They are. Joining the sampled OOD predictions back to each row's chain shows
**118 of 120 name a MIDDLE entity of the ordering, against 60.0% expected if it
picked uniformly among the chain's entities**. The OOD answer is always an
*extreme* (first or last), so naming a middle is not a near miss:

> the model learned "answer with an entity from the chain, chosen by position",
> not "extend the ordering by one more hop".

This is a clean negative result and it is reported as one.

A **curriculum arm** — adding a 3-hop *chain* question (P5) to training alongside
the 3-hop *ordinal* one — was also run, and did not fix it: the OOD margin moved
from +0.052 to +0.060, inside seed noise. Two caveats keep that from being strong
evidence, and both are stated rather than glossed: it was run on the **v1**
dataset, before P4 was widened and before the OOD probe moved from 3 hops to 4,
and it was scored with the **buggy decoder** (§5). It was not re-run, so it is
reported as suggestive, not as a measurement on equal footing with the rest of
this section. The one intervention that *was* re-measured on the corrected
harness and did move the OOD number is chain-of-thought (§6.2).

## 8. Attention: what finetuning actually changed

Full tables in [attention_analysis.md](attention_analysis.md). The Phase-2
toolkit is imported unchanged, and the comparison runs on **two probe sets**,
which is what makes it interpretable:

- `corpus` — held-out natural text, built with the exact Phase-2 call. Asks:
  *did finetuning damage the language model?*
- `reasoning` — held-out reasoning prompts. Asks: *what did finetuning build?*

**Verification first.** The `corpus` pretrained numbers reproduce the committed
`report/phase2/<tree>/attention.json` head for head, max |Δ| = 1.1e-05 over all
60 heads per language. So the pretrained baseline is provably the Phase-2 model
and every delta is attributable to the finetune.

**Finding 1 — the restructuring is task-specific, not global.** Entropy falls
3.0× more on reasoning prompts than on natural text for Hindi (−0.0984 vs
−0.0324). Content dependence rises on the reasoning probes while the corpus
probes barely move. Finetuning sharpened the attention the task uses and largely
left the language model's attention alone.

**Finding 2 — both models build a late-layer answer-retrieval circuit.**
Measured at the position whose logits actually emit the entity (the `:` of
`उत्तर: `, *not* the last prompt token, which emits the answer-marker word two
steps earlier), the share of candidate-entity attention landing on the **gold**
entity rises from 0.281 → 0.356 for Hindi and 0.243 → 0.350 for Nepali, against
an occurrence-weighted baseline of 0.250. The baseline is *not* 1/k, because
entities repeat unevenly in a chain — the pivot of a 3-entity chain appears
twice, so uniform attention already yields 0.250 rather than 0.333.

The effect concentrates in depth: for Hindi, layers 0–5 do not move (some fall)
while layers 6–9 all rise. The strongest single head reaches **0.918 ±0.012**
(Hindi) and **0.878 ±0.020** (Nepali) — and it is the same index in both,
**L9H4**. The two models share no weights, no tokenizer and no data, so that is
convergence on a solution, not a shared initialisation.

**Finding 3 — circuit strength tracks the learning rate, and tracks accuracy.**
Nepali's answer-position share is 0.243 pretrained, 0.271 at the under-trained
3e-06, and 0.350 at the selected 3e-05 — mirroring its test EM of 0.525 → 0.835.
The mechanistic measurement and the behavioural one agree, which is the main
reason to trust either.

## 9. The four questions the final report must answer

### 9.1 How did data scale and quality differ between the two languages?

**Not in the direction the labels suggest.** Model L is the lower-resource
*language*, but in this project it is not the token-starved model:

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| corpus tokens (own tokenizer) | 582,711,232 | **903,937,802** |
| manual share | 19.29% | **22.40%** |
| pretraining steps | 8,732 | **13,547** |
| near-duplicates removed | 15,503 (**1.27%**) | 120,689 (**5.24%**) |

Nepali ended up with ~55% more tokens and trained ~55% longer. What differs is
**diversity, not volume**. Nepali's near-duplicate rate is 4× Hindi's, and that
gap is structural rather than a filter artifact: Nepali's manual tier is three
news sites that syndicate each other's wire copy, whereas Hindi's is Wikipedia +
Wikisource + one news site, which overlap far less. Median true Jaccard of the
removed pairs was 0.847 (ne) and 0.803 (hi), so both sets really are near-dupes.

Sourcing difficulty is the other real difference, and it does not show up in the
token count: of 8 candidate Nepali news sites, only 1 had a genuinely fetchable
sitemap — the rest returned SPA catch-all HTML, bot-challenge pages, or
English-edition redirects. Reaching a comparable corpus took materially more
engineering on the low-resource side.

### 9.2 How do language-modelling and reasoning results compare across tiers?

**They dissociate, and in opposite directions.**

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| test bits-per-byte (the cross-language comparator) | 0.4947 | **0.4314** |
| test perplexity (per-token, *not* comparable across tokenizers) | 29.81 | 40.70 |
| **perplexity after reasoning finetuning**, same held-out text | 34.41 (**1.15×**) | 60.49 (**1.49×**) |
| reasoning test EM | **0.954 ±0.004** | 0.835 ±0.004 |
| reasoning test closed-set | **0.956 ±0.003** | 0.869 ±0.009 |
| 4-hop OOD EM | 0.023 ±0.010 | **0.140 ±0.048** |

By BPB — the metric that *is* portable across tokenizers — Model L is the better
language model, and PPL reverses the ordering only because the two vocabularies
segment text differently. By reasoning exact match, Model H is 12 points better.
So "better language model" and "better reasoner" come apart here.

The BPB result needs one honest caveat: a more redundant corpus is intrinsically
more predictable, so part of Model L's BPB advantage reflects its 5.24%
near-duplicate rate rather than a better model.

**Reasoning finetuning costs Model L more of its language modelling than it
costs Model H.** Measured on the same held-out natural text and the same code
as the Phase-2 numbers above: Model H's test perplexity rises 29.81 → 34.41
(1.15×), Model L's rises 40.70 → 60.49 (1.49×). Model L's BPB is still
numerically better post-finetune (0.4775 vs Model H's 0.5157), but the margin
narrows — the model that was the stronger language model gives up more ground
to become the weaker reasoner. That is a third respect, alongside the reasoning
gap itself and the induction-copying deficit (§9.3), in which Model L is the
more fragile model under this recipe.

And the OOD row reverses the ordering a fourth time — see §9.4.

### 9.3 Which tokenizer or corpus factors most hurt Model L?

**Not the tokenizer.** This is worth stating plainly because it is the intuitive
answer and it is wrong here:

| at the chosen 16k vocab | Model H | Model L |
|---|---|---|
| chars/token (whole corpus) | 3.892 | **4.644** |
| fertility (tokens/word) | 1.386 | **1.360** |
| UNK rate | 0.000000 | 0.000000 |

(Phase 1 reports two chars/token figures on different bases — 3.892 / 4.644 over
the whole corpus, 3.854 / 4.760 on the held-out sample used to pick the vocab.
The whole-corpus basis is used here; both orderings agree.)

Nepali encodes *more* efficiently at the same vocabulary size. Within the
reasoning dataset the tokenizer is controlled out entirely: 0 UNK tokens in
either language, and every entity name is a single token in both by
construction, so copying an entity is a one-step operation for both models.

**What does hurt Model L, in order of evidence strength:**

1. **In-context copying.** The induction probe — can the model complete a
   repeated random token sequence? — gives Nepali **0.364** against **0.680**
   for the Hindi checkpoint that was actually finetuned (0.749 for Hindi's
   pre-tuning checkpoint), so roughly 2× worse. Transitive reasoning over named
   entities *requires* copying a name out of the prompt, so this is the
   mechanism most directly relevant to the task. That is why it became the
   explanation for the H-vs-L gap: Nepali being ~2× worse at this one
   decontextualised, meaning-free mechanism lines up with it being weaker
   specifically at transfer — using an entity or relation it is less used to
   seeing — while both models do fine on things they have memorised patterns
   for.
2. **Corpus redundancy.** 5.24% vs 1.27% near-duplicates means fewer effective
   distinct contexts per token, which is the plausible upstream cause of (1).
3. **Transfer to unseen relations**, where the residual gap concentrates — §9.4.

**This mechanism itself survives finetuning.** Re-run on the finetuned
checkpoints (3 seeds each), induction accuracy holds: Hindi 0.658 and Nepali
0.414, both within seed noise of their pretrained values (0.680 and 0.364) —
Nepali if anything slightly higher. The copying circuit the explanation above
rests on is stable, not something finetuning degraded further.

**One factor that looked decisive and was not.** An earlier version measured Model
L at 0.21 against Model H's 0.79 and read it as a fact about the language. It was
an artifact of how the entity pool was built (§5) — a lesson about instrument
design, not about Nepali.

### 9.4 What evidence explains the differences?

**(a) Most of the apparent gap was optimisation, not language.** Model L had been
finetuned at 3e-06 against Model H's 3e-05. Correcting that one hyperparameter
moved it 0.525 → 0.835, closing ~72% of what had looked like a 43-point language
gap. Any resource-tier claim made before that was measuring the recipe (§6.1).

**(b) Both models build the same circuit**, so the residual gap is not "Model L
cannot do it". Gold-entity attention at the answer position rises 0.281 → 0.356
(H) and 0.243 → 0.350 (L) against a 0.250 baseline, strongest head **L9H4 in
both**. Circuit strength tracks the LR (0.243 → 0.271 → 0.350), so the mechanistic
and behavioural measurements agree.

**(c) The residual gap concentrates in transfer to unseen relations.**

| | seen relations | held-out relations | drop |
|---|---|---|---|
| Model H | 0.983 ±0.006 | 0.848 ±0.001 | −0.135 |
| Model L | 0.923 ±0.007 | 0.518 ±0.015 | **−0.405** |

Both order entities well for relations they trained on; Model L loses three times
as much when the relation word is new. That is the profile predicted by weaker
in-context copying (§9.3): a novel relation word forces reliance on the prompt's
structure rather than a memorised mapping, and that is exactly what the induction
probe says Model L is worse at.

**(d) Length generalisation fails for both, but the failure shapes differ.**

| sampled OOD predictions, 3 seeds | correct | middle entity | extreme | off-chain |
|---|---|---|---|---|
| Model H | 1 / 120 | **118 (98.3%)** | 2 | 0 |
| Model L | 12 / 120 | 98 (81.7%) | 13 | 9 |

Uniform guessing among the chain's entities would give 60.0%. Model H is almost
perfectly committed to answering by position; Model L keeps some probability on
the extremes. A plausible reading, offered as hypothesis not result: Model H's
training loss reaching 0.0000 by step ~1,900 locked in the heuristic that fits
every training example. Testing it would need a checkpoint sweep this phase did
not run. Supporting evidence: the only intervention that moved Model H's OOD score
was chain-of-thought (0.020 → 0.097, §6.2) — an intervention against answering by
position — while the curriculum arm, which changed hop distribution but not output
format, did not.

**Synthesis.** With the recipe genuinely equalised the low-resource model is *not*
dramatically worse at comparative reasoning: 0.835 vs 0.954, both far above a
0.398 chance line, same attention circuit, same symbolic-distance signature. The
honest resource-tier findings are narrower than "Model L reasons worse" — it
transfers markedly worse to unseen relation words, and copies from context about
half as well. Everything else that looked like a resource-tier effect turned out
to be an instrument or optimisation artifact, twice over.

## 10. Reproduction

The datasets are **not** committed — they regenerate exactly from the committed
generator and its seed, and `test_generation_is_deterministic` asserts the
regeneration is byte-identical. The Kaggle kernel regenerates them in-kernel, which
doubles as a live proof of that claim.

```bash
# 1. data — deterministic in (split, n, seed); repeat for val / test / ood
python <lang>/reasoning/generate.py --split train --n 60000
python scripts/build_reasoning_stats.py      # -> dataset_stats.md
python <lang>/reasoning/verify_leakage.py    # the three structural holdouts

# 2. finetune  (Kaggle, 2x T4, ~8 min/run; MODE is a literal in run_sft.py
#    because Kaggle cannot receive env vars at push time)
python scripts/kaggle/run.py --kernel-dir scripts/kaggle/sft_kernel --no-params

# 3. evaluate, and re-run the Phase-2 attention toolkit on both checkpoints
python <lang>/eval/run_reasoning_eval.py --pretrained <pre.pt> \
    --finetuned "<sft>/step_*.pt" --splits test,ood
python <lang>/eval/run_attention_compare.py --pretrained <pre.pt> \
    --finetuned "<sft>/step_*.pt"
python scripts/build_error_analysis.py       # -> error_analysis.md + figures
python scripts/build_attention_report.py     # -> attention_analysis.md
```

> Run the attention step **locally**. The Kaggle image ships only DejaVu Sans and
> renders every Devanagari tick label as an empty box — which is why the Phase-3
> heatmaps show real Devanagari where the Phase-2 ones do not.

## 11. Deliverables checklist

| spec item | where |
|---|---|
| own synthetic reasoning dataset, per language, in-script and natural phrasing | `<lang>/reasoning/` (generator, templates, entity pools) |
| train/val/test splits with documented anti-leakage | §3, [dataset_stats.md](dataset_stats.md), `<lang>/reasoning/verify_leakage.py` |
| generation scripts released | `<lang>/reasoning/generate.py`, deterministic in (split, n, seed) |
| finetune from own pretrained checkpoint, own tokenizer frozen | `<lang>/train/finetune.py`, §6 |
| resume-capable checkpoints | `<lang>/train/finetune.py` (`--init-from` vs `--resume`), model + optimizer + scheduler + step + config |
| pretrained vs finetuned accuracy / exact match | §7, [error_analysis.md](error_analysis.md) |
| qualitative wins and failures | [error_analysis.md](error_analysis.md) §7 |
| H vs L comparison | §9 |
| Phase-2 attention toolkit re-run on finetuned checkpoints | §8, [attention_analysis.md](attention_analysis.md) |
| final report answering the four questions | §9 |
