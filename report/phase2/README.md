# Phase 2 — Model Implementation, Pretraining, and Evaluation

**Model H = Hindi (`hi`) · Model L = Nepali (`ne`)** · branch `phase-2`

Two decoder-only Transformers built from PyTorch primitives, pretrained one full
epoch each on their own corpus, and evaluated side by side. Every number is measured
from artifacts in this repository; the side-by-side tables in
[`comparison_tables.md`](comparison_tables.md) are generated from the result JSONs.

## Results

**Model** — identical architecture for both, so the comparison isolates the data

| | value |
|---|---|
| parameters | **24,070,656** (96.3% of the ~25M budget) |
| d_model / layers / heads / context | 384 / 10 / 6 / 512 |
| positional scheme | learned absolute; LM head weight-tied |

**Language modelling** — held-out test split

| | Model H (Hindi) | Model L (Nepali) | Model L @ matched tokens |
|---|---|---|---|
| optimizer steps | 8,732 | 13,547 | 8,732 |
| train tokens | 572.3M | 887.9M | 572.3M |
| cross-entropy (nats/token) | 3.3949 | 3.7062 | 3.7915 |
| perplexity *(within-model only)* | 29.81 | 40.70 | 44.32 |
| **bits-per-byte** *(cross-language comparator)* | 0.4947 | **0.4314** | 0.4413 |

Three things to take from this:

- **BPB reverses the perplexity ordering, and BPB is the one that transfers.** The two
  models use different tokenizers, so perplexity is denominated in different units and
  is comparable *within* a model only. By bits-per-byte — which normalises by bytes of
  text — the lower-resource model is the better language model. §7 works this through
  with a measured decomposition rather than an assertion.
- **Model L is evaluated at two token budgets.** The matched checkpoint stops at Model
  H's exact training-set size, pricing the extra 316M tokens at ≈0.010 BPB — so data
  volume is separated from language and corpus effects.
- **Generation-overlap metrics were largely uninformative here** (§8), and the attention
  analysis (§9) is what actually distinguishes the two models.

Model H is reported before and after hyperparameter tuning; `hindi_baseline/` holds the
earlier run at identical architecture, steps and token budget (§4).

---

## 1. Architecture

Decoder-only transformer built only from `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout` and raw tensor ops. No `nn.Transformer*`, no
HuggingFace classes, no fused attention kernel.

```
ids (B,T)
  -> token embedding + learned positional embedding      (B,T,384)
  -> embedding dropout
  -> 10 x [ pre-norm attention residual, pre-norm FFN residual ]
  -> final LayerNorm
  -> LM head (weight-tied to token embedding)            (B,T,16000)
  -> cross-entropy against ids shifted by one
```

| choice | value | why |
|---|---|---|
| positional scheme | **learned absolute** | Simplest scheme we can explain end to end. It imposes a **hard** context bound: `nn.Embedding(512, 384)` has no row 512, so `forward` raises rather than degrading. We are not claiming length generalisation. |
| norm placement | **pre-norm** | The residual path is an unmodified identity, so gradients reach layer 0 without being rescaled at every depth. Post-norm's output variance grows with depth and needs LR warmup to train a 10-layer stack. |
| attention scaling | `QKᵀ/√d_k` | `q·k` sums `d_k` products, so its variance grows like `d_k`. Large scores saturate softmax and its Jacobian `diag(p) − ppᵀ` → 0. This is about **gradient flow**, not "smaller numbers". |
| Q/K/V | three independent projections | A shared Q/K would make every token match itself perfectly (`q·q` is maximal by construction) and dominate its own attention. |
| causal mask | additive −∞, **before** softmax | Zeroing probabilities *after* softmax leaves rows not summing to 1, silently down-weighting each position by the deleted future mass. |
| FFN | 2 linears, GELU, `d_ff = 4×d_model` | Attention is a linear recombination of values; the FFN supplies the per-token nonlinearity. |
| weight tying | **yes** | Saves `vocab × d_model` = **6,144,000** params — a quarter of the budget, redirected into depth. |

## 2. Parameter count — 24,070,656 (96.3% of the ~25M budget)

| component | params |
|---|---|
| token embedding (16,000 × 384) | 6,144,000 |
| position embedding (512 × 384) | 196,608 |
| 10 blocks × 1,772,928 | 17,729,280 |
| final LayerNorm | 768 |
| LM head (tied) | 0 — **saves 6,144,000** |
| **total trainable** | **24,070,656** |

Computed **two independent ways** — closed-form arithmetic and a live count off
the built module — which agree exactly (`python <lang>/model/param_count.py`).
One number is an assertion; two agreeing is evidence.

**Depth over width.** At this budget 384×10 spends **25.5%** of parameters on
embeddings, while 512×5 spends **33.8%** for the same total — the narrower,
deeper shape buys ~2.5M more parameters of actual computation and gives a
longer early-vs-late gradient for the Phase-2 attention analysis.

Both languages use the same shape because the vocab sweep landed on **16,000
for both** (Phase 1, T1.12), so their configs coincide. This was verified per
tree, not inferred: each tree's own `param_count.py` and its own `val.bin`
were run separately.

## 3. Correctness tests

`tests/test_model.py`, `tests/test_train.py`, `tests/test_tokenizer_roundtrip.py`
— 40 tests. The suite targets failures that are **silent**: a broken causal
mask trains happily, produces a falling loss curve, and reports a plausible
perplexity.

| test | asserts |
|---|---|
| future token cannot change past logits | perturb token *t+1*; logits at every position ≤ *t* are **bit-identical** — and the perturbation *did* change *t+1*, so the test cannot pass on a model that ignores its input |
| attention weights strictly lower-triangular | everything above the diagonal is exactly zero, rows still sum to 1 |
| head reshape is a true split | `(B,T,d) → (B,h,T,d_k)` partitions rather than duplicates |
| over-length sequence rejected | `T > max_seq_len` raises |
| weight tying shares storage | one tensor, counted once |
| gradient reaches every parameter | no orphaned weights |
| tiny-batch overfit | loss < 0.05 after 300 steps |
| tokenizer / shard round-trip | packed uint16 shard decodes back to the source corpus |
| trees byte-identical | `hindi/` and `nepali/` model and training code are identical |
| schedule horizon independent of link stop | regression guard, see §5 |

## 4. Training setup

| | |
|---|---|
| optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 |
| weight decay applies to | **2D matmul weights only** — decaying a LayerNorm gain toward zero suppresses the layer for no regularisation benefit |
| schedule | linear warmup (500) → cosine decay to `0.1 × peak` |
| peak LR | **3e-3** (6e-4 for the pre-tuning baseline) |
| dropout | **0.0** (0.1 for the pre-tuning baseline) |
| grad clipping | 1.0, applied **after** full accumulation so the norm is that of the true effective-batch gradient |
| precision | **fp16 autocast + GradScaler** — measured 3.11× faster than bf16 on the T4 (Turing tensor cores are fp16; bf16 arrives with Ampere). 0 skipped steps in 8,732. |
| batch | 8 × 16 accumulation = 128 sequences = **65,536 tokens/step** |
| hardware | Kaggle **2× Tesla T4** (sm_75), one language per GPU, in parallel |
| seed | 1337 |

### How the learning rate and dropout were chosen

Not by default: by a sweep, run against the completed baseline. The baseline
showed validation loss **below** training loss at all 33 checkpoints, which is
the signature of a model that is not overfitting — so dropout was regularising
against a problem that does not occur. Four configurations were trained for
1,500 steps each, all using `--steps 8732 --stop-at 1500` so the LR schedule
shape matched the baseline exactly and val@1500 was directly comparable:

| dropout | peak_lr | val@1500 | vs baseline |
|---|---|---|---|
| 0.1 | 6e-4 | 4.1865 | baseline |
| 0.0 | 6e-4 | 4.1167 | −0.070 |
| 0.0 | 1.5e-3 | 3.9924 | −0.194 |
| **0.0** | **3e-3** | **3.9492** | **−0.237** |
| 0.0 | 6e-3 | 3.9729 | −0.214 |

The optimum is *interior* — 3e-3 beats both 1.5e-3 and 6e-3 — so it is
bracketed rather than at an edge. Fitting the parabola through the three points
puts the true optimum at 3.32e-3 and the remaining gain from refining further
at **0.0007 nats**, which is why the sweep stopped there. Removing dropout was
worth −0.070 on its own; the learning rate was worth 2.4× more.

Carried to the full 8,732-step run, the combined change gave **−0.177 nats**
(val 3.5574 → 3.3803), a **16.2% perplexity reduction**.

**Why warmup, given pre-norm doesn't need it.** Pre-norm removes the
*architectural* need. Warmup is kept for the *optimizer*: at step 0 AdamW's
second-moment estimate is near zero, so early updates divide by a noisy
gradient-scale estimate. 500 steps is <6% of the run and removes a failure
mode we would otherwise discover by watching a loss curve diverge on rented
GPU time.

**Parallelism is process isolation, not DDP.** Each language runs as its own
OS process with `CUDA_VISIBLE_DEVICES` pinned, so the trainer contains no
distributed code and only ever sees one device. This enforces the model
independence constraint at the strongest boundary available — DDP would put
both languages in one process group and make independence a property of
careful coding rather than of the operating system.

## 5. Checkpointing and resume

A checkpoint carries **weights + optimizer moments + step + config +
`lang_code` + RNG state**, written atomically (temp file then `os.replace`).

**`lang_code` is a hard gate.** Hindi and Nepali share vocab 16,000, so every
tensor shape matches and a cross-language load would succeed **silently**.
Shape checking cannot catch this; only an explicit identity check can.
`load_checkpoint` requires `expected_lang_code` as a keyword-only argument
with no default — a gate you can skip by forgetting a parameter is not a gate
— and raises a dedicated `LanguageMismatchError`.

**Kill-and-resume proof** (`scripts/kill_resume_proof.py`): SIGKILL a live
training subprocess mid-step, resume from its checkpoint with a *different*
seed, and compare against an uninterrupted reference run.

| | Hindi | Nepali |
|---|---|---|
| kill mechanism | SIGKILL, `returncode -9` | same |
| batch hashes at N+1, N+2 | identical | identical |
| loss / grad-norm delta | **0.0e+00** | **0.0e+00** |
| weights, AdamW m and v | max abs diff **0.0** | **0.0** |
| LR at resume, step, config, lang_code | restored | restored |

Bit-identical resume requires the RNG state as well as the weights and
optimizer moments: dropout masks are drawn from the global generator, so
without it a resumed run follows a different random path — statistically
similar, but not the same run.

**Chained training.** A Kaggle batch kernel is killed at 12 hours, so a full
epoch is a chain of runs, each resuming from the previous. The LR **schedule
horizon** is the whole run and is identical on every link; only the stop point
moves. If each link planned its own cosine instead, the schedule would anneal
to its floor at every boundary and restart — a sawtooth rather than the single
decay the run is meant to follow. Measured at the link boundary:

| | link 1 last | link 2 first |
|---|---|---|
| Hindi | step 7,080 · lr 1.12e-04 · gnorm 0.38 | step 7,100 · lr 1.11e-04 · gnorm 0.38 |
| Nepali | step 6,760 · lr 3.47e-04 · gnorm 0.39 | step 6,780 · lr 3.46e-04 · gnorm 0.39 |

## 6. Pretraining results

Both models trained **one full epoch** on their own corpus.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| train tokens | 572,264,559 | 887,852,764 |
| optimizer steps | 8,732 | 13,547 |
| throughput | **1.718 s/step** | 5.999 s/step |
| peak VRAM | 2.49 GiB / 14.6 (17%) | 2.64 GiB / 14.6 (18%) |
| wall-clock | **4.16 h** | 23.0 h |

Loss curves: [`hindi/loss_curves.png`](hindi/loss_curves.png),
[`nepali/loss_curves.png`](nepali/loss_curves.png). Full histories in
`training/*_train_history.jsonl`.

Validation during training was a 40-batch sample taken every 250 steps. **The
authoritative numbers are the full-split evaluation in §7**, which scores every
token of val and test; the two differ by 0.0002 nats (Hindi) and 0.0121 nats
(Nepali).

**Matched-token checkpoint.** Model L also checkpoints at step 8,732, where it
has consumed 572,260,352 tokens — Hindi's *entire* training set. This gives two
separable comparisons instead of one confounded number, and is used throughout
§7 and §10.

---

## 7. Intrinsic evaluation — loss, perplexity, BPB

Scored over the **complete** val and test splits (not a sample):
5.80M/5.85M tokens for Hindi, 9.24M/9.03M for Nepali.

| | Model H (Hindi) | Model H (pre-tuning) | Model L (Nepali) | Model L @ matched |
|---|---|---|---|---|
| val cross-entropy | **3.3803** | 3.5574 | 3.7161 | 3.8013 |
| val perplexity | **29.38** | 35.07 | 41.10 | 44.76 |
| **val BPB** | **0.4932** | 0.5190 | **0.4348** | **0.4448** |
| test cross-entropy | **3.3949** | 3.5716 | 3.7062 | 3.7915 |
| test perplexity | **29.81** | 35.57 | 40.70 | 44.32 |
| **test BPB** | **0.4947** | 0.5205 | **0.4314** | **0.4413** |

Val and test agree to within 0.015 nats on both models, so neither is overfit
to its validation set.

### Discussing the H-vs-L gap

**The two metrics disagree, and only one of them is valid across languages.**

By perplexity Model H looks clearly better (35.57 vs 40.70). By **BPB, Model L
is better (0.4314 vs 0.5205)**. This is not a contradiction — it is the reason
the spec asks for BPB at all.

Perplexity is `exp(mean cross-entropy per token)`, so its unit is *this
tokenizer's token*. The two tokenizers are different: at vocab 16,000 Nepali
encodes **4.760 chars/token against Hindi's 3.854** (fertility 1.360 vs 1.386,
Phase 1 §7). A Nepali token therefore carries ~23% more text, making each
prediction harder and inflating perplexity, while the per-byte cost of the text
is lower. BPB divides by bytes — a property of the text, not of the
segmentation — and is the only figure on which a cross-language claim can rest.

### Perplexity is not portable across vocabularies — a worked example

Perplexity is denominated in tokens, so **it is not comparable to any published
figure unless that model used the same tokenizer**. The effect is large enough
to be worth quantifying, because it is easy to read our 29.81 as weak when
compared against a number produced on a different scale.

The identity connecting the two metrics, with `N` tokens carrying `B` bytes and
total cross-entropy `L_total` nats:

```
loss_per_token = L_total / N
bytes_per_token = B / N
BPB            = (L_total / ln 2) / B

  =>  loss_per_token = BPB x ln(2) x bytes_per_token
```

Against our own measurements this is exact:

```
0.4947 x 0.693147 x 9.8998 = 3.3949   ==  measured Hindi test loss
```

Holding **BPB fixed** — the tokenizer-invariant quantity, which is why the spec
asks for it — only `bytes_per_token` changes, and it scales with characters per
token, which Phase 1 measured for every vocabulary in its sweep. The resulting
table is **generated** in
[`comparison_tables.md`](comparison_tables.md) rather than written by hand, so
it cannot keep an old model's numbers after a retrain. In summary:

| vocab | perplexity of this same model |
|---|---|
| 5,000 *(extrapolated)* | ~17.1 |
| 8,000 | 21.9 |
| **16,000 (ours)** | **29.8** |
| 48,000 | 42.2 |

**The model is identical in every row.** Its weights and its per-byte modelling
quality do not change; only the unit of the denominator does. A larger
vocabulary packs more text into each token, making every prediction harder and
perplexity higher, while the cost per byte of text stays the same.

This is the arithmetic consequence of the Phase-1 vocabulary decision. 16,000
was chosen on parameter grounds — 48,000 would have spent 74% of a 25M model on
the embedding matrix to buy ~9% better fertility (Phase 1 §7) — and a higher
reported perplexity is the price of that choice, not evidence of a worse model.

Comparisons against models using small vocabularies (5k–8k is common in
low-resource work) are therefore comparisons of tokenizers as much as of
models, and **BPB is the only figure here on which such a comparison should
rest**.

One caveat: this assumes BPB is exactly invariant to vocabulary size. It is approximately but not perfectly so — a smaller vocabulary
is marginally less efficient to model — so the extrapolated rows are close
estimates, not exact predictions.

**Isolating data volume from everything else.** The matched-token checkpoint
answers "how much of the difference is just more data?":

- Model L at **Hindi's exact token budget**: 0.4413 BPB
- Model L with its **full 888M tokens**: 0.4314 BPB

So the extra 316M tokens are worth **≈0.010 BPB**, and the remaining
**≈0.063 BPB** advantage over Hindi is attributable to language and corpus
properties rather than to scale. Given Model L is the *lower*-resource side,
that is the opposite of the naive expectation and needs the explanation in §10.

---

## 8. Generation quality

Protocol: prompts are the first **64 tokens** of held-out **test** documents,
taken at document boundaries so the model is never asked to continue text whose
beginning it never saw. Reference = the true continuation of the same document.
Up to **128 new tokens**. 191 prompts (Hindi), 197 (Nepali).

Full tables for all five settings and all three checkpoints:
[`comparison_tables.md`](comparison_tables.md).

| Hindi | BLEU-4 | chrF | chrF++ | ROUGE-L | rep-4 | D-1 | D-2 |
|---|---|---|---|---|---|---|---|
| greedy | 3.12 | 15.53 | 13.91 | 0.1124 | **0.578** | 0.080 | 0.176 |
| temp 0.5 | **3.53** | 18.56 | 16.88 | **0.1252** | 0.244 | 0.143 | 0.419 |
| temp 1.0 | 2.00 | **21.33** | **18.93** | 0.1138 | 0.002 | 0.338 | 0.835 |
| temp 1.5 | 0.21 | 19.64 | 15.80 | 0.0483 | 0.000 | 0.605 | **0.993** |
| top-p 0.95 | 2.39 | 21.20 | **18.94** | 0.1172 | 0.002 | 0.309 | 0.813 |

Model H improved on almost every generation metric after tuning (pre-tuning
figures in [`hindi_baseline/`](hindi_baseline/) and
[`comparison_tables.md`](comparison_tables.md)): greedy repetition 0.617 →
**0.578**, greedy BLEU-4 2.28 → **3.12**, temp-0.5 chrF 16.99 → **18.56**. The
one regression is temperature 1.5 (chrF 20.03 → 19.64), which is expected — a
sharper, better-calibrated distribution has more to lose when flattened.

| Nepali | BLEU-4 | chrF | chrF++ | ROUGE-L | rep-4 | D-1 | D-2 |
|---|---|---|---|---|---|---|---|
| greedy | 1.55 | 14.42 | 12.35 | 0.0916 | **0.586** | 0.094 | 0.161 |
| temp 0.5 | **2.30** | 17.97 | 15.63 | **0.1112** | 0.235 | 0.207 | 0.452 |
| temp 1.0 | 1.60 | 24.16 | 20.39 | 0.0990 | 0.006 | 0.439 | 0.884 |
| temp 1.5 | 0.14 | 23.31 | 18.28 | 0.0402 | 0.000 | 0.595 | **0.987** |
| top-p 0.95 | 1.60 | **24.94** | **21.08** | 0.1044 | 0.005 | 0.406 | 0.862 |

BLEU is **corpus-level, order 4**, sacrebleu with the `intl` tokenizer
(sacrebleu's default `13a` is built around Latin script). chrF++ is
`word_order=2`.

### Why each metric is or is not informative here

**All three overlap metrics are weak for open-ended generation, structurally.**
They score precision/recall against a *single* reference, while the space of
valid continuations is enormous. A fluent, on-topic continuation that simply
differs from the one the corpus happened to contain scores near zero: the
metrics cannot tell "wrong" from "different but fine".

**BLEU-4 is the weakest.** It requires exact 4-gram matches, and matching four
consecutive word forms by chance in a free continuation is vanishingly unlikely
— hence absolute values of 0.14–2.59 that carry almost no quality signal.

**chrF/chrF++ are the most useful of the three for Devanagari**, because they
match *character* n-grams. Hindi and Nepali are morphologically rich with heavy
inflection and agglutinative postpositions, so two forms of the same lemma share
most of their characters while being completely different word tokens. chrF
awards that partial credit; BLEU does not.

**ROUGE-L** (LCS, order-sensitive but not contiguity-requiring) sits between
them.

**The decisive evidence that BLEU misleads here is in our own numbers.** BLEU-4
and ROUGE-L both peak at **temperature 0.5**, while chrF/chrF++ peak at
**temperature 1.0–1.5**. Temperature 0.5 still repeats **24.4%** of its 4-grams
(Hindi) and has Distinct-2 of only 0.419. BLEU is rewarding conservative,
partly-repetitive output that the diversity diagnostics show is degenerate.
Greedy makes this starker: it scores a *respectable* BLEU-4 of 2.28 while
repeating **57.8%** of its 4-grams.

Part of that is measurable rather than inferred: greedy's **prompt-copy rate is
8.6%** of 4-grams (Hindi) against 0.5% at temperature 1.0. It partly "continues"
by paraphrasing the prompt back, which inflates overlap against a reference
drawn from the same document while being a generation failure.

**Conclusion: the reference-free diagnostics — repetition rate and Distinct-1/2
— are the informative ones here**, and the overlap metrics are useful mainly as
*within-model* comparisons across decoding settings, not as absolute quality.

### Repetition, and why top-p was run

The course notes (Holtzman et al.) report greedy decoding at 73.66% repetition
against 0.28% for human text, and nucleus sampling at 0.36%. Our models
reproduce that: **greedy 57.8% (Hindi) / 58.6% (Nepali)**, versus ~0% for every
sampled setting. Top-p 0.95 was run as a diagnostic — not required by the spec —
so the repetition figures could be attributed to the decoding strategy rather
than to the model.

Figures: [`hindi/generation_repetition.png`](hindi/generation_repetition.png),
[`hindi/generation_distinct2.png`](hindi/generation_distinct2.png) and the
Nepali equivalents.

### Qualitative notes

Full write-ups: [`hindi/qualitative_notes.md`](hindi/qualitative_notes.md),
[`nepali/qualitative_notes.md`](nepali/qualitative_notes.md). Samples:
`*/generation_samples.md`. In brief:

**Language correctness is strong in both.** 95–97% of non-space characters are
Devanagari for Hindi, **97–99% for Nepali**, with Latin under 2% and confined to
the parenthesised English technical terms both corpora contain. Neither model
drifts out of script.

**Coherence fails in two distinct ways.** Greedy loops verbatim at clause level
while remaining locally grammatical. High temperature keeps syntax well-formed
but loses meaning — given an insider-trading prompt, temperature 1.0 wanders
through "parallel debt policy" and "soya credit card", and 1.5 invents
plausible-looking non-existent proper nouns.

**High temperature also abandons sentence structure**, which no diversity metric
shows: danda (sentence terminator) density falls **7.34 → 1.65 per 100 words**
(Hindi) and **10.35 → 2.43** (Nepali) from greedy to temperature 1.5. The
diversity at 1.5 is bought by producing unbounded run-on text.

**Model L reproduces a census template** — see §10, it is a corpus-quality
finding rather than a modelling one.

---

## 9. Attention analysis

Probes are held-out test passages of 64 tokens; the sentences themselves are
recorded in `*/attention_probes.md` and inside `attention.json`, and appear as
tick labels on the heatmaps.

Heatmaps (query × key, colour = attention weight, 3 heads each):
**early layer 0** — [`hindi/attention_layer0.png`](hindi/attention_layer0.png),
[`nepali/attention_layer0.png`](nepali/attention_layer0.png);
**late layer 9** — [`hindi/attention_layer9.png`](hindi/attention_layer9.png),
[`nepali/attention_layer9.png`](nepali/attention_layer9.png).

| layer | H entropy (norm) | H distance | L entropy (norm) | L distance |
|---|---|---|---|---|
| 0 | 0.818 | 12.18 | 0.974 | 14.95 |
| 1 | 0.893 | 16.73 | 0.904 | 16.74 |
| 2 | 0.804 | 16.81 | 0.776 | 12.43 |
| 3 | 0.792 | 10.31 | 0.717 | 7.17 |
| 4 | 0.706 | 8.24 | 0.672 | **5.38** |
| 5 | **0.623** | 8.86 | **0.652** | 6.43 |
| 6 | 0.721 | 13.85 | 0.723 | 11.64 |
| 7 | 0.696 | 16.15 | 0.714 | 18.92 |
| 8 | 0.650 | **21.04** | 0.795 | 17.74 |
| 9 | 0.698 | 20.99 | 0.815 | **18.69** |

Entropy is normalised by `log(i+1)` — the entropy of a uniform distribution over
the keys query *i* is allowed to see — because raw entropy is confounded by
position: early rows can only attend to few keys and are forced to look sharp.

### Which heads are local/positional, and which are content-based

Reading this off a heatmap is guesswork, so it is **measured**. The
content-dependence diagnostic re-runs the model with token identities
**permuted within the sequence** — same length, same positions, same token
distribution, only *which token sits where* changes. A purely positional head is
unaffected (score ≈ 0); a content-based head re-routes.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| most local (min distance) | L5H1 (1.07) | L5H1 (1.06) |
| most distant | L7H0 (27.51) | L6H1 (24.11) |
| sharpest (min norm. entropy) | L5H1 (0.108) | L5H1 (0.075) |
| **most positional** | **L5H1 (0.00656)** | L2H5 (0.00468) |
| **most content-based** | L6H4 (0.04109) | L5H3 (0.04223) |

Grid figures: `*/attention_entropy_grid.png`, `*/attention_distance_grid.png`,
`*/attention_content_grid.png`.

**Both models independently learned a dedicated previous-token head, and both
placed it at the same index — L5H1.** Hindi: mean distance **1.07**,
previous-token weight **0.927**. Nepali: **1.06**, **0.938**. In both models
that head is also the *sharpest*, and in Hindi it is additionally the **most
positional** head by the shuffle diagnostic — exactly as the mechanism predicts,
since a head that always looks one step back does not care which token is there.

That the two models converged on the same layer and head index, from different
languages, different corpora and independent initialisation, is a stronger
result than any single-model observation.

**Depth profile.** Both models sharpen through the middle of the stack and
reach furthest back at the end. Entropy bottoms at layer 5 in both (Hindi
0.623, Nepali 0.652) and the shortest attention distances sit in layers 4–5
(Hindi 8.24, Nepali 5.38), while layers 8–9 reach ~19–21 tokens — consistent
with mid-stack heads tracking local syntax and late layers aggregating context.

Model H's layer 0 is notably sharper than Model L's (0.818 vs 0.974): after
tuning it specialises earlier rather than leaving the first layer near-uniform.

**Content-based heads sit mid-stack in both models** — layers 5–8 — which is
where the shuffle diagnostic peaks. That is the layer band doing token-identity
work, sandwiched between positional early layers and broad late ones.

### How Model H and Model L differ

Two differences are robust:

1. **Model H specialises earlier in the stack.** Its layer 0 normalised entropy
   is 0.818 against Model L's 0.974, and it stays below Model L from layer 7
   onward (0.696/0.650/0.698 vs 0.714/0.795/0.815).
2. **Model H reaches further back in its late layers** — 21.04 tokens at layer 8
   against Model L's 17.74.
3. **Model L goes more sharply local mid-stack** — minimum distance 5.38 at
   layer 4 against Model H's 8.24.

The similarity is more striking than the differences: **both models placed their
previous-token head at exactly L5H1**, with near-identical distance (1.07 vs
1.06) and previous-token weight (0.927 vs 0.938), despite different languages,
different corpora and independent initialisation.

---

## 10. Resource-level comparison

The headline is counter-intuitive and needs stating carefully.

**Model L — the designated lower-resource model — has the better bits-per-byte**
(0.4314 vs 0.4947 on test), and keeps that advantage even when its data is cut
to Hindi's exact token budget (0.4413). Three factors explain it, and only one
of them is about the models:

**1. Tokenizer fertility, not modelling quality.** Nepali's tokenizer encodes
4.760 chars/token against Hindi's 3.854 at the same 16,000 vocabulary — a
finding already recorded in Phase 1. More text per token means fewer, harder
predictions per byte. This inflates Nepali's perplexity and deflates its BPB
simultaneously, which is exactly the pattern observed.

**2. Corpus homogeneity — Model L's corpus is easier.** Nepali's manual tier is
85% news from three sites that syndicate each other's wire copy; its Phase-1
near-duplicate rate was **5.24%, four times Hindi's 1.27%**. A more repetitive,
more templated corpus is more predictable per byte. This shows up directly in
the generations: Model L reproduces a **2068 census template with varying
numeric slots** —

> …जनगणना २०६८मा यहाँको जनसङ्ख्या १८९७ रहेको थियो जसमध्ये १२९७ पुरुष र १२९७ महिला…

— at greedy *and* at temperature 1.0, with the frame fixed and the numbers
changing. Nepali Wikipedia contains thousands of near-identical village stubs
built from that census, and at 30,037 total articles they are a large share of
its encyclopedic tier. The template even leaks a factual inconsistency
(identical male and female counts), because slot-filling does not constrain
arithmetic.

**3. Data volume helps, but less than expected.** The matched-token checkpoint
puts a number on it: going from 572M to 888M tokens is worth **0.010 BPB**,
about a sixth of the 0.063 BPB gap to Hindi. Scale is the smallest of the three
effects.

**So a lower BPB does not mean Model L is the better language model.** It means
Nepali text, as segmented by its own tokenizer and as sampled by our corpus, is
cheaper to predict per byte. The cross-tier conclusion is that **BPB
comparisons across languages confound model quality with tokenizer granularity
and corpus redundancy**, and that the matched-token control is what makes any
part of the comparison interpretable.

**Model L's weakness is visible qualitatively rather than in the aggregate
metrics:** a meaningful share of its fluency is template
reproduction, which is a data-quality ceiling that more parameters would not
lift. That is the finding to carry into Phase 3.

**A note on training configuration.** Model H's reported results come from a
run using dropout 0.0 and peak LR 3e-3, selected by the sweep in §4; the
pre-tuning run (dropout 0.1, peak LR 6e-4) is retained in
[`hindi_baseline/`](hindi_baseline/) and reported alongside throughout. Model L
uses the configuration recorded in `nepali/configs/train.yaml`. Architecture,
parameter count (24,070,656), context length, step count and token budget are
identical across all of them, so the comparison remains one of data and
language rather than of model shape.

Tuning Model H narrowed the H-vs-L BPB gap from 0.089 to **0.063** — roughly
29% of what would otherwise have been attributed to language and corpus
properties was optimisation headroom. The remaining gap is better evidence for
the argument above precisely because that component has been removed.

**Corpus context** (Phase 1): Hindi 572.3M train tokens, 19.29% manually
collected; Nepali 887.9M, 22.40% manual.

---

## 11. Figures index

| figure | what it shows |
|---|---|
| `<lang>/loss_curves.png` | training and validation loss against step |
| `<lang>/attention_layer0.png` | early-layer heatmaps, 3 heads, query × key |
| `<lang>/attention_layer9.png` | late-layer heatmaps, 3 heads |
| `<lang>/attention_entropy_grid.png` | normalised entropy, layer × head |
| `<lang>/attention_distance_grid.png` | mean attention distance, layer × head |
| `<lang>/attention_content_grid.png` | content-dependence, layer × head |
| `<lang>/attention_layer_profile.png` | entropy and distance against depth |
| `<lang>/generation_repetition.png` | 4-gram repetition by decoding setting |
| `<lang>/generation_distinct2.png` | Distinct-2 by decoding setting |

`hindi/` holds Model H's final results; `hindi_baseline/` holds the
pre-tuning run.

`viz.py` **raises** if a figure lacks a title or axis labels, so an unlabelled
figure cannot be saved.

---

## 12. Reproduction

```bash
python <lang>/data_prep/tokenize_corpus.py         # -> data/tokenized/*.bin
python <lang>/model/param_count.py                 # verify 24,070,656
python -m pytest tests/ -q                         # 40 tests
python scripts/kill_resume_proof.py --tree hindi   # SIGKILL resume proof

# Pretraining on Kaggle (2x T4), chained past the 12h per-run cap
python scripts/kaggle/sync_code.py  --user <user> --create
python scripts/kaggle/sync_data.py  --user <user> --slug lma-tokens --create
python scripts/kaggle/chain.py --links 2 --steps 13547 --stop-every 6774 \
    --stop-hi 7089 8732 --stop-ne 6764 13547 --data-slug lma-tokens --real

# Evaluation
python <lang>/eval/run_eval.py --checkpoint <ckpt.pt>
python scripts/build_comparison.py                 # regenerates comparison_tables.md
python scripts/check_independence.py               # 5/5 must pass
```

Seed 1337 throughout. Every checkpoint stores its resolved config.

---

## 13. Deliverables

| # | deliverable | where |
|---|---|---|
| 1 | Transformer implementation (MHA, positional embeddings, causal mask) | `<lang>/model/` |
| 2 | Model configuration files (H and L) | `<lang>/configs/model.yaml`, `train.yaml` |
| 3 | Parameter counts; training scripts; Drive links to checkpoints | §2, `<lang>/train/`, root `README.md` |
| 4 | Training logs and loss curves | `training/`, `<lang>/loss_curves.png` |
| 5 | PPL / BPB tables; BLEU, chrF, ROUGE-L results | §7, §8, `comparison_tables.md` |
| 6 | Generated samples and diversity / repetition stats | `<lang>/generation_samples.md`, `generation.json` |
| 7 | Attention heatmaps plus entropy / distance summaries | §9, `<lang>/attention_*.png`, `attention.json` |
| 8 | Resource-level comparison write-up | §10 |

---

## 14. Bonus — no-positional-embeddings ablation

Completed separately from the graded Phase-2 work, using this phase's own model
code and evaluation suite unchanged. Model L was retrained from scratch with the
positional table removed (23,874,048 parameters) and scored with the full suite
at both the 13,547-step and 8,732-step horizons.

Removing the entire positional mechanism costs **+0.0255 nats / ×1.026
perplexity** on the test split, and the two horizons agree to 0.0005 nats.

- Write-up: [`bonus_ablation.md`](bonus_ablation.md)
- Generated tables: [`bonus_ablation_tables.md`](bonus_ablation_tables.md)
- Results: `nepali_nope/`, `nepali_nope_matched/`
