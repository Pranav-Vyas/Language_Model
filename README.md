# Decoder-Only Transformer Language Models for Hindi and Nepali

Two **completely independent** decoder-only Transformer language models built from scratch in PyTorch — one for Hindi and one for Nepali — spanning the full pipeline from raw corpus collection through BPE tokenisation, pretraining, and supervised reasoning finetuning, with a rigorous cross-language evaluation framework.

| | Model H | Model L |
|---|---|---|
| Language | **Hindi** (`hi`) | **Nepali** (`ne`) |
| Script | Devanagari | Devanagari |
| Resource tier | Higher-resource | Lower-resource |
| Directory | `hindi/` | `nepali/` |

The two models share **no data, tokenizer, vocabulary, or weights**. This is enforced structurally, not by convention: each language tree holds its own copy of the code and never imports from its sibling, and `scripts/check_independence.py` fails the build if that invariant is ever violated.

---

## Results at a Glance

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Corpus tokens | 582,711,232 | 903,937,802 |
| Parameters | 24,070,656 | 24,070,656 |
| Test perplexity | 29.81 | 40.70 |
| **Test bits-per-byte** (cross-language comparator) | 0.4947 | **0.4314** |
| **Reasoning exact match** (chance 0.398) | **0.954** | 0.835 |
| 4-hop generalisation (chance 0.200) | 0.023 | 0.140 |

The low-resource model is the **better language model** by the portable BPB metric, yet the **weaker reasoner** — the two measures come apart. Full argument in [`report/phase3/README.md`](report/phase3/README.md) §9.

---

## Project Overview

The project is structured across three sequential phases:

| Phase | Focus | Status |
|---|---|---|
| 1 — Data Collection & Tokenization | Corpus scraping, cleaning, deduplication, BPE training | **Complete** — [`report/phase1/README.md`](report/phase1/README.md) |
| 2 — Model, Pretraining & Evaluation | Architecture, training on Kaggle T4 GPUs, PPL/BPB evaluation, attention analysis | **Complete** — [`report/phase2/README.md`](report/phase2/README.md) |
| 3 — Reasoning Finetuning & Attention | Synthetic reasoning dataset, SFT, evaluation, corrections | **Complete** — [`report/phase3/README.md`](report/phase3/README.md) |

---

## Phase 1 — Data Collection & Tokenization

Corpora were assembled from multiple sources: Wikipedia, Wikisource, news archives, and the publicly available Sangraha (`ai4bharat/sangraha`) HuggingFace dataset. Scrapers are polite — `robots.txt` is honoured, requests are rate-limited, and all runs are resumable. Split assignment is a hash of each document's own content, so the same document always lands in the same split regardless of corpus extension order.

| | Hindi | Nepali |
|---|---|---|
| Documents | 1,202,072 | 2,180,424 |
| **Tokens** (own trained tokenizer) | **582,711,232** | **903,937,802** |
| Manual share of corpus | 19.29% | 22.40% |
| Vocabulary size | 16,000 | 16,000 |
| Fertility (tokens/word) | 1.386 | 1.360 |
| Chars per token | 3.892 | 4.644 |
| UNK rate | 0.000000 | 0.000000 |

Both corpora exceed 500M training tokens. Tokenizers are BPE models trained with SentencePiece; a sweep over vocabulary sizes of 8k, 16k, 32k, and 48k was run for each language, with 16k selected for both. Every committed tokenizer file (`.model`, `.vocab`) is the exact model used in training. Full details: [`report/phase1/README.md`](report/phase1/README.md).


---

## Phase 2 — Model Architecture & Pretraining

### Architecture

Both models use the same decoder-only Transformer architecture:

- **Parameters:** 24,070,656 (analytic count matches `module.parameters()`)
- **d_model:** 384
- **Layers:** 10
- **Attention heads:** 6
- **Context length:** 512 tokens
- **Weight-tied LM head** (embedding and output projection share weights)

### Training

Each model was trained for **one full epoch** on its own corpus on Kaggle T4 GPUs. Because a single Kaggle session is capped at 12 hours and one full epoch requires 14.2 h (Hindi) / 23.0 h (Nepali), training is structured as a chain of resumed sessions using `scripts/kaggle/chain.py`. Checkpoints store weights, optimizer moments, step, config, `lang_code`, and RNG state, and resume bit-identically after a `SIGKILL` — proven in `scripts/kill_resume_proof.py`.

Model H's hyperparameters (dropout 0.0, peak LR 3e-3, fp16) were selected by a sweep; the pre-tuning baseline run is retained in `report/phase2/hindi_baseline/` for direct comparison.

### Pretraining Results

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| train tokens | 572,264,559 | 887,852,764 |
| optimizer steps | 8,732 | 13,547 |
| **full-split val loss** | **3.3803** | 3.7161 |
| **full-split val perplexity** | **29.38** | 41.10 |
| checkpoint | `hindi/checkpoints/step_8732.pt` | `nepali/checkpoints/step_13547.pt` |

Model L also has a **matched-token checkpoint** at step 8,732, where it has seen
572,260,352 tokens — exactly Hindi's entire training set — so the H-vs-L comparison
can be made at equal token budget as well as at each model's own data scale.

**Evaluation (full val + test splits):**

| | Model H (Hindi) | Model L (Nepali) | Model L @ matched tokens |
|---|---|---|---|
| test cross-entropy | **3.3949** | 3.7062 | 3.7915 |
| test perplexity | **29.81** | 40.70 | 44.32 |
| **test bits-per-byte** | **0.4947** | **0.4314** | **0.4413** |

Perplexities are **not** comparable across the two languages — the tokenizers are
separate and differ in fertility (4.760 vs 3.854 chars/token), so raw PPL measures
different units. The cross-language claim rests on **BPB**, and note that BPB
*reverses* the PPL ordering. `report/phase2/README.md` §10 decomposes why.

Checkpoints store weights, optimizer moments, step, config, `lang_code` and RNG state,
and resume bit-identically after a `SIGKILL` — proven in `scripts/kill_resume_proof.py`.


### Bonus — ablation: Model L without positional embeddings

Model L retrained from scratch with the learned positional table removed
(**23,874,048** parameters, −196,608 = 512 × 384), everything else held to the
standard run's own stored config. Full write-up:
[`report/phase2/bonus_ablation.md`](report/phase2/bonus_ablation.md); generated
tables in [`bonus_ablation_tables.md`](report/phase2/bonus_ablation_tables.md).

| test split | standard | no positional embeddings | Δ |
|---|---|---|---|
| cross-entropy, 13,547 steps | 3.7062 | 3.7317 | **+0.0255 nats** |
| cross-entropy, 8,732 steps | 3.7915 | 3.8175 | **+0.0260 nats** |
| perplexity, 13,547 steps | 40.70 | 41.75 | ×1.026 |
| bits-per-byte, 13,547 steps | 0.4314 | 0.4343 | +0.0030 |

Removing the entire positional mechanism costs **2.6% perplexity** — 30% of what
Model L's final 4,815 training steps bought, and two independent horizons agree to
0.0005 nats. The model does not become order-blind because a **causal** decoder
recovers position from its own mask: attention entropy shows layers 1–2 turning
near-uniform (0.965 / 0.967 vs 0.904 / 0.776), which is what averaging over a
prefix whose *size is the position* looks like. The clearest signature is that
the most positional head available goes from content-dependence 0.0047 to 0.0103
— **the floor more than doubles, because near-pure positional heads can no longer
be built.**


---

## Phase 3 — Reasoning Finetuning & Attention Analysis

### Synthetic Reasoning Dataset

A synthetic comparative/transitive reasoning dataset was generated separately per language (deterministic in split, n, and seed). Each model was finetuned from **its own** pretrained checkpoint with **its own** frozen tokenizer.

| Split | Size |
|---|---|
| Train | 60,000 |
| Validation | 6,000 |
| Test | 6,000 |
| Out-of-domain (4-hop) | 2,000 |

The generator is committed and the Drive mirror is bit-identical to what `python <lang>/reasoning/generate.py` produces; `tests/test_reasoning_data.py::test_generation_is_deterministic` asserts this.

### Finetuning Results

Results are means ± half-range over three seeds. The pretrained models score exactly at chance before finetuning, confirming that the reasoning capability is **taught** by SFT rather than merely unlocked. Full report: [`report/phase3/README.md`](report/phase3/README.md).

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| pretrained closed-set (test) | 0.399 | 0.378 |
| **finetuned exact match (test)** | **0.954 ±0.004** | **0.835 ±0.004** |
| finetuned closed-set (test) | 0.956 ±0.003 | 0.869 ±0.009 |
| well-formed answers | 0.996 | 0.957 |
| 4-hop OOD exact match | 0.023 ±0.010 | 0.140 ±0.048 |
| *chance / last-mention baseline* | *0.398 / 0.384* | *0.398 / 0.388* |

### Key Findings

- **Model L is not the data-starved model.** It collected *more* tokens than Model H (903.9M vs 582.7M) and achieves *better* bits-per-byte (0.4314 vs 0.4947). The residual 12-point reasoning gap concentrates in transfer to **unseen relations** (0.518 vs 0.848) and tracks a 2× deficit in in-context copying (induction head score: 0.364 vs 0.680).
- **Both models build the same attention circuit.** At the position that emits the answer, gold-entity attention rises 0.281 → 0.356 (H) and 0.243 → 0.350 (L) against a 0.250 occurrence-weighted baseline — and the strongest head is **L9H4 in both models**, which share no weights, tokenizer, or data.
- **Length generalisation fails for both, and the failure has a shape.** On 4 hops, 118 of 120 sampled Model H predictions name a *middle* entity of the chain (98.3% vs 60.0% for uniform guessing) when the answer is always an extreme: the model learned "answer with a chain entity, picked by position", not "extend the ordering". Chain-of-thought targets are the only intervention that moved this (~5×, at a 9.5-point in-domain cost).

> **Note on corrections.** A left-padding bug in batched decoding invalidated an earlier round of accuracies, and the Nepali finetuning learning rate chosen on those numbers was 10× too low. Both are documented in the report (§5.1, §6.2). The superseded results are kept in the repository rather than deleted — a correction is only checkable against what it replaced. Read `V1_BASELINE_NOTE.md` before quoting anything from `v1` files.

---

## Repository Layout

```
hindi/ , nepali/          one self-contained tree per language; never import across them
  configs/                <lang>.yaml (corpus), model.yaml, train.yaml
  config.py               config loader + independence guard
  data_prep/              scrapers, cleaning, dedup, splits, statistics, shard packing
  tokenizer/              training code + COMMITTED .model / .vocab
  model/                  embeddings, causal MHA, FFN, block, transformer, param count
  train/                  schedule, data loader, checkpointing, trainer, entry point
                          + finetune.py, sft_data.py       (Phase 3 supervised finetuning)
  reasoning/              Phase 3: corpus-mined templates, entity pools, the dataset
                          generator (deterministic in split/n/seed) and leakage verifier
  eval/                   LM metrics (PPL/BPB), generation, gen metrics, attention, plots
                          + reasoning_metrics.py           the single scorer (EM, closed-set,
                                                           heuristics), shared with training
                          + run_reasoning_eval.py          pretrained vs finetuned
                          + run_attention_compare.py       Phase-2 toolkit on both checkpoints
  corpus/ , data/         gitignored artifacts (raw -> clean -> splits -> uint16 shards)
scripts/
  check_independence.py   cross-language gate; holds no language data
  kill_resume_proof.py    SIGKILL-and-resume proof for the checkpoint format
  build_comparison.py     generates report/phase2/comparison_tables.md from the result JSONs
  build_reasoning_stats.py   -> report/phase3/dataset_stats.md
  build_error_analysis.py    -> report/phase3/error_analysis.md + figures
  build_attention_report.py  -> report/phase3/attention_analysis.md
  collect_sft_results.py     pulls Kaggle finetuning logs into report/phase3/sft_*/
  shortcut_audit.py       scores seven trivial rules per pattern as the honest comparator
  induction_probe.py      in-context copying, measured per pretrained checkpoint
  kaggle/                 dataset sync, push/poll/fetch driver, chained multi-session training
report/phase1/            Phase-1 report, statistics, labelled figures
report/phase2/            Phase-2 report, per-model results, figures, training logs
report/phase3/            Phase-3 report + dataset stats, error analysis, attention
                          re-analysis, per-model results and figures
tests/                    68 tests: config, model correctness, trainer, checkpointing,
                          tokenizer, and the reasoning dataset + decoding invariants
```

---

## Reproduction

Requires Python 3.12.

```bash
# Phase 1 (corpus + tokenizer)
pip install sentencepiece numpy matplotlib pyarrow requests beautifulsoup4 lxml pyyaml mwparserfromhell
# Phase 2 (model, training, evaluation) additionally needs
pip install torch sacrebleu
```

```bash
# 1. collect  (polite: robots.txt honoured, rate-limited, resumable)
python hindi/data_prep/wiki_scrape.py
python hindi/data_prep/wikisource_scrape.py
python hindi/data_prep/news_scrape.py --workers 32
python nepali/data_prep/wiki_scrape.py
python nepali/data_prep/news_scrape.py --workers 20

# 2. clean -> dedup -> split   (per language)
python <lang>/data_prep/clean.py
python <lang>/data_prep/near_dedup.py
python <lang>/data_prep/split.py

# 3. tokenizer sweep, then statistics with the chosen model
python <lang>/tokenizer/train_spm.py --vocab-sizes 8000 16000 32000 48000
python <lang>/data_prep/stats.py --tokenizer <lang>/tokenizer/<code>_bpe_16000.model

# 4. independence gate  (run before every phase push)
python scripts/check_independence.py

# 5. tests
python -m pytest tests/ -q

# ---- Phase 2 ----------------------------------------------------------

# 6. pack the splits into uint16 token shards
python <lang>/data_prep/tokenize_corpus.py

# 7. verify the architecture before spending GPU time
python <lang>/model/param_count.py                  # 24,070,656, analytic == module
python scripts/kill_resume_proof.py --tree hindi    # SIGKILL, then bit-identical resume

# 8. pretrain. A Kaggle batch kernel is capped at 12h and one epoch needs
#    14.2h (H) / 23.0h (L), so training is a CHAIN of resumed sessions.
python scripts/kaggle/sync_code.py --user <user> --create
python scripts/kaggle/sync_data.py --user <user> --slug lma-tokens --create
python scripts/kaggle/chain.py --links 2 --steps 13547 --stop-every 6774 \
    --stop-hi 7089 8732 --stop-ne 6764 13547 --data-slug lma-tokens --real
#    Locally on one GPU, a single language is simply:
#    python <lang>/train/loop.py

# 9. evaluate (PPL/BPB, generation, attention, figures)
python <lang>/eval/run_eval.py --checkpoint <checkpoint.pt>
python scripts/build_comparison.py                  # side-by-side H/L tables

# 10. PHASE 3 — reasoning data (deterministic in split/n/seed; not committed)
python <lang>/reasoning/generate.py --split train --n 60000   # + val/test/ood
python scripts/build_reasoning_stats.py             # -> report/phase3/dataset_stats.md
python <lang>/reasoning/verify_leakage.py           # the three structural holdouts

# 11. finetune (Kaggle, 2x T4, ~8 min/run; MODE is a literal in run_sft.py)
python scripts/kaggle/run.py --kernel-dir scripts/kaggle/sft_kernel --no-params \
    --out kaggle_v2_out

# 12. evaluate reasoning + re-run the attention toolkit on the finetuned models
python <lang>/eval/run_reasoning_eval.py --pretrained <pre.pt> \
    --finetuned "<sft-glob>/step_*.pt" --splits test,ood
python <lang>/eval/run_attention_compare.py --pretrained <pre.pt> \
    --finetuned "<sft-glob>/step_*.pt"              # run LOCALLY: Devanagari fonts
python scripts/build_error_analysis.py              # -> report/phase3/error_analysis.md
python scripts/build_attention_report.py            # -> report/phase3/attention_analysis.md
```

Seed **1337** throughout. Split assignment is a hash of each document's own content, so the
same document lands in the same split on any machine, and extending the corpus does not
reshuffle the existing split. Every stage writes its resolved config beside its outputs.

---

## Datasets & Artifacts

**All large artifacts (corpora, checkpoints, tokenized shards):**
[Google Drive](https://drive.google.com/drive/folders/1KVpR6f12MbJikW7-BhSbV_-Xy3lIVstb) — viewer access, no request needed.

```
LMA-Individual-Project/
├── hindi/
│   ├── raw/              scraped text (wikipedia, wikisource, livehindustan)
│   ├── raw_downloaded/   Sangraha verified/hin, 5 parquet shards
│   ├── splits/           train / val / test + ID manifests
│   ├── tokenizer/        BPE .model + .vocab, all 4 vocab sizes
│   ├── tokenized/        packed uint16 token shards (train/val/test .bin)
│   ├── checkpoints/      step_8732.pt  — Model H, 1 epoch (Phase 2)
│   ├── checkpoints_sft/  Model H finetuned for reasoning (Phase 3), 3 seeds
│   │                     hindi_lr3e-05_ans_mask_s{1337,42,7}_*.pt
│   ├── reasoning_data/   synthetic reasoning finetune set (Phase 3)
│   │                     train 60k / val 6k / test 6k / ood 2k
│   └── manifest.jsonl    per-document provenance
└── nepali/
    ├── raw/              scraped text (wikipedia, onlinekhabar, khabarhub, nepalpress)
    ├── raw_downloaded/   Sangraha verified/nep, 10 parquet shards
    ├── clean/            cleaned corpus before splitting
    ├── splits/           train / val / test + ID manifests
    ├── tokenizer/        BPE .model + .vocab, all 4 vocab sizes
    ├── tokenized/        packed uint16 token shards (train/val/test .bin)
    ├── checkpoints/      step_13547.pt — Model L, 1 epoch (Phase 2)
    │                     step_8732.pt  — Model L at Hindi's token budget
    ├── checkpoints_nope/ step_13547.pt — Model L, no positional embeddings (bonus)
    │                     step_8732.pt  — the same, at Hindi's token budget
    ├── checkpoints_sft/  Model L finetuned for reasoning (Phase 3), 3 seeds
    │                     nepali_lr3e-05_ans_mask_s{1337,42,7}_*.pt
    ├── reasoning_data/   synthetic reasoning finetune set (Phase 3)
    │                     train 60k / val 6k / test 6k / ood 2k
    └── manifest.jsonl    per-document provenance
```

Every upload was verified by MD5 against the local file, not by filename or size.

**The finetuned checkpoints are the `lr3e-05` runs for both languages.** An earlier Nepali set trained at `lr3e-06` is *not* uploaded: it was superseded when the learning rate was re-selected (see [`report/phase3/README.md`](report/phase3/README.md) §6.2), and its results are kept in-repo as `reasoning_eval_v2_lr3e-06.json`.

**The reasoning datasets are mirrored, not archived.** They are regenerated exactly by `python <lang>/reasoning/generate.py --split train --n 60000` from the committed generator and its seed, and `tests/test_reasoning_data.py::test_generation_is_deterministic` asserts byte-identical reproduction. The Drive copy exists for convenience without running the generator locally.

**Not uploaded, by design:** the Sangraha shards are a public HuggingFace dataset (`ai4bharat/sangraha`, configs `verified/hin` and `verified/nep`), reproducible from their shard numbers.

---

## Reading the Results

**Perplexity is not comparable between the two models.** The tokenizers are
separate and differ in fertility (4.644 vs 3.892 characters per token over the
whole corpus; 4.760 vs 3.854 on the held-out vocab-selection sample), so
perplexity is denominated in different units per model. Cross-language claims in
this project rest on **bits-per-byte**, which normalises by bytes of text. BPB
reverses the perplexity ordering; `report/phase2/README.md` §10 decomposes the
causes.

**Model L is evaluated at two token budgets.** In addition to its full 887.9M
tokens, a checkpoint is taken at 572.3M — Model H's exact training-set size — so
data volume can be separated from language and corpus effects. The difference
between them prices the extra 316M tokens at ≈0.010 BPB.

**Model H is reported before and after hyperparameter tuning.**
`report/phase2/hindi/` holds the final model; `report/phase2/hindi_baseline/`
holds the earlier run. Architecture, parameter count, context length, step count
and token budget are identical between them.

**Corpus artifacts.** Split manifests (`*_ids.txt`) are committed and reproduce
the exact train/val/test partition without shipping the corpora. All four
tokenizer vocabulary sizes from the Phase-1 sweep are committed, not only the
chosen one.

**Phase-3 numbers include corrections, and the superseded ones are kept.** A
left-padding bug in batched decoding invalidated an earlier round of accuracies,
and the Nepali finetuning learning rate that had been chosen on those numbers was
10x too low. Both are documented in `report/phase3/README.md` §5.1 and §6.2. The
replaced results stay in the repo — `<tree>/reasoning_eval_v1.json`,
`nepali/*_lr3e-06.json`, `sft_baseline_v1/` — because a correction is only
checkable against what it replaced. Read `V1_BASELINE_NOTE.md` before quoting
anything from the v1 files: none of those accuracies are valid.

**Reasoning results are reported as three numbers, never one.** Exact match
(greedy), closed-set scoring, and heuristic baselines (first-mention,
last-mention, majority). Closed-set scoring is what gives the *pretrained* model
a fair test, since free generation scores ~0 exact match regardless of what it
knows; the heuristics catch a score that is really a positional shortcut.

**Limitations** are recorded in Phase 1 §11, Phase 2 §8, and Phase 3 §7.2
(length generalisation fails for both models) and §6.3 (the chain-of-thought
trade-off, documented rather than taken).
