# Phase 1 — Data Collection & Tokenizer Construction

**Model H = Hindi (`hi`) · Model L = Nepali (`ne`)** · both Devanagari · branch `phase-1`

Two independent corpora collected, cleaned and split, and one BPE tokenizer trained
per language from scratch. Every number is measured from the artifacts in this
repository — token counts come from each language's own trained tokenizer, never an
estimate.

## Results

**Corpus**

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| documents | 1,202,072 | 2,180,424 |
| words | 440,575,921 | 652,216,805 |
| **tokens** (own tokenizer) | **582,711,232** | **903,937,802** |
| manual share of tokens | 19.29% | **22.40%** |
| near-duplicates removed | 15,503 (1.27%) | 120,689 (5.24%) |
| train / val / test | 98.00 / 0.99 / 1.00 % | 97.99 / 1.01 / 1.00 % |

**Tokenizer** — SentencePiece BPE, trained separately per language, vocabularies never shared

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| vocabulary size | 16,000 | 16,000 |
| fertility (tokens/word) | 1.386 | **1.360** |
| avg chars/token | 3.892 | **4.644** |
| **UNK rate** | **0.000000** | **0.000000** |
| distinct tokens observed | 15,788 | 15,798 |

Three things to take from this:

- Both corpora clear the ~500M training-token target. Hindi sits at **19.29%** manual
  against a 20% floor; §2 explains why that is reported honestly rather than engineered.
- **Nepali encodes more efficiently than Hindi** at the same vocabulary size — fewer
  tokens per word and more characters per token, despite being the lower-resource
  language. Real, not an artifact (§7).
- Nepali's near-duplicate rate is **4× Hindi's**, and the cause is structural: its manual
  tier is news sites that syndicate each other's copy (§5).

Vocabulary was chosen on parameter arithmetic, not convention: at a 25M budget a 48k
vocab spends 74% of the model on the embedding matrix to buy ~9–11% better fertility
than 16k (§7).

---

## 1. Language selection

| Model | Language | Tier | Justification |
|---|---|---|---|
| **H** | Hindi | higher-resource | Largest public text pool of any non-English Indian language (Sangraha, IndicCorp, OSCAR, Wikipedia, dense news web). 500M tokens is comfortably reachable, making Model H a clean resource-rich control. |
| **L** | Nepali | lower-resource | On the spec's allowed list (§1.1). Devanagari, like Hindi. |

**Why this pair.** Both languages use the same script and belong to the same broad
family, so script and tokenizer-family effects are held roughly constant. The H-vs-L gap
that Phase 3 must explain is therefore attributable mainly to **corpus scale and quality**
rather than orthography — a sharper comparison than e.g. Devanagari vs Latin would give.

**Why it is also the main risk.** Shared script means a language-ID error is simultaneously
a contamination of the independence constraint. §5 reports the measured rates.

---

## 2. Corpus summary

| | Hindi (H) | Nepali (L) |
|---|---|---|
| documents (final) | 1,202,072 | 2,180,424 |
| characters | 2,267,898,230 | 4,197,850,537 |
| words | 440,575,921 | 652,216,805 |
| **tokens** (own tokenizer) | **582,711,232** | **903,937,802** |
| manual share (tokens) | 19.29% | **22.40%** |
| avg chars / token | 3.892 | 4.644 |

Both exceed the spec's ~500M training-token target.

### On Hindi's 19.29%

The spec requires ≥20% of final training tokens from manual collection. Nepali clears this
outright at 22.40%. Hindi reads 19.29%, and the reason is deliberate over-collection:

- Manual sources were collected **to exhaustion** — all 174,691 `hi.wikipedia` articles,
  all 4,634 usable `hi.wikisource` pages, 288,733 `livehindustan` articles.
- **82.7M tokens of surplus** above the 500M target were collected as a cleaning buffer,
  and that surplus is **entirely downloaded-tier** (Sangraha).
- The same manual token count against a larger denominator gives a smaller percentage.

Against the 500M target the model actually trains on, manual is **22.48%** — *if* the
training set is built manual-first (keep every manual token, top up from downloaded).
That conditional is stated explicitly rather than quoted bare, because under proportional
sampling the ratio is unchanged at 19.29%. Both figures, and the assumption attached to
the first, are recorded in `corpus_stats.json → target_analysis`.

**Downloaded documents were not trimmed to raise the headline above 20%.** Discarding
usable data to change a reported percentage would misrepresent the corpus that was
actually collected.

---

## 3. Sources

### Hindi

| source | bucket | documents | tokens |
|---|---|---|---|
| Sangraha `verified/hin` (5 shards) | downloaded | 842,071 | 470,299,892 |
| `hi.wikipedia.org` | manual | 93,970 | 62,944,409 |
| `livehindustan.com` | manual | 261,942 | 35,699,204 |
| `hi.wikisource.org` | manual | 4,089 | 13,767,727 |

### Nepali

| source | bucket | documents | tokens |
|---|---|---|---|
| Sangraha `verified/nep` (10 shards) | downloaded | 1,655,996 | 701,459,522 |
| `onlinekhabar.com` | manual | 230,135 | 104,459,501 |
| `khabarhub.com` | manual | 214,574 | 69,556,635 |
| `nepalpress.com` | manual | 56,418 | 18,265,441 |
| `ne.wikipedia.org` | manual | 23,301 | 10,196,703 |

All manual text was gathered by our own scrapers (`data_prep/*_scrape.py`), robots.txt
checked per site and rate-limited. Sites rejected during survey, with reasons, are
documented in each `news_scrape.py` module docstring — 6 of 8 Nepali candidates and
several Hindi candidates failed on direct content verification rather than a status code.

**Collection ethics.** `robots.txt` was fetched and parsed directly for every site
(`robotparser.read()` gave false negatives on 3 sites). One site — `annapurnapost.com` —
was dropped because it sits behind a Cloudflare interactive challenge, making its crawl
policy unreadable; bypassing that would mean evading an access control. `nepalpress.com`
began returning HTTP 403 mid-scrape and was stopped immediately rather than retried.

---

## 4. Cleaning pipeline

Modelled on the **Yi pipeline** (arXiv 2403.04652 Fig. 1) as presented in course lecture
L3. Implemented in `data_prep/clean.py`, duplicated per language.

| Yi stage | Our implementation | Status |
|---|---|---|
| Language filtering | Devanagari script ratio + Hindi↔Nepali function-word discriminator | ✅ |
| Text metric filtering | length, symbol ratio, short-line ratio | ✅ |
| Repetitive document removal | repeated-line ratio + repeated 10-gram ratio | ✅ |
| Rule-based correction | strip See-also/References navigation tails | ✅ |
| Perplexity filtering | — needs a trained LM (Phase 2) | skipped, documented |
| Exact dedup | SHA-256 over normalised text | ✅ |
| MinHash dedup | k=100, b=20, r=5, s\*=0.549 | ✅ |
| Semantic / topic / safety | — needs trained classifiers; corpus is curated, not raw crawl | skipped, documented |

Unicode is normalised to **NFC** first, since Devanagari can encode the same grapheme two
ways and the tokenizer would otherwise learn two inventories for identical text.

### Drop counts

| rule | Hindi | Nepali |
|---|---|---|
| `too_short_chars` | 75,285 | 8,068 |
| `exact_duplicate` | 19,576 | 6,076 |
| `low_script_ratio` | 10,286 | 1,300 |
| `mostly_short_lines` | 9,695 | 855 |
| `empty` | 6,545 | 884 |
| `high_symbol_ratio` | 2,573 | 480 |
| `wrong_language` | 11 (ne in hi) | 402 (hi in ne) |
| `repetitive_lines` | 232 | 106 |
| `repetitive_ngrams` | 90 | 57 |
| **raw → cleaned** | 1,341,872 → 1,217,575 | 2,319,342 → 2,301,113 |

**Rule-based correction was added because of a measured false positive**, not as a
precaution. Without it, `short_line_ratio` discarded ~30% of Hindi Wikipedia: real stub
articles whose trailing see-also link lists dragged the ratio over threshold and took the
prose with them. Following L3 slide 30 ("repair what you can rather than discarding it"),
the apparatus is cut and the remaining prose re-judged. A stub too thin to survive then
fails `min_chars` — dropped for the right reason.

**Threshold set differently for the low-resource side.** `max_short_line_ratio` is 0.80 for
Hindi but **0.92 for Nepali**. On Hindi that threshold cost ~16% of Wikipedia including
list-shaped but genuine articles; affordable on the resource-rich side, not on the
constrained one. This is the one filter deliberately tuned per language.

---

## 5. Deduplication

`data_prep/near_dedup.py` — 3-word shingles → MinHash (k=100) → LSH (b=20, r=5).
Threshold s\* ≈ (1/b)^(1/r) = **0.549**; Pr[candidate] = 0.9996 at s=0.8, 0.047 at s=0.3.

| | Hindi | Nepali |
|---|---|---|
| in → out | 1,217,575 → 1,202,072 | 2,301,113 → 2,180,424 |
| near-duplicates removed | 15,503 (1.27%) | 120,689 (5.24%) |
| candidate groups verified | 97,596 | 916,181 |
| candidate pairs rejected below threshold | 15,236 | 62,820 |
| median true Jaccard of removals | 0.803 | 0.847 |

**The 4× gap is structural, not a bug.** Nepali's manual tier is three news sites that
syndicate each other's wire copy — exactly the "same story, different headline" case.
Hindi's manual tier is Wikipedia + Wikisource + one news site, which overlaps far less.

**LSH candidates are verified against true Jaccard before removal.** Bucket collision is
not sufficient evidence of duplication: 61.7% of flagged pairs fall below the threshold,
and removing them unchecked would discard ~115,000 legitimate documents. The verification
pass re-signs only documents appearing in a candidate group and compares signatures.

---

## 6. Splits

Document-level, never sentence-level: splitting sentences would let the model memorise
validation text during training.

| split | Hindi | Nepali |
|---|---|---|
| train | 1,178,056 (98.00%) | 2,136,505 (97.99%) |
| val | 11,960 (0.99%) | 22,014 (1.01%) |
| test | 12,056 (1.00%) | 21,905 (1.00%) |
| manual share of train (words) | 18.59% | 23.09% |

The train-split manual share is computed on a **word** basis, because the split
statistics are counted before tokenization; §2's corpus-level share is on a
**token** basis. The two are close but not the same measurement.

Assignment is a **hash of the document's own content** (seed 1337), so the same document
lands in the same split on any machine and adding data does not reshuffle the existing
split. ID manifests (`*_ids.txt`) are committed and reproduce the split exactly without
shipping the corpus.

**Decontamination.** L3 slide 24 says hold out the test set before cleaning. We run
`clean → dedup → split` instead, because near-dedup is precisely the machinery that finds
near-duplicates; running it across the whole corpus before splitting guarantees no
near-duplicate of a validation document survives in train. The property is then **verified
**: `verify_no_leakage` re-checks train against the held-out splits.

| | Hindi | Nepali |
|---|---|---|
| train docs checked | 1,178,056 | 2,136,505 |
| leaks found | **15** (0.0013%) | **26** (0.0012%) |

Residual leaks arise because dedup compares each group member against the group
representative only, so a pair resembling each other but not the representative can
survive into different splits. The offending line numbers are written to
`train_leaked_lines.txt` for exclusion rather than left as an unverified claim.

---

## 7. Tokenizers

**SentencePiece BPE**, trained separately per language, vocabularies never shared.
Byte fallback enabled. Trained on the **train split only** — fitting on held-out text
would leak evaluation data into the vocabulary.

### Vocabulary sweep

**Hindi**

| vocab | fertility (tok/word) | chars/token | UNK rate |
|---|---|---|---|
| 8,000 | 1.526 | 3.501 | 0.000000 |
| **16,000** | **1.386** | **3.854** | **0.000000** |
| 32,000 | 1.296 | 4.123 | 0.000000 |
| 48,000 | 1.258 | 4.247 | 0.000000 |

**Nepali**

| vocab | fertility (tok/word) | chars/token | UNK rate |
|---|---|---|---|
| 8,000 | 1.529 | 4.235 | 0.000000 |
| **16,000** | **1.360** | **4.760** | **0.000000** |
| 32,000 | 1.252 | 5.171 | 0.000000 |
| 48,000 | 1.208 | 5.359 | 0.000000 |

### Chosen: 16,000 for both — on parameter arithmetic

Fertility falls monotonically with vocab size, but the embedding matrix grows linearly.
Against a ~25M trainable-parameter budget with weight tying, at d_model = 384:

| vocab | embedding params | % of 25M budget | fertility gain vs 8k |
|---|---|---|---|
| 8,000 | 3.1M | 12.3% | — |
| **16,000** | **6.1M** | **24.6%** | 9.1% |
| 32,000 | 12.3M | 49.2% | 15.1% |
| 48,000 | 18.4M | 73.7% | 17.6% |

48k would spend **74% of the entire model** on the embedding matrix (98% at d_model=512)
to buy 9.2% better fertility than 16k. 16,000 is also within the spec's recommended
"tens of thousands", where 8,000 is not.

### Token statistics (at the chosen vocab)

| | Hindi | Nepali |
|---|---|---|
| distinct tokens observed | 15,788 | 15,798 |
| top-100 coverage | 38.34% | 29.47% |
| top-1,000 coverage | 65.49% | 62.03% |
| top-10,000 coverage | 97.50% | 97.68% |
| avg chars / token | 3.892 | 4.644 |
| UNK rate | 0.000000 | 0.000000 |

Two chars/token figures appear in this report and they measure different text:
**3.892 / 4.644** are over the whole corpus (total characters ÷ total tokens),
while the sweep table's **3.854 / 4.760** are measured on the held-out sample
used to choose the vocabulary. Both are correct on their own basis.

Most frequent pieces — Hindi: `▁के ▁में ▁है ▁की ▁और ▁से` (postpositions and copula);
Nepali: `▁र को ▁थियो हरू मा` (conjunction, genitive, past copula, plural marker).
Both are function-word inventories, which is what a healthy BPE vocabulary should surface.

**UNK rate is 0.000000 for every vocab size** thanks to byte fallback — any unseen byte
still round-trips rather than collapsing to UNK.

**Round-trip fidelity.** A character-multiset diff over 3,000 held-out documents found
**zero characters lost**. Decoded text differs from input only through SentencePiece's own
normaliser: newline→space, NFKC compatibility mappings (`³`→`3`), and ZWJ/ZWNJ→space.
The last affects ~1% of documents and is a real trade-off: `clean.py` deliberately
preserves ZWJ/ZWNJ (they control Devanagari conjunct formation) while `nmt_nfkc` strips
them. Reported rather than silently accepted.

### A finding worth noting

**Nepali encodes more efficiently than Hindi** — 4.760 vs 3.854 chars/token at identical
vocab size — despite being the lower-resource language. Nepali also needs fewer tokens per
word (1.360 vs 1.386). This is real, not an artifact, and is a candidate explanation to
carry into Phase 3's resource-tier comparison.

---

## 8. Independence

`scripts/check_independence.py` — the only script permitted to see both trees, and it
holds no language data of its own. **All five checks pass** (`independence_check.json`):

| check | result | evidence |
|---|---|---|
| no cross-tree imports | PASS | 0 modules reference a sibling tree |
| config isolation (paths cannot escape their tree) | PASS | lang_codes distinct, 0 path problems |
| no shared documents (SHA-256) | PASS | 49,016 hi hashes vs 68,919 ne docs, **0 overlaps** |
| no cross-corpus near-duplicates (MinHash) | PASS | 49,016 indexed, 68,919 compared, **0 pairs** at s\*=0.549 |
| separate tokenizer models and vocabularies | PASS | distinct SHA for both `.model` and `.vocab` |

The document checks run on a bounded sample (`--limit 25000` per split file, so the
whole of val and test plus the first 25,000 train documents per language). That
bound is stated explicitly: the exhaustive within-corpus dedup in §5
already covers each corpus in full, and this check exists to test the *cross*-corpus
property.

**Language filtering** deserves special mention because Hindi and Nepali share Devanagari,
which L3 slide 27 names as the headline Indic failure mode. Marker function words were
derived by **log-odds over 10,000 documents per language**, not intuited. Two candidates
were removed after they misfired: `को` is a Hindi accusative *and* an ordinary Nepali
genitive, and `हो` is common in both. Validated on held-out slices:

| corpus | correct | cross-language error |
|---|---|---|
| hi.wikipedia | 98.45% | 0.025% |
| ne.wikipedia | 85.88% | 0.150% |
| livehindustan | 96.88% | 0.000% |
| nepalpress | 98.95% | 0.000% |

All 6 `ne.wikipedia` flags were read by hand: **2 were genuine Hindi pages** sitting in
ne.wikipedia (correctly removed), 4 were false positives on Sanskrit-heavy religious prose.
The filter found 402 Hindi documents inside the Nepali corpus and 11 Nepali documents
inside the Hindi corpus.

---

## 9. Figures

Per language in `report/phase1/<lang>/`:

| figure | shows |
|---|---|
| `sources.png` | corpus contribution per source, coloured by manual/downloaded |
| `manual_share.png` | manual share on each basis, with the 20% floor drawn |
| `doc_lengths.png` | document length distribution after cleaning |
| `cleaning_drops.png` | documents removed by each cleaning rule (log scale) |
| `vocab_sweep.png` | fertility and UNK rate against vocabulary size |

The plotting helper **raises** if a figure lacks title, x-label or y-label, so an
unlabelled figure cannot be saved even by accident.

---

## 10. Reproduction

```bash
python <lang>/data_prep/wiki_scrape.py        # + wikisource_scrape.py, news_scrape.py
python <lang>/data_prep/clean.py              # -> corpus/clean/clean.jsonl
python <lang>/data_prep/near_dedup.py         # -> corpus/dedup/duplicate_indices.txt
python <lang>/data_prep/split.py              # -> corpus/splits/ + leakage check
python <lang>/tokenizer/train_spm.py          # -> tokenizer/*.model, *.vocab
python <lang>/data_prep/stats.py --tokenizer <lang>/tokenizer/<code>_bpe_16000.model
python scripts/check_independence.py
```

Every stage is resumable and writes its resolved config next to its outputs.
Seed 1337 throughout.

---

## 11. Honest notes

- **Hindi manual share is 19.29%**, below the 20% floor when measured across the full
  over-collected corpus. §2 explains the cause; downloaded documents were not trimmed to
  raise the percentage.
- **41 documents** (15 Hindi + 26 Nepali) remain in train that are near-duplicates of
  held-out text — 0.001% of training data. Line numbers are recorded for exclusion.
- **ZWJ/ZWNJ are stripped by the tokenizer's normaliser** on ~1% of documents, contradicting
  `clean.py`'s deliberate preservation of them. Alternatives exist (`user_defined_symbols`)
  at the cost of vocabulary slots; left as-is and reported.
- **Perplexity and semantic/topic/safety filtering are not implemented.** Both need models
  we do not have at Phase 1; the omission is deliberate and marked in the pipeline table.
