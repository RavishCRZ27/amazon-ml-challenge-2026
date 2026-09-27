# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TheAnarchy
**Team Members:** Ravish Pandey ([@RavishCRZ27](https://github.com/RavishCRZ27)), Samar Nathani ([@SammySN-car](https://github.com/SammySN-car)), Ayush Rajdeep ([@rajdeep3456](https://github.com/rajdeep3456))
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

The pipeline has four stages:
1. Per-country, IDF-weighted sparse retrieval generates about 50 candidates per record
   (97.2% of true pairs retrieved).
2. A 53-feature XGBoost model scores every candidate pair. Its features include
   *reverse-competition* features: how a pool record ranks this S1 among all S1s that
   retrieved it.
3. A multilingual cross-encoder (bge-reranker-v2-m3, Apache-2.0, fine-tuned on training
   pairs) re-scores only the pairs the GBM is unsure about.
4. The two scores are combined, and a second-stage LightGBM re-scores every pair using the S1's
   context and sibling evidence.
5. The threshold is chosen by exact macro-F0.5 on held-out training records, followed by a
   one-to-one pass on the Source-2/3 side.

On 100K held-out training records that were never used for any choice (HOLDOUT), the result
is **F0.5 0.9800**, and it scored **0.968** on the public leaderboard. Submission 1 (a smaller
cross-encoder and no stage 2, HOLDOUT 0.9764) scored 0.965. The gap between HOLDOUT and the
leaderboard is France, which has no labels: its implied F0.5 is about 0.91 (Section 5).

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale.**
- Train: 2.21M S1 records and a pool of 10.3M S2/S3 records.
- Test: 1.73M S1 and 9.97M pool records.
- 7.64M true pairs, a mean of 3.46 matches per S1 (max 11), and 5.58% singletons.

**Every match is within the same country, and each pool record belongs to at most one S1**
(checked: 0 violations). That justifies per-country blocking and a one-to-one pass on the
pool side.

**Names are noisy in every way.**
- Legal suffixes (LLC / PVT LTD / SARL / S.A.R.L.), word order, typos, digit-for-letter,
  domains and hashtags, and renames ("X formerly: Y").
- 18–28% of Indian pool names are in Indic scripts.
- 17% of Indian pool records (10% US) share their exact normalized name with more than 50
  other records (chains, generic names). Name-only matching is therefore unreliable.

**Addresses.**
- Truncation is common, so containment works better than Jaccard.
- A shared house number is strong evidence. Postal codes are mostly absent, and leading
  zeros vary. About 3% of pool addresses are empty.

**France appears only in test** (15% of test S1s), so there are no French labels.
- Its data is templated: "{City} {Word} {legal form}" names, legal-form variants
  ("S.A.R.L.", "5AS", "[SAS]"), number formats ("N° 23", "023"), street abbreviations
  ("R.", "Q.", "Rte.").
- S1 addresses end with the region ("Hauts-de-France"), while pool records often use the
  department ("Nord").
- The pipeline therefore treats `country` as an open set and never uses it as a feature. It
  only partitions by it.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (sparse retrieval), then a GBM classifier, then a
fine-tuned cross-encoder on the uncertain band, then a calibrated combiner, then a stage-2
GBM re-scorer, an exact-metric threshold and a one-to-one pass.

**Core Innovation:**
1. Reverse-competition features. For each candidate pair we compute how the pool record ranks
   this S1 among **all** S1s that retrieved it. Train blocks all 2.2M train S1s, so these
   features see the same density as test. They are the two most important features, 7× the
   next one, and add +0.0030 F0.5 on their own.
2. A multilingual cross-encoder, used only where the GBM is unsure (stage-1 p in
   [0.002, 0.998], ≈ 10% of pairs). It is fine-tuned on training pairs from that band and
   combined with the GBM through a logistic combiner fit on validation. It generalizes to
   unseen names: AUC 0.994 on validation pairs whose pool names never appeared in training.
3. The threshold is chosen by the **exact** challenge metric: per-S1 macro-F0.5, singletons
   included, recall against the full ground truth, so blocking misses count. It is the centre
   of the plateau of the F curve, not a spiky argmax.

**Honest evaluation.**
- Train S1s are split into TRAIN (180K), early stopping (20K), VAL (100K, all choices) and
  HOLDOUT (100K, never used for any choice), stratified by country.
- Every number carries a standard error, and variants are compared by paired SE on the same
  S1s.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Per-country IDF-weighted sparse vectors, retrieving the top-k by
cosine with `sparse_dot_topn`:
- **W_all:** word tokens of the normalized name and address, top 40.
- **C_all:** character 3-grams of a consonant skeleton of the name, plus address tokens,
  top 40. It is robust to typos and transliteration variants.

**Normalization (S0).**
- Custom Indic → Latin transliteration and accent folding.
- Legal-suffix removal (US/India/France forms).
- Street-type and US/Indian state codes.
- House-number extraction without leading zeros.

**Candidate order and cap.** The union of the two views is ordered by best view rank and
capped at **50 per S1**, with up to 80 stored for analysis.

**Candidate pairs generated.** Test: **85,986,019** (49.6 per S1). Train (400K-S1
modelling scope): 19.9M.

**How we made sure true matches were not lost.** From the blocking study on 50K VAL S1s:

| Views | Recall |
|---|---|
| Name-only + address-only + name-3-gram views, 20 each | .934 |
| W_all at 50 | .961 |
| **W_all 40 + C_all 40 (chosen)** | **.972** (US .985, India .952) |

- Per-country partitions stop common tokens of one country from crowding out another's.
- The skeleton 3-gram view recovers typo, transliteration and domain-style names that the
  word view misses.
- Blocking-oracle macro-F0.5 (predicting exactly the true pairs among the candidates): 0.990.

---

## 4. Matching Model

**Features used** (53, float32, missing = NaN):
- **Name:** exact and skeleton-exact match, first token, length difference, script codes and a
  cross-script flag. rapidfuzz ratio, token-sort, token-set, partial and Jaro-Winkler on the
  suffix-stripped name and on its skeleton. Space-less ratio and partial. IDF-weighted name
  containment. Name frequency on each side.
- **Address:** token containment (shared / min size), Jaccard, IDF-weighted containment and
  shared weight, token-set, partial, last-two-token overlap, missing-address flags.
- **Numbers:** shared house numbers, conflict, longest shared, first-number match.
- **Blocking and context:** per-view score, rank and gap to best; number of views; number of
  candidates; candidate position. Rank and gap of name/address similarity within the S1's
  candidates. **Reverse competition:** rank and gap of this S1 among all S1s that retrieved
  the pool record. Source flag (S2/S3). Never `country`.

**Model type.**
- **Stage 1:** XGBoost on GPU (depth 8, lr 0.05, early stopping on a separate 20K-S1 slice;
  1,994 trees). It was benchmarked against LightGBM: equal F0.5, XGBoost 3× faster.
- **Stage 2:** cross-encoder `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M parameters, local
  weights).
  - Fine-tuned for 1 epoch (BCE, lr 2e-5, batch 64) on 300K TRAIN pairs: 195K pairs with
    stage-1 p in [0.05, 0.95] plus random positives and negatives.
  - Input: the raw "name | address" of both records, in their original script.
  - At inference it scores pairs with stage-1 p in [0.002, 0.998].
- **Combiner:** logistic regression on [logit p, cross-encoder logit, their product], fit on
  VAL band pairs. It is cross-fitted over 2 VAL folds when choosing the threshold, and refit
  on all of VAL for test.
- **Stage 2** (LightGBM, CPU): re-scores every candidate pair.
  - Inputs: the combined probability, the S1's context over its candidates (rank, max, 2nd
    best, count above threshold, sum, gap), sibling evidence against the S1's best other
    candidate (its probability, name and address similarity, same address), counts of
    candidates sharing the exact address or name, and the 53 stage-1 features.
  - Fit on VAL, cross-fitted over the same 2 folds for the threshold choice, then refit on all
    of VAL (181 trees).

**Threshold selection method.**
- Exact macro-F0.5 on VAL, sweeping t (0.02–0.98) and a relative cut r·max-p within the S1.
- Chosen: **t = 0.71, r = 0**, the centre of the plateau (on the stage-2 score: [0.55, 0.87]).
- Then a **one-to-one pass on the pool side**: each S2/S3 record keeps only its best S1 (the
  ground truth is injective), while S1s keep any number of matches.

---

## 5. Results & Error Analysis

**F_0.5 score (macro, all S1s, singletons included, recall against the full ground truth):**

| System | VAL (choices) | HOLDOUT (report only) | India / US (HOLDOUT) |
|---|---|---|---|
| Stage-1 XGBoost | 0.9484 ± .0005 | 0.9484 | 0.931 / 0.960 |
| + e5-small cross-encoder, band [.01,.99] (submission 1, **public LB 0.965**) | 0.9768 ± .0003 | 0.9764 | 0.968 / 0.982 |
| + bge-reranker-v2-m3, band [.002,.998] | 0.9796 ± .0003 | 0.9794 ± .0003 | 0.972 / 0.985 |
| + stage-2 re-scorer (**final**, **public LB 0.968**) | **0.9800 ± .0003** | **0.9800 ± .0003** | **0.972 / 0.985** |

The final system beats submission 1 on HOLDOUT by **+0.0036 ± 0.0002** (paired; India +0.0046,
US +0.0029).

**What is left to gain** (HOLDOUT, 1 − F = 0.0206):

| Part | Loss | What it is |
|---|---|---|
| True match never retrieved | **0.0104** | Half the remaining loss. Typos or transliteration in the name together with a weak address, and renamed businesses. |
| True match retrieved but missed | 0.0069 | Hardest pairs. |
| False positives on matched S1s | 0.0025 | Wrong merges. |
| Predictions on singletons | 0.0009 | Emit rate 1.6%. |

**Common false positives (wrong merges):**
- Different businesses at the same address, or with near-identical names: branches of a chain,
  "{City} Club" vs "{City} Comité" in the same street.
- Records sharing a generic name but with an empty address.
- The larger cross-encoder removes many of these. On HOLDOUT, 80% of the pairs it drops with a
  different name but the same house number are false matches.

**Common false negatives (missed matches):**
- Mostly blocking misses: the name changed beyond token or 3-gram overlap (renames,
  transliteration into another script) and the address is weak or missing.
- Also exact-name pairs where the pool address is empty. On HOLDOUT only about 10% of those
  are true, so they are usually and correctly rejected.

**France (no labels).**
- The public score is a country-weighted mean (test shares: India .468, US .383, France .150).
  With India and US taken from HOLDOUT, the implied F_France is ≈ 0.912 for submission 1 and
  ≈ 0.911 for the final system (± 0.003 from the 3-decimal leaderboard). France gained nothing
  measurable from the larger cross-encoder and stage 2, while India and US gained 0.005 and 0.003.
- We also tried a French address normalization: regions and their departments mapped to one
  code, "N°"/"NUMERO" prefixes dropped, French street abbreviations expanded. It aligned S1 and
  pool addresses: on each French S1's best candidate, the last two address tokens agreed in 90% of
  cases, up from 41%, and address containment rose from 69% to 97%. HOLDOUT was unchanged, and
  without French labels there was no evidence of a France gain, so it was not adopted.
- Label-free checks point to hard pairs and to "same address, different business" false
  positives as France's main loss. Better-aligned addresses do not separate those pairs.
- The final cross-encoder changes 17% of French S1s' predictions (vs 3–4% elsewhere), mostly
  by dropping pairs of the type whose drops are 80% correct on HOLDOUT.

**Test sanity** (predicted matches per S1 / empty-prediction rate): France 3.01 / 6.9%,
India 3.23 / 6.4%, US 3.37 / 5.8%. Train ground truth: 3.46 / 5.6%.

---

## 6. Conclusion

- Retrieval at 97% recall, a reverse-competition-aware GBM, and a multilingual cross-encoder
  confined to the GBM's uncertain band, plus a stage-2 re-scorer, reach **0.980 macro-F0.5** on
  untouched held-out data and **0.968** on the public leaderboard. Every decision was made on
  validation with standard errors.
- France, the unseen country, is at about 0.91, and the stronger cross-encoder did not move it.
  Its errors are most likely hard same-address and look-alike pairs.
- The biggest lessons:
  1. Measure the exact metric, singletons and blocking misses included.
  2. Spend the expensive model only where the cheap one is unsure.
  3. Treat the unseen country as an open set, checking it with label-free diagnostics.
- The remaining loss is dominated by pairs retrieval never surfaces. That is the next thing
  to improve.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: pipeline stages S0–S6 plus `common.py`, `translit.py`, `threshold.py`.
- `config.yaml`: every setting.
- `run_all.sh`: runs the stages with checkpoint and resume.
- `preflight.sh`: environment, data and secret checks.
- `README.md`: exact commands.
- `requirements.txt`: pinned versions.

Reproduce:
```
bash preflight.sh
bash run_all.sh --force all
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the
official validator. Wall time is about 5 h on 32 vCPU + 1× L4. Most of it is the
cross-encoder: about 34 min of training plus about 2.6 h scoring ≈ 8.3M uncertain test pairs.

### B. Additional Results

**Experiments** (paired ΔF0.5 on VAL ± SE; each option was adopted only if > 2 SE and it fit
the time budget):

| Experiment | ΔF0.5 on VAL | Decision |
|---|---|---|
| Reverse-competition features | +0.0030 ± .0002 | in |
| One-to-one pass | +0.00005 ± .00002 | in (larger effect expected at test density) |
| e5-small cross-encoder on band [.05,.95] | +0.0263 ± .0004 | in (run 1) |
| Band widened to [.01,.99] | +0.0021 | in (run 1) |
| Cross-encoder → bge-reranker-v2-m3 | +0.0023 ± .0001 | in |
| Band widened to [.002,.998] | +0.0005 ± .0001 | in |
| S1-level "emit gate" against singletons | +0.0001 ± .0001 | out |
| Context-aware LightGBM combiner (band pairs, context features only) | +0.00001 ± .0001 | out |
| Stage-2 re-scorer: context + sibling features only | +0.00007 ± .00009 | out |
| Stage-2 re-scorer: + stage-1 features | +0.00043 ± .00009 (HOLDOUT +0.00065) | in |
| French address normalization (France-only change) | HOLDOUT +0.0001 (no French labels to measure a gain) | out |
| Stage 2 bagged over 4 LightGBM seeds | +0.00009 vs the submitted single fit (HOLDOUT +0.00078 vs +0.00065; single fits +0.0002 to +0.00065 by seed) | found after the final run was scored; too small to replace it. With the French address normalization, single fits were unstable (HOLDOUT +0.0006 to −0.0034) and bagging fixed it |
| Qwen2.5-7B yes/no log-prob on band pairs | no gain on top of the cross-encoder, 40× slower | out |
| Embedding view as a third blocking view | oracle +0.0008 to +0.0016 | deferred (needs re-blocking) |
| Candidate cap 80 instead of 50 | oracle +0.0009 | small; not adopted |

**Leave-one-country-out** (stage 1 only; a proxy for an unseen country):
- India trained on US data only: 0.846 (−0.085).
- US trained on India data only: 0.929 (−0.031).

This is why the transferable cross-encoder matters for France.
