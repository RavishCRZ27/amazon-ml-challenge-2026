# EDA Findings

## GT Injectivity
✓ GT is injective on S2/S3 side (0 duplicates found)
Decision: Set `injective: auto` in config.yaml (will be enabled)

## Data Stats
- Train S1: 2,206,821 / S2: 5,034,616 / S3: 5,285,603
- Test S1: 1,732,544 (India 46.8%, US 38.3%, France 15.0%) / S2: 4,887,273 / S3: 5,082,316
- GT: 7,638,365 positive pairs, mean 3.46 matches per S1 (max 11), 5.58% singletons, 100% same-country

## Blocking Ceiling
Blocking ceiling ≈ 0.995 (pairs share ≥ 2 tokens per EDA-11, EDA-13)

## Key Insights
- Names are noisy in every way (suffixes, order, typos, domains/hashtags, renames)
- ~18–28% of Indian pool names are Indic script; France and part of US are accented Latin
- Addresses: containment beats Jaccard (truncation); shared numbers are strong; postal codes mostly absent

## EDA-14: blocking recall (S2, 2026-09-26)
50K VAL S1s (stratified: India 20,010, US 29,990) against the FULL train pool (10.32M).
Recall = GT pairs retrieved / ALL GT pairs of those S1s (172,725 pairs; 47,221 S1s with GT).
Tool: `tools/eval_blocking.py` on `work/eval/s2_train_50000/`.

Duplicated names break name-only retrieval: 17% of India pool rows (10% US) share their exact
`name_s` with > 50 other rows (e.g. "HEARTLAND INITIATIVE" x138, "BACK ALLEY YOGA" x197).

Per-view recall@k (single view):
| view | @1 | @5 | @10 | @20 | @30 | @40/50 |
|---|---|---|---|---|---|---|
| W_all (name + addr tokens) | .250 | .827 | .919 | .944 | .953 | .958 (@40) |
| C_all (name-skeleton 3-grams + addr tokens) | .252 | .812 | .895 | .921 | .931 | .938 (@40) |
| W_addr | .231 | .728 | .812 | .851 | .865 | .878 (@50) |
| W_name | .124 | .398 | .478 | .540 | .573 | .611 (@50) |
| C_name | .053 | .184 | .240 | .292 | .322 | .361 (@50) |

Union at cap 50 (candidates ordered by best view rank, then #views, then best score):
| views | recall | S1 all found | mean cands/S1 |
|---|---|---|---|
| W_name 20 + W_addr 20 + C_name 20 (spec starting point) | .9339 | .825 | 48.0 |
| W_all 50 | .9614 | .888 | 50.0 |
| W_all 30 + C_all 30 | .9699 | .913 | 45.5 |
| **W_all 40 + C_all 40 (chosen)** | **.9717** | .918 | 49.7 |
| all 5 views at 50, uncapped | .9798 | .939 | 176.1 |
Orderings tried for the cap (RRF c=1..60, sum of scores, W_all+C_all score): within +/-0.001, kept.

Chosen config, by slice:
- country: US .985, India .952
- script pair (S1 is always ASCII): same .977, ASCII->Latin-accented .985, ASCII->Indic .893
- Indic by pool script: Devanagari .895 (6,898 pairs), Telugu .909, Kannada .875, Tamil .836
  (906 pairs = 0.5% of GT), Gujarati .880, Bengali .878, Malayalam .957, Gurmukhi .965, Oriya .945

Misses (not retrieved by any view at k=50: 4,591 = 2.7%): 3,283 share >= 3 address tokens, but
the name has typos or is transliterated; 1,074 have an empty pool address + generic name.
200-sample categorization (at chosen config): ASCII typos/domains/digit-for-letter with weak
address 50%, Indic transliteration with weak address 29%, names essentially different 12%,
empty pool address 10%, translated (not transliterated) Indic names 0%.
Decisions: no Tamil K/G fold (6/200 misses Tamil, no K/G confusion seen; max gain ~0.03 pt);
no `embed_view` (0 translated names among misses).

Timing (32 vCPU, ~165M postings/s aggregate):
- train slice: 163 s (matrix build India 59 s + US 67 s; retrieval India 13.4 s, US 24.0 s)
- test 30K stratified slice: 130 s (build France 15, India 70, US 42 s; retrieval France 0.6,
  India 10.9, US 5.7 s)
- extrapolated S2b full test (1.73M S1): France 35 s + India 10.5 min + US 5.5 min + builds
  2.1 min ~= 19 min (plan 30, kill 60). S2a full train (300K S1): ~6 min (plan 10).

## EDA-15: matcher experiments (step 3, 2026-09-26)
Scope: S3/S4 on TRAIN 180K / ES 20K / VAL 100K / HOLDOUT 100K S1s (19.9M candidate pairs, cap 50).
Metric: exact macro-F0.5 over ALL S1s of the split (full-GT recall, singletons in); ± = SE; Δ = paired.
| Variant | VAL F0.5 | Note |
|---|---|---|
| Blocking oracle (GT ∩ candidates) | .9899 ± .0002 | ceiling at cap 50 |
| Stage-1 XGBoost (53 features) | .94843 ± .00045 | HOLDOUT .94836; India .9314, US .9598 |
| − reverse-competition (rc_*) | Δ −.00303 ± .00023 | rc_rank/gap_W_all are the top features |
| injective pass (on VAL) | Δ +.00005 ± .00002 | understated: VAL S1s compete only among VAL |
| S1 emit gate (cross-fitted) | Δ +.00008 ± .00013 | n.s.; singleton emit 7.9% → 6.6% |
| + cross-encoder e5-small on band [.05,.95] | .97471 ± .00031 (Δ +.0263 ± .0004) | HOLDOUT .9744 |
Loss decomposition (stage 1 → with xenc): singleton .0044 → .0017, blocking .0101 (unchanged),
model-miss .0259 → .0086, false-pos .0111 → .0049.
LightGBM vs XGBoost (50K slice): .9395 vs .9397 ± .0014 (tie); XGB 3× faster → xgboost.
LOCO (France proxy, stage 1): India from a US-only model .846 (−.085 vs in-country), US from an
India-only model .929 (−.031).
Cross-encoder band-pair AUC .9805 (stage-1 p .8621); on pool names never seen in TRAIN candidates
.994 → it generalizes, it is not memorizing.
