
# S3 features (vectorized, chunks of `chunk_s1` S1s, float32)

- Blocking: per-view score and rank, number of views that retrieved the pair, candidate
  count for this S1, gap to this S1's best score per view.
- Name, on `name_s` and `name_sk`: rapidfuzz ratio, token_sort_ratio, token_set_ratio,
  partial_ratio, JaroWinkler (prefix-weighted; covers "first words matter more"); spaceless
  ratio and partial_ratio (domains, hashtags: `#zinetclinic`, `innovativegood.com`);
  exact match; first-token match; length difference; `name_script` of both sides
  (cross-script flag).
- Address: token containment |a∩b|/min, Jaccard, IDF-weighted overlap, token_set_ratio;
  shared-number count, first-number match, number conflict (both have numbers, none
  shared), longest shared number; missing-address flags; overlap of the last two tokens
  (city/state).
- Group context: this pair's name and address rank within its S1's candidates, and the gap
  to the best.
- Source (S2/S3). Never `country`.
- Reverse competition (`features.reverse_competition`, `rc_*`): this S1's rank and score gap
  among ALL S1s whose capped candidates include the pool row. No raw counts (train has ~20%
  more S1s per pool row than test).
- Extras: name-token IDF containment; log1p count of pool rows sharing the exact `name_s`
  (pool name and S1 name; EDA-14 duplicated names); `cand_pos`; `emb_cos` when the optional
  embedding stage is on.
- Scope: S2's `queries.parquet` labelled with split (train/early_stop/val/holdout; test),
  `--limit-s1` samples per split; saved as `s1_scope.parquet` (S4's denominators).
- Element-wise string scores via `rapidfuzz.process.cpdist(a, b, scorer=...,
  workers=n_workers)`. Set operations via polars list ops or numba over token-id arrays.
- IMPORTANT: no Python loops over pairs. At ~80M test pairs, 20 µs per pair is ~27 min per
  feature.
- Labels (train/val splits only): 1 if the candidate is in that S1's GT set.
- Output: `work/<split>/s3_features/part-<country>-<chunk>.parquet`, atomic writes. Test
  parts may be deleted once S5 and S6 (stage 2 reads them) have finished (disk).
- Kill criterion: S3b (test) extrapolating past 60 min → shrink k to the recall knee
  (EDA-14), then drop address partial_ratio (the slowest scorer).
