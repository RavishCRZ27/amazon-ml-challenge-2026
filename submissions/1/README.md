# Submission 1 (run 1)

- Code: commit `24af83a` (clean tree), `bash run_all.sh --force all`, 2026-09-27 02:56–05:03 UTC
  (wall 2 h 07 min; stage sum 125.1 min, see `timings_run1.txt`) vs `budget_hours` 8.
- Contents: stage-1 XGBoost (53 features incl. rc_, best_iteration 1994) → e5-small cross-encoder
  (train band .05–.95, score band .01–.99) → VAL-fit logistic combiner → t = 0.70, r = 0,
  injective pass on S2/S3 ids. Blocking: per-country IDF top-k, W_all + C_all, cap 50.
- Files: `matching_results.tsv`, `candidate_pairs.tsv` (moved to S3 `$BER_S3_BUCKET/v1/submissions/1/` on
  2026-09-27 with md5 metadata; restore with `aws s3 cp`; md5 in `md5_run1.txt`),
  `config.yaml`, `thresholds.json` (stage 1), `thresholds_final.json`, `s4x_report.json`,
  `france_sanity.json`.

## md5
```
e5f673739eec1cd46d82bf91a03737c6  matching_results.tsv
d7154e62fc4735656114b0c1a5ca919c  candidate_pairs.tsv
```

## Train-side F0.5 (macro, full GT, singletons in; ± SE)
| Split | Overall | India | US |
|---|---|---|---|
| VAL (100K, used for choosing) | .97682 ± .00029 | .96909 ± .00054 | .98198 ± .00032 |
| HOLDOUT (100K, never used to choose) | **.97642 ± .00029** | .96772 ± .00056 | .98222 ± .00031 |

- Stage 1 alone: VAL .94843, HOLDOUT .94836. Cross-encoder gain on HOLDOUT: +.02806 ± .00037 (paired).
- HOLDOUT loss: blocking_miss .0104, model_miss .0080, false_pos .0037, singleton_emit .0014
  (emit rate 2.6%).
- Bar: the 0.924-LB reference scored 0.908 on the same honest metric.

## Test sanity (by country)
| Country | S1 | mean matches | empty rate | mean candidates |
|---|---|---|---|---|
| France | 259,452 | 3.12 | 6.1% | 49.7 |
| India | 809,986 | 3.24 | 6.3% | 49.4 |
| US | 663,106 | 3.35 | 5.8% | 49.9 |

Train GT reference: 3.46 matches per S1, 5.6% empty. Cross-encoder band: 5,495,967 pairs (6.39% of
85,986,019), scored in 1,902 s.

## Gate
1. Validator PASS, also with `--check-ids`. Matches ⊆ candidates, one row per test S1, no
   duplicate ids.
2. Timed clean run; all 10 stages are in `logs/timings.tsv`.
3. HOLDOUT is above.
4. France sanity is above.
5. S5 rerun md5: see PROGRESS.
6. Secret scan: clean.

## Public LB (reported by the user)
- Score: **0.965** (reported 2026-09-27 ~06:15 UTC) -> F_Fr ≈ (0.965 − 0.8291)/0.150 ≈ 0.906 if India/US match HOLDOUT
- France estimate from the formula: `F_Fr ≈ (LB − 0.468·.96772 − 0.383·.98222) / 0.150 =
  (LB − 0.8291) / 0.150`.
  - For example: LB .976 → F_Fr ≈ .98; LB .970 → .94; LB .960 → .87.
  - The public LB is a subset of test, so read this estimate as ±~.02.
