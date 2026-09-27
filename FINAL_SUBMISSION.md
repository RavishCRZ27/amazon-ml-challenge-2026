# Final submission report — Team TheAnarchy

**Amazon ML Challenge 2026 · Business Entity Resolution**
Team: [Ravish Pandey](https://github.com/RavishCRZ27), [Samar Nathani](https://github.com/SammySN-car), [Ayush Rajdeep](https://github.com/rajdeep3456) · Submitted 2026-09-27

| | |
|---|---|
| **Public leaderboard** | **0.968** macro-F0.5 (first submission: 0.965) |
| **HOLDOUT F0.5** (100K held-out training records, never used for a choice) | **0.98000 ± 0.00027** — India 0.97235, US 0.98511 |
| **VAL F0.5** (cross-fitted, used for every choice) | 0.98004 ± 0.00027 — India 0.97319, US 0.98461 |
| Gain over the first submission (HOLDOUT, paired) | +0.00359 ± 0.00016 (India +0.0046, US +0.0029) |
| France (test only, no labels) | implied F0.5 ≈ 0.911 from the public score and the India/US HOLDOUT scores |

Full methodology (the document submitted with the package):
[`docs/Documentation_filled.md`](docs/Documentation_filled.md). Pipeline overview and how to run it:
[`README.md`](README.md).

## Download

All files are attached to the release
[**final-run2-0.968**](https://github.com/RavishCRZ27/amazon-ml-challenge-2026/releases/tag/final-run2-0.968)
(gzip-compressed; `gunzip` restores the exact files). `SHA256SUMS.txt` in the release covers every asset.

**Submission files**

| Asset | Contents | md5 of the uncompressed file |
|---|---|---|
| `matching_results.tsv.gz` | final matches, the file scored on the leaderboard (1,732,544 rows, 95.1 MB) | `3d61e100489f7047641294922eb5bff1` |
| `candidate_pairs.tsv.gz` | candidate set fed to the matcher (1,732,544 rows, 1.13 GB) | `afc30cca32502ca9eb1baa1922713823` |
| `TheAnarchy_submission.zip` | the submitted package: both TSVs, code, methodology | `4ff3b2043adc85f57a27fa4cedc8f999` (zip itself) |

**Challenge dataset** (as provided by the organisers; not covered by this repository's license)

| Asset | Rows | md5 of the uncompressed file |
|---|---|---|
| `train_source1.tsv.gz` | 2,206,821 | `1a0c98e43dad1babed3c99bca2c7b5bd` |
| `train_source2.tsv.gz` | 5,034,616 | `a7bddba33ead4a85e6c088ad3b377fc2` |
| `train_source3.tsv.gz` | 5,285,603 | `5df07bf109f55a292bdfaa8c983cfa20` |
| `train_ground_truth.tsv.gz` | 2,206,821 | `3643443570b2c881c425d69c2d46e95d` |
| `test_source1.tsv.gz` | 1,732,544 | `51a155b8c84bfefcf049e983f06c3cf4` |
| `test_source2.tsv.gz` | 4,887,273 | `ff8aba5701829a622ac05b2765c4eb5b` |
| `test_source3.tsv.gz` | 5,082,316 | `5f48d48594dea3ea1455549eb5233ad8` |

To run the pipeline on them, unpack into `data/dataset/train/` and `data/dataset/test/` (see the README).

## The submitted system

- **Retrieval:** per-country IDF-weighted sparse top-k over two views (name + address words; name-skeleton
  3-grams + address words), union capped at 50 candidates per record: 85,986,019 test pairs, 97.2% recall on VAL.
- **Stage 1:** XGBoost on 53 pair features, including reverse-competition features.
- **Cross-encoder:** `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) fine-tuned 1 epoch on 300K training pairs; it scores
  only pairs with stage-1 probability in [0.002, 0.998] — 8,348,891 test pairs (9.7%) — and a logistic combiner
  merges both scores.
- **Stage 2:** LightGBM (181 trees) re-scores every candidate from the combined probability, the record's candidate
  context, sibling evidence and the stage-1 features.
- **Decision:** threshold t = 0.71 (centre of the VAL plateau [0.55, 0.87]) chosen by exact macro-F0.5, then a
  one-to-one pass on the Source-2/3 side.

## Output statistics (test)

| | France | India | US |
|---|---|---|---|
| Source-1 records | 259,452 | 809,986 | 663,106 |
| Predicted matches per record | 3.01 | 3.23 | 3.37 |
| Records with no match | 6.9% | 6.4% | 5.8% |
| Candidates per record | 49.7 | 49.4 | 49.9 |

Totals: 5,633,855 predicted matches over 1,732,544 records (108,287 with none; the training ground truth averages
3.46 matches and 5.6% singletons). 24 records have no candidate at all.

## Checks passed before the upload

- Official validator (`validate_submission.py --check-ids`): **PASS** — every test Source-1 id exactly once, every
  matched id exists in the test Sources 2/3, no duplicates.
- Every match is among that record's candidates.
- Reproducibility: re-scoring cross-encoder parts from scratch gave bitwise-identical scores, and re-running stage 2
  rewrote both TSVs byte for byte (same md5s as above).
- Secret scan of the package: clean.

## Submission history

| # | System | HOLDOUT F0.5 | Public LB |
|---|---|---|---|
| 1 | Stage 1 + e5-small cross-encoder on band [0.01, 0.99] | 0.97642 | 0.965 |
| **2 (final)** | + bge-reranker-v2-m3 on band [0.002, 0.998] + stage-2 re-scorer | **0.98000** | **0.968** |
