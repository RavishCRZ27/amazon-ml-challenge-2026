
# S2 blocking (per split × country)

Why: flat DF-capped token keys can't reach 0.95 recall inside budget (EDA-11), while
sharing ≥ 2 tokens covers 0.995 (EDA-12). IDF-weighted top-k retrieval is the graded form
of that: a candidate sharing several rare-ish tokens outranks one sharing a common token.
The cost is the sum of postings, in compiled code.

- Pool = S2∪S3 rows of that split and country; queries = that split's S1 rows of that
  country. IDF = log(N_pool/df) over that pool only; binary tf; rows L2-normalized.
- Views (`config.blocking.views`; starting point, tune on the slice):
  - `W_name`: word tokens of `name_s` (plus digits in the name)
  - `W_addr`: address word tokens + number tokens, prefixed (e.g. `#107`) so numbers only
    match numbers
  - `C_name`: char 3-grams of `name_sk` with spaces removed (concatenations, typos,
    transliteration drift)
- Skip-cap: remove from the QUERY matrix any feature whose df exceeds
  `skip_cap_frac × partition size` (start at 1% for word views, 0.5% for C_name). Pool
  rows keep every feature.
- Top-k per view: `sparse_dot_topn.sp_matmul_topn(Q, P.T.tocsr(), top_n=k,
  n_threads=n_workers)` (v1.2.0, Apache-2.0), or a numba prange accumulator as fallback.
- IMPORTANT: never materialize (query, candidate) pairs from token joins in polars or
  duckdb (EDA-11: billions of rows), and never loop over queries in Python.
- Candidates = deduplicated union of per-view top-k, ordered (best view rank, #views, best
  score) and stored up to `store_candidates` with `cand_pos`. S3 applies `max_candidates`, so
  it is not part of S2's hash and run 2 can change it without re-blocking. Keep each view's
  score and rank (they become S3 features).
- Train (no limit) blocks ALL train S1s, not only the split ids: reverse-competition features
  need every S1's candidates, and run 2 can grow the train set without re-blocking. The
  `--limit-s1` train mode stays the VAL-only tuning slice (EDA-14).
- Write `queries.parquet` (s1_row, country): the exact S1 universe, including S1s that get
  zero candidates (they still count in macro-F0.5).
- Output: `work/<split>/s2_block/part-<country>-<chunk>.parquet` with columns (s1_row,
  pool_row, cand_pos, score_<view>, rank_<view>), plus `queries.parquet`; atomic writes.
- Tune on a 50K VAL slice against the full train pool. Report recall@k per view and for
  the union (against FULL GT), mean union size, and wall time. Target union recall ≥ 0.97
  (stretch 0.985) at ≤ 50 candidates per S1. Record it as EDA-14 in eda-findings: per
  country, and for cross-script vs same-script pairs.
- Also save the misses (true pairs no view retrieved, ~200 examples). They decide whether
  the optional `embed_view` is worth it.
- Cost reference, measured in a sandbox with scipy on 1 core and synthetic Zipf data:
  20–25M postings/s per core (pessimistic). The test word views are ≈ 2.4e11 postings,
  about 10 min on 16–20 effective cores. C_name is unmeasured, so measure it.
- Kill criterion: S2b (test) extrapolating past 60 min → lower the skip-caps or k, or run
  C_name only for queries without a strong word-view hit.
