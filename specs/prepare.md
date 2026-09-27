
# S0 prepare + S1 GT checks and split

## S0 prepare (7 TSVs, ~24.2M records)
- `pl.read_csv(path, separator="\t", infer_schema=False)`; assert each row count equals
  `wc -l` − 1, and if it doesn't, retry with `quote_char=None`.
- Write `work/<split>/s0_prepare/{s1,pool}.parquet`. The pool gets a global row id per
  split (S2 → [0, n2), S3 → [n2, n2+n3)); keep the entity_id strings and a `src` column.
- Names: `translit.to_latin` only on rows containing non-ASCII (multiprocessing,
  `n_workers`; ~28K names/s per core measured) → uppercase → delete apostrophes and periods
  (`L.L.P.` → `LLP`) → other ASCII punctuation → space → collapse whitespace.
  Columns: `name_n`, `name_s` (suffix-stripped), `name_sk` (`translit.skeleton(name_s)`),
  `name_script` (`translit.script_class` of the raw name).
- Suffix tokens are removed anywhere in the name (as EDA-03 measured) unless that empties
  it: LLC, INC, INCORPORATED, CORP, CORPORATION, CO, COMPANY, LTD, LIMITED, PVT, PRIVATE,
  LLP, PLLC, PC, PLC, LP, OPC; French SAS, SASU, SARL, EURL, SA, SCI, SNC, SELARL, EIRL.
  For romanized Indic names only, also the romanized forms (seen: PRAIVET, PRAIVATT,
  LIMITTAD, LIMATID, PRA, LI). Derive the full list from the ~200 most frequent tokens of
  romanized pool names, show them to the user, and hard-code the approved list.
- Addresses: `to_latin` → uppercase → ALL ASCII punctuation → space (`NO:107/2` →
  `NO 107 2`; the notebook's `N()` deleted punctuation and produced `NO1072`) → collapse
  → a small hand-written map of street types (ST/STREET, RD/ROAD, AVE/AVENUE, BLVD, DR, LN,
  HWY, STE, APT, FL, NR/NEAR, OPP; France R/RUE, AV/AVENUE, BD/BOULEVARD, PL/PLACE,
  CHE/CHEMIN, IMP/IMPASSE, ALL/ALLEE) and US/India state names ↔ codes. These maps are
  documented in the methodology as hand-written normalization, not a lookup.
- Numbers: digit runs from the raw address, leading zeros stripped (`003544` → `3544`).
- Do not use the notebook's `fold()`: it strips Indic viramas (`प्रोडक्ट्स` → `परोडकटस`).
- Self-check: `.venv/bin/python src/translit.py` must print matching skeletons.

## S1 GT checks + split
- Explode GT (an empty cell = 0 matches; `''.split(',')` gives `['']`). Count S2/S3 ids
  that appear under more than one S1. Record the result in eda-findings as "GT
  injectivity". If it's more than a trace, set `injective: off` and tell the user.
- Split train S1s by country (seed from config): TRAIN (`split.train_s1`, default 200K),
  VAL (default 100K), plus an early-stopping slice from TRAIN (default 20K). Save the id
  lists to `work/train/s1_gt/`. Everything is blocked against the FULL train pool.
- HOLDOUT (`split.holdout_s1`, default 100K, seed + 2) is drawn from the S1s outside TRAIN ∪
  VAL ∪ ES, so adding it leaves those splits byte-identical. It is never used for any choice
  (see the methodology, "Submissions").
- Kill criterion: S0 extrapolating past 25 min → profile the transliteration and regex
  steps first.
