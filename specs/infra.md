
# Pipeline infrastructure: config, checkpoints, preflight, S3, scaling

## Config
- Every tunable lives in `config.yaml`: budget, `n_workers` (`auto` = `os.cpu_count()`),
  chunk size, split sizes, per-view k and skip-cap, model backend and params, threshold
  grid, injective, S3, disk floor, optional stages. Nothing is hard-coded to this VM.
- `config.v1.yaml` is the frozen baseline; never edit it after the first full run.

## Checkpoints and resume (a crash must never repeat finished work)
- Each stage writes to `work/<split>/<stage>/` and finishes by writing `_DONE.json` with
  rows, seconds, git commit, and a hash of (its config section + the input stages'
  `_DONE.json` hashes). On start, a stage whose `_DONE.json` matches is skipped.
  Changing a stage's config invalidates it and everything after it, never the stages
  before it.
- Chunked stages (S2, S3, S5 predict) write `part-<country>-<chunk>.parquet` via
  `*.tmp` + `os.replace`, so a killed process never leaves a half file that looks
  finished. On restart, existing parts are skipped. Chunk = one country × `chunk_s1` S1s.
- `src/common.py` provides this once, for every stage: config load, paths, a `stage()`
  context manager (skip check, timing, `_DONE.json`, appending `stage\trows\tseconds\tsplit`
  to `logs/timings.tsv`), `atomic_write_parquet`, `s3_sync(stage)`, logging to
  `logs/<stage>.log`.
- `run_all.sh`: resumes by default; `--from <stage>`, `--force <stage|all>`, `--only
  <stage>`, `--limit-s1 N`. It starts with `preflight.sh` and ends with the validator. It
  runs detached under `tmux`.

## S3 offsite copy
- After each stage: `aws s3 sync work/ s3://$BER_S3_BUCKET/$RUN_ID/work/ --exclude
  "*/s3_features/*"`, plus `output/` and `logs/timings.tsv`. Skipped silently when
  `BER_S3_BUCKET` is unset. The pipeline must never need S3 to run.
- Restore on a new VM: `aws s3 sync s3://$BER_S3_BUCKET/$RUN_ID/work/ work/` →
  `preflight.sh` → `run_all.sh` (resumes).
- Credentials only via the aws CLI's own config (`~/.aws`). Never read, print, or write
  keys.

## Disk (100 GB SSD)
- Check `df -h .` before S3 and S5; stop if under `disk_min_free_gb` (25). Rough sizes:
  raw TSVs ~4 GB, S0 parquet ~5 GB, test candidates ~3 GB, test features 8–12 GB,
  outputs ~1.5 GB.

## preflight.sh (PASS / WARN / FAIL per line; exits non-zero on any FAIL; writes logs/preflight.txt)
- Hardware: `nproc`, `free -g`, `df -h .` (FAIL < 60 GB free), `nvidia-smi` (WARN only).
- Python: `.venv/bin/python` is 3.12; import and print versions of polars, pyarrow,
  numpy, scipy, rapidfuzz (and `process.cpdist` exists), sparse_dot_topn (or numba),
  lightgbm, xgboost, pyyaml; `src/translit.py` self-check.
- Data: the 7 TSVs and the validator exist; row counts match the known dataset sizes.
- AWS (only if `BER_S3_BUCKET` is set): `aws sts get-caller-identity` succeeds, then
  put/get/delete a tiny object under `s3://$BER_S3_BUCKET/$RUN_ID/_preflight/`. On FAIL,
  tell the user to run `aws configure`. Never ask for keys in chat.
- Hugging Face: v1 downloads nothing. If an optional model stage is enabled, its
  `models/<name>/` directory must exist, and runs set `HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1`.
- Secrets: grep `src/ code/ docs/ config*.yaml` for `hf_[A-Za-z0-9]{20,}`,
  `AKIA[0-9A-Z]{16}`, `PRIVATE KEY`. FAIL on any hit.
- Git: working tree clean or not, and whether `notebooks/` is ignored.

## Scaling (the budget may grow to ~10 h; the VM may change)
- Parallelism comes from `n_workers`; memory is bounded by chunk size. A bigger VM means
  more workers and bigger chunks; a smaller VM still finishes.
- Moving VMs: restore from S3 → preflight → resume.
