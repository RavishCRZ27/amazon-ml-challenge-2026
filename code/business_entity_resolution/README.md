# Business Entity Resolution — Team TheAnarchy

This pipeline matches every Source-1 business record to all its Source-2/3 records and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. It uses only the provided
data and two local open models (Apache-2.0 / MIT). There are no external lookups, and nothing
touches the network at inference.

## 1. Environment (tested)
- Machine: 32 vCPU, 128 GB RAM, 1× NVIDIA L4 (24 GB), about 60 GB free disk.
  - Stages 2–3 scale with `n_workers` (default: all cores).
  - Memory use is bounded by `chunk_s1`.
  - The two GPU steps (XGBoost and the cross-encoder) need a CUDA GPU with ≥ 20 GB.
- Python 3.12 and a CUDA 12.x driver.

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## 2. Data and model
```
data/dataset/train/*.tsv, data/dataset/test/*.tsv   # the challenge's dataset/ folder
data/utils/validate_submission.py                    # the challenge's utils/ folder
models/bge-reranker-v2-m3/                           # base cross-encoder weights (Apache-2.0, 568M)
```
- Copy or symlink the challenge's `dataset/` and `utils/` folders into `data/`.
- Download the model once:
  `hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3`
- All runs set `HF_HUB_OFFLINE=1`, so nothing is fetched while the pipeline runs.

## 3. Run (from this folder)
```bash
bash preflight.sh               # environment, data row counts, secrets; exits non-zero on failure
bash run_all.sh --force all     # clean end-to-end run: raw TSVs -> both TSVs -> validator
```
- `run_all.sh` resumes from checkpoints by default. `--force <stage|all>` recomputes.
- Logs are in `logs/<stage>.log`, per-stage wall time in `logs/timings.tsv`, checkpoints in
  `work/`.
- To validate by hand:
  `.venv/bin/python data/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/dataset/test`

Wall time on the machine above:
- Run 1 (same stages, small cross-encoder), clean: 2 h 07 min.
- This configuration, clean: about 5 h (budget 8 h). That includes training the cross-encoder
  (≈ 34 min), scoring ≈ 8.3M uncertain test pairs with it (≈ 2.6 h at ~880 pairs/s; the submitted
  run measured 3.1 h while another job shared the GPU) and stage 2 (≈ 10 min on a free CPU; 28 min
  measured with a second job sharing it).

## 4. Stages (`src/`, all settings in `config.yaml`)

| Stage | File | Output |
|---|---|---|
| S0 normalize names and addresses | `s0_prepare.py` (+ `translit.py`) | `work/<split>/s0_prepare/` |
| S1 ground truth and splits: TRAIN / ES / VAL / HOLDOUT (train only) | `s1_gt.py` | `work/train/s1_gt/` |
| S2 blocking: per-country IDF-weighted sparse top-k over 2 views, ≤ 80 stored per record, cap 50 | `s2_block.py` | `work/<split>/s2_block/` |
| S3 pair features: 53, incl. reverse competition | `s3_features.py` | `work/<split>/s3_features/` |
| S4 stage-1 XGBoost (GPU) + exact macro-F0.5 threshold sweep on VAL | `s4_train.py`, `threshold.py` | `output/artifacts/model.json`, `thresholds.json` |
| S4x cross-encoder fine-tuned on TRAIN pairs + combiner fit on VAL | `s4x_xenc.py` | `output/artifacts/xenc/`, `thresholds_final.json` |
| S5 test inference: stage 1 + cross-encoder on the band + combiner → threshold → injective pass → TSVs | `s5_infer.py` | `work/test/s5_infer/`, `output/*.tsv` |
| S6 stage 2 (CPU): LightGBM on the S5 probability + S1 context + sibling evidence + stage-1 features, fit on VAL → final TSVs | `s6_stage2.py` | `output/*.tsv`, `output/artifacts/stage2.txt` |

- `common.py` handles config loading, checkpoints (`_DONE.json` with config/code hashes and
  atomic per-part parquet), logging and an optional S3 backup. The backup runs only when
  `BER_S3_BUCKET` is set.
- Optional stages that are off in this configuration: `s2e_embed.py` (embedding view) and
  `s4l_llm.py` (LLM triage).
- Methodology and results: `Documentation_template.md` at the root of the submission zip.

## 5. Determinism
- S5 and S6 are deterministic. Scoring runs in a fixed order, ties break by S1 index, and
  LightGBM runs with `deterministic`, so a rerun from the same checkpoints reproduces both TSVs
  byte for byte. S6 writes the final TSVs; if S5 is ever re-run afterwards, rerun S6.
- Training on CUDA (XGBoost, cross-encoder) is not bit-reproducible. A clean rerun gives a
  statistically equivalent model, not identical scores.
- The submitted run reused the cross-encoder evaluated during development, via
  `optional.cross_encoder.init_dir`. That model was trained with this exact recipe on TRAIN
  pairs only. This package leaves `init_dir` unset, so a clean run retrains it.
