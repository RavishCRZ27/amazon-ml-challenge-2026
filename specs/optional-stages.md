
# Optional stages (off in v1; enable only with measured evidence and budget)

Each stage sits behind a flag in `config.optional`, has its own `_DONE.json`, and only
adds candidates or features; it never changes earlier stages.

## embed_view (sentence-transformers) → src/s2e_embed.py
- When: the EDA-14 misses (true pairs no lexical view retrieves) are large enough to
  matter, and the budget has room. Report recall with and without.
- Models (MIT/Apache-2.0, ≤ 8B, loaded from `models/<name>/` with `HF_HUB_OFFLINE=1`):
  `intfloat/multilingual-e5-small` (MIT, ~118M) first; `sentence-transformers/all-MiniLM-L6-v2`
  (Apache-2.0, faster, works on the romanized text); `BAAI/bge-m3` (MIT, 568M) only with a
  ~10 h budget (≈ 2–4 h of L4 time for ~22M records).
- Measure records/s on a 200K sample before committing. fp16, length-sorted batches.
- Per-country faiss-cpu indices on host RAM. Never search across countries. Its top-k
  becomes a 4th view; cosine becomes a feature.
- Run as a separate process, and confirm `nvidia-smi` shows the GPU memory released
  before any other GPU stage.

## llm_triage (Qwen2.5-7B-Instruct, Apache-2.0, vLLM on the L4) → src/s4l_llm.py
- Only on the GBM's uncertain band (`optional.llm_triage.band`) and only if that band is
  ≤ `max_pairs` on test (~30K at tens of pairs/s). Score it by the log-prob of yes/no,
  used as a feature for a second-stage model or a calibrated override, tuned on VAL.
- Separate process; the GPU must be free (no embedding or XGBoost job).

## Other upgrades, in order
1. Reverse-competition features: block all 2.2M train S1s so each pool record knows which
   S1s retrieved it and at what rank (also makes VAL mimic test for the injective pass).
2. A token-synonym table learned from train GT pairs (MH↔MAHARASHTRA), using training
   data only.

## Setup and GPU discipline (2026-09-26)
- 2026-09-27: Qwen weights, bge-m3 and `.venv-llm` were deleted to free disk (Qwen and bge-m3
  are out of the pipeline). Recreate with `hf download` + the steps below if ever needed.
- Weights in `models/` (multilingual-e5-small MIT, bge-m3 MIT, Qwen2.5-7B-Instruct Apache-2.0),
  fetched once with `hf download`, no token. Runs set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- vLLM lives in `.venv-llm` (its own torch); `llm_triage.python` points at it. The main `.venv`
  never gets vLLM.
- One GPU job at a time: embeddings → cross-encoder → XGBoost → vLLM. Check `nvidia-smi` shows
  no compute apps before each.
- vLLM + FlashInfer JIT: run `bash tools/setup_llm_cuda.sh` once per VM. It pins the pip CUDA
  compiler in `.venv-llm` to torch's CUDA (13.0; CCCL requires nvcc == toolkit headers) and
  builds `.venv-llm/cuda` (a CUDA_HOME of symlinks, incl. lib64 and libcudart.so). s4l_llm.py
  uses it when present, else falls back to `VLLM_USE_FLASHINFER_SAMPLER=0`. No system CUDA.
  Measured Qwen2.5-7B bf16 on the L4, 2,000 prompts of ~166 tokens: FlashInfer sampler 48.7
  pairs/s, 64 s load; torch sampler 44.2 pairs/s, 145 s load. First JIT build ~104 s (cached
  in ~/.cache/flashinfer).

## cross_encoder (fine-tuned, `src/s4x_xenc.py`) — IN for run 1 (EDA-15: +0.0284 VAL)
- Backbone multilingual-e5-small (bge-m3 is a run-2 option). Trained one epoch (no early
  stopping) on TRAIN pairs with stage-1 p in `train_band` + TRAIN positives + random negatives
  (≤ `max_train`); input raw "name | address" of both records.
- Scores every pair with stage-1 p in `band` ([.01,.99]); a logistic combiner on [logit p,
  xenc, logit p·xenc] (cross-fitted over VAL S1s for the (t, r) choice, refit on all VAL for
  test) gives p'. Measured: train 1,000 pairs/s, score ~2,500 pairs/s on the L4.
- The trained weights live in `work/train/s4x_xenc/xenc` (resume) and are copied to
  `output/artifacts/xenc`. CUDA training is not bit-reproducible: run 2 must RESUME S4x (never
  `--force` it) unless it deliberately retrains; S5 reruns are byte-identical.
- Run 2 (2026-09-27): backbone `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M, a pre-trained
  pair reranker). 300K TRAIN pairs, batch 64 (128 runs out of memory on the L4), lr 2e-5.
  +.00228 ± .00013 VAL vs e5-small; train 150 pairs/s (34 min); scoring ~880 pairs/s end to end.
- `init_dir` (optional): adopt a model trained earlier with the SAME recipe (RECIPE keys in
  s4x_xenc.py must equal the source's `xc.json`) on the CURRENT S4's TRAIN pairs (its mtime must
  be after S4's `_DONE.json`), e.g. `tools/exp_xenc.py`'s model. The submitted model is then the
  evaluated one. A clean run leaves `init_dir` unset and retrains.
- Scoring runs length-sorted and training uses length-bucketed batches (`batches(lengths=…)`)
  to trim padding: e5-small 2.7K → 4.4K pairs/s end to end.

## stage2 (`src/s6_stage2.py`, CPU, after S5) — IN for run 2
- LightGBM on the final S5 probability pf (stage-1 p + cross-encoder via the VAL-fit combiner), S1 context
  over its candidates (rank, max, 2nd, count >= t, sum, gap), sibling evidence vs the S1's top OTHER
  candidate (its pf, name ratio / token_set, address token_set, same address), counts of the S1's
  candidates sharing the exact address / name, and the 53 stage-1 features. Never `country`.
- Fit on VAL, cross-fitted over the combiner's 2 VAL S1 folds (early stopping on 10% of each training
  fold) for the (t, r) choice; refit on all VAL (rounds = mean best iteration); HOLDOUT once.
- Apply per S5 part (reads S5's pred-/xenc-part checkpoints, never rewrites them), then threshold.select
  over all test S1s and both TSVs, like S5. Model and per-part p2 are checkpointed (resume-safe);
  LightGBM `deterministic`, `force_row_wise`.
- Evidence (run 2): VAL +.00043 ± .00009 paired over S5's rule, HOLDOUT +.00065 ± .0001 (report
  only). The context/sibling features alone gave +.00007 ± .00009 (n.s.): the gain needs the stage-1
  features. France (label-free, 200K S1s): 6.2% of S1s change; HOLDOUT per-type truth rates project
  ~3,100 fewer false positives at unchanged true matches.

## emit_gate (S1-level, CPU)
- A small GBM on S1 aggregates (top-1/top-2 p, gap, n above t, n_cands, best name/address
  scores) predicting "≥ 1 GT match among the candidates"; suppress the S1's output below g.
  Cross-fitted over VAL S1s; g swept jointly with t. Targets singleton emissions.
