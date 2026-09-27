
# S4 train + threshold

- Backend from `model.backend`: `lightgbm` (CPU, MIT) or `xgboost` (`device="cuda"`,
  `tree_method="hist"`, Apache-2.0). Both read the same feature parquet. The first time,
  benchmark both on the slice (train time, predict time extrapolated to test, VAL
  macro-F0.5), report the result, and keep the winner in config. XGBoost-GPU needs the
  GPU free (`nvidia-smi`): no embedding or vLLM process may be running.
- Train on TRAIN candidates, with early stopping on the early-stopping slice (never VAL).
- Score VAL. Sweep t over `threshold.t_grid`, optionally with a relative rule (keep if
  p ≥ t and p ≥ r·max_p within the S1, r ∈ `threshold.relative_r`). Apply the injective
  pass within VAL (if enabled), then compute exact macro-F0.5 over ALL VAL S1s.
  - Singletons are included: 1.0 if the prediction is empty, else 0.0.
  - Recall denominators come from the FULL GT, so blocking misses count.
  - An S1 with matches but an empty prediction scores 0.
- Report: the F0.5 curve, the chosen (t, r) and why (the argmax, and how flat the curve is
  around it; prefer the flat region's centre over a spiky argmax), per-country scores,
  injective on vs off, and the blocking-oracle F0.5 (predict GT ∩ candidates) as the
  ceiling. State the caveat: VAL S1s compete only with each other for pool ids, so the
  injective gain is understated relative to test.
- Save `output/artifacts/model.{txt,json}`, `thresholds.json`, and feature importances.
- Don't hand-tune thresholds per country (France has no labels).

## Reporting discipline (2 submissions left)
- Every F0.5 comes with SE = std(per-S1 F)/√n; compare variants with the paired SE of per-S1
  differences. Benchmark winner: higher F if |ΔF| > 2 paired SE, else the faster backend
  (train + extrapolated test predict).
- Loss decomposition of 1 − F (`threshold.loss_decomposition`): singleton emissions, blocking
  misses, model misses among candidates, false positives; per country; save the 200 costliest
  VAL S1s. It decides which experiment comes next.
- HOLDOUT is scored once, at the (t, r) chosen on VAL, and is report-only.
- `--ablate <prefix>` (feature group, paired ΔF on VAL), `--loco` (train on one country, score
  the other: a France proxy; diagnostic only, never per-country thresholds).
- `n_estimators` must not bind: if best_iteration hits the cap, raise it.
