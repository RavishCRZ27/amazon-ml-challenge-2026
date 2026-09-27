#!/usr/bin/env python3
"""Submission-2 criterion: current system (output/artifacts + S4x HOLDOUT scores) vs submission 1
(frozen copies) on HOLDOUT, per-S1 F0.5 with the exact test rule (combiner, t, r, injective),
paired SE overall and per country. HOLDOUT is report-only: this compares, it never tunes.

usage: .venv/bin/python tools/compare_holdout.py [stage2_dir, e.g. work/test/s6_stage2] [--dump f.parquet]
  --dump writes the per-S1 HOLDOUT F of this checkout's final system (s1_row, country, f) so two
  checkouts with different candidate sets (run 3 re-blocks) can be paired: tools/paired_f.py a b.
  Without submissions/1's frozen scores in this checkout (run 3), the sub1 line is skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import threshold as th         # noqa: E402

SYSTEMS = {"sub1": ("submissions/1/thresholds_final.json", "work/eval/xenc_exp_e5run1/scores_holdout.parquet"),
           "now": ("output/artifacts/thresholds_final.json", "work/train/s4x_xenc/scores_holdout.parquet")}

scope = (pl.read_parquet("work/train/s3_features/s1_scope.parquet").filter(pl.col("split") == "holdout").sort("s1_row")
           .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx")))
n_gt = scope.join(pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt"),
                  on="s1_row", how="left").fill_null(0)["n_gt"].to_numpy()
base = pl.read_parquet("work/eval/s4_full/pred_holdout.parquet").join(scope.select("s1_row", "idx"), on="s1_row").sort("s1_row", "pool_row")
y = base["label"].to_numpy().astype(np.float64)
dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
args = [a for a in sys.argv[1:] if a != "--dump" and a != dump]
F = {}
for name, (thr_f, sc_f) in SYSTEMS.items():
    if not Path(sc_f).exists():
        print(f"{name}: {sc_f} missing in this checkout, skipped")
        continue
    thr = json.loads(Path(thr_f).read_text())
    x = (base.select("s1_row", "pool_row").join(pl.read_parquet(sc_f).select("s1_row", "pool_row", "xenc"),
         on=["s1_row", "pool_row"], how="left", maintain_order="left")["xenc"].fill_null(0.0).to_numpy())
    lo, hi = thr["xenc"]["band"]
    inb = ((base["p"] >= lo) & (base["p"] <= hi)).to_numpy()
    pf = xe.combine_apply(base["p"].to_numpy(), x, inb, thr["xenc"]["combiner"])
    keep = th.select(th.Cands(base["idx"].to_numpy(), base["pool_row"].to_numpy(), pf, scope.height),
                     thr["t"], thr["r"], thr["injective"])
    F[name] = th.per_s1_f05(n_gt, base["idx"].to_numpy(), keep, y, scope.height)
    print(f"{name}: HOLDOUT {F[name].mean():.5f} ± {th.se(F[name]):.5f} (t {thr['t']}, band {thr['xenc']['band']})")
if args:                                    # + stage 2 (S6): model + report dir, e.g. work/test/s6_stage2
    import lightgbm as lgb
    import s6_stage2 as s6
    sd = Path(args[0])
    rep = json.loads((sd / "stage2_report.json").read_text())
    thr = json.loads(Path(SYSTEMS["now"][0]).read_text())
    s3cols = json.loads(Path("output/artifacts/thresholds.json").read_text())["features"]
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    sch, gh = s6.train_split("holdout", thr, s3cols, pool_txt, 16)
    assert gh.select("s1_row", "pool_row").equals(base.select("s1_row", "pool_row")), "row order mismatch"
    p2 = lgb.Booster(model_file=str(sd / "stage2.txt")).predict(gh.select(rep["features"]).to_numpy().astype(np.float32))
    keep = th.select(th.Cands(base["idx"].to_numpy(), base["pool_row"].to_numpy(), p2, scope.height),
                     rep["choice"]["t"], rep["choice"]["r"], rep["injective"])
    F["now"] = th.per_s1_f05(n_gt, base["idx"].to_numpy(), keep, y, scope.height)
    print(f"now + stage 2 ({sd}): HOLDOUT {F['now'].mean():.5f} ± {th.se(F['now']):.5f} (t {rep['choice']['t']})")
if dump:
    scope.select("s1_row", "country").with_columns(pl.Series("f", F["now"])).write_parquet(dump)
    print(f"per-S1 HOLDOUT F of the final system -> {dump}")
if "sub1" not in F:
    sys.exit(0)
d = F["now"] - F["sub1"]
print(f"now - sub1: {d.mean():+.5f} ± {th.paired_se(F['now'], F['sub1']):.5f} "
      f"({d.mean() / th.paired_se(F['now'], F['sub1']):.1f} paired SE)")
cc = scope["country"].to_numpy()
for c in sorted(set(cc)):
    m = cc == c
    print(f"  {c}: sub1 {F['sub1'][m].mean():.5f} -> now {F['now'][m].mean():.5f} "
          f"({d[m].mean():+.5f} ± {th.paired_se(F['now'][m], F['sub1'][m]):.5f})")
