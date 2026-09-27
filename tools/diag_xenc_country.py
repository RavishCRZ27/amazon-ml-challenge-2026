#!/usr/bin/env python3
"""Label-free: how would a new cross-encoder (tools/exp_xenc.py result) change TEST predictions per
country vs run 1? Samples S1s per country, scores their band pairs with the new model, applies
each system's own VAL-fit combiner and (t, r), and reports predicted matches, empty rate,
self-estimated F and the share of S1s whose predicted set changes. Suggestive only (no labels);
France has none anywhere, so this is the only test-side look at it.

usage: .venv/bin/python tools/diag_xenc_country.py --exp rr300k [--n-s1 20000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import threshold as th         # noqa: E402


def predict(d, n, band, coef, t, r, inj):
    inb = (d["p"].is_between(*band) & d["x"].is_not_null()).to_numpy()
    pf = xe.combine_apply(d["p"].to_numpy(), d["x"].fill_null(0.0).to_numpy(), inb, coef)
    keep = th.select(th.Cands(d["idx"].to_numpy(), d["pool_row"].to_numpy(), pf, n), t, r, inj)
    return pf, keep


def main(a):
    res = json.loads(Path(f"work/eval/xenc_exp_{a.exp}/result.json").read_text())
    xc = res["xc"]
    src = json.loads(Path(f"work/eval/xenc_exp_{a.exp}/xc.json").read_text()).get("model_from") or a.exp
    mdir = Path(f"work/eval/xenc_exp_{src}/model")                # --model-from runs reuse another run's model
    run1 = json.loads(Path("output/artifacts/thresholds_final.json").read_text())
    w = "work/test/s5_infer"
    s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "country"])
    samp = pl.concat([g.sample(min(a.n_s1, g.height), seed=5) for _, g in s1.group_by("country", maintain_order=True)])
    samp = samp.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    d = (pl.read_parquet(f"{w}/pred-part-*.parquet").join(samp, on="s1_row")
           .join(pl.read_parquet(f"{w}/xenc-part-*.parquet").rename({"xenc": "x1"}), on=["s1_row", "pool_row"], how="left")
           .sort("s1_row", "pool_row"))
    band = xc["band"]
    todo = d["p"].is_between(*band).to_numpy()
    tok, model = xe.load_xenc(mdir)
    x2 = np.full(d.height, np.nan, np.float32)
    x2[todo] = xe.score_pairs("test", d.filter(pl.Series(todo)), tok, model, xc["max_len"])
    d = d.with_columns(pl.Series("x2", x2).fill_nan(None))
    n = samp.height
    pf1, k1 = predict(d.rename({"x1": "x"}), n, run1["xenc"]["band"], run1["xenc"]["combiner"], run1["t"], run1["r"], run1["injective"])
    v = res["variant"]
    pf2, k2 = predict(d.rename({"x2": "x"}), n, band, v["combiner"], v["choice"]["t"], v["choice"]["r"], v["choice"]["injective"])
    d = d.with_columns(pl.Series("pf1", pf1), pl.Series("k1", k1), pl.Series("pf2", pf2), pl.Series("k2", k2))
    g = d.group_by("idx", "country").agg(
        pl.col("k1").sum().alias("n1"), pl.col("k2").sum().alias("n2"), (pl.col("k1") != pl.col("k2")).any().alias("changed"),
        (pl.col("pf1") * pl.col("k1")).sum().alias("sk1"), pl.col("pf1").sum().alias("sa1"),
        (pl.col("pf2") * pl.col("k2")).sum().alias("sk2"), pl.col("pf2").sum().alias("sa2"))
    ef = lambda i: pl.when(pl.col(f"n{i}") == 0).then((-pl.col(f"sa{i}")).exp()).otherwise(
        1.25 * pl.col(f"sk{i}") / (0.25 * pl.col(f"sa{i}") + pl.col(f"n{i}"))).clip(0, 1)
    out = (g.with_columns(ef(1).alias("selfF1"), ef(2).alias("selfF2")).group_by("country").agg(
        pl.len().alias("s1"), pl.col("n1").mean().round(3).alias("matches_run1"), pl.col("n2").mean().round(3).alias("matches_new"),
        (pl.col("n1") == 0).mean().round(4).alias("empty_run1"), (pl.col("n2") == 0).mean().round(4).alias("empty_new"),
        pl.col("selfF1").mean().round(4).alias("selfF_run1"), pl.col("selfF2").mean().round(4).alias("selfF_new"),
        pl.col("changed").mean().round(4).alias("s1_changed")).sort("country"))
    band_pairs = d.filter(pl.col("x1").is_not_null() & pl.col("x2").is_not_null())
    corr = band_pairs.group_by("country").agg(pl.corr("x1", "x2").round(3).alias("xenc_corr"),
                                              ((pl.col("x1") > 0) != (pl.col("x2") > 0)).mean().round(4).alias("sign_disagree"))
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(20)
    print(out.join(corr, on="country"))
    d.write_parquet(f"work/eval/diag_xenc_country_{a.exp}.parquet")   # pairs with both systems' p/keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-s1", type=int, default=20000)
    main(ap.parse_args())
