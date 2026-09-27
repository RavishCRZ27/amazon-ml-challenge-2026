#!/usr/bin/env python3
"""Label-free per-country diagnostics: HOLDOUT (has labels) vs test (none), same statistics.

For each (split, country): band share, S1 confidence profile, cross-encoder vs stage-1
disagreement, predicted matches, and a self-estimated F0.5 from the final probabilities
(E[F] ~ 1.25 sum_{kept} p / (0.25 (sum_all p + blocking_miss) + n_kept)). On HOLDOUT the true F is
printed next to it, which shows how far the self-estimate can be trusted before reading it on test.

usage: .venv/bin/python tools/diag_country.py
"""

import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import threshold as th         # noqa: E402

thr = json.loads(open("output/artifacts/thresholds_final.json").read())
LO, HI = thr["xenc"]["band"]
T, R, INJ = thr["t"], thr["r"], thr["injective"]


def final_p(d: pl.DataFrame) -> pl.DataFrame:
    inb = (d["p"].is_between(LO, HI) & d["xenc"].is_not_null()).to_numpy()   # HOLDOUT file stores xenc=0 off-band
    p2 = xe.combine_apply(d["p"].to_numpy(), d["xenc"].fill_null(0.0).to_numpy(), inb, thr["xenc"]["combiner"])
    return d.with_columns(pl.Series("pf", p2), pl.Series("inb", inb))


def stats(d: pl.DataFrame, s1: pl.DataFrame, label: str):
    s1 = s1.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    d = d.join(s1.select("s1_row", "idx", "country"), on="s1_row").sort("s1_row", "pool_row")
    keep = th.select(th.Cands(d["idx"].to_numpy(), d["pool_row"].to_numpy(), d["pf"].to_numpy(), s1.height), T, R, INJ)
    d = d.with_columns(pl.Series("keep", keep))
    g = (d.group_by("idx").agg(
            pl.col("pf").max().alias("top"),
            pl.col("keep").sum().alias("n_pred"),
            (pl.col("pf") * pl.col("keep")).sum().alias("sp_kept"),
            pl.col("pf").sum().alias("sp_all"),
            pl.col("inb").sum().alias("n_band"))
         .join(s1.select("idx", "country"), on="idx", how="right")
         .with_columns(pl.col("n_pred", "sp_kept", "sp_all", "n_band", "top").fill_null(0)))
    # self-estimated F per S1 (singleton case: F = 1 if nothing kept, weighted by P(no match))
    ef = pl.when(pl.col("n_pred") == 0).then((-pl.col("sp_all")).exp()).otherwise(
        1.25 * pl.col("sp_kept") / (0.25 * pl.col("sp_all") + pl.col("n_pred")))
    g = g.with_columns(ef.clip(0, 1).alias("selfF"))
    band = d.filter(pl.col("inb"))
    dis = band.group_by("country").agg(
        ((pl.col("xenc") > 0) != (pl.col("p") >= 0.5)).mean().alias("xenc_vs_p1_disagree"),
        (pl.col("xenc") > 0).mean().alias("xenc_pos_rate"))
    out = (g.group_by("country").agg(
              pl.len().alias("s1"),
              pl.col("n_pred").mean().round(3).alias("mean_pred"),
              (pl.col("n_pred") == 0).mean().round(4).alias("empty"),
              ((pl.col("top") > 0.2) & (pl.col("top") < 0.9)).mean().round(4).alias("top_uncertain"),
              (pl.col("n_band") > 0).mean().round(4).alias("s1_with_band"),
              pl.col("selfF").mean().round(4).alias("selfF"))
           .join(d.group_by("country").agg(pl.col("inb").mean().round(4).alias("band_share"),
                                            (pl.len() / pl.col("idx").n_unique()).round(1).alias("cands")),
                 on="country")
           .join(dis, on="country").sort("country")
           .with_columns(pl.lit(label).alias("split")))
    return out, d, s1


def main():
    rows = []
    # HOLDOUT (labels available)
    ho = pl.read_parquet("work/eval/s4_full/pred_holdout.parquet")
    xs = pl.read_parquet("work/train/s4x_xenc/scores_holdout.parquet")
    s1h = (pl.read_parquet("work/train/s3_features/s1_scope.parquet").filter(pl.col("split") == "holdout")
             .select("s1_row").join(pl.read_parquet("work/train/s0_prepare/s1.parquet", columns=["s1_row", "country"]), on="s1_row"))
    d = final_p(ho.join(xs, on=["s1_row", "pool_row"], how="left"))
    out, dh, s1h = stats(d, s1h, "holdout")
    n_gt = s1h.join(pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt"),
                    on="s1_row", how="left").fill_null(0)["n_gt"].to_numpy()
    F = th.per_s1_f05(n_gt, dh["idx"].to_numpy(), dh["keep"].to_numpy(), dh["label"].to_numpy(), s1h.height)
    trueF = {c: round(float(F[s1h["country"].to_numpy() == c].mean()), 4) for c in ("India", "US")}
    rows.append(out.with_columns(pl.col("country").replace_strict(trueF, default=None).alias("trueF")))
    # TEST (no labels)
    w = "work/test/s5_infer"
    pr = pl.read_parquet(f"{w}/pred-part-*.parquet")
    xt = pl.read_parquet(f"{w}/xenc-part-*.parquet")
    s1t = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "country"])
    out, _, _ = stats(final_p(pr.join(xt, on=["s1_row", "pool_row"], how="left")), s1t, "test")
    rows.append(out.with_columns(pl.lit(None, pl.Float64).alias("trueF")))
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    print(pl.concat(rows).select("split", "country", "s1", "trueF", "selfF", "mean_pred", "empty", "top_uncertain",
                                 "s1_with_band", "band_share", "cands", "xenc_pos_rate", "xenc_vs_p1_disagree"))


if __name__ == "__main__":
    main()
