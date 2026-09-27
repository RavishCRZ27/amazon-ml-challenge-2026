#!/usr/bin/env python3
"""Where does a new cross-encoder change decisions vs run 1, and are the changes right?

HOLDOUT (labels): pairs dropped (run 1 kept, new drops) and added (new keeps, run 1 dropped), by pair
type, with the share that are true matches. TEST (no labels, from tools/diag_xenc_country.py's saved
sample): the same pair types per country. If France's changes fall in types whose HOLDOUT changes are
mostly right, the new model is likely right on France too; if France drops pairs of a type that are
mostly TRUE on HOLDOUT, that is a red flag.

Pair types: name exact (name_s equal), first house number equal, pool address empty.

usage: .venv/bin/python tools/diag_changes.py --exp rr_w002
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


def types(d, split):
    s0 = f"work/{split}/s0_prepare"
    s1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "name_s", "addr_nums"])
    pool = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "name_s", "addr_n", "addr_nums"])
    d = (d.join(s1, on="s1_row", how="left")
          .join(pool.rename({"name_s": "pname", "addr_nums": "pnums"}), on="pool_row", how="left"))
    return d.with_columns(
        (pl.col("name_s") == pl.col("pname")).fill_null(False).alias("name_exact"),
        (pl.col("addr_nums").list.first() == pl.col("pnums").list.first()).fill_null(False).alias("num_first_eq"),
        (pl.col("addr_n").fill_null("") == "").alias("pool_addr_empty"))


def summarize(d, by):
    return (d.filter(pl.col("change") != "same").group_by(by + ["change"])
              .agg(pl.len().alias("pairs"), *([pl.col("label").mean().round(3).alias("true_rate")] if "label" in d.columns else []))
              .sort(by + ["change"]))


def main(a):
    res = json.loads(Path(f"work/eval/xenc_exp_{a.exp}/result.json").read_text())
    v = res["variant"]
    band2 = res["xc"]["band"]
    r1 = json.loads(Path("work/eval/xenc_exp_e5run1/xc.json").read_text())
    run1 = json.loads(Path("submissions/1/thresholds_final.json").read_text())
    # HOLDOUT: both systems, same pairs
    ho = pl.read_parquet("work/eval/s4_full/pred_holdout.parquet")
    for tag, f in (("x1", "work/eval/xenc_exp_e5run1/scores_holdout.parquet"), ("x2", f"work/eval/xenc_exp_{a.exp}/scores_holdout.parquet")):
        ho = ho.join(pl.read_parquet(f).select("s1_row", "pool_row", pl.col("xenc").alias(tag)), on=["s1_row", "pool_row"], how="left")
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet").filter(pl.col("split") == "holdout").sort("s1_row")
    scope = scope.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    ho = ho.join(scope.select("s1_row", "idx", "country"), on="s1_row").sort("s1_row", "pool_row")
    p = ho["p"].to_numpy()
    for tag, band, coef, t, kk in (("x1", r1["band"], run1["xenc"]["combiner"], run1["t"], "k1"),
                                   ("x2", band2, v["combiner"], v["choice"]["t"], "k2")):
        inb = ((ho["p"] >= band[0]) & (ho["p"] <= band[1])).to_numpy()
        pf = xe.combine_apply(p, ho[tag].fill_null(0.0).to_numpy(), inb, coef)
        ho = ho.with_columns(pl.Series(kk, th.select(th.Cands(ho["idx"].to_numpy(), ho["pool_row"].to_numpy(), pf, scope.height), t, 0.0, True)))
    chg = pl.when(pl.col("k1") & ~pl.col("k2")).then(pl.lit("dropped")).when(~pl.col("k1") & pl.col("k2")).then(pl.lit("added")).otherwise(pl.lit("same"))
    ho = types(ho.with_columns(chg.alias("change")), "train")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    print("HOLDOUT changes (label = true match?):")
    print(summarize(ho, ["name_exact", "num_first_eq", "pool_addr_empty"]))
    # TEST sample saved by diag_xenc_country.py
    te = pl.read_parquet(f"work/eval/diag_xenc_country_{a.exp}.parquet")
    te = types(te.with_columns(chg.alias("change")), "test")
    print("TEST changes by country (no labels):")
    print(summarize(te, ["country", "name_exact", "num_first_eq", "pool_addr_empty"]))
    print(te.group_by("country").agg((pl.col("change") == "dropped").sum().alias("dropped"),
                                     (pl.col("change") == "added").sum().alias("added"),
                                     pl.col("s1_row").n_unique().alias("s1")).sort("country"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    main(ap.parse_args())
