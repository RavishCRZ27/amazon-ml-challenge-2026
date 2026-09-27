#!/usr/bin/env python3
"""Pairs whose POOL name carries a corporate-structure token that the S1 name lacks ("X HOLDING"
next to "X"): how often the final score keeps them. HOLDOUT (labels: true rate) vs test per country.

usage: .venv/bin/python tools/diag_corp_tokens.py
"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe  # noqa: E402

TOKS = ["HOLDING", "HOLDINGS", "GROUP", "GROUPE", "PARTICIPATIONS", "DISTRIBUTION", "INTERNATIONAL",
        "DEVELOPPEMENT", "DEVELOPMENT", "ASSOCIES", "ASSOCIATES", "ET", "AND", "VENTURES", "INDUSTRIES"]
thr = json.loads(open("output/artifacts/thresholds_final.json").read())
T, (LO, HI), comb = thr["t"], thr["xenc"]["band"], thr["xenc"]["combiner"]


def final(pred, xs):
    d = pred.join(xs, on=["s1_row", "pool_row"], how="left")
    inb = d["xenc"].is_not_null().to_numpy()
    return d.with_columns(pl.Series("pf", xe.combine_apply(d["p"].to_numpy(), d["xenc"].fill_null(0.0).to_numpy(), inb, comb)))


def tokens(d, split):
    s0 = f"work/{split}/s0_prepare"
    s1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "country", "name_n"])
    pool = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "name_n"])
    d = (d.join(s1, on="s1_row", how="left").join(pool, on="pool_row", how="left", suffix="_p")
          .with_columns(pl.col("name_n").str.split(" ").alias("t1"), pl.col("name_n_p").str.split(" ").alias("tp")))
    return d.with_columns([(pl.col("tp").list.contains(w) & ~pl.col("t1").list.contains(w)).alias(w) for w in TOKS])


def report(d, label, has_gt):
    rows = []
    for c in sorted(d["country"].unique()):
        x = d.filter(pl.col("country") == c)
        for w in TOKS:
            y = x.filter(pl.col(w))
            if y.height < 200:
                continue
            kept = y.filter(pl.col("pf") > T)
            r = {"split": label, "country": c, "tok": w, "pairs": y.height, "kept": kept.height,
                 "kept_rate": round(kept.height / y.height, 3)}
            if has_gt:
                r["true_rate"] = round(y["gt"].mean(), 3)
                r["kept_prec"] = round(kept["gt"].mean(), 3) if kept.height else None
            rows.append(r)
    return pl.DataFrame(rows)


pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
# HOLDOUT (labels)
ho = final(pl.read_parquet("work/eval/s4_full/pred_holdout.parquet").select("s1_row", "pool_row", "p"),
           pl.read_parquet("work/train/s4x_xenc/scores_holdout.parquet").select("s1_row", "pool_row", "xenc"))
gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").with_columns(pl.lit(True).alias("gt"))
ho = ho.join(gt, on=["s1_row", "pool_row"], how="left").with_columns(pl.col("gt").fill_null(False))
print(report(tokens(ho, "train"), "holdout", True))
# test: parts whose cross-encoder scores exist
from pathlib import Path  # noqa: E402
w = Path("work/test/s5_infer")
parts = [p for p in sorted(w.glob("pred-part-*.parquet")) if (w / p.name.replace("pred-", "xenc-")).exists()]
te = final(pl.concat([pl.read_parquet(p) for p in parts]),
           pl.concat([pl.read_parquet(w / p.name.replace("pred-", "xenc-")) for p in parts]))
print(f"test parts with xenc: {[p.name for p in parts]}")
print(report(tokens(te, "test"), "test", False))
