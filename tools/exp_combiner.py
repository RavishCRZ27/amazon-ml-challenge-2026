#!/usr/bin/env python3
"""Run-2 experiment: S1-context combiner (LightGBM) vs the logistic combiner, same xenc scores.

The logistic combiner sees one pair at a time ([logit p, xenc, logit p * xenc]). This one adds
the S1's context over its candidates: xenc rank and gap to the S1's best band pair, p rank,
number of confident (p > hi) candidates and band pairs. Both are cross-fitted over the same 2
folds of VAL S1s (s4x_xenc FOLDS, hash seed 7) for the (t, r) choice and refit on all VAL for
HOLDOUT (report only). Paired SE over the same S1s.

usage: .venv/bin/python tools/exp_combiner.py [--scores work/train/s4x_xenc] [--band 0.01 0.99]
"""

import argparse
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe                  # noqa: E402
import threshold as th                 # noqa: E402
from common import load_config         # noqa: E402

CTX = ["lp", "x", "lpx", "x_rank", "x_gap", "p_rank", "n_conf", "n_band"]


def context(d: pl.DataFrame, lo: float, hi: float) -> pl.DataFrame:
    inb = pl.col("p").is_between(lo, hi)
    xb = pl.when(inb).then(pl.col("xenc"))
    lp = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log()
    return d.with_columns(
        inb.alias("inb"), lp.alias("lp"), pl.col("xenc").alias("x"), (lp * pl.col("xenc")).alias("lpx"),
        xb.rank("ordinal", descending=True).over("s1_row").alias("x_rank"),
        (xb.max().over("s1_row") - xb).alias("x_gap"),
        pl.col("p").rank("ordinal", descending=True).over("s1_row").alias("p_rank"),
        (pl.col("p") > hi).sum().over("s1_row").alias("n_conf"),
        inb.sum().over("s1_row").alias("n_band"))


def fit(X, y, seed):
    import lightgbm as lgb
    return lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
                      "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
                      "verbose": -1, "num_threads": 0}, lgb.Dataset(X, y), num_boost_round=300)


def main(a):
    cfg = load_config()
    lo, hi = a.band
    inj = th.resolve_injective(cfg)
    tcfg = cfg["threshold"]
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    E = {}
    for name in ("val", "holdout"):
        sc, d = xe.ev_arrays(name, pl.read_parquet(f"work/eval/s4_full/pred_{name}.parquet"), scope, n_gt)
        d = d.join(pl.read_parquet(f"{a.scores}/scores_{name}.parquet").select("s1_row", "pool_row", "xenc"),
                   on=["s1_row", "pool_row"], how="left", maintain_order="left").with_columns(pl.col("xenc").fill_null(0.0))
        E[name] = (sc, context(d, lo, hi))
    sc, v = E["val"]
    inb = v["inb"].to_numpy()
    X = v.select(CTX).to_numpy().astype(np.float32)
    y = v["label"].to_numpy()
    fold = (sc["s1_row"].hash(seed=7) % xe.FOLDS).to_numpy()[v["idx"].to_numpy()]
    res = {}
    for kind in ("logistic", "context_gbm"):
        pv = v["p"].to_numpy().astype(np.float64).copy()
        for k in range(xe.FOLDS):
            tr, te = inb & (fold != k), inb & (fold == k)
            if kind == "logistic":
                pv[te] = xe.combine_apply(pv, v["xenc"].to_numpy(), te, xe.fit_combiner(v["p"].to_numpy()[tr], v["xenc"].to_numpy()[tr], y[tr]))[te]
            else:
                pv[te] = fit(X[tr], y[tr], cfg["seed"]).predict(X[te])
        if kind == "logistic":
            coef = xe.fit_combiner(v["p"].to_numpy()[inb], v["xenc"].to_numpy()[inb], y[inb])
        else:
            m = fit(X[inb], y[inb], cfg["seed"])
        out = {}
        for name in ("val", "holdout"):
            scn, d = E[name]
            p_ = pv if name == "val" else d["p"].to_numpy().astype(np.float64).copy()
            if name == "holdout":
                ib = d["inb"].to_numpy()
                if kind == "logistic":
                    p_ = xe.combine_apply(p_, d["xenc"].to_numpy(), ib, coef)
                else:
                    p_[ib] = m.predict(d.select(CTX).to_numpy().astype(np.float32)[ib])
            s1i, n = d["idx"].to_numpy(), scn.height
            c = th.Cands(s1i, d["pool_row"].to_numpy(), p_, n)
            yy, ng = d["label"].to_numpy().astype(np.float64), scn["n_gt"].to_numpy()
            if name == "val":
                ch = th.choose(th.sweep(c, yy, ng, tcfg["t_grid"], tcfg["relative_r"], [inj]), inj, tcfg["plateau_tol"])
            F = th.per_s1_f05(ng, s1i, th.select(c, ch["t"], ch["r"], inj), yy, n)
            cc = scn["country"].to_numpy()
            out[name] = {"F": F, "f05": round(float(F.mean()), 5), "se": round(th.se(F), 5),
                         "by_country": {k: round(float(F[cc == k].mean()), 5) for k in sorted(set(cc))}}
        out["choice"] = {k: ch[k] for k in ("t", "r", "plateau")}
        res[kind] = out
        print(kind, {n: {k: out[n][k] for k in ("f05", "se", "by_country")} for n in ("val", "holdout")}, out["choice"], flush=True)
    for name in ("val", "holdout"):
        F1, F0 = res["context_gbm"][name]["F"], res["logistic"][name]["F"]
        print(f"{name}{' (report only)' if name == 'holdout' else ''}: context - logistic = "
              f"{float((F1 - F0).mean()):+.5f} ± {th.paired_se(F1, F0):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="work/train/s4x_xenc")
    ap.add_argument("--band", type=float, nargs=2, default=[0.01, 0.99])
    main(ap.parse_args())
