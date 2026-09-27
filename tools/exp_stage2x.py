#!/usr/bin/env python3
"""Run-2 experiment: stage 2 AFTER the cross-encoder (CPU only; applies at S5 assembly, no GPU re-scoring).

Input per pair: pf = the current system's final probability (stage-1 p + cross-encoder xenc through the
VAL-fit combiner, output/artifacts/thresholds_final.json). Stage-2 features:
  A: pf, logit pf, stage-1 p, xenc, in-band flag; S1 context over all its candidates by pf (rank, max,
     2nd, count >= t, sum, gap to max); sibling evidence vs the S1's top OTHER candidate by pf (its pf,
     name ratio / token_set, address token_set, same-address flag); how many of the S1's candidates share
     this candidate's exact address / exact name.
  B: A + the 53 stage-1 features (S3).
LightGBM, cross-fitted over 2 folds of VAL S1s (xe.FOLDS, hash seed 7) with early stopping on 10% of the
training fold's S1s; (t, r) chosen on the cross-fitted VAL predictions; compared paired with the current
system (same combiner, its own VAL-chosen t). Final model = all VAL, rounds = mean best iteration, applied
to HOLDOUT once (report only). CPU threads capped (run under `nice`) so S5 on the GPU is not slowed.

usage: nice -n 10 .venv/bin/python tools/exp_stage2x.py [--threads 20]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, "src")
import s4x_xenc as xe                  # noqa: E402
import threshold as th                 # noqa: E402
from common import load_config         # noqa: E402

THR = json.loads(Path("output/artifacts/thresholds_final.json").read_text())
LO, HI = THR["xenc"]["band"]
A_FEATS = ["pf", "lpf", "p1", "xenc", "inb", "pf_rank", "pf_max", "pf_2nd", "pf_n_t", "pf_sum", "pf_gap",
           "ref_pf", "sib_n_ratio", "sib_n_tset", "sib_a_tset", "sib_same_addr", "n_same_addr", "n_same_name"]


def build(split, pool_txt, s3cols, nw):
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    sc, d = xe.ev_arrays(split, pl.read_parquet(f"work/eval/s4_full/pred_{split}.parquet"), scope, n_gt)
    d = d.join(pl.read_parquet(f"work/train/s4x_xenc/scores_{split}.parquet").select("s1_row", "pool_row", "xenc"),
               on=["s1_row", "pool_row"], how="left", maintain_order="left").with_columns(pl.col("xenc").fill_null(0.0))
    inb = ((d["p"] >= LO) & (d["p"] <= HI)).to_numpy()
    pf = xe.combine_apply(d["p"].to_numpy(), d["xenc"].to_numpy(), inb, THR["xenc"]["combiner"])
    d = d.with_columns(pl.Series("pf", pf), pl.Series("inb", inb.astype(np.float32)), pl.col("p").alias("p1"),
                       pl.int_range(pl.len()).alias("i"))
    t0 = time.time()
    g = d.with_columns(
        pl.col("pf").rank("ordinal", descending=True).over("idx").alias("pf_rank"),
        pl.col("pf").max().over("idx").alias("pf_max"),
        pl.col("pf").sort(descending=True).get(1, null_on_oob=True).over("idx").alias("pf_2nd"),
        (pl.col("pf") >= THR["t"]).sum().over("idx").alias("pf_n_t"),
        pl.col("pf").sum().over("idx").alias("pf_sum"))
    g = g.with_columns((pl.col("pf_max") - pl.col("pf")).alias("pf_gap"),
                       (pl.col("pf").clip(1e-6, 1 - 1e-6) / (1 - pl.col("pf").clip(1e-6, 1 - 1e-6))).log().alias("lpf"))
    top = g.filter(pl.col("pf_rank") <= 2).select("idx", "pf_rank", "pool_row", "pf")
    t1 = top.filter(pl.col("pf_rank") == 1).select("idx", pl.col("pool_row").alias("r1"), pl.col("pf").alias("q1"))
    t2 = top.filter(pl.col("pf_rank") == 2).select("idx", pl.col("pool_row").alias("r2"), pl.col("pf").alias("q2"))
    g = (g.join(t1, on="idx", how="left").join(t2, on="idx", how="left")
          .with_columns(pl.when(pl.col("pf_rank") == 1).then(pl.col("r2")).otherwise(pl.col("r1")).alias("ref"),
                        pl.when(pl.col("pf_rank") == 1).then(pl.col("q2")).otherwise(pl.col("q1")).alias("ref_pf"))
          .join(pool_txt, on="pool_row", how="left")
          .join(pool_txt.rename({"pool_row": "ref", "name_s": "name_r", "addr_n": "addr_r"}), on="ref", how="left")
          .with_columns(pl.col("name_s", "addr_n", "name_r", "addr_r").fill_null(""))
          .with_columns(pl.len().over("idx", "addr_n").alias("n_same_addr"), pl.len().over("idx", "name_s").alias("n_same_name"))
          .with_columns(pl.when(pl.col("addr_n") == "").then(None).otherwise(pl.col("n_same_addr")).alias("n_same_addr"),
                        ((pl.col("addr_n") == pl.col("addr_r")) & (pl.col("addr_n") != "")).cast(pl.Float32).alias("sib_same_addr"))
          .sort("i"))
    a, b = g["name_s"].to_list(), g["name_r"].to_list()
    x, y = g["addr_n"].to_list(), g["addr_r"].to_list()
    sib = {"sib_n_ratio": process.cpdist(a, b, scorer=fuzz.ratio, workers=nw, dtype=np.float32),
           "sib_n_tset": process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32),
           "sib_a_tset": process.cpdist(x, y, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32)}
    no_ref = g["ref"].is_null().to_numpy()
    empty_a = ((g["addr_n"] == "") | (g["addr_r"] == "")).to_numpy()
    sib["sib_a_tset"][empty_a] = np.nan
    for k in sib:
        sib[k][no_ref] = np.nan
    g = g.with_columns(*[pl.Series(k, v) for k, v in sib.items()])
    if s3cols:
        s3 = pl.read_parquet("work/train/s3_features/part-*.parquet", columns=["s1_row", "pool_row"] + s3cols)
        g = g.join(s3, on=["s1_row", "pool_row"], how="left", maintain_order="left")
    secs = time.time() - t0
    return sc, g, secs


def fit(X, y, Xes, yes, threads, seed):
    import lightgbm as lgb
    prm = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "verbose": -1,
           "num_threads": threads}
    m = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=3000, valid_sets=[lgb.Dataset(Xes, yes)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, prm


def main(a):
    import lightgbm as lgb
    cfg = load_config()
    tcfg, inj, seed = cfg["threshold"], th.resolve_injective(cfg), cfg["seed"]
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    s3cols = [f for f in json.loads(Path("output/artifacts/thresholds.json").read_text())["features"]]
    t0 = time.time()
    V, H = build("val", pool_txt, s3cols, a.threads), build("holdout", pool_txt, s3cols, a.threads)
    print(f"features built in {time.time() - t0:.0f}s (VAL {V[1].height:,} pairs: {V[2]:.0f}s)", flush=True)

    def score(sc, g, p_):
        c = th.Cands(g["idx"].to_numpy(), g["pool_row"].to_numpy(), p_, sc.height)
        return c, g["label"].to_numpy().astype(np.float64), sc["n_gt"].to_numpy()

    def at(sc, g, p_, t, r):
        c, y, ng = score(sc, g, p_)
        return th.per_s1_f05(ng, g["idx"].to_numpy(), th.select(c, t, r, inj), y, sc.height)

    def choose(sc, g, p_):
        c, y, ng = score(sc, g, p_)
        return th.choose(th.sweep(c, y, ng, tcfg["t_grid"], tcfg["relative_r"], [inj]), inj, tcfg["plateau_tol"])

    (scv, gv, _), (sch, gh, _) = V, H
    base_ch = choose(scv, gv, gv["pf"].to_numpy())             # current system (refit combiner), own t
    Fb_v = at(scv, gv, gv["pf"].to_numpy(), base_ch["t"], base_ch["r"])
    Fb_h = at(sch, gh, gh["pf"].to_numpy(), base_ch["t"], base_ch["r"])
    print(f"current: VAL {Fb_v.mean():.5f} (t {base_ch['t']}), HOLDOUT {Fb_h.mean():.5f}", flush=True)
    fold = (scv["s1_row"].hash(seed=7) % xe.FOLDS).to_numpy()[gv["idx"].to_numpy()]
    es = ((scv["s1_row"].hash(seed=11) % 10) == 0).to_numpy()[gv["idx"].to_numpy()]
    y = gv["label"].to_numpy()
    out = {}
    for name, feats in (("A_context_sibling", A_FEATS), ("B_plus_stage1", A_FEATS + s3cols)):
        X = gv.select(feats).to_numpy().astype(np.float32)
        p2 = np.empty(len(y))
        its = []
        t1 = time.time()
        for k in range(xe.FOLDS):
            tr, te = (fold != k) & ~es, fold == k
            esm = (fold != k) & es
            m, prm = fit(X[tr], y[tr], X[esm], y[esm], a.threads, seed)
            its.append(m.best_iteration)
            p2[te] = m.predict(X[te], num_iteration=m.best_iteration, num_threads=a.threads)
        ch = choose(scv, gv, p2)
        F2v = at(scv, gv, p2, ch["t"], ch["r"])
        mfull = lgb.train({**prm, "num_threads": a.threads}, lgb.Dataset(X, y), num_boost_round=int(np.mean(its)))
        p2h = mfull.predict(gh.select(feats).to_numpy().astype(np.float32), num_threads=a.threads)
        F2h = at(sch, gh, p2h, ch["t"], ch["r"])
        cc_v, cc_h = scv["country"].to_numpy(), sch["country"].to_numpy()
        res = {"val": round(float(F2v.mean()), 5), "dF_val": round(float((F2v - Fb_v).mean()), 5),
               "paired_se_val": round(th.paired_se(F2v, Fb_v), 5),
               "holdout": round(float(F2h.mean()), 5), "dF_holdout_report_only": round(float((F2h - Fb_h).mean()), 5),
               "paired_se_holdout": round(th.paired_se(F2h, Fb_h), 5),
               "dF_val_by_country": {c: round(float((F2v - Fb_v)[cc_v == c].mean()), 5) for c in sorted(set(cc_v))},
               "dF_holdout_by_country": {c: round(float((F2h - Fb_h)[cc_h == c].mean()), 5) for c in sorted(set(cc_h))},
               "choice": {k: ch[k] for k in ("t", "r", "plateau")}, "best_iterations": its,
               "fit_seconds": round(time.time() - t1, 1),
               "importance_top": sorted(zip(feats, mfull.feature_importance("gain").round(0).tolist()), key=lambda z: -z[1])[:12]}
        out[name] = res
        print(name, json.dumps(res, default=str), flush=True)
        mfull.save_model(f"work/eval/stage2x_{name}.txt")
    Path("work/eval/stage2x.json").write_text(json.dumps(
        {"current": {"val": round(float(Fb_v.mean()), 5), "holdout": round(float(Fb_h.mean()), 5), "choice": base_ch}, **out},
        indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=20)
    main(ap.parse_args())
