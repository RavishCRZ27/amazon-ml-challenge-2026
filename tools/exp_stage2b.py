#!/usr/bin/env python3
"""Step-3 experiment: stage-2 GBM on stage-1 p + S1-group context + sibling similarity.

GT S1s have ~3.5 matches that are records of ONE business, so they resemble each other. Stage 2
adds, per pair: p1, its rank / gap / max / 2nd-max / count>=0.5 / sum within the S1, and the
name/address similarity of this candidate to the S1's most confident OTHER candidate.

Leakage control: TRAIN p1 is out-of-fold (2 folds by S1); ES / VAL / HOLDOUT p1 come from the
stage-1 model trained on all of TRAIN (out of sample for them). Threshold chosen on VAL for
both stages; compared by paired SE on VAL; HOLDOUT reported once. LightGBM (CPU) by default
so it can run beside a GPU job.

usage: .venv/bin/python tools/exp_stage2b.py [--backend lightgbm]
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
import s4_train as s4                                  # noqa: E402
import threshold as th                                 # noqa: E402
from common import load_config                         # noqa: E402

NON = {"s1_row", "pool_row", "label", "split"}


def group_feats(s: s4.Split, p1: np.ndarray, pool_txt: pl.DataFrame, nw: int) -> np.ndarray:
    d = pl.DataFrame({"i": np.arange(len(p1)), "s1": s.s1_idx, "pool_row": s.pool_row, "p1": p1})
    d = d.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1").alias("rk"),
        pl.col("p1").max().over("s1").alias("pmax"),
        pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("s1").alias("p2nd"),
        (pl.col("p1") >= 0.5).sum().over("s1").alias("n50"),
        pl.col("p1").sum().over("s1").alias("psum"))
    # reference = the S1's top candidate; for the top candidate itself, the runner-up
    top = d.filter(pl.col("rk") <= 2).select("s1", "rk", "pool_row", pl.col("p1").alias("p_ref"))
    t1 = top.filter(pl.col("rk") == 1).drop("rk").rename({"pool_row": "ref1", "p_ref": "p_ref1"})
    t2 = top.filter(pl.col("rk") == 2).drop("rk").rename({"pool_row": "ref2", "p_ref": "p_ref2"})
    d = (d.join(t1, on="s1", how="left").join(t2, on="s1", how="left")
          .with_columns(pl.when(pl.col("rk") == 1).then(pl.col("ref2")).otherwise(pl.col("ref1")).alias("ref"),
                        pl.when(pl.col("rk") == 1).then(pl.col("p_ref2")).otherwise(pl.col("p_ref1")).alias("p_ref"))
          .join(pool_txt, on="pool_row", how="left")
          .join(pool_txt.rename({"pool_row": "ref", "name_s": "name_r", "addr_n": "addr_r"}), on="ref", how="left")
          .sort("i"))
    has = d["ref"].is_not_null().to_numpy()
    a, b = d["name_s"].fill_null("").to_list(), d["name_r"].fill_null("").to_list()
    x, y = d["addr_n"].fill_null("").to_list(), d["addr_r"].fill_null("").to_list()
    sib = {"sib_n_ratio": process.cpdist(a, b, scorer=fuzz.ratio, workers=nw, dtype=np.float32),
           "sib_n_tset": process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32),
           "sib_a_tset": process.cpdist(x, y, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32)}
    ae = ((d["addr_n"].fill_null("") == "") | (d["addr_r"].fill_null("") == "")).to_numpy()
    sib["sib_a_tset"][ae] = np.nan
    for k in sib:
        sib[k][~has] = np.nan
    cols = [d[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in ("p1", "rk", "pmax", "p2nd", "n50", "psum", "p_ref")]
    return np.column_stack(cols + list(sib.values())).astype(np.float32)


G_NAMES = ["p1", "p1_rank", "p1_max", "p1_2nd", "p1_n50", "p1_sum", "p_ref",
           "sib_n_ratio", "sib_n_tset", "sib_a_tset"]


def main(backend):
    cfg = load_config()
    mcfg, tcfg, nw, seed = cfg["model"], cfg["threshold"], cfg["n_workers"], cfg["seed"]
    inj = th.resolve_injective(cfg)
    t0 = time.time()
    s3 = Path("work/train/s3_features")
    scope = pl.read_parquet(s3 / "s1_scope.parquet")
    df = pl.read_parquet(str(s3 / "part-*.parquet")).join(scope.select("s1_row", "split"), on="s1_row")
    feats = [c for c in df.columns if c not in NON]
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    sp = {n: s4.Split(df, scope, n_gt, feats, n) for n in ("train", "early_stop", "val", "holdout")}
    del df
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    tr, es, va, ho = sp["train"], sp["early_stop"], sp["val"], sp["holdout"]
    allc = np.arange(len(feats))
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    # stage 1: full model (ES/VAL/HOLDOUT p1) + 2 out-of-fold models (TRAIN p1)
    m1, _, b1 = s4.fit(backend, mcfg, tr, es, feats, allc, nw, seed)
    p1 = {n: s4.predict(backend, m1, sp[n].X, b1, nw) for n in ("early_stop", "val", "holdout")}
    fold = (pl.Series(tr.s1_row).hash(seed=11) % 2).to_numpy()[tr.s1_idx]
    p1["train"] = np.empty(tr.X.shape[0], np.float32)
    for k in (0, 1):
        m, _, b = s4.fit(backend, mcfg, s4._subset(tr, fold != k), es, feats, allc, nw, seed)
        p1["train"][fold == k] = s4.predict(backend, m, tr.X[fold == k], b, nw)
    ev1 = s4.evaluate(p1["val"], va, tcfg, inj)
    print(f"stage 1 ({backend}): VAL {s4.fse(ev1['F'])} at {ev1['choice']}", flush=True)

    # stage 2
    for n, s in sp.items():
        s.X = np.hstack([s.X, group_feats(s, p1[n], pool_txt, nw)])
    f2 = feats + G_NAMES
    m2, _, b2 = s4.fit(backend, mcfg, tr, es, f2, np.arange(len(f2)), nw, seed)
    p2v = s4.predict(backend, m2, va.X, b2, nw)
    ev2 = s4.evaluate(p2v, va, tcfg, inj)
    ch1, ch2 = ev1["choice"], ev2["choice"]
    rep2 = s4.full_report(p2v, va, ch2["t"], ch2["r"], inj)
    od = Path("work/eval/s4_stage2b")          # p2 for the xenc-stacking experiment (exp_stage2 --tag stage2b)
    od.mkdir(parents=True, exist_ok=True)
    p2h = s4.predict(backend, m2, ho.X, b2, nw)
    for name, s_, p_ in (("val", va, p2v), ("holdout", ho, p2h)):
        pl.DataFrame({"s1_row": s_.s1_row[s_.s1_idx], "pool_row": s_.pool_row, "p": p_,
                      "label": s_.y.astype(np.int8)}).write_parquet(od / f"pred_{name}.parquet")
    H1 = s4.at(p1["holdout"], ho, ch1["t"], ch1["r"], inj)[0]
    H2 = s4.at(p2h, ho, ch2["t"], ch2["r"], inj)[0]
    out = {"backend": backend,
           "stage1": {"val": s4.fse(ev1["F"]), "choice": ch1, "holdout": s4.fse(H1),
                      "val_loss": s4.full_report(p1["val"], va, ch1["t"], ch1["r"], inj)["loss"]},
           "stage2": {"val": s4.fse(ev2["F"]), "choice": ch2, "holdout": s4.fse(H2),
                      "val_by_country": rep2["by_country"], "val_loss": rep2["loss"]},
           "dF_val": round(float(np.mean(ev2["F"] - ev1["F"])), 5),
           "paired_se_val": round(th.paired_se(ev2["F"], ev1["F"]), 5),
           "dF_holdout_report_only": round(float(np.mean(H2 - H1)), 5),
           "importance_top": sorted(s4.importance(backend, m2, f2).items(), key=lambda x: -x[1])[:15]}
    print(json.dumps(out, indent=1, default=str), flush=True)
    Path("work/eval/stage2b.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lightgbm")
    main(ap.parse_args().backend)
