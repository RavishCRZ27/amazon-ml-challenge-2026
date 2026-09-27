#!/usr/bin/env python3
"""Run-2 experiment: a cross-encoder variant (backbone / lr / data size) vs the run-1 cross-encoder.

Trains on the same TRAIN pairs as S4x (run-1 stage-1 preds in work/eval/s4_full), scores the
VAL/HOLDOUT band pairs, evaluates with s4x_xenc.evaluate (combiner cross-fitted on VAL, (t, r)
chosen on VAL), and compares per-S1 F with the run-1 cross-encoder (work/train/s4x_xenc
scores, same pairs, same evaluation) by paired SE. HOLDOUT is reported, never used to choose.
Measures train and score pairs/s -> S5 cost for the 5.50M run-1 test band pairs.

usage: .venv/bin/python tools/exp_xenc.py --name rr --model-dir models/bge-reranker-v2-m3 --lr 2e-5 --bs 64
       .venv/bin/python tools/exp_xenc.py --name rr_w --model-from rr --band 0.002 0.998 [same model args]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe                                   # noqa: E402
import threshold as th                                  # noqa: E402
from common import atomic_write_parquet, load_config    # noqa: E402

TEST_BAND_PAIRS = 5_495_967        # run-1 S5: test pairs with stage-1 p in [.01, .99]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s - %(message)s", datefmt="%H:%M:%S")


def scores_for(d, f):
    return (d.select("s1_row", "pool_row").join(pl.read_parquet(f).select("s1_row", "pool_row", "xenc"),
                                                on=["s1_row", "pool_row"], how="left", maintain_order="left")
             ["xenc"].fill_null(0.0).to_numpy())


def main(a):
    cfg = load_config()
    xc = {**cfg["optional"]["cross_encoder"], "model_dir": a.model_dir, "lr": a.lr, "batch_size": a.bs,
          "epochs": a.epochs, "max_train": a.max_train, "max_len": a.max_len}
    if a.band:
        xc["band"] = a.band
    out = Path(f"work/eval/xenc_exp_{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "xc.json").write_text(json.dumps({**xc, "model_from": a.model_from}, indent=1))
    ev = Path("work/eval/s4_full")
    src = Path(f"work/eval/xenc_exp_{a.model_from}") if a.model_from else out
    t = time.time()
    tok, model = xe.train_xenc(ev, xc, cfg["seed"], src / "model")       # resumes if src/model exists
    t_train = round(time.time() - t, 1)
    lo, hi = xc["band"]
    prev_band = json.loads((src / "xc.json").read_text())["band"] if a.model_from else None
    RUN1 = Path("work/eval/xenc_exp_e5run1")           # frozen copy of run 1's cross-encoder scores + recipe
    base_band = json.loads((RUN1 / "xc.json").read_text())["band"]
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    E, B, rate = {}, {}, {}
    for name in ("val", "holdout"):
        sc, d = xe.ev_arrays(name, pl.read_parquet(ev / f"pred_{name}.parquet"), scope, n_gt)
        inb = ((d["p"] >= lo) & (d["p"] <= hi)).to_numpy()
        f = out / f"scores_{name}.parquet"
        if f.exists():
            s = scores_for(d, f)
        else:
            s = np.zeros(d.height, np.float32)
            todo = inb.copy()
            if prev_band:                         # reuse the source run's scores inside its band
                pb = ((d["p"] >= prev_band[0]) & (d["p"] <= prev_band[1])).to_numpy() & inb
                s[pb] = scores_for(d, src / f"scores_{name}.parquet")[pb]
                todo &= ~pb
            t = time.time()
            s[todo] = xe.score_pairs("train", d.filter(pl.Series(todo)), tok, model, xc["max_len"])
            rate[name] = round(float(todo.sum() / (time.time() - t)), 1)
            logging.info(f"{name}: scored {int(todo.sum()):,} new band pairs at {rate[name]:,.0f} pairs/s")
            atomic_write_parquet(d.select("s1_row", "pool_row", "p").with_columns(pl.Series("xenc", s)), f)
        assert np.isfinite(s).all()
        base = scores_for(d, RUN1 / f"scores_{name}.parquet")
        b_inb = ((d["p"] >= base_band[0]) & (d["p"] <= base_band[1])).to_numpy()   # run-1 band, not the variant's
        common = {"sc": sc, "d": d, "y": d["label"].to_numpy().astype(np.float64), "p": d["p"].to_numpy()}
        E[name], B[name] = {**common, "inb": inb, "s": s}, {**common, "inb": b_inb, "s": base}
    s4_thr = json.loads(Path("output/artifacts/thresholds.json").read_text())
    inj = th.resolve_injective(cfg)
    rep, _, F = xe.evaluate(E, s4_thr, cfg["threshold"], inj)
    rep0, _, F0 = xe.evaluate(B, s4_thr, cfg["threshold"], inj)
    r = rate.get("holdout") or rate.get("val")
    res = {"name": a.name, "xc": xc, "train_seconds": t_train, "score_pairs_per_s": rate,
           "s5_test_band_minutes_est": round(TEST_BAND_PAIRS / r / 60, 1) if r else None,
           "variant": {k: rep[k] for k in ("choice", "val", "holdout", "combiner")},
           "run1_xenc": {k: rep0[k] for k in ("choice", "val", "holdout")},
           "dF_val": round(float((F["val"] - F0["val"]).mean()), 5),
           "paired_se_val": round(th.paired_se(F["val"], F0["val"]), 5),
           "dF_holdout_report_only": round(float((F["holdout"] - F0["holdout"]).mean()), 5),
           "paired_se_holdout": round(th.paired_se(F["holdout"], F0["holdout"]), 5)}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("name", "train_seconds", "score_pairs_per_s", "s5_test_band_minutes_est",
                                          "dF_val", "paired_se_val", "dF_holdout_report_only", "paired_se_holdout")}))
    print("variant VAL", rep["val"]["f05"], rep["val"]["by_country"], "HOLDOUT", rep["holdout"]["f05"], rep["holdout"]["by_country"])
    print("run1    VAL", rep0["val"]["f05"], rep0["val"]["by_country"], "HOLDOUT", rep0["holdout"]["f05"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--band", type=float, nargs=2, help="override the scoring/combiner band")
    ap.add_argument("--model-from", help="reuse the trained model (and in-band scores) of xenc_exp_<name>")
    main(ap.parse_args())
