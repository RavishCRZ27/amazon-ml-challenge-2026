#!/usr/bin/env python3
"""S6 fit variance: refit stage 2 exactly as src/s6_stage2.py does (cross-fitted on VAL for the (t, r) choice,
refit on all VAL, HOLDOUT report only) with n LightGBM seeds, then the seed-averaged (bagged) predictions.
Seed 0 = the config seed, so its line must reproduce S6's own report. Diagnostic for run 3, whose S6 HOLDOUT
gain (+.00002) fell far below run 2's (+.00065) at a similar cross-fitted VAL gain (+.00039 vs +.00043).

usage: .venv/bin/python tools/exp_s6_seeds.py [n_seeds=4] [threads=n_workers]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import s6_stage2 as s6         # noqa: E402
import threshold as th         # noqa: E402
from common import load_config  # noqa: E402


def main(n_seeds, nw):
    import lightgbm as lgb
    cfg = load_config()
    nw = nw or cfg["n_workers"]
    thr = json.loads(Path("output/artifacts/thresholds_final.json").read_text())
    s3cols = thr["features"]
    s2, tcfg, seed0, inj = cfg["optional"]["stage2"], cfg["threshold"], cfg["seed"], thr["injective"]
    feats = s6.CTX + s3cols
    t0 = time.time()
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    scv, gv = s6.train_split("val", thr, s3cols, pool_txt, nw)
    sch, gh = s6.train_split("holdout", thr, s3cols, pool_txt, nw)
    print(f"{Path.cwd().name}: features in {time.time() - t0:.0f}s ({gv.height:,} VAL / {gh.height:,} HOLDOUT pairs)",
          flush=True)

    def F(sc, g, p_, t, r):
        c = th.Cands(g["idx"].to_numpy(), g["pool_row"].to_numpy(), p_, sc.height)
        return th.per_s1_f05(sc["n_gt"].to_numpy(), g["idx"].to_numpy(), th.select(c, t, r, inj),
                             g["label"].to_numpy().astype(np.float64), sc.height)

    def choose(p2):
        c = th.Cands(gv["idx"].to_numpy(), gv["pool_row"].to_numpy(), p2, scv.height)
        return th.choose(th.sweep(c, y.astype(np.float64), scv["n_gt"].to_numpy(), tcfg["t_grid"], tcfg["relative_r"],
                                  [inj]), inj, tcfg["plateau_tol"])

    def line(tag, p2, ph):
        ch = choose(p2)
        Fv, Fh = F(scv, gv, p2, ch["t"], ch["r"]), F(sch, gh, ph, ch["t"], ch["r"])
        print(f"{tag}: t {ch['t']}  VAL {Fv.mean():.5f} ({(Fv - Fv0).mean():+.5f} ± {th.paired_se(Fv, Fv0):.5f})  "
              f"HOLDOUT {Fh.mean():.5f} ({(Fh - Fh0).mean():+.5f} ± {th.paired_se(Fh, Fh0):.5f})", flush=True)
        return Fh

    X, y = gv.select(feats).to_numpy().astype(np.float32), gv["label"].to_numpy()
    Xh = gh.select(feats).to_numpy().astype(np.float32)
    fold = (scv["s1_row"].hash(seed=7) % xe.FOLDS).to_numpy()[gv["idx"].to_numpy()]
    es = ((scv["s1_row"].hash(seed=11) % 10) == 0).to_numpy()[gv["idx"].to_numpy()]
    Fv0, Fh0 = F(scv, gv, gv["pf"].to_numpy(), thr["t"], thr["r"]), F(sch, gh, gh["pf"].to_numpy(), thr["t"], thr["r"])
    print(f"S5 rule: VAL {Fv0.mean():.5f}  HOLDOUT {Fh0.mean():.5f}", flush=True)
    P2, PH, FH = [], [], []
    for i in range(n_seeds):
        t1 = time.time()
        prm = s6.lgb_params(s2, nw, seed0 + i)
        p2, its = np.empty(len(y)), []
        for k in range(xe.FOLDS):
            tr, te, esm = (fold != k) & ~es, fold == k, (fold != k) & es
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), num_boost_round=s2["max_rounds"],
                          valid_sets=[lgb.Dataset(X[esm], y[esm])], callbacks=[lgb.early_stopping(100, verbose=False)])
            its.append(m.best_iteration)
            p2[te] = m.predict(X[te], num_iteration=m.best_iteration)
        m = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=int(round(np.mean(its))))
        ph = m.predict(Xh)
        P2.append(p2)
        PH.append(ph)
        FH.append(line(f"seed {seed0 + i} its {its} ({time.time() - t1:.0f}s)", p2, ph))
        if i:
            line(f"  bagged seeds {seed0}..{seed0 + i}", np.mean(P2, axis=0), np.mean(PH, axis=0))
    if n_seeds > 1:
        g = np.array([f.mean() - Fh0.mean() for f in FH])
        print(f"single-seed HOLDOUT gain: mean {g.mean():+.5f}, sd {g.std(ddof=1):.5f} over {n_seeds} seeds")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
