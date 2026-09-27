#!/usr/bin/env python3
"""Step-3 experiments on S4's scored pairs: stage-2 combiner and S1 emit gate.

Everything that learns is cross-fitted over 2 folds of VAL S1s, so VAL F stays honest; the
threshold is then chosen on VAL exactly like S4's baseline, and variants are compared by the
paired SE of per-S1 F. HOLDOUT is scored once at the VAL-chosen settings (report only).

  combiner: on band pairs (lo <= p <= hi), logistic regression of the label on
            [logit p, s_k, logit p * s_k] for extra scores s_k (xenc, llm); p' = p elsewhere.
  gate:     S1-level LightGBM on aggregates of p predicting "any GT among candidates";
            an S1 emits only if gate >= g (g swept with t).

usage: .venv/bin/python tools/exp_stage2.py --tag 50000 [--score work/eval/xenc_x] [--gate]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import threshold as th                   # noqa: E402
from common import load_config           # noqa: E402

FOLDS = 2


def ev_set(split: str, tag: str, scores=(), restrict=None):
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet").filter(pl.col("split") == split)
    if restrict:     # evaluate only on the S1s a subset scorer (e.g. the LLM) covered
        keep = pl.read_parquet(Path(restrict) / f"scores_{split}.parquet").select("s1_row").unique()
        scope = scope.join(keep, on="s1_row")
    sc = scope.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    sc = (sc.join(n_gt, on="s1_row", how="left", maintain_order="left")
            .with_columns(pl.col("n_gt").fill_null(0)).sort("idx"))
    d = pl.read_parquet(f"work/eval/s4_{tag}/pred_{split}.parquet").join(sc.select("s1_row", "idx"), on="s1_row")
    for s in scores:
        f = pl.read_parquet(Path(s) / f"scores_{split}.parquet")
        d = d.join(f, on=["s1_row", "pool_row"], how="left")
    fold = (sc["s1_row"].hash(seed=7) % FOLDS).to_numpy()
    return {"d": d, "n_s1": sc.height, "n_gt": sc["n_gt"].to_numpy(), "country": sc["country"].to_numpy(),
            "fold_s1": fold, "s1_idx": d["idx"].to_numpy(), "pool": d["pool_row"].to_numpy(),
            "y": d["label"].to_numpy().astype(np.float64), "p": d["p"].to_numpy()}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(lp, S):
    cols = [lp]
    for s in S:
        cols += [s, lp * s]
    return np.column_stack(cols)


def combine(ev, score_cols, band, fit_on=None):
    """p' for ev. fit_on=None: cross-fit within ev; else fit once on fit_on (VAL) and apply."""
    from sklearn.linear_model import LogisticRegression
    lo, hi = band
    d = ev["d"]
    inb = ((d["p"] >= lo) & (d["p"] <= hi)).to_numpy()
    S = [d[c].fill_null(0.0).to_numpy() for c in score_cols]
    X = design(logit(ev["p"]), S)
    p2 = ev["p"].astype(np.float64).copy()
    if fit_on is None:
        f = ev["fold_s1"][ev["s1_idx"]]
        for k in range(FOLDS):
            tr, te = inb & (f != k), inb & (f == k)
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], ev["y"][tr])
            p2[te] = m.predict_proba(X[te])[:, 1]
        return p2, None
    fd = fit_on["d"]
    finb = ((fd["p"] >= lo) & (fd["p"] <= hi)).to_numpy()
    FX = design(logit(fit_on["p"]), [fd[c].fill_null(0.0).to_numpy() for c in score_cols])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(FX[finb], fit_on["y"][finb])
    p2[inb] = m.predict_proba(X[inb])[:, 1]
    return p2, m


def s1_aggregates(ev, p):
    d = pl.DataFrame({"idx": ev["s1_idx"], "p": p})
    a = (d.sort("p", descending=True).group_by("idx", maintain_order=True)
          .agg(pl.col("p").first().alias("top1"), pl.col("p").get(1, null_on_oob=True).alias("top2"),
               (pl.col("p") >= 0.5).sum().alias("n50"), (pl.col("p") >= 0.9).sum().alias("n90"),
               pl.col("p").sum().alias("psum"), pl.len().alias("n")))
    full = pl.DataFrame({"idx": np.arange(ev["n_s1"])}).join(a, on="idx", how="left").fill_null(0)
    X = full.select(pl.exclude("idx")).with_columns((pl.col("top1") - pl.col("top2")).alias("gap")).to_numpy()
    return X.astype(np.float32)


def gate_scores(ev, p, fit_on=None, p_fit=None):
    import lightgbm as lgb
    X = s1_aggregates(ev, p)
    gtc = np.bincount(ev["s1_idx"], weights=ev["y"], minlength=ev["n_s1"])
    y = (gtc > 0).astype(np.float64)
    prm = {"objective": "binary", "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50,
           "verbosity": -1, "seed": 0}
    g = np.zeros(ev["n_s1"])
    if fit_on is None:
        for k in range(FOLDS):
            tr, te = ev["fold_s1"] != k, ev["fold_s1"] == k
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), 300)
            g[te] = m.predict(X[te])
        return g
    Xf = s1_aggregates(fit_on, p_fit)
    yf = (np.bincount(fit_on["s1_idx"], weights=fit_on["y"], minlength=fit_on["n_s1"]) > 0).astype(float)
    return lgb.train(prm, lgb.Dataset(Xf, yf), 300).predict(X)


def choose_and_score(ev, p, tcfg, inj, gate=None, g_grid=(0.0,)):
    c = th.Cands(ev["s1_idx"], ev["pool"], p, ev["n_s1"])
    best = None
    for g in g_grid:
        ok = np.ones(ev["n_s1"], bool) if gate is None else gate >= g
        rows = []
        for r in tcfg["relative_r"]:
            for t in th.t_values(tcfg["t_grid"]):
                keep = th.select(c, t, r, inj) & ok[ev["s1_idx"]]
                rows.append({"t": float(t), "r": float(r), "injective": inj,
                             "f05": float(th.per_s1_f05(ev["n_gt"], ev["s1_idx"], keep, ev["y"], ev["n_s1"]).mean())})
        ch = th.choose(rows, inj, tcfg["plateau_tol"])
        ch["g"] = g
        if best is None or ch["f05_at_t"] > best["f05_at_t"]:
            best = ch
    return best


def f_at(ev, p, ch, inj, gate=None):
    c = th.Cands(ev["s1_idx"], ev["pool"], p, ev["n_s1"])
    keep = th.select(c, ch["t"], ch["r"], inj)
    if gate is not None:
        keep &= (gate >= ch["g"])[ev["s1_idx"]]
    F = th.per_s1_f05(ev["n_gt"], ev["s1_idx"], keep, ev["y"], ev["n_s1"])
    gtc = np.bincount(ev["s1_idx"], weights=ev["y"], minlength=ev["n_s1"])
    loss = th.loss_decomposition(ev["n_gt"], gtc, ev["s1_idx"], keep, ev["y"], ev["n_s1"])
    return F, loss


def summ(F):
    return {"f05": round(float(F.mean()), 5), "se": round(th.se(F), 5)}


def main(tag, score_dirs, use_gate, band, restrict=None):
    cfg = load_config()
    tcfg, inj = cfg["threshold"], th.resolve_injective(cfg)
    va, ho = ev_set("val", tag, score_dirs, restrict), ev_set("holdout", tag, score_dirs, restrict)
    score_cols = [c for c in va["d"].columns if c not in ("s1_row", "pool_row", "p", "label", "idx")]
    out = {"tag": tag, "scores": score_cols, "band": band, "injective": inj, "restrict": restrict,
           "n_val_s1": va["n_s1"]}
    ch0 = choose_and_score(va, va["p"], tcfg, inj)
    F0, L0 = f_at(va, va["p"], ch0, inj)
    H0, _ = f_at(ho, ho["p"], ch0, inj)
    out["baseline"] = {"val": summ(F0), "holdout": summ(H0), "choice": ch0, "val_loss": L0}
    variants = {}
    if score_cols:
        variants["combiner"] = (combine(va, score_cols, band)[0], combine(ho, score_cols, band, fit_on=va)[0])
    variants_g = [("gate", va["p"], ho["p"])]
    if score_cols:
        variants_g.append(("combiner+gate", variants["combiner"][0], variants["combiner"][1]))
    for name, (pv, ph) in variants.items():
        ch = choose_and_score(va, pv, tcfg, inj)
        F, L = f_at(va, pv, ch, inj)
        H, _ = f_at(ho, ph, ch, inj)
        out[name] = {"val": summ(F), "dF_vs_base": round(float((F - F0).mean()), 5),
                     "paired_se": round(th.paired_se(F, F0), 5), "holdout": summ(H),
                     "holdout_dF": round(float((H - H0).mean()), 5), "choice": ch, "val_loss": L}
    if use_gate:
        g_grid = tuple(np.round(np.arange(0.0, 0.95, 0.1), 2))
        for name, pv, ph in variants_g:
            gv = gate_scores(va, pv)
            gh = gate_scores(ho, ph, fit_on=va, p_fit=pv)
            ch = choose_and_score(va, pv, tcfg, inj, gv, g_grid)
            F, L = f_at(va, pv, ch, inj, gv)
            H, _ = f_at(ho, ph, ch, inj, gh)
            out[name] = {"val": summ(F), "dF_vs_base": round(float((F - F0).mean()), 5),
                         "paired_se": round(th.paired_se(F, F0), 5), "holdout": summ(H),
                         "holdout_dF": round(float((H - H0).mean()), 5), "choice": ch, "val_loss": L}
    for k, v in out.items():
        if isinstance(v, dict) and "val" in v:
            print(f"{k:15s} VAL {v['val']}  dF {v.get('dF_vs_base', 0):+.5f} ± {v.get('paired_se', 0):.5f}  "
                  f"HOLDOUT {v['holdout']}  loss {v['val_loss']}")
    name = "_".join(["stage2", tag] + [Path(s).name for s in score_dirs] + (["gate"] if use_gate else [])
                    + (["restricted"] if restrict else []))
    Path(f"work/eval/{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> work/eval/{name}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--score", action="append", default=[], help="dir with scores_{val,holdout}.parquet")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--band", type=float, nargs=2, default=None)
    ap.add_argument("--restrict", help="score dir whose S1s define the evaluation subset")
    a = ap.parse_args()
    band = a.band or load_config()["optional"]["cross_encoder"]["band"]
    main(a.tag, a.score, a.gate, band, a.restrict)
