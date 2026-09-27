#!/usr/bin/env python3
"""S4 GBM training + threshold sweep. Rule: specs/train.md.

Train on TRAIN S1s' candidates with early stopping on ES (never VAL). VAL chooses (t, r) by
exact macro-F0.5 over ALL VAL S1s (full-GT recall, singletons in). HOLDOUT is scored once at
the chosen settings and is report-only: nothing is ever chosen on it.

Extras (reports only, never change the saved artifacts):
  --benchmark   train both backends on the same data (time, test-predict extrapolation, F)
  --loco        leave-one-country-out: a proxy for France, which has no labels
  --ablate P    retrain without feature columns starting with P; paired dF vs the full model
"""

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

import threshold as th
from common import atomic_write_parquet, code_hash, input_hash, load_config, stage

log = logging.getLogger("s4_train")
ART = Path("output/artifacts")
NON_FEAT = {"s1_row", "pool_row", "label"}
PRED_BATCH = 2_000_000


def gpu_free() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    others = {x.strip() for x in out.stdout.split()} - {"", str(os.getpid())}   # our own XGBoost context is fine
    return out.returncode == 0 and not others


class Split:
    """Rows of one split, aligned: X, y, dense S1 index over the split's FULL scope."""

    def __init__(self, df: pl.DataFrame, scope: pl.DataFrame, n_gt: pl.DataFrame, feats, name):
        sc = (scope.filter(pl.col("split") == name).sort("s1_row")
                   .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
                   .join(n_gt, on="s1_row", how="left", maintain_order="left")
                   .with_columns(pl.col("n_gt").fill_null(0)).sort("idx"))
        sub = (df.filter(pl.col("split") == name).join(sc.select("s1_row", "idx"), on="s1_row")
                 .sort("s1_row", "pool_row"))      # fixed row order: deterministic bagging
        self.name, self.feats = name, feats
        self.X = sub.select(feats).to_numpy().astype(np.float32, copy=False)
        self.y = sub["label"].to_numpy().astype(np.float64)
        self.s1_idx = sub["idx"].to_numpy()
        self.pool_row = sub["pool_row"].to_numpy()
        self.n_s1 = sc.height
        self.n_gt = sc["n_gt"].to_numpy()
        self.s1_row = sc["s1_row"].to_numpy()
        self.country = sc["country"].to_numpy()
        self.gtc = np.bincount(self.s1_idx, weights=self.y, minlength=self.n_s1)

    def cols(self, keep):
        return self.X[:, keep]


# ---------- backends ----------
def fit(backend, mcfg, tr, es, feats, keep, nw, seed):
    p = dict(mcfg[backend])
    n_est, esr = p.pop("n_estimators"), p.pop("early_stopping_rounds")
    t = time.time()
    if backend == "lightgbm":
        import lightgbm as lgb
        params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
                  "num_threads": nw, "seed": seed, **p}
        dtr = lgb.Dataset(tr.cols(keep), tr.y, feature_name=feats, free_raw_data=True)
        des = lgb.Dataset(es.cols(keep), es.y, reference=dtr)
        m = lgb.train(params, dtr, n_est, valid_sets=[des],
                      callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(250)])
        best = m.best_iteration
    else:
        import xgboost as xgb
        if not gpu_free():
            raise RuntimeError("xgboost backend needs a free GPU (nvidia-smi shows compute apps)")
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": seed,
                  "nthread": nw, **p}
        dtr = xgb.QuantileDMatrix(tr.cols(keep), tr.y, feature_names=feats)
        des = xgb.QuantileDMatrix(es.cols(keep), es.y, ref=dtr, feature_names=feats)
        m = xgb.train(params, dtr, n_est, evals=[(des, "es")], early_stopping_rounds=esr,
                      verbose_eval=250)
        best = m.best_iteration + 1
    secs = time.time() - t
    log.info(f"[{backend}] trained in {secs:.1f}s, best iteration {best} (cap {n_est})")
    if best > n_est - esr:      # early stopping could not have fired: the cap binds
        log.warning(f"[{backend}] best iteration {best} within {esr} of the n_estimators cap: raise it")
    return m, secs, best


def predict(backend, m, X, best, nw) -> np.ndarray:
    out = np.empty(X.shape[0], np.float32)
    if backend == "xgboost":
        import xgboost as xgb
    for i in range(0, X.shape[0], PRED_BATCH):
        xb = X[i:i + PRED_BATCH]
        if backend == "lightgbm":
            out[i:i + PRED_BATCH] = m.predict(xb, num_iteration=best, num_threads=nw)
        else:
            out[i:i + PRED_BATCH] = m.predict(xgb.DMatrix(xb, feature_names=m.feature_names),
                                              iteration_range=(0, best))
    return out


def importance(backend, m, feats):
    if backend == "lightgbm":
        return dict(zip(feats, m.feature_importance("gain").tolist()))
    g = m.get_score(importance_type="total_gain")
    return {f: float(g.get(f, 0.0)) for f in feats}


# ---------- evaluation ----------
def evaluate(p, ev: Split, tcfg, inj: bool):
    c = th.Cands(ev.s1_idx, ev.pool_row, p, ev.n_s1)
    rows = th.sweep(c, ev.y, ev.n_gt, tcfg["t_grid"], tcfg["relative_r"], [False, True])
    ch = {m: th.choose(rows, m, tcfg["plateau_tol"]) for m in (False, True)}
    F = {m: th.per_s1_f05(ev.n_gt, ev.s1_idx, th.select(c, ch[m]["t"], ch[m]["r"], m), ev.y, ev.n_s1)
         for m in (False, True)}
    return {"rows": rows, "choice": ch[inj], "choice_other": ch[not inj], "F": F[inj],
            "F_other": F[not inj], "cands": c}


def at(p, ev: Split, t, r, inj):
    c = th.Cands(ev.s1_idx, ev.pool_row, p, ev.n_s1)
    keep = th.select(c, t, r, inj)
    return th.per_s1_f05(ev.n_gt, ev.s1_idx, keep, ev.y, ev.n_s1), keep


def fse(F) -> dict:
    return {"f05": round(float(np.mean(F)), 5), "se": round(th.se(F), 5), "n_s1": int(len(F))}


def by_country(F, ev: Split) -> dict:
    return {c: fse(F[ev.country == c]) for c in sorted(set(ev.country.tolist()))}


def oracle(ev: Split):
    return th.per_s1_f05(ev.n_gt, ev.s1_idx, ev.y > 0, ev.y, ev.n_s1)


def full_report(p, ev: Split, t, r, inj) -> dict:
    F, keep = at(p, ev, t, r, inj)
    Fo = oracle(ev)
    P, n_pred, single = th.loss_parts(ev.n_gt, ev.gtc, ev.s1_idx, keep, ev.y, ev.n_s1)
    rnd = lambda d: {k: round(v, 5) for k, v in d.items()}
    return {"overall": fse(F), "by_country": by_country(F, ev), "oracle": fse(Fo),
            "oracle_by_country": by_country(Fo, ev),
            "loss": rnd(th.summarize_parts(P, n_pred, single, np.ones(ev.n_s1, bool))),
            "loss_by_country": {c: rnd(th.summarize_parts(P, n_pred, single, ev.country == c))
                                for c in sorted(set(ev.country.tolist()))}}


def costliest(p, ev: Split, t, r, inj, n=200) -> pl.DataFrame:
    F, keep = at(p, ev, t, r, inj)
    n_pred = np.bincount(ev.s1_idx[keep], minlength=ev.n_s1)
    tp = np.bincount(ev.s1_idx[keep], weights=ev.y[keep], minlength=ev.n_s1)
    return (pl.DataFrame({"s1_row": ev.s1_row, "country": ev.country, "n_gt": ev.n_gt,
                          "gt_in_cands": ev.gtc, "n_pred": n_pred, "tp": tp, "f05": F})
              .sort("f05", "s1_row").head(n))


def test_pairs_estimate(rows_per_s1: float) -> float:
    n = pl.scan_parquet("work/test/s0_prepare/s1.parquet").select(pl.len()).collect().item()
    return n * rows_per_s1


def main(split, limit_s1, force, benchmark=False, loco=False, ablate=()):
    cfg = load_config()
    if split != "train":
        log.info("s4_train only applies to train")
        return
    mcfg, tcfg, nw, seed = cfg["model"], cfg["threshold"], cfg["n_workers"], cfg["seed"]
    backend = mcfg["backend"]
    inj = th.resolve_injective(cfg)
    section = {"model": {"backend": backend, backend: mcfg[backend]}, "threshold": tcfg,
               "injective": inj, "limit_s1": limit_s1,
               "code": code_hash("src/s4_train.py", "src/threshold.py")}
    inputs = {"s3_features": input_hash("train", "s3_features"), "s1_gt": input_hash("train", "s1_gt")}
    force = force or benchmark or loco or bool(ablate)     # extras are reports: always run them
    with stage("s4_train", split, cfg, section, limit_s1=limit_s1, force=force,
               input_stages=inputs) as st:
        if st.skip:
            return
        t0 = time.time()
        s3 = Path("work/train/s3_features")
        scope = pl.read_parquet(s3 / "s1_scope.parquet")
        df = pl.read_parquet(str(s3 / "part-*.parquet")).join(scope.select("s1_row", "split"), on="s1_row")
        feats = [c for c in df.columns if c not in NON_FEAT | {"split"}]
        assert not any("country" in f for f in feats), feats
        n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
        sp = {n: Split(df, scope, n_gt, feats, n) for n in ("train", "early_stop", "val", "holdout")}
        rows_per_s1 = df.height / scope.height
        del df
        log.info(f"loaded {sum(s.X.shape[0] for s in sp.values()):,} rows x {len(feats)} features in "
                 f"{time.time() - t0:.1f}s; rows: { {k: v.X.shape[0] for k, v in sp.items()} }; "
                 f"injective={inj}")
        all_cols = np.arange(len(feats))
        tr, es, va, ho = sp["train"], sp["early_stop"], sp["val"], sp["holdout"]
        report = {"backend": backend, "injective": inj, "features": feats,
                  "caveat": "VAL S1s compete only with each other for pool rows, so the injective "
                            "gain on VAL understates the gain on test"}

        # ---- main model (+ benchmark of the other backend on the same data) ----
        backends = ["lightgbm", "xgboost"] if benchmark else [backend]
        bench, main = {}, None
        for b in backends:
            if b == "xgboost" and b != backend and not gpu_free():
                log.warning("benchmark: GPU busy or absent, xgboost skipped")
                continue
            m, train_s, best = fit(b, mcfg, tr, es, feats, all_cols, nw, seed)
            t = time.time()
            p = predict(b, m, va.X, best, nw)
            pred_s = time.time() - t
            ev = evaluate(p, va, tcfg, inj)
            per_row = pred_s / max(va.X.shape[0], 1)
            bench[b] = {"train_seconds": round(train_s, 1), "best_iteration": best,
                        "val_predict_seconds": round(pred_s, 2),
                        "test_predict_seconds_est": round(per_row * test_pairs_estimate(rows_per_s1), 1),
                        "val": fse(ev["F"]), "choice": ev["choice"]}
            log.info(f"[{b}] VAL F0.5 {bench[b]['val']} at {ev['choice']}; predict {pred_s:.1f}s "
                     f"-> test est {bench[b]['test_predict_seconds_est']:.0f}s")
            if b == backend:
                main = (m, best, p, ev)
            else:
                bench[b]["_F"] = ev["F"]
        if benchmark and len(bench) == 2:
            other = [b for b in bench if b != backend][0]
            F_other = bench[other].pop("_F")
            dF = float(np.mean(main[3]["F"] - F_other))
            pse = th.paired_se(main[3]["F"], F_other)
            cost = {b: v["train_seconds"] + v["test_predict_seconds_est"] for b, v in bench.items()}
            if abs(dF) > 2 * pse:
                winner = backend if dF > 0 else other
                why = f"|dF| {abs(dF):.4f} > 2 paired SE {2 * pse:.4f}: higher F wins"
            else:
                winner = min(cost, key=cost.get)
                why = f"|dF| {abs(dF):.4f} <= 2 paired SE {2 * pse:.4f}: faster wins ({cost})"
            bench["comparison"] = {f"dF_{backend}_minus_{other}": round(dF, 5), "paired_se": round(pse, 5),
                                   "winner": winner, "why": why}
            log.info(f"benchmark: winner {winner} ({why})")
        for v in bench.values():
            v.pop("_F", None)
        report["benchmark"] = bench

        m, best, p_val, ev = main
        ch = ev["choice"]
        t_, r_, = ch["t"], ch["r"]
        report["val"] = full_report(p_val, va, t_, r_, inj)
        report["choice"] = ch
        report["injective_on_vs_off"] = {
            "chosen_mode": inj, "other_mode_choice": ev["choice_other"],
            "dF_chosen_minus_other": round(float(np.mean(ev["F"] - ev["F_other"])), 5),
            "paired_se": round(th.paired_se(ev["F"], ev["F_other"]), 5)}
        curve = pl.DataFrame(ev["rows"])
        log.info(f"VAL: {report['val']['overall']} (oracle {report['val']['oracle']}) at t={t_} r={r_} "
                 f"inj={inj}; plateau {ch['plateau']}, argmax t={ch['argmax_t']} F={ch['f05_max']:.5f}")
        log.info(f"VAL by country: {report['val']['by_country']}")
        log.info(f"VAL loss decomposition: {report['val']['loss']}")
        log.info(f"VAL loss by country: {report['val']['loss_by_country']}")
        log.info(f"injective on vs off: {report['injective_on_vs_off']}")
        best_r = curve.filter((pl.col("r") == r_) & (pl.col("injective") == inj))
        log.info("VAL F curve (every 0.05): " + ", ".join(
            f"{a:.2f}:{b:.4f}" for a, b in best_r.filter(((pl.col("t") * 100).round() % 5) == 0)
                                              .select("t", "f05").iter_rows()))

        # ---- ablations (paired against the main model on VAL) ----
        report["ablations"] = {}
        for pre in ablate:
            keep = np.array([i for i, f in enumerate(feats) if not f.startswith(pre)])
            fa = [feats[i] for i in keep]
            ma, _, besta = fit(backend, mcfg, tr, es, fa, keep, nw, seed)
            eva = evaluate(predict(backend, ma, va.cols(keep), besta, nw), va, tcfg, inj)
            d = float(np.mean(ev["F"] - eva["F"]))
            report["ablations"][pre] = {"dropped": len(feats) - len(fa), "val_without": fse(eva["F"]),
                                        "dF_full_minus_ablated": round(d, 5),
                                        "paired_se": round(th.paired_se(ev["F"], eva["F"]), 5)}
            log.info(f"ablate {pre}: {report['ablations'][pre]}")

        # ---- leave-one-country-out (France proxy; diagnostic only) ----
        if loco:
            report["loco"] = {}
            for c in sorted(set(va.country.tolist())):
                trm = np.isin(tr.s1_idx, np.flatnonzero(tr.country != c))
                esm = np.isin(es.s1_idx, np.flatnonzero(es.country != c))
                if not trm.any() or trm.all():
                    continue
                sub_tr, sub_es = _subset(tr, trm), _subset(es, esm)
                mc, _, bc_ = fit(backend, mcfg, sub_tr, sub_es, feats, all_cols, nw, seed)
                pv = predict(backend, mc, va.X, bc_, nw)
                va_in, va_out = _subset_s1(va, va.country != c), _subset_s1(va, va.country == c)
                pin, pout = pv[_row_mask(va, va.country != c)], pv[_row_mask(va, va.country == c)]
                ch_in = evaluate(pin, va_in, tcfg, inj)["choice"]
                F_tr, _ = at(pout, va_out, ch_in["t"], ch_in["r"], inj)
                ev_out = evaluate(pout, va_out, tcfg, inj)
                F_main = at(p_val, va, t_, r_, inj)[0][va.country == c]
                report["loco"][c] = {"trained_on_other_countries": fse(F_tr),
                                     "at_its_own_optimum": fse(ev_out["F"]),
                                     "in_country_model": fse(F_main),
                                     "transferred_t_r": [ch_in["t"], ch_in["r"]],
                                     "own_optimum_t_r": [ev_out["choice"]["t"], ev_out["choice"]["r"]]}
                log.info(f"LOCO held-out {c}: {report['loco'][c]}")

        # ---- HOLDOUT: once, at the settings chosen on VAL; report only ----
        t = time.time()
        p_ho = predict(backend, m, ho.X, best, nw)
        report["holdout"] = full_report(p_ho, ho, t_, r_, inj)
        report["holdout"]["note"] = "report only; never used for any choice"
        log.info(f"HOLDOUT (report only): {report['holdout']['overall']} by country "
                 f"{report['holdout']['by_country']} oracle {report['holdout']['oracle']} "
                 f"({time.time() - t:.1f}s)")
        log.info(f"HOLDOUT loss decomposition: {report['holdout']['loss']}")

        # ---- artifacts ----
        ev_dir = Path(f"work/eval/s4_{limit_s1 or 'full'}")
        ev_dir.mkdir(parents=True, exist_ok=True)
        art = ART if limit_s1 is None else ev_dir / "artifacts"    # slices never touch output/
        art.mkdir(parents=True, exist_ok=True)
        model_file = art / ("model.txt" if backend == "lightgbm" else "model.json")
        stale = art / ("model.json" if backend == "lightgbm" else "model.txt")
        if backend == "lightgbm":
            m.save_model(str(model_file), num_iteration=best)
        else:
            m.save_model(str(model_file))
        stale.unlink(missing_ok=True)
        imp = sorted(importance(backend, m, feats).items(), key=lambda x: -x[1])
        (art / "feature_importance.tsv").write_text(
            "feature\tgain\n" + "".join(f"{f}\t{g:.1f}\n" for f, g in imp))
        thr = {"backend": backend, "model_file": str(model_file), "features": feats,
               "best_iteration": best, "t": t_, "r": r_, "injective": inj,
               "plateau": ch["plateau"], "val": report["val"]["overall"],
               "holdout": report["holdout"]["overall"], "limit_s1": limit_s1,
               "s3_scope_s1": scope.height, "s4_hash": st.hash}
        tmp = art / "thresholds.json.tmp"
        tmp.write_text(json.dumps(thr, indent=1))
        os.replace(tmp, art / "thresholds.json")
        # scored pairs for the stage-2 experiments (emit gate, cross-encoder, LLM band)
        p_tr = predict(backend, m, tr.X, best, nw)
        for name, s_, p_ in (("train", tr, p_tr), ("val", va, p_val), ("holdout", ho, p_ho)):
            atomic_write_parquet(pl.DataFrame({"s1_row": s_.s1_row[s_.s1_idx], "pool_row": s_.pool_row,
                                               "p": p_, "label": s_.y.astype(np.int8)}),
                                 ev_dir / f"pred_{name}.parquet")
        curve.write_csv(ev_dir / "curve.tsv", separator="\t")
        atomic_write_parquet(costliest(p_val, va, t_, r_, inj), ev_dir / "costliest_val_s1.parquet")
        (ev_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        (st.work_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        log.info(f"top features: {[f for f, _ in imp[:12]]}")
        st.rows = int(tr.X.shape[0])


def _subset(s: Split, row_mask) -> Split:
    """Row subset for training (only X, y are used by fit)."""
    o = object.__new__(Split)
    o.X, o.y = s.X[row_mask], s.y[row_mask]
    return o


def _row_mask(s: Split, s1_mask) -> np.ndarray:
    return s1_mask[s.s1_idx]


def _subset_s1(s: Split, s1_mask) -> Split:
    """S1 subset for evaluation: re-indexes S1s densely and keeps their rows."""
    o = object.__new__(Split)
    new = np.full(s.n_s1, -1, np.int64)
    new[s1_mask] = np.arange(int(s1_mask.sum()))
    rm = _row_mask(s, s1_mask)
    o.X, o.y = s.X[rm], s.y[rm]
    o.s1_idx, o.pool_row = new[s.s1_idx[rm]], s.pool_row[rm]
    o.n_s1, o.n_gt, o.gtc = int(s1_mask.sum()), s.n_gt[s1_mask], s.gtc[s1_mask]
    o.s1_row, o.country = s.s1_row[s1_mask], s.country[s1_mask]
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit-s1", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="also train the other backend")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out diagnostic")
    ap.add_argument("--ablate", action="append", default=[], help="feature prefix to drop (repeatable)")
    a = ap.parse_args()
    main(a.split, a.limit_s1, a.force, a.benchmark, a.loco, tuple(a.ablate))
