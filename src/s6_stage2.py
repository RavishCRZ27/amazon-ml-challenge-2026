#!/usr/bin/env python3
"""S6 optional: stage 2 after the cross-encoder (CPU). Rule: specs/optional-stages.md (stage2).

Stage 2 re-scores every candidate pair from the final S5 probability pf (stage-1 p + cross-encoder xenc
through the VAL-fit combiner), the S1's context over its candidates (rank, max, 2nd, count >= t, sum,
gap), sibling evidence vs the S1's top OTHER candidate (its pf, name ratio / token_set, address
token_set, same address), how many of the S1's candidates share the exact address / name, and the 53
stage-1 features. Never `country`.

Fit (train artifacts): LightGBM on VAL pairs, cross-fitted over the combiner's 2 VAL S1 folds (early
stopping on 10% of each training fold) to choose (t, r) honestly; refit on all VAL (rounds = mean best
iteration); HOLDOUT scored once (report only). The fitted model is checkpointed in the work dir.
Apply (test): per S5 part (all candidates of an S1 live in one part) -> p2 checkpoints; then
threshold.select over all test S1s (t, r, injective) -> both TSVs, exactly like S5.
Run-2 evidence (tools/exp_stage2x.py, variant B): VAL +.00043 +/- .00009 paired, HOLDOUT +.00065.
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

import s4x_xenc as xe
import threshold as th
from common import atomic_write_parquet, code_hash, input_hash, load_config, load_done, stage
from s5_infer import write_tsv

log = logging.getLogger("s6_stage2")
ART = Path("output/artifacts")
CTX = ["pf", "lpf", "p1", "xenc", "inb", "pf_rank", "pf_max", "pf_2nd", "pf_n_t", "pf_sum", "pf_gap",
       "ref_pf", "sib_n_ratio", "sib_n_tset", "sib_a_tset", "sib_same_addr", "n_same_addr", "n_same_name"]


def features(d: pl.DataFrame, pool_txt: pl.DataFrame, t: float, nw: int) -> pl.DataFrame:
    """d: candidate pairs (s1_row, pool_row, p1, xenc, inb, pf) sorted by (s1_row, pool_row), holding ALL
    candidates of each S1 present. Returns d in the same row order with the CTX columns added."""
    d = d.with_columns(pl.int_range(pl.len()).alias("_i"))
    g = d.with_columns(
        pl.col("pf").rank("ordinal", descending=True).over("s1_row").alias("pf_rank"),
        pl.col("pf").max().over("s1_row").alias("pf_max"),
        pl.col("pf").sort(descending=True).get(1, null_on_oob=True).over("s1_row").alias("pf_2nd"),
        (pl.col("pf") >= t).sum().over("s1_row").alias("pf_n_t"),
        pl.col("pf").sum().over("s1_row").alias("pf_sum"))
    g = g.with_columns((pl.col("pf_max") - pl.col("pf")).alias("pf_gap"),
                       (pl.col("pf").clip(1e-6, 1 - 1e-6) / (1 - pl.col("pf").clip(1e-6, 1 - 1e-6))).log().alias("lpf"))
    top = g.filter(pl.col("pf_rank") <= 2).select("s1_row", "pf_rank", "pool_row", "pf")
    t1 = top.filter(pl.col("pf_rank") == 1).select("s1_row", pl.col("pool_row").alias("r1"), pl.col("pf").alias("q1"))
    t2 = top.filter(pl.col("pf_rank") == 2).select("s1_row", pl.col("pool_row").alias("r2"), pl.col("pf").alias("q2"))
    g = (g.join(t1, on="s1_row", how="left").join(t2, on="s1_row", how="left")
          .with_columns(pl.when(pl.col("pf_rank") == 1).then(pl.col("r2")).otherwise(pl.col("r1")).alias("ref"),
                        pl.when(pl.col("pf_rank") == 1).then(pl.col("q2")).otherwise(pl.col("q1")).alias("ref_pf"))
          .join(pool_txt, on="pool_row", how="left")
          .join(pool_txt.rename({"pool_row": "ref", "name_s": "name_r", "addr_n": "addr_r"}), on="ref", how="left")
          .with_columns(pl.col("name_s", "addr_n", "name_r", "addr_r").fill_null(""))
          .with_columns(pl.len().over("s1_row", "addr_n").alias("n_same_addr"),
                        pl.len().over("s1_row", "name_s").alias("n_same_name"))
          .with_columns(pl.when(pl.col("addr_n") == "").then(None).otherwise(pl.col("n_same_addr")).alias("n_same_addr"),
                        ((pl.col("addr_n") == pl.col("addr_r")) & (pl.col("addr_n") != "")).cast(pl.Float32).alias("sib_same_addr"))
          .sort("_i"))
    a, b = g["name_s"].to_list(), g["name_r"].to_list()
    x, y = g["addr_n"].to_list(), g["addr_r"].to_list()
    sib = {"sib_n_ratio": process.cpdist(a, b, scorer=fuzz.ratio, workers=nw, dtype=np.float32),
           "sib_n_tset": process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32),
           "sib_a_tset": process.cpdist(x, y, scorer=fuzz.token_set_ratio, workers=nw, dtype=np.float32)}
    no_ref = g["ref"].is_null().to_numpy()
    sib["sib_a_tset"][((g["addr_n"] == "") | (g["addr_r"] == "")).to_numpy()] = np.nan
    for k in sib:
        sib[k][no_ref] = np.nan
    g = g.with_columns(*[pl.Series(k, v) for k, v in sib.items()])
    return g.drop("_i", "name_s", "addr_n", "name_r", "addr_r", "r1", "r2", "q1", "q2", "ref")


def combined(d: pl.DataFrame, band, coef) -> pl.DataFrame:
    """Add inb / pf exactly as S4x and S5 compute them (xenc = 0 off-band)."""
    inb = ((d["p1"] >= band[0]) & (d["p1"] <= band[1])).to_numpy()
    pf = xe.combine_apply(d["p1"].to_numpy(), d["xenc"].to_numpy(), inb, coef)
    return d.with_columns(pl.Series("pf", pf), pl.Series("inb", inb.astype(np.float32)))


def lgb_params(s2, nw, seed):
    return {"objective": "binary", "learning_rate": s2["learning_rate"], "num_leaves": s2["num_leaves"],
            "min_data_in_leaf": s2["min_data_in_leaf"], "feature_fraction": s2["feature_fraction"],
            "bagging_fraction": s2["bagging_fraction"], "bagging_freq": 1, "seed": seed, "verbose": -1,
            "num_threads": nw, "deterministic": True, "force_row_wise": True}


def train_split(name, thr, s3cols, pool_txt, nw):
    """VAL/HOLDOUT pairs with stage-2 features, labels and the eval scope (S1 index, n_gt, country)."""
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    sc, d = xe.ev_arrays(name, pl.read_parquet(f"work/eval/s4_full/pred_{name}.parquet"), scope, n_gt)
    d = (d.rename({"p": "p1"})
          .join(pl.read_parquet(f"work/train/s4x_xenc/scores_{name}.parquet").select("s1_row", "pool_row", "xenc"),
                on=["s1_row", "pool_row"], how="left", maintain_order="left")
          .with_columns(pl.col("xenc").fill_null(0.0)))
    d = features(combined(d, thr["xenc"]["band"], thr["xenc"]["combiner"]), pool_txt, thr["t"], nw)
    s3 = pl.read_parquet("work/train/s3_features/part-*.parquet", columns=["s1_row", "pool_row"] + s3cols)
    return sc, d.join(s3, on=["s1_row", "pool_row"], how="left", maintain_order="left")


def fit(st, cfg, thr, s3cols, nw):
    """Cross-fit on VAL -> (t, r); refit on all VAL; HOLDOUT once. Checkpointed model (resume-safe)."""
    import lightgbm as lgb
    s2, tcfg, seed = cfg["optional"]["stage2"], cfg["threshold"], cfg["seed"]
    inj = thr["injective"]
    feats = CTX + s3cols
    mfile, rfile = st.work_dir / "stage2.txt", st.work_dir / "stage2_report.json"
    if mfile.exists() and rfile.exists():                   # checkpoint from an interrupted run of this hash
        log.info(f"reusing fitted stage-2 model {mfile}")
        return lgb.Booster(model_file=str(mfile)), json.loads(rfile.read_text())
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    scv, gv = train_split("val", thr, s3cols, pool_txt, nw)
    sch, gh = train_split("holdout", thr, s3cols, pool_txt, nw)

    def F(sc, g, p_, t, r):
        c = th.Cands(g["idx"].to_numpy(), g["pool_row"].to_numpy(), p_, sc.height)
        return th.per_s1_f05(sc["n_gt"].to_numpy(), g["idx"].to_numpy(), th.select(c, t, r, inj),
                             g["label"].to_numpy().astype(np.float64), sc.height)

    X, y = gv.select(feats).to_numpy().astype(np.float32), gv["label"].to_numpy()
    fold = (scv["s1_row"].hash(seed=7) % xe.FOLDS).to_numpy()[gv["idx"].to_numpy()]
    es = ((scv["s1_row"].hash(seed=11) % 10) == 0).to_numpy()[gv["idx"].to_numpy()]
    prm = lgb_params(s2, nw, seed)
    p2, its = np.empty(len(y)), []
    for k in range(xe.FOLDS):
        tr, te, esm = (fold != k) & ~es, fold == k, (fold != k) & es
        m = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), num_boost_round=s2["max_rounds"],
                      valid_sets=[lgb.Dataset(X[esm], y[esm])], callbacks=[lgb.early_stopping(100, verbose=False)])
        its.append(m.best_iteration)
        p2[te] = m.predict(X[te], num_iteration=m.best_iteration)
    c = th.Cands(gv["idx"].to_numpy(), gv["pool_row"].to_numpy(), p2, scv.height)
    ch = th.choose(th.sweep(c, y.astype(np.float64), scv["n_gt"].to_numpy(), tcfg["t_grid"], tcfg["relative_r"], [inj]),
                   inj, tcfg["plateau_tol"])
    Fv, Fv0 = F(scv, gv, p2, ch["t"], ch["r"]), F(scv, gv, gv["pf"].to_numpy(), thr["t"], thr["r"])
    m = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=int(round(np.mean(its))))
    p2h = m.predict(gh.select(feats).to_numpy().astype(np.float32))
    Fh, Fh0 = F(sch, gh, p2h, ch["t"], ch["r"]), F(sch, gh, gh["pf"].to_numpy(), thr["t"], thr["r"])
    cv, chh = scv["country"].to_numpy(), sch["country"].to_numpy()
    rep = {"features": feats, "rounds": int(round(np.mean(its))), "best_iterations": its,
           "choice": {k: ch[k] for k in ("t", "r", "plateau", "argmax_t")}, "injective": inj,
           "val": {"f05": round(float(Fv.mean()), 5), "se": round(th.se(Fv), 5),
                   "dF_vs_s5": round(float((Fv - Fv0).mean()), 5), "paired_se": round(th.paired_se(Fv, Fv0), 5),
                   "by_country": {cc: round(float(Fv[cv == cc].mean()), 5) for cc in sorted(set(cv))}},
           "holdout": {"f05": round(float(Fh.mean()), 5), "se": round(th.se(Fh), 5),
                       "dF_vs_s5": round(float((Fh - Fh0).mean()), 5), "paired_se": round(th.paired_se(Fh, Fh0), 5),
                       "by_country": {cc: round(float(Fh[chh == cc].mean()), 5) for cc in sorted(set(chh))}}}
    log.info(f"stage 2 VAL (cross-fitted): {rep['val']}")
    log.info(f"stage 2 HOLDOUT (report only): {rep['holdout']}")
    log.info(f"chosen {rep['choice']}")
    tmp = mfile.with_suffix(".tmp")
    m.save_model(str(tmp))
    tmp.replace(mfile)
    write_json(rfile, rep)                                  # after the model: report present => model complete
    return m, rep


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


TSVS = (Path("output/matching_results.tsv"), Path("output/candidate_pairs.tsv"))


def main(split, limit_s1, force):
    cfg = load_config()
    s2 = cfg["optional"].get("stage2", {"enabled": False})
    if split != "test" or not s2.get("enabled"):
        log.info("s6_stage2 applies to test, when optional.stage2 is enabled")
        return
    if not cfg["optional"]["cross_encoder"]["enabled"]:
        raise RuntimeError("stage 2 is defined on top of the cross-encoder (enable optional.cross_encoder)")
    if limit_s1 is not None:
        log.info("s6_stage2 runs on the full test set only: skipped for --limit-s1 slices")
        return
    nw = cfg["n_workers"]
    thr = json.loads((ART / "thresholds_final.json").read_text())
    s3cols = thr["features"]                                # stage-1 features, guarded by s4x_hash below
    s5_done = load_done(Path("work/test/s5_infer/_DONE.json"))
    inputs = {"s4x_xenc": input_hash("train", "s4x_xenc"), "s5_infer": input_hash("test", "s5_infer"),
              # any re-run of S5 rewrites output/*.tsv, so it must invalidate S6 even at an unchanged S5 hash
              "s5_run": f"{s5_done['timestamp']:.6f}",
              "s3_features_train": input_hash("train", "s3_features"), "s3_features_test": input_hash("test", "s3_features")}
    assert thr.get("s4x_hash") == inputs["s4x_xenc"], "thresholds_final.json is not from the finished S4x run"
    section = {"stage2": s2, "threshold": cfg["threshold"], "code": code_hash("src/s6_stage2.py", "src/threshold.py",
                                                                            "src/s4x_xenc.py", "src/s5_infer.py")}
    with stage("s6_stage2", split, cfg, section, limit_s1=None, force=force, input_stages=inputs) as st:
        if st.skip:                                         # the TSVs in output/ must still be S6's
            want = json.loads((st.work_dir / "tsv_md5.json").read_text())
            got = {p.name: md5(p) for p in TSVS}
            if got != want:
                raise RuntimeError(f"output TSVs are not S6's (overwritten after S6?): {got} != {want}; "
                                   "rerun `src/s6_stage2.py --force`")
            return
        t0 = time.time()
        model, rep = fit(st, cfg, thr, s3cols, nw)
        feats = rep["features"]
        log.info(f"fit/report in {time.time() - t0:.1f}s")
        s5 = Path("work/test/s5_infer")
        pool_txt = pl.read_parquet("work/test/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
        names = sorted(p.name.removeprefix("pred-") for p in s5.glob("pred-part-*.parquet"))   # S5's parts
        s3 = Path("work/test/s3_features")
        missing = [n for n in names if not (s3 / n).exists() or not (s5 / f"xenc-{n}").exists()]
        assert names and not missing, f"S3/S5 parts missing for S6: {missing[:5]}"
        t0 = time.time()
        for name in names:                                   # per-part p2 checkpoints: resumable
            out = st.work_dir / f"p2-{name}"
            if out.exists():
                continue
            d = (pl.read_parquet(s5 / f"pred-{name}").rename({"p": "p1"})
                   .join(pl.read_parquet(s5 / f"xenc-{name}"), on=["s1_row", "pool_row"], how="left")
                   .with_columns(pl.col("xenc").fill_null(0.0)).sort("s1_row", "pool_row"))
            d = features(combined(d, thr["xenc"]["band"], thr["xenc"]["combiner"]), pool_txt, thr["t"], nw)
            d = d.join(pl.read_parquet(s3 / name, columns=["s1_row", "pool_row"] + s3cols), on=["s1_row", "pool_row"],
                       how="left", maintain_order="left")
            assert d.select(pl.col(s3cols).null_count()).sum_horizontal().item() == 0, f"S3 join gaps in {name}"
            p2 = model.predict(d.select(feats).to_numpy().astype(np.float32))
            assert np.isfinite(p2).all(), f"non-finite stage-2 scores in {name}"
            atomic_write_parquet(d.select("s1_row", "pool_row", "pf").with_columns(pl.Series("p", p2.astype(np.float32))), out)
        pred = pl.read_parquet(str(st.work_dir / "p2-part-*.parquet"))
        n5 = pl.scan_parquet(str(s5 / "pred-part-*.parquet")).select(pl.len()).collect().item()
        assert pred.height == n5, f"S6 scored {pred.height:,} pairs, S5 has {n5:,}"
        log.info(f"stage 2 scored {pred.height:,} pairs from {len(names)} parts in {time.time() - t0:.1f}s")

        s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "entity_id", "country"])
        assert s5_done["rows"] == s1.height, "S5 did not cover every test S1"
        s1 = s1.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
        pred = pred.join(s1.select("s1_row", "idx"), on="s1_row").sort("s1_row", "pool_row")
        ch = rep["choice"]
        keep = th.select(th.Cands(pred["idx"].to_numpy(), pred["pool_row"].to_numpy(), pred["p"].to_numpy(), s1.height),
                         ch["t"], ch["r"], rep["injective"])
        keep0 = th.select(th.Cands(pred["idx"].to_numpy(), pred["pool_row"].to_numpy(), pred["pf"].to_numpy(), s1.height),
                          thr["t"], thr["r"], thr["injective"])
        pred = pred.with_columns(pl.Series("keep", keep), pl.Series("keep0", keep0))
        pool_ids = pl.read_parquet("work/test/s0_prepare/pool.parquet", columns=["pool_row", "entity_id"])
        cand = pred.join(pool_ids, on="pool_row", how="left").sort("s1_row", "p", "pool_row", descending=[False, True, False])
        assert cand["entity_id"].null_count() == 0
        match = cand.filter(pl.col("keep"))
        write_tsv(Path("output/matching_results.tsv"), "source1_entity_id\tmatched_entity_ids", s1, match)
        write_tsv(Path("output/candidate_pairs.tsv"), "source1_entity_id\tcandidate_entity_ids", s1, cand)
        chg = (pred.join(s1.select("s1_row", "country"), on="s1_row")
                   .group_by("country").agg(pl.col("keep").sum().alias("kept"), pl.col("keep0").sum().alias("kept_s5"),
                                            (pl.col("keep") & ~pl.col("keep0")).sum().alias("added"),
                                            (~pl.col("keep") & pl.col("keep0")).sum().alias("dropped")).sort("country"))
        for r in chg.iter_rows(named=True):
            log.info(f"vs S5 {r}")
        n_m = s1.join(match.group_by("s1_row").len("m"), on="s1_row", how="left").fill_null(0)
        n_c = s1.join(cand.group_by("s1_row").len("c"), on="s1_row", how="left").fill_null(0)
        san = (n_m.join(n_c.select("s1_row", "c"), on="s1_row")
                  .group_by("country").agg(pl.len().alias("s1"), pl.col("m").mean().round(3).alias("mean_matches"),
                                           (pl.col("m") == 0).mean().round(4).alias("empty_rate"),
                                           pl.col("c").mean().round(1).alias("mean_cands")).sort("country"))
        for r in san.iter_rows(named=True):
            log.info(f"sanity {r}")
        write_json(st.work_dir / "sanity.json", san.to_dicts())
        tsv_md5 = {p.name: md5(p) for p in TSVS}
        write_json(st.work_dir / "tsv_md5.json", tsv_md5)
        tmp = ART / "stage2.txt.tmp"
        model.save_model(str(tmp))
        tmp.replace(ART / "stage2.txt")
        write_json(ART / "thresholds_stage2.json", {**rep, "s6_hash": st.hash, "tsv_md5": tsv_md5})
        log.info(f"matches {match.height:,} / candidates {cand.height:,} for {s1.height:,} S1s; md5 {tsv_md5}")
        st.rows = s1.height


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force)
