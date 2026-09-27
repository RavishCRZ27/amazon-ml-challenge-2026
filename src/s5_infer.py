#!/usr/bin/env python3
"""S5 test inference -> output TSVs. Rule: specs/output.md.

Scores every test S3 part with S4's model (per-part checkpoints), applies the VAL-chosen
threshold rule and the injective pass with `threshold.select` (the exact rule VAL was scored
with; deterministic tie-break), and writes both TSVs with one row per test S1 (built from
s0's s1.parquet, so S1s with zero candidates get an empty row).
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

import threshold as th
from common import atomic_write_parquet, code_hash, input_hash, load_config, stage

log = logging.getLogger("s5_infer")
ART = Path("output/artifacts")
OUT = Path("output")


def load_model(thr):
    if thr["backend"] == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=thr["model_file"])
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(thr["model_file"])
    return b


def predict(thr, m, X, nw, batch=2_000_000):
    out = np.empty(X.shape[0], np.float32)
    if thr["backend"] == "xgboost":
        import xgboost as xgb
        m.set_param({"device": "cuda", "nthread": nw})
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        if thr["backend"] == "lightgbm":
            out[i:i + batch] = m.predict(xb, num_iteration=thr["best_iteration"], num_threads=nw)
        else:
            out[i:i + batch] = m.predict(xgb.DMatrix(xb, feature_names=thr["features"]),
                                         iteration_range=(0, thr["best_iteration"]))
    return out


def write_tsv(path: Path, header: str, s1: pl.DataFrame, pairs: pl.DataFrame):
    """One row per test S1 in s1's order; ids comma-joined (no duplicates), empty when none."""
    lists = pairs.group_by("s1_row").agg(pl.col("entity_id").unique(maintain_order=True).str.join(","))
    rows = (s1.join(lists, on="s1_row", how="left", maintain_order="left")
              .select(pl.col("entity_id").alias("a"), pl.col("entity_id_right").fill_null("").alias("b")))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        rows.write_csv(f, separator="\t", include_header=False, quote_style="never")
    tmp.replace(path)


def main(split, limit_s1, force):
    cfg = load_config()
    if split != "test":
        log.info("s5_infer only applies to test")
        return
    use_x = cfg["optional"]["cross_encoder"]["enabled"]
    thr = json.loads((ART / ("thresholds_final.json" if use_x else "thresholds.json")).read_text())
    if thr.get("limit_s1") is not None:
        raise RuntimeError("output/artifacts holds a slice model; run s4_train without --limit-s1")
    nw = cfg["n_workers"]
    inputs = {"s0_prepare": input_hash("test", "s0_prepare"), "s2_block": input_hash("test", "s2_block"),
              "s3_features": input_hash("test", "s3_features"), "s4_train": input_hash("train", "s4_train")}
    if use_x:
        inputs["s4x_xenc"] = input_hash("train", "s4x_xenc")
    # artifacts are fixed paths: they must come from the runs recorded in _DONE.json
    assert thr.get("s4_hash") == inputs["s4_train"], "output/artifacts model is not from the finished S4 run"
    if use_x:
        assert thr.get("s4x_hash") == inputs["s4x_xenc"], "thresholds_final.json is not from the finished S4x run"
    section = {"thresholds": {k: thr[k] for k in ("backend", "t", "r", "injective", "best_iteration")},
               "xenc": thr.get("xenc"), "limit_s1": limit_s1,
               "code": code_hash("src/s5_infer.py", "src/threshold.py", "src/s4x_xenc.py")}
    with stage("s5_infer", split, cfg, section, limit_s1=limit_s1, force=force,
               input_stages=inputs) as st:
        if st.skip:
            return
        m = load_model(thr)
        feats = thr["features"]
        s3 = Path("work/test/s3_features")
        parts = sorted(s3.glob("part-*.parquet"))
        t = time.time()
        for part in parts:                         # per-part predictions: resumable
            out = st.work_dir / f"pred-{part.name}"
            if out.exists():
                continue
            d = pl.read_parquet(part)
            missing = [f for f in feats if f not in d.columns]
            assert not missing, f"{part.name} lacks model features {missing}"
            p = predict(thr, m, d.select(feats).to_numpy().astype(np.float32, copy=False), nw)
            atomic_write_parquet(d.select("s1_row", "pool_row").with_columns(pl.Series("p", p)), out)
        pred = pl.read_parquet(str(st.work_dir / "pred-part-*.parquet"))
        log.info(f"scored {pred.height:,} pairs from {len(parts)} parts in {time.time() - t:.1f}s")
        if use_x:        # cross-encoder on the stage-1 uncertain band, then the VAL-fit combiner
            import s4x_xenc as xe
            xi = thr["xenc"]
            lo, hi = xi["band"]
            tok, xm = xe.load_xenc(xi["model_dir"])
            t = time.time()
            todo = [q for q in parts if not (st.work_dir / f"xenc-{q.name}").exists()]   # resumable per part

            def prep(part):                        # CPU side: band pairs + tokens (same as score_pairs)
                d = pl.read_parquet(st.work_dir / f"pred-{part.name}").filter(pl.col("p").is_between(lo, hi))
                return d, (xe.tokenize(tok, *xe.texts("test", d), xi["max_len"]) if d.height else None)

            with ThreadPoolExecutor(1) as ex:      # part i+1 is prepared while the GPU scores part i
                nxt = ex.submit(prep, todo[0]) if todo else None
                for i, part in enumerate(todo):
                    d, enc = nxt.result()
                    if i + 1 < len(todo):
                        nxt = ex.submit(prep, todo[i + 1])
                    s = xe.score(xm, *enc) if d.height else np.zeros(0, np.float32)
                    atomic_write_parquet(d.select("s1_row", "pool_row").with_columns(pl.Series("xenc", s, pl.Float32)),
                                         st.work_dir / f"xenc-{part.name}")
            xs = pl.read_parquet(str(st.work_dir / "xenc-part-*.parquet"))
            pred = pred.join(xs, on=["s1_row", "pool_row"], how="left")
            inb = pred["xenc"].is_not_null().to_numpy()
            p2 = xe.combine_apply(pred["p"].to_numpy(), pred["xenc"].fill_null(0.0).to_numpy(), inb, xi["combiner"])
            assert np.isfinite(p2).all(), "non-finite combined scores"
            pred = pred.with_columns(pl.Series("p", p2.astype(np.float32))).drop("xenc")
            log.info(f"cross-encoder: {int(inb.sum()):,} band pairs ({inb.mean():.2%}) in {time.time() - t:.1f}s")

        s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "entity_id", "country"])
        scope = pl.read_parquet(s3 / "s1_scope.parquet").select("s1_row")
        if limit_s1 is None:
            assert scope.height == s1.height, f"S3 covered {scope.height:,} of {s1.height:,} test S1s"
        s1 = s1.join(scope, on="s1_row").sort("s1_row").with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
        pred = pred.join(s1.select("s1_row", "idx"), on="s1_row").sort("s1_row", "pool_row")
        c = th.Cands(pred["idx"].to_numpy(), pred["pool_row"].to_numpy(), pred["p"].to_numpy(), s1.height)
        pred = pred.with_columns(pl.Series("keep", th.select(c, thr["t"], thr["r"], thr["injective"])))
        pool_ids = pl.read_parquet("work/test/s0_prepare/pool.parquet", columns=["pool_row", "entity_id"])
        cand = pred.join(pool_ids, on="pool_row", how="left").sort("s1_row", "p", "pool_row",
                                                                   descending=[False, True, False])
        assert cand["entity_id"].null_count() == 0
        match = cand.filter(pl.col("keep"))
        out_dir = OUT if limit_s1 is None else Path(f"work/eval/s5_{limit_s1}")   # slices never touch output/
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tsv(out_dir / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids", s1, match)
        write_tsv(out_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids", s1, cand)

        # sanity by country (train GT reference: 3.46 matches / S1, 5.6% empty)
        n_m = s1.join(match.group_by("s1_row").len("m"), on="s1_row", how="left").fill_null(0)
        n_c = s1.join(cand.group_by("s1_row").len("c"), on="s1_row", how="left").fill_null(0)
        rep = (n_m.join(n_c.select("s1_row", "c"), on="s1_row")
                  .group_by("country").agg(pl.len().alias("s1"), pl.col("m").mean().round(3).alias("mean_matches"),
                                           (pl.col("m") == 0).mean().round(4).alias("empty_rate"),
                                           pl.col("c").mean().round(1).alias("mean_cands"))
                  .sort("country"))
        for r in rep.iter_rows(named=True):
            log.info(f"sanity {r}")
        (st.work_dir / "sanity.json").write_text(json.dumps(rep.to_dicts(), indent=1))
        log.info(f"matches {match.height:,} / candidates {cand.height:,} for {s1.height:,} S1s")
        st.rows = s1.height


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force)
