#!/usr/bin/env python3
"""S1 GT checks + train/val split. Rule: specs/prepare.md."""

import argparse
import json
import logging

import polars as pl

from common import atomic_write_parquet, code_hash, input_hash, load_config, stage

log = logging.getLogger("s1_gt")


def explode_gt(path) -> pl.DataFrame:
    """(source1_entity_id, match_id); an empty cell means 0 matches."""
    gt = pl.read_csv(path, separator="\t", infer_schema=False)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .with_columns(pl.col("matched_entity_ids").str.strip_chars())
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"matched_entity_ids": "match_id"}))


def stratified(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Sample n rows with each country's share preserved (countries taken from the data)."""
    parts = []
    for (c,), g in df.group_by("country", maintain_order=True):
        k = round(n * g.height / df.height)
        parts.append(g.sample(min(k, g.height), seed=seed, shuffle=True))
    return pl.concat(parts)


def main(split, limit_s1, force):
    cfg = load_config()
    if split != "train":
        log.info("s1_gt only applies to train")
        return
    sc = cfg["split"]
    with stage("s1_gt", split, cfg, {"split": sc, "seed": cfg["seed"], "code": code_hash("src/s1_gt.py")}, limit_s1=limit_s1,
               force=force, input_stages={"s0_prepare": input_hash("train", "s0_prepare")}) as st:
        if st.skip:
            return
        s0 = "work/train/s0_prepare"
        s1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "entity_id", "country"])
        pool = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "entity_id", "country"])

        pairs = explode_gt("data/dataset/train/train_ground_truth.tsv")
        n_gt_s1 = pairs["source1_entity_id"].n_unique()
        pairs = (pairs.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
                      .join(pool.rename({"entity_id": "match_id", "country": "pool_country"}),
                            on="match_id", how="left"))
        unresolved = pairs["pool_row"].null_count()
        per_id = pairs.group_by("match_id").agg(pl.col("s1_row").n_unique().alias("n"))
        multi = per_id.filter(pl.col("n") > 1).height
        same_country = (pairs["country"] == pairs["pool_country"]).mean()
        counts = s1.join(pairs.group_by("s1_row").len(), on="s1_row", how="left").fill_null(0)
        stats = {
            "positive_pairs": pairs.height, "s1_with_gt_rows": n_gt_s1,
            "unresolved_match_ids": unresolved, "pool_ids_under_multiple_s1": multi,
            "same_country_frac": round(same_country, 5),
            "mean_matches_per_s1": round(counts["len"].mean(), 4),
            "max_matches": int(counts["len"].max()),
            "singleton_frac": round((counts["len"] == 0).mean(), 5),
        }
        for k, v in stats.items():
            log.info(f"  {k}: {v}")
        assert unresolved == 0, f"{unresolved} GT ids not found in the train pool"

        pairs = pairs.filter(pl.col("pool_row").is_not_null()).select("s1_row", "pool_row")
        atomic_write_parquet(pairs, st.work_dir / "gt_pairs.parquet")

        seed = cfg["seed"]
        # a limited S0 has few S1s: scale the split so VAL is never empty (use <= half of S1)
        f = min(1.0, 0.5 * s1.height / (sc["train_s1"] + sc["val_s1"]))
        n_train, n_val, n_early, n_hold = (int(sc[k] * f) for k in
                                           ("train_s1", "val_s1", "early_stop_s1", "holdout_s1"))
        if f < 1:
            log.warning(f"  S1 has {s1.height:,} rows: split scaled by {f:.3f}")
        held = stratified(s1, n_train + n_val, seed)
        train = stratified(held, n_train, seed)
        val = held.join(train, on="s1_row", how="anti")
        early = stratified(train, n_early, seed + 1)
        train = train.join(early, on="s1_row", how="anti")
        # drawn from the S1s outside TRAIN/VAL/ES, so adding it leaves those splits unchanged
        holdout = stratified(s1.join(held, on="s1_row", how="anti"), n_hold, seed + 2)
        for name, df in [("train_ids", train), ("val_ids", val), ("early_stop_ids", early),
                         ("holdout_ids", holdout)]:
            atomic_write_parquet(df.select("s1_row", "entity_id", "country").sort("s1_row"),
                                 st.work_dir / f"{name}.parquet")
            by_c = dict(df.group_by("country").len().sort("country").iter_rows())
            log.info(f"  {name}: {df.height:,} {by_c}")
        (st.work_dir / "gt_stats.json").write_text(json.dumps(stats, indent=1))
        st.rows = pairs.height


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force)
