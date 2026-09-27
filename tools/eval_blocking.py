#!/usr/bin/env python3
"""EDA-14: blocking recall on a train slice, from the uncapped per-view top-k saved by S2.

usage: .venv/bin/python tools/eval_blocking.py [--dir work/eval/s2_train_50000] [--k 20] [--cap 50]
Recall is against the FULL GT of the slice's S1s, so blocking misses count.
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from s2_block import VIEWS, order_and_cap  # noqa: E402

KS = (1, 5, 10, 20, 30, 50)
UNION_KS = (10, 20, 30, 50)
# per-view k mixes (views missing from a mix are off)
COMBOS = [
    {"W_name": 20, "W_addr": 20, "C_name": 20},
    {"W_all": 50},
    {"W_all": 30, "W_addr": 20},
    {"W_all": 30, "W_addr": 10, "W_name": 5, "C_name": 5},
    {"W_all": 40, "W_addr": 10, "W_name": 5, "C_name": 5},
    {"W_all": 30, "W_addr": 20, "W_name": 10, "C_name": 10},
    {"W_all": 50, "W_addr": 20, "W_name": 10, "C_name": 10},
    {"W_all": 30, "C_all": 30},
    {"W_all": 25, "C_all": 25},
    {"W_all": 30, "C_all": 20, "W_addr": 10},
    {"W_all": 25, "C_all": 20, "W_addr": 10, "W_name": 5},
    {"W_all": 30, "C_all": 30, "W_addr": 10, "W_name": 5, "C_name": 5},
    {"W_all": 50, "C_all": 50, "W_addr": 50, "W_name": 50, "C_name": 50},
]
CAPS = (30, 50)


def ranked(unc: pl.DataFrame, views, k) -> pl.DataFrame:
    """Keep each view's top-k only (k: int or {view: k}, 0 = view off), then order like S2 (no cap)."""
    kv = k if isinstance(k, dict) else {v: k for v in views}
    w = unc.with_columns([pl.when(pl.col(f"rank_{v}") <= kv.get(v, 0)).then(pl.col(c)).alias(c)
                          for v in views for c in (f"rank_{v}", f"score_{v}")])
    w = w.filter(pl.any_horizontal([pl.col(f"rank_{v}").is_not_null() for v in views]))
    return order_and_cap(w, views, 10**9).select("s1_row", "pool_row", "_pos")


def union_stats(gt, r, n_s1, cap):
    hit = gt.join(r.filter(pl.col("_pos") < cap), on=["s1_row", "pool_row"], how="left")
    found = hit["_pos"].is_not_null()
    per_s1 = hit.with_columns(found.alias("f")).group_by("s1_row").agg(pl.col("f").all())
    size = r.group_by("s1_row").len().select(pl.col("len").clip(upper_bound=cap).sum()).item()
    return {"recall": round(found.mean(), 4), "s1_all_found": round(per_s1["f"].mean(), 4),
            "mean_union": round(size / n_s1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/eval/s2_train_50000")
    ap.add_argument("--k", type=int, default=20, help="per-view k for the breakdown and misses")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--misses", type=int, default=200)
    a = ap.parse_args()
    d = Path(a.dir)
    unc = pl.read_parquet(d / "uncapped.parquet")
    views = [v for v in VIEWS if f"rank_{v}" in unc.columns]
    q = pl.read_parquet(d / "queries.parquet").rename({"name_script": "s1_script"})
    pool = pl.read_parquet("work/train/s0_prepare/pool.parquet",
                           columns=["pool_row", "name_script", "business_name", "business_address"])
    gt = (pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").join(q, on="s1_row")
            .join(pool.select("pool_row", pl.col("name_script").alias("pool_script")), on="pool_row")
            .with_columns(pl.when(pl.col("s1_script") == pl.col("pool_script")).then(pl.lit("same"))
                          .otherwise(pl.lit("cross")).alias("script_pair")))
    n_s1 = q.height
    out = {"n_s1": n_s1, "n_s1_with_gt": gt["s1_row"].n_unique(), "n_gt_pairs": gt.height,
           "max_rank_available": int(max(unc[f"rank_{v}"].max() for v in views))}

    g = gt.join(unc, on=["s1_row", "pool_row"], how="left")
    out["per_view_recall"] = {v: {k: round((g[f"rank_{v}"] <= k).fill_null(False).mean(), 4) for k in KS} for v in views}
    out["union"] = {f"k{k}_cap{c}": union_stats(gt, ranked(unc, views, k), n_s1, c)
                    for k in UNION_KS for c in CAPS}
    out["union"]["k50_uncapped"] = union_stats(gt, ranked(unc, views, 50), n_s1, 10**9)
    out["combos"] = {}
    for combo in COMBOS:
        if not set(combo) <= set(views):
            continue
        r = ranked(unc, views, combo)
        name = " ".join(f"{v}={k}" for v, k in combo.items())
        out["combos"][name] = {f"cap{c}": union_stats(gt, r, n_s1, c) for c in (*CAPS, 10**9)}

    r = ranked(unc, views, a.k).filter(pl.col("_pos") < a.cap)
    hit = gt.join(r, on=["s1_row", "pool_row"], how="left").with_columns(
        pl.col("_pos").is_not_null().alias("found"))
    for col in ("country", "script_pair"):
        out[f"recall_by_{col}@k{a.k}_cap{a.cap}"] = {
            k: {"recall": round(v, 4), "pairs": n} for k, v, n in
            hit.group_by(col).agg(pl.col("found").mean(), pl.len()).sort(col).iter_rows()}
    out[f"recall_by_script_combo@k{a.k}_cap{a.cap}"] = {
        f"{s}->{p}": {"recall": round(v, 4), "pairs": n} for s, p, v, n in
        hit.group_by("s1_script", "pool_script").agg(pl.col("found").mean(), pl.len())
           .sort("len", descending=True).iter_rows()}
    per_view_c = g.group_by("country").agg([(pl.col(f"rank_{v}") <= a.k).fill_null(False).mean().round(4).alias(v)
                                            for v in views]).sort("country")
    out[f"per_view_recall_by_country@k{a.k}"] = {row["country"]: {v: row[v] for v in views}
                                                for row in per_view_c.iter_rows(named=True)}

    s1 = pl.read_parquet("work/train/s0_prepare/s1.parquet",
                         columns=["s1_row", "business_name", "business_address"])
    misses = (hit.filter(~pl.col("found")).sample(min(a.misses, hit.height - hit["found"].sum()), seed=0)
                 .join(s1.rename({"business_name": "s1_name", "business_address": "s1_addr"}), on="s1_row")
                 .join(pool.drop("name_script").rename({"business_name": "pool_name",
                                                        "business_address": "pool_addr"}), on="pool_row")
                 .join(unc, on=["s1_row", "pool_row"], how="left")
                 .select(["country", "script_pair", "s1_script", "pool_script", "s1_name", "pool_name",
                          "s1_addr", "pool_addr"] + [f"rank_{v}" for v in views]))
    misses.write_csv(d / "misses.tsv", separator="\t")
    (d / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
