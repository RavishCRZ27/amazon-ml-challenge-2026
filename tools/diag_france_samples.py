#!/usr/bin/env python3
"""Print sampled test S1s of one country with their top candidates (final p, stage-1 p, xenc, kept).
usage: .venv/bin/python tools/diag_france_samples.py [country] [n_per_group]"""
import json, sys
import numpy as np, polars as pl
sys.path.insert(0, "src")
import s4x_xenc as xe, threshold as th
C = sys.argv[1] if len(sys.argv) > 1 else "France"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
thr = json.loads(open("output/artifacts/thresholds_final.json").read()); lo, hi = thr["xenc"]["band"]
w = "work/test/s5_infer"
d = pl.read_parquet(f"{w}/pred-part-{C}-*.parquet").join(pl.read_parquet(f"{w}/xenc-part-{C}-*.parquet"), on=["s1_row", "pool_row"], how="left")
inb = (d["p"].is_between(lo, hi) & d["xenc"].is_not_null()).to_numpy()
d = d.with_columns(pl.Series("pf", xe.combine_apply(d["p"].to_numpy(), d["xenc"].fill_null(0.0).to_numpy(), inb, thr["xenc"]["combiner"])))
s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "country", "business_name", "business_address"]).filter(pl.col("country") == C)
pool = pl.read_parquet("work/test/s0_prepare/pool.parquet", columns=["pool_row", "src", "business_name", "business_address"])
d = d.with_columns((pl.col("pf") >= thr["t"]).alias("keep"))
g = d.group_by("s1_row").agg(pl.col("pf").max().alias("top"), pl.col("keep").sum().alias("n"),
                             ((pl.col("xenc") > 0) != (pl.col("p") >= 0.5)).filter(pl.col("xenc").is_not_null()).sum().alias("dis"))
groups = {"EMPTY prediction": g.filter(pl.col("n") == 0),
          "top uncertain .2-.9": g.filter(pl.col("top").is_between(0.2, 0.9)),
          "xenc/stage-1 disagree": g.filter((pl.col("dis") > 0) & (pl.col("n") > 0))}
cut = lambda s, k=62: (s or "")[:k]
for name, gg in groups.items():
    print(f"\n######## {name}: {gg.height:,} S1s ({gg.height / s1.height:.1%})")
    for r in gg.sample(min(N, gg.height), seed=3).iter_rows(named=True):
        a = s1.filter(pl.col("s1_row") == r["s1_row"]).row(0, named=True)
        print(f"S1  {cut(a['business_name'], 40):40s} | {cut(a['business_address'], 70)}")
        cc = d.filter(pl.col("s1_row") == r["s1_row"]).sort("pf", descending=True).head(6).join(pool, on="pool_row", how="left", maintain_order="left")
        for c in cc.iter_rows(named=True):
            x = "  -  " if c["xenc"] is None else f"{c['xenc']:+5.1f}"
            print(f"  {'*' if c['keep'] else ' '} pf {c['pf']:.2f} p1 {c['p']:.2f} x {x} {c['src']} {cut(c['business_name'], 36):36s} | {cut(c['business_address'], 60)}")
