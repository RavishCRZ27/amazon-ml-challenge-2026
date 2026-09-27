#!/usr/bin/env python3
"""Paired comparison of two per-S1 HOLDOUT F dumps (tools/compare_holdout.py --dump) over the same S1s.

usage: .venv/bin/python tools/paired_f.py base.parquet new.parquet
"""
import sys

import polars as pl

sys.path.insert(0, "src")
import threshold as th  # noqa: E402

a, b = (pl.read_parquet(f) for f in sys.argv[1:3])
d = a.join(b.select("s1_row", pl.col("f").alias("f_new")), on="s1_row", how="inner")
assert d.height == a.height == b.height, "the dumps cover different S1s"
x, y = d["f"].to_numpy(), d["f_new"].to_numpy()
print(f"base {x.mean():.5f} -> new {y.mean():.5f}: {(y - x).mean():+.5f} ± {th.paired_se(y, x):.5f} "
      f"({(y - x).mean() / th.paired_se(y, x):.1f} paired SE), n {d.height:,}")
for c in sorted(d["country"].unique()):
    m = (d["country"] == c).to_numpy()
    print(f"  {c}: {x[m].mean():.5f} -> {y[m].mean():.5f} ({(y[m] - x[m]).mean():+.5f} ± {th.paired_se(y[m], x[m]):.5f})")
