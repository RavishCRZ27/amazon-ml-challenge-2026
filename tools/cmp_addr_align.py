#!/usr/bin/env python3
"""Label-free address alignment per test country: for each S1, the pair stage 1 is most sure
of (max p, kept only if p > .9) is almost always a true match; its address features show how
well a true pair's addresses agree after normalization. Run 3 (S0 v3) should bring France to
the US/India level if the region/department mismatch was what kept France apart.

usage: .venv/bin/python tools/cmp_addr_align.py [root=.]
"""

import sys
from pathlib import Path

import polars as pl

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FEATS = ["a_contain", "a_jacc", "a_last2", "a_tset", "a_idf_cont", "a_missing_pool"]
rows = []
for part in sorted((root / "work/test/s3_features").glob("part-*.parquet")):
    pred = pl.read_parquet(root / "work/test/s5_infer" / f"pred-{part.name}")
    d = pl.read_parquet(part, columns=["s1_row", "pool_row"] + FEATS)
    assert d.select("s1_row", "pool_row").equals(pred.select("s1_row", "pool_row")), part.name
    d = d.with_columns(pred["p"])
    top = d.sort(["s1_row", "p", "pool_row"], descending=[False, True, False]).group_by("s1_row", maintain_order=True).first()
    rows.append(top.filter(pl.col("p") > 0.9).with_columns(pl.lit(part.name.split("-")[1]).alias("country")))
t = pl.concat(rows)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_tbl_cols(20)
print(f"{root}: top pair per S1 with stage-1 p > .9")
print(t.group_by("country").agg(pl.len().alias("n_s1"),
                                *[pl.col(f).fill_nan(None).mean().round(3).alias(f) for f in FEATS],
                                (pl.col("a_last2") >= 0.999).mean().round(3).alias("last2_full"),
                                (pl.col("a_contain") >= 0.999).mean().round(3).alias("contain_full"))
       .sort("country"))
