#!/usr/bin/env python3
"""Submission gate item 1: both TSVs have one row per test S1 (exactly the test ids), no duplicate
ids within a row, no empty ids, and matches ⊆ candidates.

usage: .venv/bin/python tools/gate_check.py [dir=output]
"""

import sys

import polars as pl

d = sys.argv[1] if len(sys.argv) > 1 else "output"


def load(name, col):
    df = pl.read_csv(f"{d}/{name}", separator="\t", quote_char=None, infer_schema=False)
    assert df.columns == ["source1_entity_id", col], df.columns
    return df.rename({col: "ids"}).with_columns(pl.col("ids").fill_null(""))


m = load("matching_results.tsv", "matched_entity_ids")
c = load("candidate_pairs.tsv", "candidate_entity_ids")
s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["entity_id"])
for name, df in (("matching", m), ("candidate", c)):
    assert df.height == s1.height == df["source1_entity_id"].n_unique(), (name, df.height)
    assert df.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height == 0, name
print(f"rows: matching {m.height:,}, candidate {c.height:,} (test S1 {s1.height:,}); ids = test S1 ids exactly")


def pairs(df):
    return (df.filter(pl.col("ids") != "").with_columns(pl.col("ids").str.split(","))
              .explode("ids").rename({"ids": "eid"}))


mp, cp = pairs(m), pairs(c)
for name, p in (("matching", mp), ("candidate", cp)):
    assert p.filter(pl.col("eid") == "").height == 0, f"{name}: empty id inside a list"
    dup = p.height - p.unique().height
    assert dup == 0, f"{name}: {dup} duplicate ids within rows"
miss = mp.join(cp, on=["source1_entity_id", "eid"], how="anti").height
assert miss == 0, f"{miss} matches not in candidates"
print(f"pairs: matches {mp.height:,}, candidates {cp.height:,}; matches ⊆ candidates; no duplicates")
print(f"empty match rows {(m['ids'] == '').sum():,} ({(m['ids'] == '').mean():.2%}); empty candidate rows {(c['ids'] == '').sum():,}")
print("GATE-1 OK")
