#!/usr/bin/env python3
"""France F0.5 implied by a public LB score: LB = sum_c share_c * F_c over test countries, with F_c
for the train countries taken from the same system's HOLDOUT (tools/compare_holdout.py --dump).
Assumes test India/US score like HOLDOUT (checked label-free by tools/diag_country.py) and that the
public subset keeps the test country mix.

usage: .venv/bin/python tools/france_from_lb.py <holdout_f.parquet> <LB score> [<holdout_f 2> <LB 2>]
"""
import sys

import polars as pl

s1 = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["country"])
share = {c: n / s1.height for c, n in s1.group_by("country").len().iter_rows()}
res = []
args = sys.argv[1:]
for f, lb in zip(args[0::2], args[1::2]):
    h = dict(pl.read_parquet(f).group_by("country").agg(pl.col("f").mean()).iter_rows())
    known = {c: v for c, v in h.items() if c in share}
    unknown = [c for c in share if c not in known]
    assert len(unknown) == 1, f"expected exactly one country without labels, got {unknown}"
    u = unknown[0]
    fu = (float(lb) - sum(share[c] * v for c, v in known.items())) / share[u]
    res.append(fu)
    print(f"{f}: LB {float(lb):.4f}; HOLDOUT " + ", ".join(f"{c} {v:.5f}" for c, v in sorted(known.items()))
          + f" -> implied {u} F0.5 {fu:.4f} (share {share[u]:.3f})")
if len(res) == 2:
    print(f"implied change for the unlabeled country: {res[1] - res[0]:+.4f}")
