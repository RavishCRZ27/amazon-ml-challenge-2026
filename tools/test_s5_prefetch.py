#!/usr/bin/env python3
"""Smoke test: S5's threaded prepare-while-scoring loop == the sequential score_pairs path, bitwise.

Runs on the two smallest run-1 test pred parts with the run-1 e5-small cross-encoder (fast).
usage: .venv/bin/python tools/test_s5_prefetch.py
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402

W = Path("work/test/s5_infer")
lo, hi, max_len = 0.002, 0.998, 96
parts = sorted(W.glob("pred-part-*.parquet"), key=lambda f: f.stat().st_size)[:2]
tok, xm = xe.load_xenc("work/eval/xenc_exp_e5run1/model")

t = time.time()
seq = {}
for f in parts:
    d = pl.read_parquet(f).filter(pl.col("p").is_between(lo, hi))
    seq[f.name] = xe.score_pairs("test", d, tok, xm, max_len)
t_seq = time.time() - t


def prep(f):
    d = pl.read_parquet(f).filter(pl.col("p").is_between(lo, hi))
    return d, (xe.tokenize(tok, *xe.texts("test", d), max_len) if d.height else None)


t = time.time()
thr = {}
with ThreadPoolExecutor(1) as ex:
    nxt = ex.submit(prep, parts[0])
    for i, f in enumerate(parts):
        d, enc = nxt.result()
        if i + 1 < len(parts):
            nxt = ex.submit(prep, parts[i + 1])
        thr[f.name] = xe.score(xm, *enc) if d.height else np.zeros(0, np.float32)
t_thr = time.time() - t

for k in seq:
    assert np.array_equal(seq[k], thr[k]), f"{k}: threaded scores differ"
print(f"OK: {sum(len(v) for v in seq.values()):,} pairs in {len(parts)} parts identical; "
      f"sequential {t_seq:.1f}s, threaded {t_thr:.1f}s")
