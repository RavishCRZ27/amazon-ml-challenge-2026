#!/usr/bin/env python3
"""Gate item 5 (GPU half): re-score S5 cross-encoder parts and compare bitwise with the saved
xenc-part files. Default: the smallest part of each country.

usage: .venv/bin/python tools/check_xenc_determinism.py [part-name ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402

W = Path("work/test/s5_infer")
xi = json.loads(Path("output/artifacts/thresholds_final.json").read_text())["xenc"]
lo, hi = xi["band"]
names = sys.argv[1:]
if not names:
    by_c = {}
    for f in sorted(W.glob("pred-part-*.parquet"), key=lambda f: f.stat().st_size):
        by_c.setdefault(f.name.split("-")[2], f.name.removeprefix("pred-"))
    names = list(by_c.values())
tok, model = xe.load_xenc(xi["model_dir"])
ok = True
for n in names:
    d = pl.read_parquet(W / f"pred-{n}").filter(pl.col("p").is_between(lo, hi))
    t = time.time()
    s = xe.score_pairs("test", d, tok, model, xi["max_len"]) if d.height else np.zeros(0, np.float32)
    saved = pl.read_parquet(W / f"xenc-{n}")
    same_keys = saved.select("s1_row", "pool_row").equals(d.select("s1_row", "pool_row"))
    same = same_keys and np.array_equal(saved["xenc"].to_numpy(), s.astype(np.float32))
    ok &= same
    print(f"{n}: {d.height:,} pairs re-scored in {time.time() - t:.0f}s -> "
          f"{'IDENTICAL' if same else 'DIFFERENT'} (keys equal: {same_keys}; "
          f"max|diff| {np.abs(saved['xenc'].to_numpy() - s).max() if same_keys and d.height else 'n/a'})", flush=True)
print("DETERMINISM OK" if ok else "DETERMINISM FAILED")
