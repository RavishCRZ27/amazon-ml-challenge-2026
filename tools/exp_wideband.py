#!/usr/bin/env python3
"""Step-3 experiment: widen the cross-encoder band from [.05,.95] to [lo, hi].

Scores the extra VAL/HOLDOUT pairs (lo <= p < .05 and .95 < p <= hi) with the already-trained
experiment cross-encoder, merges them with the in-band scores, and writes a score dir that
tools/exp_stage2.py can evaluate with `--band lo hi`.

usage: .venv/bin/python tools/exp_wideband.py --lo 0.01 --hi 0.99
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402

SRC = Path("work/eval/xenc_full_multilingual-e5-small")


def main(lo, hi, pred_tag="full", max_len=96):
    out = Path(f"work/eval/xenc_{pred_tag}_{lo}_{hi}")
    out.mkdir(parents=True, exist_ok=True)
    tok, model = xe.load_xenc(SRC / "model")
    for split in ("val", "holdout"):
        pr = pl.read_parquet(f"work/eval/s4_{pred_tag}/pred_{split}.parquet")
        old = pl.read_parquet(SRC / f"scores_{split}.parquet")
        extra = (pr.filter(pl.col("p").is_between(lo, hi))            # band pairs not scored yet
                   .join(old.select("s1_row", "pool_row"), on=["s1_row", "pool_row"], how="anti"))
        t = time.time()
        s = xe.score_pairs("train", extra, tok, model, max_len)
        print(f"{split}: {extra.height:,} extra pairs in {time.time() - t:.0f}s", flush=True)
        new = extra.select("s1_row", "pool_row").with_columns(pl.Series("xenc", s))
        pl.concat([old, new]).unique(["s1_row", "pool_row"]).write_parquet(out / f"scores_{split}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--pred-tag", default="full", help="work/eval/s4_<tag>/pred_*.parquet defines the band")
    a = ap.parse_args()
    main(a.lo, a.hi, a.pred_tag)
