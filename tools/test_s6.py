#!/usr/bin/env python3
"""Check S6 before it runs for real: (1) its fit code reproduces tools/exp_stage2x.py variant B on
VAL/HOLDOUT (model checkpointed in work/eval/s6_fitcheck/); (2) label-free France check: apply it to the
S5 parts finished so far and compare decisions with S5's (no injective pass, within part), by country and
by pair type, next to the same pair types' HOLDOUT changes (with labels).

usage: nice -n 10 .venv/bin/python tools/test_s6.py [--threads 16]
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import s6_stage2 as s6                 # noqa: E402
from common import load_config         # noqa: E402
from diag_changes import types as pair_types   # noqa: E402


def main(a):
    cfg = load_config()
    thr = json.loads(Path("output/artifacts/thresholds_final.json").read_text())
    s3cols = json.loads(Path("output/artifacts/thresholds.json").read_text())["features"]
    st = types.SimpleNamespace(work_dir=Path("work/eval/s6_fitcheck"))
    st.work_dir.mkdir(parents=True, exist_ok=True)
    model, rep = s6.fit(st, cfg, thr, s3cols, a.threads)
    print("fit:", json.dumps({k: rep[k] for k in ("val", "holdout", "choice", "rounds")}, default=str), flush=True)
    t2, feats = rep["choice"]["t"], rep["features"]
    chg = lambda k, k0: (pl.when(pl.col(k) & ~pl.col(k0)).then(pl.lit("added"))
                          .when(~pl.col(k) & pl.col(k0)).then(pl.lit("dropped")).otherwise(pl.lit("same")))
    # HOLDOUT change types with labels (no injective, like the test-side check)
    pool_txt = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    _, gh = s6.train_split("holdout", thr, s3cols, pool_txt, a.threads)
    gh = gh.with_columns(pl.Series("k2", model.predict(gh.select(feats).to_numpy().astype(np.float32)) >= t2),
                         (pl.col("pf") >= thr["t"]).alias("k0"))
    gh = pair_types(gh.with_columns(chg("k2", "k0").alias("change")).select("s1_row", "pool_row", "label", "change"), "train")
    by = ["name_exact", "num_first_eq", "pool_addr_empty", "change"]
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_width_chars(200)
    print("HOLDOUT changes stage 2 vs S5 rule (label = true match?):")
    print(gh.filter(pl.col("change") != "same").group_by(by).agg(pl.len().alias("pairs"), pl.col("label").mean().round(3).alias("true_rate")).sort(by))
    # TEST: parts S5 has finished
    s5 = Path("work/test/s5_infer")
    pool_t = pl.read_parquet("work/test/s0_prepare/pool.parquet", columns=["pool_row", "name_s", "addr_n"])
    s1c = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "country"])
    rows = []
    for xp in sorted(s5.glob("xenc-part-*.parquet")):
        name = xp.name.removeprefix("xenc-")
        d = (pl.read_parquet(s5 / f"pred-{name}").rename({"p": "p1"})
               .join(pl.read_parquet(xp), on=["s1_row", "pool_row"], how="left").with_columns(pl.col("xenc").fill_null(0.0))
               .sort("s1_row", "pool_row"))
        d = s6.features(s6.combined(d, thr["xenc"]["band"], thr["xenc"]["combiner"]), pool_t, thr["t"], a.threads)
        d = d.join(pl.read_parquet(Path("work/test/s3_features") / name, columns=["s1_row", "pool_row"] + s3cols),
                   on=["s1_row", "pool_row"], how="left", maintain_order="left")
        d = d.with_columns(pl.Series("k2", model.predict(d.select(feats).to_numpy().astype(np.float32)) >= t2),
                           (pl.col("pf") >= thr["t"]).alias("k0"))
        rows.append(d.select("s1_row", "pool_row", "k2", "k0").with_columns(chg("k2", "k0").alias("change")))
        print(f"applied to {name}: {d.height:,} pairs", flush=True)
    te = pair_types(pl.concat(rows).join(s1c, on="s1_row"), "test")
    print("TEST (finished S5 parts) per country: S1s, kept S5 rule -> stage 2, added, dropped, S1s changed")
    print(te.group_by("country").agg(pl.col("s1_row").n_unique().alias("s1"), pl.col("k0").sum().alias("kept_s5"),
                                     pl.col("k2").sum().alias("kept_s2"), (pl.col("change") == "added").sum().alias("added"),
                                     (pl.col("change") == "dropped").sum().alias("dropped"),
                                     pl.col("s1_row").filter(pl.col("change") != "same").n_unique().alias("s1_changed")))
    print(te.filter(pl.col("change") != "same").group_by(["country"] + by).len("pairs").sort(["country"] + by))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    main(ap.parse_args())
