#!/usr/bin/env python3
"""Step-3 experiment: does an embedding view (4th blocking view) raise the blocking ceiling?

Embeds unique raw business names of the whole train pool + the EDA-14 50K VAL slice with
the configured sentence-transformer (e5-small), exact per-country top-k by cosine on the GPU
(torch matmul + topk in query batches), then compares the blocking-oracle macro-F0.5 of the
current capped candidates vs their union with the embedding top-k. Embeddings are cached in
work/eval/embview/ (reusable if the view is adopted).

usage: .venv/bin/python tools/exp_embed_view.py [--k 10 20]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import threshold as th                              # noqa: E402
from common import load_config                      # noqa: E402
from s2_block import order_and_cap                  # noqa: E402
from s2e_embed import encode, load_model            # noqa: E402

OUT = Path("work/eval/embview")


def embed_names(names: pl.Series, ecfg) -> tuple:
    OUT.mkdir(parents=True, exist_ok=True)
    tf, ef = OUT / "texts.parquet", OUT / "emb.npy"
    if tf.exists() and ef.exists():
        return pl.read_parquet(tf), np.load(ef, mmap_mode="r")
    texts = names.fill_null("").unique().sort().to_frame("text").with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).alias("eid"))
    t = time.time()
    emb = encode(load_model(ecfg), texts["text"].to_list(), ecfg)
    print(f"embedded {texts.height:,} unique names in {time.time() - t:.0f}s "
          f"({texts.height / (time.time() - t):,.0f}/s)", flush=True)
    texts.write_parquet(tf)
    np.save(ef, emb)
    return texts, emb


def topk(q: np.ndarray, P: np.ndarray, k: int, bs: int = 512):
    import torch
    Pt = torch.from_numpy(np.ascontiguousarray(P)).cuda()          # fp16 on the GPU
    out_i = np.empty((q.shape[0], k), np.int64)
    out_s = np.empty((q.shape[0], k), np.float32)
    for i in range(0, q.shape[0], bs):
        s = torch.from_numpy(np.ascontiguousarray(q[i:i + bs])).cuda() @ Pt.T
        v, j = torch.topk(s, k, dim=1)
        out_i[i:i + bs], out_s[i:i + bs] = j.cpu().numpy(), v.float().cpu().numpy()
    del Pt
    torch.cuda.empty_cache()
    return out_i, out_s


def oracle_f(cands: pl.DataFrame, q: pl.DataFrame, gt: pl.DataFrame) -> dict:
    s = q.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    n_gt = s.join(gt.group_by("s1_row").len("n"), on="s1_row", how="left").fill_null(0)["n"].to_numpy()
    c = cands.join(s.select("s1_row", "idx"), on="s1_row").join(
        gt.with_columns(pl.lit(1.0).alias("y")), on=["s1_row", "pool_row"], how="left").with_columns(
        pl.col("y").fill_null(0.0))
    y = c["y"].to_numpy()
    F = th.per_s1_f05(n_gt, c["idx"].to_numpy(), y > 0, y, s.height)
    per = {k: round(float(F[s["country"].to_numpy() == k].mean()), 4) for k in sorted(set(s["country"]))}
    return {"oracle_f05": round(float(F.mean()), 5), "se": round(th.se(F), 5),
            "recall": round(float(y.sum() / n_gt.sum()), 4), "cands_per_s1": round(c.height / s.height, 1),
            "by_country": per, "_F": F}


def main(ks):
    cfg = load_config()
    ecfg = cfg["optional"]["embed_view"]
    ev = Path("work/eval/s2_train_50000")
    q = pl.read_parquet(ev / "queries.parquet").select("s1_row", "country")
    base = order_and_cap(pl.read_parquet(ev / "uncapped.parquet"), ["W_all", "C_all"],
                         cfg["blocking"]["max_candidates"]).select("s1_row", "pool_row")
    gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").join(q.select("s1_row"), on="s1_row")
    pool = pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["pool_row", "country", "business_name"])
    s1 = pl.read_parquet("work/train/s0_prepare/s1.parquet", columns=["s1_row", "business_name"]).join(q, on="s1_row")
    texts, E = embed_names(pl.concat([pool["business_name"], s1["business_name"]]), ecfg)
    tb = texts.rename({"text": "business_name"})
    pool = pool.with_columns(pl.col("business_name").fill_null("")).join(tb, on="business_name", how="left")
    s1 = s1.with_columns(pl.col("business_name").fill_null("")).join(tb, on="business_name", how="left")
    r0 = oracle_f(base, q, gt)
    F0 = r0.pop("_F")
    print(f"current (W_all 40 + C_all 40, cap {cfg['blocking']['max_candidates']}): {r0}", flush=True)
    kmax = max(ks)
    rows = []
    for c in sorted(q["country"].unique().to_list()):
        pc, sc = pool.filter(pl.col("country") == c), s1.filter(pl.col("country") == c)
        t = time.time()
        idx, _ = topk(E[sc["eid"].to_numpy()], E[pc["eid"].to_numpy()], kmax)
        prow = pc["pool_row"].to_numpy()[idx]
        rows.append(pl.DataFrame({"s1_row": np.repeat(sc["s1_row"].to_numpy(), kmax),
                                  "pool_row": prow.ravel(), "erank": np.tile(np.arange(kmax), sc.height)}))
        print(f"[{c}] {sc.height:,} queries x {pc.height:,} pool: top-{kmax} in {time.time() - t:.1f}s", flush=True)
    emb = pl.concat(rows)
    for k in ks:
        u = pl.concat([base, emb.filter(pl.col("erank") < k).select("s1_row", "pool_row")]).unique()
        r = oracle_f(u, q, gt)
        F = r.pop("_F")
        print(f"union with embed top-{k}: {r}  dF_oracle {float((F - F0).mean()):+.5f} "
              f"± {th.paired_se(F, F0):.5f}", flush=True)
        eo = oracle_f(emb.filter(pl.col("erank") < k).select("s1_row", "pool_row"), q, gt)
        eo.pop("_F")
        print(f"   embed top-{k} alone: {eo}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    main(ap.parse_args().k)
