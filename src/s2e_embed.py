#!/usr/bin/env python3
"""S2e optional: sentence embeddings of raw business names. Rule: specs/optional-stages.md.

Embeds each UNIQUE raw `business_name` (original script: the cross-script / transliteration gap
is what lexical features miss) of the S1 scope and of the pool rows among their capped
candidates. Output: texts.parquet (text -> eid) + emb.npy (float16, L2-normalized). S3 turns it
into `emb_cos` per pair. `--bench N` only measures names/s on N pool names (no outputs).
"""

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import polars as pl

from common import code_hash, input_hash, load_config, stage

log = logging.getLogger("s2e_embed")


def load_model(ecfg):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ecfg["model_dir"], device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = ecfg.get("max_len", 64)
    return m


def encode(m, texts, ecfg) -> np.ndarray:
    pre = ecfg.get("prefix", "")
    return m.encode([pre + t for t in texts], batch_size=ecfg["batch_size"], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False).astype(np.float16)


def bench(n, ecfg):
    names = (pl.read_parquet("work/train/s0_prepare/pool.parquet", columns=["business_name"])
               ["business_name"].drop_nulls().unique().sample(n, seed=0).to_list())
    m = load_model(ecfg)
    encode(m, names[:2000], ecfg)                      # warm-up
    t = time.time()
    e = encode(m, names, ecfg)
    dt = time.time() - t
    log.info(f"bench {ecfg['model_dir']}: {len(names):,} names in {dt:.1f}s = {len(names) / dt:,.0f}/s, "
             f"dim {e.shape[1]}")


def main(split, limit_s1, force):
    cfg = load_config()
    ecfg, max_c = cfg["optional"]["embed_view"], cfg["blocking"]["max_candidates"]
    inputs = {"s0_prepare": input_hash(split, "s0_prepare"), "s2_block": input_hash(split, "s2_block")}
    if split == "train":
        inputs["s1_gt"] = input_hash("train", "s1_gt")
    section = {"embed": ecfg, "max_candidates": max_c, "limit_s1": limit_s1, "seed": cfg["seed"],
               "code": code_hash("src/s2e_embed.py", "src/s3_features.py")}
    with stage("s2e_embed", split, cfg, section, limit_s1=limit_s1, force=force,
               input_stages=inputs) as st:
        if st.skip:
            return
        from s3_features import build_scope        # same S1 scope S3 will use
        scope = build_scope(split, limit_s1, cfg["seed"]).select("s1_row")
        cand = (pl.scan_parquet(f"work/{split}/s2_block/part-*.parquet")
                  .filter(pl.col("cand_pos") < max_c).select("s1_row", "pool_row").collect()
                  .join(scope, on="s1_row").select("pool_row").unique())
        s0 = f"work/{split}/s0_prepare"
        t1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "business_name"]).join(scope, on="s1_row")
        tp = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "business_name"]).join(cand, on="pool_row")
        texts = (pl.concat([t1["business_name"], tp["business_name"]]).fill_null("").unique().sort()
                   .to_frame("text").with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("eid")))
        log.info(f"{scope.height:,} S1 + {cand.height:,} pool rows -> {texts.height:,} unique names")
        m = load_model(ecfg)
        t = time.time()
        emb = encode(m, texts["text"].to_list(), ecfg)
        dt = time.time() - t
        log.info(f"encoded {texts.height:,} names in {dt:.1f}s ({texts.height / dt:,.0f}/s), dim {emb.shape[1]}")
        texts.write_parquet(st.work_dir / "texts.parquet")
        tmp = st.work_dir / "emb.tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, st.work_dir / "emb.npy")
        st.rows = texts.height


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit-s1", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bench", type=int, help="only measure names/s on N pool names")
    ap.add_argument("--model-dir", help="override optional.embed_view.model_dir")
    ap.add_argument("--prefix", help="override the text prefix (e5: 'query: ')")
    a = ap.parse_args()
    if a.bench:
        from common import setup_logging
        setup_logging("s2e_embed")
        e = dict(load_config()["optional"]["embed_view"])
        if a.model_dir:
            e["model_dir"] = a.model_dir
        if a.prefix is not None:
            e["prefix"] = a.prefix
        bench(a.bench, e)
    else:
        main(a.split, a.limit_s1, a.force)
