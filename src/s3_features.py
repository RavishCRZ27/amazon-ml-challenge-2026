#!/usr/bin/env python3
"""S3 pair features. Rule: specs/features.md.

Per country: pool-side record features (token-id lists, IDF sums, name frequency) are built
once; then each S2 part (one country x chunk_s1 S1s) is joined with both sides' record columns
and scored with vectorized ops only: rapidfuzz cpdist for strings, polars list ops + numpy
bincount for token sets. Output parts mirror S2's part names. Never emits `country`.
"""

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import atomic_write_parquet, code_hash, input_hash, load_config, stage
from s1_gt import stratified
from s2_block import safe, view_tokens

log = logging.getLogger("s3_features")
ID_COLS = ["s1_row", "pool_row"]
SPLITS = ("train", "early_stop", "val", "holdout")
SCRIPT = {"ascii": 0, "latin_accented": 1, "indic": 2, "other": 3}
REC_COLS = ["name_s", "name_sk", "name_script", "addr_n", "addr_nums"]


def non_features(df_cols) -> list:
    """Columns S4 must not train on."""
    return [c for c in df_cols if c in ("s1_row", "pool_row", "label")]


# ---------- scope ----------
def build_scope(split: str, limit_s1, seed: int) -> pl.DataFrame:
    """(s1_row, country, split) for every S1 this stage covers, incl. zero-candidate S1s."""
    q = pl.read_parquet(f"work/{split}/s2_block/queries.parquet")
    if split == "train":
        lab = pl.concat([pl.read_parquet(f"work/train/s1_gt/{n}_ids.parquet", columns=["s1_row"])
                           .with_columns(pl.lit(n).alias("split")) for n in SPLITS])
        q = q.join(lab, on="s1_row")
        if q.filter(pl.col("split") == "train").height == 0:
            raise RuntimeError("S2 blocked no TRAIN S1s (VAL-only tuning slice?): "
                               "run s2_block --split train without --limit-s1")
    else:
        q = q.with_columns(pl.lit("test").alias("split"))
    q = q.sort("s1_row")
    if limit_s1 and limit_s1 < q.height:
        parts = [stratified(g, round(limit_s1 * g.height / q.height), seed)
                 for _, g in q.group_by("split", maintain_order=True)]
        q = pl.concat(parts).sort("s1_row")
    return q


# ---------- record features (once per record, not per pair) ----------
def build_vocab(tok: pl.DataFrame, n_pool: int) -> pl.DataFrame:
    return (tok.group_by("tok").len("df").sort("tok")
               .with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("tid"),
                             (np.log(n_pool) - pl.col("df").log()).cast(pl.Float32).alias("idf")))


def encode(tok: pl.DataFrame, vocab: pl.DataFrame, n_pool: int, p: str) -> pl.DataFrame:
    """Per idx: sorted in-vocab token ids, token count, IDF sum (OOV tokens at ln N)."""
    t = tok.join(vocab.select("tok", "tid", "idf"), on="tok", how="left").sort("idx", "tid")
    return t.group_by("idx", maintain_order=True).agg(
        pl.col("tid").drop_nulls().alias(f"{p}_tids"),
        pl.len().cast(pl.Float32).alias(f"{p}_n"),
        pl.col("idf").fill_null(float(np.log(n_pool))).sum().cast(pl.Float32).alias(f"{p}_idf"))


def records(df: pl.DataFrame, key: str, vocab_a, vocab_n, n_pool, name_freq) -> pl.DataFrame:
    df = df.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    a = encode(view_tokens(df, "W_addr", {}), vocab_a, n_pool, "a")
    n = encode(view_tokens(df, "W_name", {}), vocab_n, n_pool, "n")
    empty_u32 = pl.lit([], dtype=pl.List(pl.UInt32))
    extra = [c for c in ("src", "business_name") if c in df.columns]
    out = (df.select(
            key, "idx", "name_s", "name_sk", "addr_n", *extra,
            pl.col("name_s").str.replace_all(" ", "").alias("name_sp"),
            pl.col("name_s").str.split(" ").list.first().alias("ftok"),
            pl.col("name_s").str.len_chars().cast(pl.Float32).alias("nlen"),
            pl.col("name_script").replace_strict(SCRIPT, default=4, return_dtype=pl.Int8).alias("script"),
            pl.col("addr_nums").list.unique().alias("nums"),
            pl.col("addr_nums").list.first().alias("num1"),
            pl.col("addr_n").str.split(" ").list.tail(2)
              .list.eval(pl.element().filter(pl.element() != "")).alias("last2"))
           .join(a, on="idx", how="left").join(n, on="idx", how="left")
           .join(name_freq, on="name_s", how="left")
           .with_columns(pl.col("a_tids").fill_null(empty_u32), pl.col("n_tids").fill_null(empty_u32),
                         pl.col(["a_n", "a_idf", "n_n", "n_idf"]).fill_null(0.0),
                         pl.col("name_freq").fill_null(0.0))
           .drop("idx"))
    return out


# ---------- reverse competition ----------
def reverse_competition(parts, max_c: int, views) -> pl.DataFrame:
    """Per (s1_row, pool_row): this S1's rank / gap among ALL S1s whose capped candidates
    include the pool row. Counts are left out: train has more S1s per pool row than test."""
    cols = ["s1_row", "pool_row", "cand_pos"] + [f"score_{v}" for v in views]
    d = pl.concat([pl.read_parquet(p, columns=cols) for p in parts]).filter(pl.col("cand_pos") < max_c)
    exprs = []
    for v in views:
        s = pl.col(f"score_{v}").fill_null(0.0)
        exprs += [s.rank("min", descending=True).over("pool_row").cast(pl.Float32).alias(f"rc_rank_{v}"),
                  (s.max().over("pool_row") - s).cast(pl.Float32).alias(f"rc_gap_{v}")]
    return d.select("s1_row", "pool_row", *exprs)


# ---------- per-chunk pair features ----------
def cp(a, b, scorer, nw):
    return process.cpdist(a, b, scorer=scorer, workers=nw, dtype=np.float32)


def idf_sum(sh: pl.Series, idf: np.ndarray) -> np.ndarray:
    ex = pl.DataFrame({"i": np.arange(len(sh)), "t": sh}).explode("t").drop_nulls("t")
    return np.bincount(ex["i"].to_numpy(), weights=idf[ex["t"].to_numpy()],
                       minlength=len(sh)).astype(np.float32)


def safe_div(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


def emb_cos(ea: np.ndarray, eb: np.ndarray, E: np.ndarray, batch=500_000) -> np.ndarray:
    out = np.empty(ea.size, np.float32)
    for i in range(0, ea.size, batch):
        a, b = E[ea[i:i + batch]].astype(np.float32), E[eb[i:i + batch]].astype(np.float32)
        out[i:i + batch] = np.einsum("ij,ij->i", a, b)
    return out


def pair_features(d: pl.DataFrame, views, idf_a, idf_n, nw, addr_partial: bool, E=None) -> pl.DataFrame:
    """d: pairs joined with S1 columns (plain names) and pool columns (suffix _p)."""
    F = {}
    # blocking
    for v in views:
        F[f"score_{v}"] = pl.col(f"score_{v}").cast(pl.Float32)
        F[f"rank_{v}"] = pl.col(f"rank_{v}").cast(pl.Float32)
        F[f"gap_{v}"] = (pl.col(f"score_{v}").max().over("s1_row") - pl.col(f"score_{v}")).cast(pl.Float32)
    F["n_views"] = pl.sum_horizontal([pl.col(f"rank_{v}").is_not_null() for v in views]).cast(pl.Float32)
    F["n_cands"] = pl.len().over("s1_row").cast(pl.Float32)
    F["cand_pos"] = pl.col("cand_pos").cast(pl.Float32)
    # set features
    d = d.with_columns(pl.col("a_tids").list.set_intersection(pl.col("a_tids_p")).alias("_ash"),
                       pl.col("n_tids").list.set_intersection(pl.col("n_tids_p")).alias("_nsh"),
                       pl.col("nums").list.set_intersection(pl.col("nums_p")).alias("_numsh"),
                       pl.col("last2").list.set_intersection(pl.col("last2_p")).list.len().alias("_l2"))
    a_sh = d["_ash"].list.len().to_numpy().astype(np.float32)
    a_n, a_np = d["a_n"].to_numpy(), d["a_n_p"].to_numpy()
    a_idf_sh = idf_sum(d["_ash"], idf_a)
    n_idf_sh = idf_sum(d["_nsh"], idf_n)
    num_sh = d["_numsh"].list.len().to_numpy().astype(np.float32)
    n_nums, n_nums_p = d["nums"].list.len().to_numpy(), d["nums_p"].list.len().to_numpy()
    arr = {
        "a_contain": safe_div(a_sh, np.minimum(a_n, a_np)),
        "a_jacc": safe_div(a_sh, a_n + a_np - a_sh),
        "a_idf_cont": safe_div(a_idf_sh, np.minimum(d["a_idf"].to_numpy(), d["a_idf_p"].to_numpy())),
        "a_idf_shared": a_idf_sh,
        "n_idf_cont": safe_div(n_idf_sh, np.minimum(d["n_idf"].to_numpy(), d["n_idf_p"].to_numpy())),
        "num_shared": num_sh,
        "num_conflict": ((n_nums > 0) & (n_nums_p > 0) & (num_sh == 0)).astype(np.float32),
        "num_longest": d["_numsh"].list.eval(pl.element().str.len_chars()).list.max()
                        .fill_null(0).to_numpy().astype(np.float32),
        "num_first": (d["num1"] == d["num1_p"]).fill_null(False).to_numpy().astype(np.float32),
        "a_last2": d["_l2"].to_numpy().astype(np.float32),
    }
    # strings (cpdist over whole columns, multi-threaded)
    ns, nsp = d["name_s"].to_list(), d["name_s_p"].to_list()
    sk, skp = d["name_sk"].to_list(), d["name_sk_p"].to_list()
    ad, adp = d["addr_n"].to_list(), d["addr_n_p"].to_list()
    for pre, x, y in (("n", ns, nsp), ("sk", sk, skp)):
        arr[f"{pre}_ratio"] = cp(x, y, fuzz.ratio, nw)
        arr[f"{pre}_tsort"] = cp(x, y, fuzz.token_sort_ratio, nw)
        arr[f"{pre}_tset"] = cp(x, y, fuzz.token_set_ratio, nw)
        arr[f"{pre}_partial"] = cp(x, y, fuzz.partial_ratio, nw)
        arr[f"{pre}_jw"] = cp(x, y, JaroWinkler.normalized_similarity, nw) * 100
    sp, spp = d["name_sp"].to_list(), d["name_sp_p"].to_list()
    arr["nsp_ratio"] = cp(sp, spp, fuzz.ratio, nw)
    arr["nsp_partial"] = cp(sp, spp, fuzz.partial_ratio, nw)
    a_empty = ((d["addr_n"] == "") | (d["addr_n_p"] == "")).to_numpy()
    arr["a_tset"] = np.where(a_empty, np.nan, cp(ad, adp, fuzz.token_set_ratio, nw)).astype(np.float32)
    if addr_partial:
        arr["a_partial"] = np.where(a_empty, np.nan, cp(ad, adp, fuzz.partial_ratio, nw)).astype(np.float32)
    if E is not None:
        arr["emb_cos"] = emb_cos(d["eid"].to_numpy(), d["eid_p"].to_numpy(), E)
    out = d.select(*ID_COLS, *[e.alias(k) for k, e in F.items()],
                   (pl.col("name_s") == pl.col("name_s_p")).cast(pl.Float32).alias("n_exact"),
                   (pl.col("name_sk") == pl.col("name_sk_p")).cast(pl.Float32).alias("sk_exact"),
                   (pl.col("ftok") == pl.col("ftok_p")).fill_null(False).cast(pl.Float32).alias("n_first_tok"),
                   (pl.col("nlen") - pl.col("nlen_p")).abs().alias("n_len_diff"),
                   pl.col("script").cast(pl.Float32).alias("script_s1"),
                   pl.col("script_p").cast(pl.Float32).alias("script_pool"),
                   (pl.col("script") != pl.col("script_p")).cast(pl.Float32).alias("cross_script"),
                   pl.col("name_freq").alias("name_freq_s1"),
                   pl.col("name_freq_p").alias("name_freq_pool"),
                   (pl.col("addr_n") == "").cast(pl.Float32).alias("a_missing_s1"),
                   (pl.col("addr_n_p") == "").cast(pl.Float32).alias("a_missing_pool"),
                   (pl.col("src_p") == "S3").cast(pl.Float32).alias("is_s3"),
                   *[c for c in d.columns if c.startswith("rc_")])
    out = out.with_columns([pl.Series(k, v, dtype=pl.Float32) for k, v in arr.items()])
    out = out.with_columns(pl.col(pl.Float32).fill_null(float("nan")))
    # group context: where this pair sits among its S1's candidates
    g = {"n_tset": "n", "a_idf_cont": "a", **({"emb_cos": "e"} if E is not None else {})}
    return out.with_columns(
        *[pl.col(c).fill_nan(0.0).rank("min", descending=True).over("s1_row").cast(pl.Float32)
            .alias(f"{p}_grp_rank") for c, p in g.items()],
        *[(pl.col(c).fill_nan(0.0).max().over("s1_row") - pl.col(c).fill_nan(0.0))
            .cast(pl.Float32).alias(f"{p}_grp_gap") for c, p in g.items()])


def main(split, limit_s1, force):
    cfg = load_config()
    fc, bc = cfg["features"], cfg["blocking"]
    max_c, nw = bc["max_candidates"], cfg["n_workers"]
    assert max_c <= bc["store_candidates"], "max_candidates > store_candidates: S2 stored fewer"
    inputs = {"s0_prepare": input_hash(split, "s0_prepare"), "s2_block": input_hash(split, "s2_block")}
    if split == "train":
        inputs["s1_gt"] = input_hash("train", "s1_gt")
    ecfg = cfg["optional"]["embed_view"]
    emb_on = bool(ecfg["enabled"] and ecfg.get("feature", True))
    if emb_on:
        inputs["s2e_embed"] = input_hash(split, "s2e_embed")
    section = {"features": fc, "max_candidates": max_c, "limit_s1": limit_s1, "seed": cfg["seed"],
               "embed": ecfg if emb_on else None,
               "code": code_hash("src/s3_features.py", "src/s2_block.py", "src/s1_gt.py")}
    with stage("s3_features", split, cfg, section, limit_s1=limit_s1, force=force,
               input_stages=inputs) as st:
        if st.skip:
            return
        views = [v for v in ("W_all", "C_all", "W_name", "W_addr", "C_name") if bc["views"][v]["enabled"]]
        scope = build_scope(split, limit_s1, cfg["seed"])
        atomic_write_parquet(scope, st.work_dir / "s1_scope.parquet")
        log.info(f"scope {scope.height:,} S1s: "
                 f"{dict(scope.group_by('split').len().sort('split').iter_rows())}; cap {max_c}")
        gt = None
        if split == "train":
            gt = (pl.read_parquet("work/train/s1_gt/gt_pairs.parquet")
                    .join(scope.select("s1_row"), on="s1_row").with_columns(pl.lit(1, pl.Int8).alias("label")))
        s0 = f"work/{split}/s0_prepare"
        bn = ["business_name"] if emb_on else []
        s1_all = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "country"] + REC_COLS + bn)
        pool_all = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "src", "country"] + REC_COLS + bn)
        E = texts = None
        if emb_on:     # names were embedded once each (s2e_embed); eid indexes emb.npy rows
            e_dir = Path(f"work/{split}/s2e_embed")
            texts = pl.read_parquet(e_dir / "texts.parquet").rename({"text": "business_name"})
            E = np.load(e_dir / "emb.npy", mmap_mode="r")
            s1_all = s1_all.with_columns(pl.col("business_name").fill_null(""))
            pool_all = pool_all.with_columns(pl.col("business_name").fill_null(""))
        s2dir = Path(f"work/{split}/s2_block")
        report, total = {"countries": {}}, 0
        for country in scope["country"].unique().sort().to_list():
            parts = sorted(s2dir.glob(f"part-{safe(country)}-*.parquet"))
            outs = [st.work_dir / p.name for p in parts]
            if not parts:
                log.warning(f"[{country}] no S2 parts (all its S1s got zero candidates?)")
                continue
            if all(o.exists() for o in outs):
                n = sum(pl.scan_parquet(o).select(pl.len()).collect().item() for o in outs)
                total += n
                log.info(f"[{country}] all {len(outs)} parts exist ({n:,} pairs), skipping")
                continue
            t = time.time()
            sc = scope.filter(pl.col("country") == country).select("s1_row")
            pc = pool_all.filter(pl.col("country") == country)
            n_pool = pc.height
            pidx = pc.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
            vocab_a = build_vocab(view_tokens(pidx, "W_addr", {}), n_pool)
            vocab_n = build_vocab(view_tokens(pidx, "W_name", {}), n_pool)
            name_freq = pc.group_by("name_s").len("name_freq").with_columns(
                pl.col("name_freq").cast(pl.Float32).log1p())
            idf_a, idf_n = vocab_a["idf"].to_numpy(), vocab_n["idf"].to_numpy()
            s1r = records(s1_all.join(sc, on="s1_row"), "s1_row", vocab_a, vocab_n, n_pool, name_freq)
            poolr = records(pc, "pool_row", vocab_a, vocab_n, n_pool, name_freq)
            if emb_on:
                s1r = s1r.join(texts, on="business_name", how="left").drop("business_name")
                poolr = poolr.join(texts, on="business_name", how="left").drop("business_name")
            poolr = poolr.rename({c: f"{c}_p" for c in poolr.columns if c != "pool_row"})
            rc = reverse_competition(parts, max_c, views) if fc.get("reverse_competition") else None
            if rc is not None:
                rc = rc.join(sc, on="s1_row")
            prep_s = time.time() - t
            log.info(f"[{country}] pool {n_pool:,}, scope S1 {sc.height:,}, vocab addr {vocab_a.height:,} "
                     f"name {vocab_n.height:,}; prep {prep_s:.1f}s")
            cstats = {"scope_s1": sc.height, "pool": n_pool, "prep_seconds": round(prep_s, 1),
                      "pairs": 0, "pair_seconds": 0.0, "labels": 0}
            for part, out in zip(parts, outs):
                if out.exists():
                    n = pl.scan_parquet(out).select(pl.len()).collect().item()
                    total += n
                    cstats["pairs"] += n
                    continue
                t = time.time()
                pairs = (pl.read_parquet(part).filter(pl.col("cand_pos") < max_c)
                           .join(sc, on="s1_row"))
                if rc is not None:
                    pairs = pairs.join(rc, on=ID_COLS, how="left")
                d = pairs.join(s1r, on="s1_row", how="left").join(poolr, on="pool_row", how="left")
                if emb_on:     # only in-scope candidates were embedded: all must resolve
                    assert d["eid"].null_count() == 0 and d["eid_p"].null_count() == 0
                feats = pair_features(d, views, idf_a, idf_n, nw, fc["address_partial_ratio"], E)
                if gt is not None:
                    feats = (feats.join(gt, on=ID_COLS, how="left")
                                  .with_columns(pl.col("label").fill_null(0)))
                    cstats["labels"] += int(feats["label"].sum())
                assert "country" not in feats.columns
                feats = feats.sort(ID_COLS)
                atomic_write_parquet(feats, out)
                dt = time.time() - t
                total += feats.height
                cstats["pairs"] += feats.height
                cstats["pair_seconds"] += dt
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                log.info(f"[{country}] {part.name}: {feats.height:,} pairs, {feats.width} cols in {dt:.1f}s "
                         f"({feats.height / max(dt, 1e-9) / 1e3:.0f}K pairs/s); peak RSS {rss:.1f} GB")
                if cstats["pairs"] == feats.height:     # first chunk: NaN fractions, once per country
                    nan = {c: round(v, 3) for c, v in feats.select(
                        pl.col(pl.Float32).is_nan().mean()).row(0, named=True).items() if v > 0}
                    log.info(f"[{country}] NaN fractions: {nan}")
            cstats["pairs_per_s"] = round(cstats["pairs"] / max(cstats["pair_seconds"], 1e-9))
            report["countries"][country] = cstats
            del poolr, s1r, rc
        if split == "train":
            ngt = (pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").join(scope, on="s1_row")
                     .group_by("split").len("gt_pairs"))
            lab = (pl.scan_parquet(str(st.work_dir / "part-*.parquet")).filter(pl.col("label") == 1)
                     .select("s1_row").collect().join(scope, on="s1_row").group_by("split").len("found"))
            rec = ngt.join(lab, on="split", how="left").with_columns(
                (pl.col("found") / pl.col("gt_pairs")).round(4).alias("recall"))
            report["recall_by_split"] = {r["split"]: r for r in rec.to_dicts()}
            log.info(f"blocking recall at cap {max_c} by split: "
                     f"{ {r['split']: r['recall'] for r in rec.to_dicts()} }")
        report["total_pairs"] = total
        ev = Path(f"work/eval/s3_{split}_{limit_s1 or 'full'}")
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "summary.json").write_text(json.dumps(report, indent=1))
        st.rows = total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force)
