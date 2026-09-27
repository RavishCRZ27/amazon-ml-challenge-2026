#!/usr/bin/env python3
"""S0 prepare: TSVs -> normalized s1/pool parquet. Rule: specs/prepare.md."""

import argparse
import logging
import multiprocessing as mp
import subprocess
import time

import polars as pl

from common import atomic_write_parquet, code_hash, load_config, stage
from translit import script_class, skeleton, to_latin

S0_VERSION = 2
log = logging.getLogger("s0_prepare")

SUFFIXES = ["LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
            "LIMITED", "PVT", "PRIVATE", "LLP", "PLLC", "PC", "PLC", "LP", "OPC",
            "SAS", "SASU", "SARL", "EURL", "SA", "SCI", "SNC", "SELARL", "EIRL"]
# Romanized Indic forms (approved 2026-09-26 from the top-200 tokens of Indic-script pool names)
SUFFIXES_INDIC = ["PRAIVET", "PRAIBHET", "PIRAIVET", "PRAIVATT", "LIMITET", "LIMITTAD",
                  "LIMATID", "PRA", "LI", "ELAELAPI", "ELELPI"]

# Hand-written address normalization (not a lookup): variant -> canonical token/code.
STREET = {
    "STREET": "ST", "ROAD": "RD", "AVENUE": "AVE", "AV": "AVE", "BOULEVARD": "BLVD",
    "BD": "BLVD", "DRIVE": "DR", "LANE": "LN", "HIGHWAY": "HWY", "SUITE": "STE",
    "APARTMENT": "APT", "FLOOR": "FL", "NEAR": "NR", "OPPOSITE": "OPP", "OPPOSIT": "OPP",
    "R": "RUE", "PLACE": "PL", "CHE": "CHEMIN", "IMPASSE": "IMP", "ALL": "ALLEE",
}
US_STATES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN",
    "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
}
IN_STATES = {
    "ANDHRA PRADESH": "AP", "ARUNACHAL PRADESH": "AR", "ASSAM": "AS", "BIHAR": "BR",
    "CHHATTISGARH": "CG", "CHATTISGARH": "CG", "GOA": "GA", "GUJARAT": "GJ", "HARYANA": "HR",
    "HIMACHAL PRADESH": "HP", "JHARKHAND": "JH", "KARNATAKA": "KA", "KERALA": "KL",
    "MADHYA PRADESH": "MP", "MAHARASHTRA": "MH", "MANIPUR": "MN", "MEGHALAYA": "ML",
    "MIZORAM": "MZ", "NAGALAND": "NL", "ODISHA": "OD", "ORISSA": "OD", "PUNJAB": "PB",
    "RAJASTHAN": "RJ", "SIKKIM": "SK", "TAMIL NADU": "TN", "TAMILNADU": "TN",
    "TELANGANA": "TS", "TRIPURA": "TR", "UTTAR PRADESH": "UP", "UTTARAKHAND": "UK",
    "UTTARANCHAL": "UK", "WEST BENGAL": "WB", "JAMMU AND KASHMIR": "JK", "JAMMU KASHMIR": "JK",
    "LADAKH": "LA", "PUDUCHERRY": "PY", "PONDICHERRY": "PY",
}
ADDR_MAP = {**STREET, **US_STATES, **IN_STATES}

ZW = r"[​-‍﻿]"                     # zero-width chars split Indic words
# apostrophe variants incl. cp1252 mojibake (\x92, "Â\x80\x99") -> "'"
APOS = r"(?:[ÂâÃ][\x80-\x9f]+|[’‘`´\x91\x92])"
NON_WORD = r"[^\p{L}\p{N}\s]"                   # any punctuation/symbol/control, any script


def _map_tokens(col: pl.Expr, mapping: dict) -> pl.Expr:
    """Whole-token / whole-phrase replacement. Doubling spaces keeps adjacent matches disjoint."""
    pats = [" " + k.replace(" ", "  ") + " " for k in mapping]
    reps = [" " + v + " " for v in mapping.values()]
    padded = pl.lit("  ") + col.str.replace_all(" ", "  ", literal=True) + pl.lit("  ")
    return padded.str.replace_many(pats, reps, leftmost=True)


def _collapse(col: pl.Expr) -> pl.Expr:
    return col.str.replace_all(r"\s+", " ").str.strip_chars()


def _par_map(fn, values, n_workers):
    if not values:
        return []
    with mp.get_context("fork").Pool(n_workers) as p:
        return p.map(fn, values, chunksize=max(1000, len(values) // (n_workers * 8)))


def _apply_unique(df, src_col, out_col, fn, n_workers, only_non_ascii=True, ascii_value=None):
    """Run a Python fn once per unique value (in parallel), then join back."""
    vals = df.get_column(src_col).drop_nulls().unique()
    if only_non_ascii:
        vals = vals.filter(vals.str.contains(r"[^\x00-\x7f]"))
    vals = vals.to_list()
    lut = pl.DataFrame({src_col: vals, out_col: _par_map(fn, vals, n_workers)},
                       schema={src_col: pl.String, out_col: pl.String})
    df = df.join(lut, on=src_col, how="left")
    if only_non_ascii:
        fallback = pl.col(src_col) if ascii_value is None else pl.lit(ascii_value)
        df = df.with_columns(pl.coalesce(pl.col(out_col), fallback).alias(out_col))
    return df


def normalize(df: pl.DataFrame, n_workers: int) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("business_name").fill_null("").str.replace_all(ZW, "")
          .str.replace_all(APOS, "'").alias("_name"),
        pl.col("business_address").fill_null("").str.replace_all(ZW, "")
          .str.replace_all(APOS, "'").alias("_addr"),
    )
    t = time.time()
    df = _apply_unique(df, "_name", "_name_lat", to_latin, n_workers)
    df = _apply_unique(df, "_addr", "_addr_lat", to_latin, n_workers)
    df = _apply_unique(df, "business_name", "name_script", script_class, n_workers,
                       ascii_value="ascii")
    log.info(f"  to_latin + script_class: {time.time() - t:.1f}s")

    name_n = _collapse(pl.col("_name_lat").str.to_uppercase()
                       .str.replace_all(r"['.]", "")
                       .str.replace_all(NON_WORD, " "))
    addr_n = _collapse(_map_tokens(_collapse(pl.col("_addr_lat").str.to_uppercase()
                                             .str.replace_all(NON_WORD, " ")), ADDR_MAP))
    df = df.with_columns(name_n.alias("name_n"), addr_n.alias("addr_n"))

    # suffix tokens removed anywhere, unless that empties the name
    toks = pl.col("name_n").str.split(" ")
    base = toks.list.eval(pl.element().filter(~pl.element().is_in(SUFFIXES)))
    indic = toks.list.eval(pl.element().filter(~pl.element().is_in(SUFFIXES + SUFFIXES_INDIC)))
    stripped = pl.when(pl.col("name_script") == "indic").then(indic).otherwise(base)
    df = df.with_columns(stripped.list.join(" ").alias("_ns"))
    df = df.with_columns(
        pl.when(pl.col("_ns") == "").then(pl.col("name_n")).otherwise(pl.col("_ns")).alias("name_s"),
        # numbers from the address, leading zeros stripped (003544 -> 3544)
        pl.col("_addr_lat").str.extract_all(r"\d+")
          .list.eval(pl.element().str.strip_chars_start("0").replace("", "0")).alias("addr_nums"),
    )
    t = time.time()
    df = _apply_unique(df, "name_s", "name_sk", skeleton, n_workers, only_non_ascii=False)
    log.info(f"  skeleton: {time.time() - t:.1f}s")
    return df.drop("_name", "_addr", "_name_lat", "_addr_lat", "_ns")


def read_tsv(path) -> pl.DataFrame:
    expected = int(subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
                   .stdout.split()[0]) - 1
    df = pl.read_csv(path, separator="\t", infer_schema=False)
    if df.height != expected:
        log.warning(f"{path}: {df.height} rows != wc {expected}; retrying with quote_char=None")
        df = pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None)
    assert df.height == expected, f"{path}: {df.height} rows, expected {expected}"
    return df


OUT_COLS = ["entity_id", "country", "business_name", "business_address", "name_n", "name_s",
            "name_sk", "name_script", "addr_n", "addr_nums"]


def main(split, limit_s1, force):
    cfg = load_config()
    section = {"version": S0_VERSION, "suffixes": SUFFIXES, "suffixes_indic": SUFFIXES_INDIC,
               "addr_map": ADDR_MAP, "limit_s1": limit_s1,
               "code": code_hash("src/s0_prepare.py", "src/translit.py")}
    with stage("s0_prepare", split, cfg, section, limit_s1=limit_s1, force=force) as st:
        if st.skip:
            return
        nw = cfg["n_workers"]
        d = f"data/dataset/{split}"

        t = time.time()
        s1 = read_tsv(f"{d}/{split}_source1.tsv")
        if limit_s1:
            s1 = s1.head(limit_s1)
        s1 = s1.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int64).alias("s1_row"))
        s1 = normalize(s1, nw).sort("s1_row")
        atomic_write_parquet(s1.select(["s1_row"] + OUT_COLS), st.work_dir / "s1.parquet")
        log.info(f"s1: {s1.height:,} rows in {time.time() - t:.1f}s")

        t = time.time()
        s2 = read_tsv(f"{d}/{split}_source2.tsv").with_columns(pl.lit("S2").alias("src"))
        s3 = read_tsv(f"{d}/{split}_source3.tsv").with_columns(pl.lit("S3").alias("src"))
        pool = pl.concat([s2, s3])                  # S2 -> [0, n2), S3 -> [n2, n2+n3)
        pool = pool.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int64).alias("pool_row"))
        pool = normalize(pool, nw).sort("pool_row")
        atomic_write_parquet(pool.select(["pool_row", "src"] + OUT_COLS), st.work_dir / "pool.parquet")
        log.info(f"pool: {pool.height:,} rows (S2 {s2.height:,}, S3 {s3.height:,}) "
                 f"in {time.time() - t:.1f}s")
        for c, n in pool.group_by("country").len().sort("country").iter_rows():
            log.info(f"  pool country {c}: {n:,}")
        st.rows = s1.height + pool.height


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force)
