#!/usr/bin/env python3
"""Shared infrastructure: config, paths, stage context, atomic writes, S3 sync, logging."""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
import yaml

# Config
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and expand config.yaml, with env overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # n_workers: auto -> os.cpu_count()
    if cfg.get("n_workers") == "auto":
        cfg["n_workers"] = os.cpu_count()

    # Env overrides
    if "RUN_ID" in os.environ:
        cfg["run_id"] = os.environ["RUN_ID"]
    if "BER_S3_BUCKET" in os.environ:
        cfg["s3"]["bucket"] = os.environ["BER_S3_BUCKET"]

    return cfg

# Paths
def make_work_dir(split: str, stage: str) -> Path:
    """work/<split>/<stage>/"""
    p = Path("work") / split / stage
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_log_dir() -> Path:
    """logs/"""
    p = Path("logs")
    p.mkdir(exist_ok=True)
    return p

def setup_logging(stage: str):
    """Configure logging to logs/<stage>.log and stderr."""
    make_log_dir()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    # File handler
    fh = logging.FileHandler(f"logs/{stage}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

# Checkpointing
def get_done_path(work_dir: Path) -> Path:
    """work/<split>/<stage>/_DONE.json"""
    return work_dir / "_DONE.json"

def config_hash(cfg_section: Dict[str, Any], prev_hashes: Dict[str, str]) -> str:
    """Hash of (config section + input stages' _DONE.json hashes)."""
    h = hashlib.sha256()
    h.update(json.dumps(cfg_section, sort_keys=True, default=str).encode())
    for stage, stage_hash in sorted(prev_hashes.items()):
        h.update(f"{stage}:{stage_hash}".encode())
    return h.hexdigest()[:16]

def code_hash(*paths: str) -> str:
    """Hash of source files, so a code edit invalidates the stage's checkpoint."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def input_hash(split: str, stage_name: str) -> str:
    """config_hash of a finished upstream stage; fails loudly if it hasn't run."""
    done = load_done(Path("work") / split / stage_name / "_DONE.json")
    if done is None:
        raise RuntimeError(f"upstream stage {stage_name}/{split} has not finished")
    return done["config_hash"]


def load_done(done_path: Path) -> Optional[Dict[str, Any]]:
    """Load _DONE.json if it exists."""
    try:
        with open(done_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def write_done(
    done_path: Path,
    rows: int,
    seconds: float,
    cfg_hash: str,
    git_commit: Optional[str] = None,
):
    """Write _DONE.json."""
    done_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rows": rows,
        "seconds": seconds,
        "config_hash": cfg_hash,
        "git_commit": git_commit or "unknown",
        "timestamp": time.time(),
    }
    tmp = done_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, done_path)

def append_timings(stage: str, split: str, limit_s1: Optional[int], rows: int, seconds: float):
    """Append to logs/timings.tsv."""
    make_log_dir()
    limit_str = f"_{limit_s1}" if limit_s1 else "full"
    with open("logs/timings.tsv", "a") as f:
        f.write(f"{stage}\t{split}\t{limit_str}\t{rows}\t{seconds}\n")

# Stage context manager
@contextlib.contextmanager
def stage(
    name: str,
    split: str,
    config: Dict[str, Any],
    cfg_section: Dict[str, Any],
    limit_s1: Optional[int] = None,
    force: bool = False,
    input_stages: Optional[Dict[str, str]] = None,
):
    """
    Context manager for a stage:
    - Skip if _DONE.json exists and config hash matches
    - Time execution
    - Write _DONE.json and append to logs/timings.tsv

    Usage:
        with stage(name, split, cfg, cfg['blocking']) as st:
            if st.skip:
                return
            ...
            st.rows = n
    """
    setup_logging(name)
    logger = logging.getLogger(name)

    work_dir = make_work_dir(split, name)
    done_path = get_done_path(work_dir)
    expected_hash = config_hash(cfg_section, input_stages or {})

    st = types.SimpleNamespace(skip=False, rows=0, work_dir=work_dir, hash=expected_hash)
    if not force:
        done = load_done(done_path)
        if done and done.get("config_hash") == expected_hash:
            logger.info(f"Skipping {name}/{split} (checkpoint exists)")
            st.skip = True
            st.rows = done.get("rows", 0)
            yield st
            return

    # Parts from an unfinished run are reused only if they were made under the same hash.
    marker = work_dir / "_HASH"
    prev = marker.read_text().strip() if marker.exists() else None
    if force or prev != expected_hash:
        stale = [p for p in work_dir.iterdir() if p.name != "_HASH"]
        if stale:
            logger.info(f"Clearing {len(stale)} stale files in {work_dir} (hash changed or --force)")
        for p in stale:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        marker.write_text(expected_hash)
    else:
        logger.info(f"Resuming {name}/{split}: keeping parts made under the same hash")

    logger.info(f"Running {name}/{split} (limit_s1={limit_s1})")
    start = time.time()
    try:
        yield st
    except Exception:
        logger.exception("Stage failed")
        raise
    seconds = time.time() - start
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    write_done(done_path, st.rows, seconds, expected_hash, commit)
    append_timings(name, split, limit_s1, st.rows, seconds)
    s3_sync(name, config)
    logger.info(f"Done {name}/{split}: {st.rows} rows in {seconds:.1f}s")

# Atomic writes
def atomic_write_parquet(df: pl.DataFrame, path: Path):
    """Write parquet atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

# S3 sync
def s3_sync(stage: str, config: Dict[str, Any]):
    """Copy work/ (minus excludes), output/ and timings.tsv to S3 in the background.

    No-op when BER_S3_BUCKET is unset. flock serializes overlapping syncs; the detached
    session lets the upload outlive the stage (and its tmux pane) without blocking the pipeline.
    """
    bucket = os.environ.get(config.get("s3", {}).get("enabled_if_env", "BER_S3_BUCKET"))
    if not bucket:
        return
    dest = f"s3://{bucket}/{config['run_id']}"
    excl = " ".join(f"--exclude {shlex.quote(e)}" for e in config["s3"].get("exclude", []))
    q = "--only-show-errors"
    # rc collects every command's failure ($? after $(date) would always read 0)
    script = (f"rc=0; aws s3 sync work/ {dest}/work/ {excl} {q} || rc=1; "
              f"if [ -d output ]; then aws s3 sync output/ {dest}/output/ {q} || rc=1; fi; "
              f"if [ -f logs/timings.tsv ]; then aws s3 cp logs/timings.tsv {dest}/logs/timings.tsv {q} || rc=1; fi; "
              f"echo \"$(date -Is) sync after {stage} exit=$rc\"")
    make_log_dir()
    with open("logs/s3_sync.log", "a") as log_f:
        subprocess.Popen(["flock", "logs/.s3sync.lock", "bash", "-c", script],
                         stdout=log_f, stderr=log_f, start_new_session=True)
    logging.getLogger("s3").info(f"S3 sync to {dest} started in background (logs/s3_sync.log)")


# Self-test
def selftest():
    """Verify skip-on-rerun and atomic writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir) / "work"
        done_path = get_done_path(work_dir)

        # Test 1: atomic write
        test_df = pl.DataFrame({"col": [1, 2, 3]})
        test_path = work_dir / "test.parquet"
        atomic_write_parquet(test_df, test_path)
        assert test_path.exists(), "Parquet file not written"
        loaded_df = pl.read_parquet(test_path)
        assert len(loaded_df) == 3, "Parquet data mismatch"

        # Test 2: checkpoint
        write_done(done_path, rows=100, seconds=1.5, cfg_hash="abc123")
        loaded_done = load_done(done_path)
        assert loaded_done["rows"] == 100, "Checkpoint rows mismatch"
        assert loaded_done["config_hash"] == "abc123", "Hash mismatch"

        # Test 3: stage() runs once, then skips on rerun; changed config reruns
        old = os.getcwd()
        os.chdir(tmpdir)
        try:
            cfg = {"run_id": "t", "s3": {"enabled_if_env": "BER_SELFTEST_NEVER_SET"}}
            with stage("t", "train", cfg, {"a": 1}) as st:
                assert not st.skip
                st.rows = 7
            with stage("t", "train", cfg, {"a": 1}) as st:
                assert st.skip and st.rows == 7, "rerun should skip"
            with stage("t", "train", cfg, {"a": 2}) as st:
                assert not st.skip, "config change should invalidate"
                (st.work_dir / "part-x-0.parquet").write_text("partial")
            # simulated crash: same hash keeps the part, a new hash clears it
            (st.work_dir / "_DONE.json").unlink()
            with stage("t", "train", cfg, {"a": 2}) as st:
                assert (st.work_dir / "part-x-0.parquet").exists(), "resume should keep parts"
            (st.work_dir / "_DONE.json").unlink()
            with stage("t", "train", cfg, {"a": 3}) as st:
                assert not (st.work_dir / "part-x-0.parquet").exists(), "new hash should clear parts"
            lines = Path("logs/timings.tsv").read_text().strip().splitlines()
            assert len(lines) == 4, f"timings should only log real runs: {lines}"
        finally:
            os.chdir(old)

        print("Common self-test passed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    else:
        print("Import this module or run with --selftest")
