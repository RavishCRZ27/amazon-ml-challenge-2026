#!/usr/bin/env python3
"""S4l optional: Qwen2.5-7B-Instruct yes/no score on the GBM's uncertain band (vLLM).
Rule: specs/optional-stages.md. Run with .venv-llm/bin/python (vLLM has its own torch).

Score = log P("Yes") - log P("No") of the first generated token, zero temperature. A fixed
system prompt + few-shot TRAIN examples form a shared prefix (vLLM prefix caching). Pairs: the
band pairs of a deterministic subset of VAL / HOLDOUT S1s from S4's scored pairs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# FlashInfer's JIT sampler needs nvcc: tools/setup_llm_cuda.sh builds a CUDA_HOME shim over the pip
# toolchain in .venv-llm; without it, fall back to vLLM's torch sampler.
_CUDA = os.path.join(sys.prefix, "cuda")
if os.path.exists(os.path.join(_CUDA, "bin", "nvcc")):
    os.environ.setdefault("CUDA_HOME", _CUDA)
    os.environ["PATH"] = os.pathsep.join([os.path.join(sys.prefix, "bin"), os.path.join(_CUDA, "bin"),
                                          os.environ.get("PATH", "")])
else:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import polars as pl
import yaml

SYSTEM = ("You decide whether two business records from different sources describe the same "
          "business. Names may be transliterated from Indian scripts, abbreviated, reordered, "
          "misspelled, or carry legal suffixes (LLC, Pvt Ltd, SARL); addresses may be truncated "
          "or formatted differently. Different branches of a chain at different addresses are "
          "NOT the same business. Answer with exactly one word: Yes or No.")


def rec(name, addr):
    return f"name: {name or ''} | address: {addr or ''}"


def question(a, b):
    return f"Record A: {a}\nRecord B: {b}\nSame business?"


def load_texts(split, pairs):
    s0 = f"work/{split}/s0_prepare"
    s1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "business_name", "business_address"])
    pool = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "business_name", "business_address"])
    d = pairs.join(s1, on="s1_row", how="left").join(pool, on="pool_row", how="left", suffix="_p")
    return [question(rec(r[0], r[1]), rec(r[2], r[3])) for r in
            d.select("business_name", "business_address", "business_name_p", "business_address_p").iter_rows()]


def main(tag, band, max_s1, few_shot, model_dir, seed):
    from vllm import LLM, SamplingParams
    ev = Path(f"work/eval/s4_{tag}")
    out = Path(f"work/eval/llm_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = band
    tr = pl.read_parquet(ev / "pred_train.parquet").filter(pl.col("p").is_between(lo, hi))
    shots = pl.concat([tr.filter(pl.col("label") == 1).sample(few_shot // 2, seed=seed),
                       tr.filter(pl.col("label") == 0).sample(few_shot - few_shot // 2, seed=seed)]
                      ).sample(fraction=1.0, shuffle=True, seed=seed)
    shot_q = load_texts("train", shots.select("s1_row", "pool_row"))
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, y in zip(shot_q, shots["label"].to_list()):
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": "Yes" if y else "No"}]
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=seed)
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")}
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    for split in ("val", "holdout"):
        pr = pl.read_parquet(ev / f"pred_{split}.parquet").filter(pl.col("p").is_between(lo, hi))
        s1s = pr.select("s1_row").unique().sort("s1_row")
        if max_s1 and s1s.height > max_s1:
            s1s = s1s.sample(max_s1, seed=seed)
        pr = pr.join(s1s, on="s1_row").sort("s1_row", "pool_row")
        prompts = [tok.apply_chat_template(msgs + [{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True)
                   for q in load_texts("train", pr.select("s1_row", "pool_row"))]
        t = time.time()
        res = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t
        score = np.empty(len(res), np.float32)
        for i, r in enumerate(res):
            lp = r.outputs[0].logprobs[0]
            y = max((v.logprob for k, v in lp.items() if k in yes_ids), default=-30.0)
            n = max((v.logprob for k, v in lp.items() if k in no_ids), default=-30.0)
            score[i] = y - n
        print(f"{split}: {len(res):,} pairs ({s1s.height:,} S1s) in {dt:.1f}s = {len(res) / dt:.1f} pairs/s; "
              f"mean Yes-No {score.mean():.2f}; AUC-ish: pos {score[pr['label'].to_numpy() == 1].mean():.2f} "
              f"neg {score[pr['label'].to_numpy() == 0].mean():.2f}", flush=True)
        pr.select("s1_row", "pool_row").with_columns(pl.Series("llm", score)).write_parquet(
            out / f"scores_{split}.parquet")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    lc = cfg["optional"]["llm_triage"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=None)
    ap.add_argument("--max-s1", type=int, default=4000, help="S1s per split (throughput test size)")
    a = ap.parse_args()
    main(a.tag, a.band or cfg["optional"]["cross_encoder"]["band"], a.max_s1, lc["few_shot"],
         lc["model_dir"], cfg["seed"])
