#!/usr/bin/env python3
"""Cross-encoder inference speed: autocast (current) vs bf16 weights, batch sizes. Same pairs,
reports GPU pairs/s and the max/mean score difference vs the current path.

usage: .venv/bin/python tools/bench_xenc_infer.py --model work/eval/xenc_exp_rr300k/model [--n 50000]
"""

import argparse
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402


def run(model, ids, mask, bs, weights_bf16):
    import torch
    lengths = mask.sum(1)
    order = np.argsort(lengths, kind="stable")
    out = np.empty(ids.shape[0], np.float32)
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not weights_bf16):
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            L = int(lengths[b].max())
            x = torch.from_numpy(ids[b, :L]).long().cuda()
            m = torch.from_numpy(mask[b, :L]).long().cuda()
            out[b] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
    torch.cuda.synchronize()
    return out, time.time() - t


def main(a):
    import torch
    d = pl.read_parquet("work/train/s4x_xenc/scores_val.parquet").filter(pl.col("p").is_between(0.01, 0.99)).head(a.n)
    tok, model = xe.load_xenc(a.model)
    model.eval()
    ids, mask = xe.tokenize(tok, *xe.texts("train", d), 96)
    run(model, ids[:2048], mask[:2048], 1024, False)                     # warm-up
    base, t0 = run(model, ids, mask, 1024, False)
    print(f"autocast bs1024: {len(d) / t0:,.0f} pairs/s (GPU only)", flush=True)
    for bs in (2048,):
        s, t = run(model, ids, mask, bs, False)
        print(f"autocast bs{bs}: {len(d) / t:,.0f} pairs/s; max|diff| {np.abs(s - base).max():.4f}", flush=True)
    model.to(torch.bfloat16)
    for bs in (1024, 2048):
        s, t = run(model, ids, mask, bs, True)
        print(f"bf16 weights bs{bs}: {len(d) / t:,.0f} pairs/s; max|diff| {np.abs(s - base).max():.4f} "
              f"mean|diff| {np.abs(s - base).mean():.5f} sign flips {(np.sign(s) != np.sign(base)).sum()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50000)
    main(ap.parse_args())
