#!/usr/bin/env python3
"""S4x optional: fine-tuned cross-encoder on the stage-1 uncertain band + combiner.
Rule: specs/optional-stages.md (cross_encoder).

1. Train (GPU): TRAIN pairs with stage-1 p in `band` + TRAIN positives + a random sample of
   easy negatives; text = raw "name | address" of both records (original script); one epoch,
   BCE on a 1-logit head over a local MIT backbone (multilingual-e5-small by default).
2. Score VAL and HOLDOUT band pairs -> `xenc` logit.
3. Combiner on band pairs: logistic regression of the label on [logit p, xenc, logit p * xenc];
   p' = p outside the band. Cross-fitted over 2 folds of VAL S1s to choose (t, r) honestly;
   refit on all of VAL for test. HOLDOUT is scored once with the VAL-fit combiner (report only).
Step-3 evidence (VAL, 100K S1s): +0.0263 +/- 0.0004 paired over stage 1.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import polars as pl

import threshold as th
from common import atomic_write_parquet, code_hash, input_hash, load_config, stage

log = logging.getLogger("s4x_xenc")
ART = Path("output/artifacts")
FOLDS = 2


# ---------- text + model helpers (shared with S5) ----------
def texts(split: str, pairs: pl.DataFrame):
    s0 = f"work/{split}/s0_prepare"
    t = lambda: (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("txt")
    s1 = pl.read_parquet(f"{s0}/s1.parquet", columns=["s1_row", "business_name", "business_address"])
    pool = pl.read_parquet(f"{s0}/pool.parquet", columns=["pool_row", "business_name", "business_address"])
    d = (pairs.select("s1_row", "pool_row")
              .join(s1.select("s1_row", t()), on="s1_row", how="left", maintain_order="left")
              .join(pool.select("pool_row", t()), on="pool_row", how="left", suffix="_p", maintain_order="left"))
    return d["txt"].to_list(), d["txt_p"].to_list()


def tokenize(tok, a, b, max_len):
    enc = tok(a, b, truncation=True, max_length=max_len, padding="max_length", return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int8)


def batches(n, bs, shuffle, seed=0, lengths=None, pool=64):
    """Index batches. With `lengths`, batches hold similar lengths (less padding): shuffled
    chunks of `pool` batches are length-sorted, cut into batches, and the batch order shuffled."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    if lengths is None:
        for i in range(0, n, bs):
            yield idx[i:i + bs]
        return
    out = []
    for i in range(0, n, bs * pool):
        ch = idx[i:i + bs * pool]
        ch = ch[np.argsort(lengths[ch], kind="stable")]
        out += [ch[j:j + bs] for j in range(0, len(ch), bs)]
    for k in (rng.permutation(len(out)) if shuffle else range(len(out))):
        yield out[k]


def score(model, ids, mask, bs=1024):
    """Logits in input order; pairs run length-sorted so each batch trims its padding."""
    import torch
    out = np.empty(ids.shape[0], np.float32)
    lengths = mask.sum(1)
    order = np.argsort(lengths, kind="stable")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            L = int(lengths[b].max())                      # trim padding per batch
            x = torch.from_numpy(ids[b, :L]).long().cuda()
            m = torch.from_numpy(mask[b, :L]).long().cuda()
            out[b] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
    return out


def load_xenc(model_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok, AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()


def score_pairs(split, pairs, tok, model, max_len):
    return score(model, *tokenize(tok, *texts(split, pairs), max_len))


# ---------- combiner (shared with S5) ----------
def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(p, s):
    lp = logit(p)
    return np.column_stack([lp, s, lp * s])


def combine_apply(p, s, inb, coef) -> np.ndarray:
    """p' = sigmoid(X w + b) on band pairs, p elsewhere."""
    out = np.asarray(p, np.float64).copy()
    z = design(out[inb], s[inb]) @ np.asarray(coef["w"]) + coef["b"]
    out[inb] = 1.0 / (1.0 + np.exp(-z))
    return out


def fit_combiner(p, s, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=1000).fit(design(p, s), y)
    return {"w": m.coef_[0].tolist(), "b": float(m.intercept_[0])}


# ---------- train ----------
def train_xenc(ev_dir, xc, seed, out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if (out_dir / "config.json").exists():       # resume: training finished under this hash
        log.info(f"reusing trained cross-encoder in {out_dir}")
        return load_xenc(out_dir)
    lo, hi = xc["train_band"]
    tr = pl.read_parquet(ev_dir / "pred_train.parquet")
    inb = tr.filter(pl.col("p").is_between(lo, hi))
    if inb.height > xc["max_train"]:
        inb = inb.sample(xc["max_train"], seed=seed)
    pos = tr.filter((pl.col("label") == 1) & ~pl.col("p").is_between(lo, hi))
    neg = tr.filter((pl.col("label") == 0) & ~pl.col("p").is_between(lo, hi))
    n_rest = max(0, xc["max_train"] - inb.height)
    take_pos = min(pos.height, n_rest // 2)
    tr_pairs = pl.concat([inb, pos.sample(take_pos, seed=seed),
                          neg.sample(min(neg.height, n_rest - take_pos), seed=seed)])
    log.info(f"train pairs {tr_pairs.height:,} (band {inb.height:,}, pos {take_pos:,}); "
             f"label rate {tr_pairs['label'].mean():.3f}")
    tok = AutoTokenizer.from_pretrained(xc["model_dir"])
    t = time.time()
    ids, mask = tokenize(tok, *texts("train", tr_pairs), xc["max_len"])
    y = tr_pairs["label"].to_numpy().astype(np.float32)
    log.info(f"tokenized in {time.time() - t:.1f}s; mean length {mask.sum(1).mean():.1f}")
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(xc["model_dir"], num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=xc["lr"], weight_decay=0.01)
    bs, epochs = xc["batch_size"], xc["epochs"]
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t, step = time.time(), 0
    lengths = mask.sum(1)
    for ep in range(epochs):
        for b in batches(len(y), bs, True, seed + ep, lengths=lengths):
            L = int(mask[b].sum(1).max())
            x = torch.from_numpy(ids[b, :L]).long().cuda()
            m = torch.from_numpy(mask[b, :L]).long().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
            loss = lossf(lg, torch.from_numpy(y[b]).cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                log.info(f"step {step}/{steps} loss {loss.item():.4f} ({step * bs / (time.time() - t):,.0f} pairs/s)")
    log.info(f"trained {steps} steps in {time.time() - t:.1f}s")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    os.replace(tmp, out_dir)                       # config.json appears only when complete
    return load_xenc(out_dir)                      # score with exactly what S5 will load


RECIPE = ("model_dir", "max_len", "epochs", "train_band", "batch_size", "lr", "max_train")


def adopt_trained(xc, dst):
    """Reuse a cross-encoder already trained with this exact recipe on the current S4's TRAIN
    pairs (`init_dir`, e.g. tools/exp_xenc.py's model) instead of retraining. CUDA training is
    not bit-reproducible, so this keeps the submitted model identical to the evaluated one."""
    import shutil
    src = Path(xc["init_dir"])
    assert (src / "config.json").exists(), f"init_dir {src} holds no trained model"
    rec = json.loads((src.parent / "xc.json").read_text())
    diff = {k: (rec.get(k), xc.get(k)) for k in RECIPE if rec.get(k) != xc.get(k)}
    assert not diff, f"init_dir {src} was trained with another recipe: {diff}"
    s4_done = Path("work/train/s4_train/_DONE.json")
    assert (src / "config.json").stat().st_mtime > s4_done.stat().st_mtime, \
        f"init_dir {src} predates the current S4 run (trained on other stage-1 predictions)"
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)
    log.info(f"adopted cross-encoder trained with the same recipe: {src}")


# ---------- evaluation ----------
def ev_arrays(split, pred, scope, n_gt):
    sc = (scope.filter(pl.col("split") == split).sort("s1_row")
               .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
               .join(n_gt, on="s1_row", how="left", maintain_order="left").with_columns(pl.col("n_gt").fill_null(0)))
    d = pred.join(sc.select("s1_row", "idx"), on="s1_row").sort("s1_row", "pool_row")
    return sc, d


def evaluate(E, s4_thr, tcfg, inj):
    """Combiner cross-fitted over VAL S1 folds for the (t, r) choice, refit on all VAL for
    HOLDOUT/test. E[split] = {sc, d, inb, s, y, p}. Returns (report, coef, per-S1 F by split)."""
    v = E["val"]
    fold = (v["sc"]["s1_row"].hash(seed=7) % FOLDS).to_numpy()[v["d"]["idx"].to_numpy()]
    pv = v["p"].astype(np.float64).copy()
    for k in range(FOLDS):
        tr_m, te_m = v["inb"] & (fold != k), v["inb"] & (fold == k)
        coef_k = fit_combiner(v["p"][tr_m], v["s"][tr_m], v["y"][tr_m])
        pv[te_m] = combine_apply(v["p"], v["s"], te_m, coef_k)[te_m]
    coef = fit_combiner(v["p"][v["inb"]], v["s"][v["inb"]], v["y"][v["inb"]])
    assert np.isfinite(pv).all()
    rep, Fs = {"combiner": coef, "injective": inj}, {}
    for name, p_ in (("val", pv), ("holdout", None)):
        e = E[name]
        if p_ is None:
            p_ = combine_apply(e["p"], e["s"], e["inb"], coef)
            assert np.isfinite(p_).all()
        s1i, pool = e["d"]["idx"].to_numpy(), e["d"]["pool_row"].to_numpy()
        n = e["sc"].height
        c = th.Cands(s1i, pool, p_, n)
        if name == "val":
            rows = th.sweep(c, e["y"], e["sc"]["n_gt"].to_numpy(), tcfg["t_grid"], tcfg["relative_r"], [inj])
            rep["choice"] = th.choose(rows, inj, tcfg["plateau_tol"])
        keep = th.select(c, rep["choice"]["t"], rep["choice"]["r"], inj)
        ng = e["sc"]["n_gt"].to_numpy()
        F = th.per_s1_f05(ng, s1i, keep, e["y"], n)
        c1 = th.Cands(s1i, pool, e["p"], n)                   # stage 1 alone, at S4's (t, r)
        F1 = th.per_s1_f05(ng, s1i, th.select(c1, s4_thr["t"], s4_thr["r"], inj), e["y"], n)
        gtc = np.bincount(s1i, weights=e["y"], minlength=n)
        country = e["sc"]["country"].to_numpy()
        rep[name] = {"f05": round(float(F.mean()), 5), "se": round(th.se(F), 5),
                     "dF_vs_stage1": round(float((F - F1).mean()), 5), "paired_se": round(th.paired_se(F, F1), 5),
                     "by_country": {cc: {"f05": round(float(F[country == cc].mean()), 5),
                                         "se": round(th.se(F[country == cc]), 5),
                                         "n_s1": int((country == cc).sum())} for cc in sorted(set(country))},
                     "loss": {k: round(v_, 5) for k, v_ in th.loss_decomposition(
                         ng, gtc, s1i, keep, e["y"], n).items()}}
        Fs[name] = F
        log.info(f"{name}{' (report only)' if name == 'holdout' else ''}: {rep[name]}")
    log.info(f"chosen {rep['choice']}")
    return rep, coef, Fs


def main(split, limit_s1, force):
    cfg = load_config()
    if split != "train":
        log.info("s4x_xenc trains on train only (S5 scores test)")
        return
    xc, tcfg, seed = cfg["optional"]["cross_encoder"], cfg["threshold"], cfg["seed"]
    if not xc["enabled"]:
        log.info("optional.cross_encoder disabled")
        return
    inj = th.resolve_injective(cfg)
    inputs = {"s4_train": input_hash("train", "s4_train")}
    section = {"cross_encoder": xc, "threshold": tcfg, "injective": inj, "limit_s1": limit_s1,
               "code": code_hash("src/s4x_xenc.py", "src/threshold.py")}
    with stage("s4x_xenc", split, cfg, section, limit_s1=limit_s1, force=force, input_stages=inputs) as st:
        if st.skip:
            return
        tag = limit_s1 or "full"
        ev_dir = Path(f"work/eval/s4_{tag}")
        art = ART if limit_s1 is None else Path(f"work/eval/s4x_{tag}/artifacts")
        s4_thr = json.loads((ART if limit_s1 is None else ev_dir / "artifacts").joinpath("thresholds.json").read_text())
        assert s4_thr["s4_hash"] == inputs["s4_train"], "S4 artifacts/preds are not from the S4 run in _DONE.json"
        if xc.get("init_dir") and not (st.work_dir / "xenc" / "config.json").exists():
            adopt_trained(xc, st.work_dir / "xenc")
        tok, model = train_xenc(ev_dir, xc, seed, st.work_dir / "xenc")
        lo, hi = xc["band"]
        scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
        n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
        E = {}
        for name in ("val", "holdout"):
            sc, d = ev_arrays(name, pl.read_parquet(ev_dir / f"pred_{name}.parquet"), scope, n_gt)
            inb = ((d["p"] >= lo) & (d["p"] <= hi)).to_numpy()
            f = st.work_dir / f"scores_{name}.parquet"
            if f.exists():                             # resume
                s = d.select("s1_row", "pool_row").join(pl.read_parquet(f).select("s1_row", "pool_row", "xenc"),
                                                        on=["s1_row", "pool_row"], how="left",
                                                        maintain_order="left")["xenc"].fill_null(0.0).to_numpy()
            else:
                s = np.zeros(d.height, np.float32)
                t = time.time()
                s[inb] = score_pairs("train", d.filter(pl.Series(inb)), tok, model, xc["max_len"])
                log.info(f"{name}: scored {int(inb.sum()):,} band pairs in {time.time() - t:.1f}s")
                atomic_write_parquet(d.select("s1_row", "pool_row", "p").with_columns(pl.Series("xenc", s)), f)
            assert np.isfinite(s).all(), f"non-finite cross-encoder scores in {name}"
            E[name] = {"sc": sc, "d": d, "inb": inb, "s": s, "y": d["label"].to_numpy().astype(np.float64),
                       "p": d["p"].to_numpy()}
        rep, coef, _ = evaluate(E, s4_thr, tcfg, inj)
        rep = {"band": [lo, hi], "train_band": xc["train_band"], **rep}
        art.mkdir(parents=True, exist_ok=True)
        import shutil
        tmp = art / "xenc.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(st.work_dir / "xenc", tmp)
        shutil.rmtree(art / "xenc", ignore_errors=True)
        os.replace(tmp, art / "xenc")
        thr = s4_thr
        final = {**thr, "xenc": {"model_dir": str(art / "xenc"), "band": [lo, hi], "max_len": xc["max_len"],
                                 "combiner": coef},
                 "t": rep["choice"]["t"], "r": rep["choice"]["r"], "injective": inj,
                 "plateau": rep["choice"]["plateau"], "val": {"f05": rep["val"]["f05"], "se": rep["val"]["se"]},
                 "holdout": {"f05": rep["holdout"]["f05"], "se": rep["holdout"]["se"]}, "s4x_hash": st.hash}
        tmp = art / "thresholds_final.json.tmp"
        tmp.write_text(json.dumps(final, indent=1))
        os.replace(tmp, art / "thresholds_final.json")
        Path(f"work/eval/s4x_{tag}").mkdir(parents=True, exist_ok=True)
        Path(f"work/eval/s4x_{tag}/report.json").write_text(json.dumps(rep, indent=1, default=str))
        st.rows = int(E["val"]["inb"].sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit-s1", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.split, a.limit_s1, a.force)
