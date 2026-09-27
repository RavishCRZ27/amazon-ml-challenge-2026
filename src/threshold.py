#!/usr/bin/env python3
"""Exact macro-F0.5, threshold rule, injective pass, loss decomposition (shared by S4 and S5).
Rule: specs/train.md.

Scoring (README): F0.5 per S1, averaged over ALL S1s in scope. A singleton scores 1.0 iff its
prediction is empty; recall denominators come from the FULL GT, so blocking misses count.
Per S1: F = 1.25*tp / (0.25*n_gt + n_pred)  (equals 1.25PR/(0.25P+R) whenever tp > 0).
"""

import argparse
import json

import numpy as np


def resolve_injective(cfg) -> bool:
    """`auto` = on iff the train GT is injective on the S2/S3 side (at most a trace of dups)."""
    mode = cfg["injective"]
    if mode in ("on", True):
        return True
    if mode in ("off", False):
        return False
    st = json.load(open("work/train/s1_gt/gt_stats.json"))
    return st["pool_ids_under_multiple_s1"] < 0.001 * st["positive_pairs"]


class Cands:
    """One scored candidate set. s1_idx is dense in [0, n_s1) over the S1 scope."""

    def __init__(self, s1_idx, pool_row, p, n_s1):
        self.s1_idx = np.asarray(s1_idx, np.int64)
        self.pool_row = np.asarray(pool_row, np.int64)
        self.p = np.asarray(p, np.float32)
        self.n_s1 = n_s1
        self.maxp = np.zeros(n_s1, np.float32)
        np.maximum.at(self.maxp, self.s1_idx, self.p)
        # injective order: per pool row, highest p first; ties broken by s1 (deterministic)
        self.order = np.lexsort((self.s1_idx, -self.p, self.pool_row))
        self.pool_sorted = self.pool_row[self.order]


def select(c: Cands, t: float, r: float, injective: bool) -> np.ndarray:
    """Keep p >= t and p >= r*max_p(S1); then each pool row keeps only its best kept S1."""
    keep = (c.p >= t) & (c.p >= r * c.maxp[c.s1_idx])
    if injective:
        idx = np.flatnonzero(keep[c.order])
        pr = c.pool_sorted[idx]
        first = np.ones(idx.size, bool)
        first[1:] = pr[1:] != pr[:-1]
        keep = np.zeros_like(keep)
        keep[c.order[idx[first]]] = True
    return keep


def per_s1_f05(n_gt, s1_idx, keep, label, n_s1) -> np.ndarray:
    n_pred = np.bincount(s1_idx[keep], minlength=n_s1).astype(np.float64)
    tp = np.bincount(s1_idx[keep], weights=label[keep], minlength=n_s1)
    return _f(np.asarray(n_gt, np.float64), n_pred, tp)


def _f(n_gt, n_pred, tp):
    den = 0.25 * n_gt + n_pred
    f = np.divide(1.25 * tp, den, out=np.zeros_like(den), where=den > 0)
    return np.where(n_gt == 0, (n_pred == 0).astype(np.float64), f)


def se(f) -> float:
    f = np.asarray(f, np.float64)
    return float(f.std(ddof=1) / np.sqrt(f.size)) if f.size > 1 else float("nan")


def paired_se(f1, f2) -> float:
    return se(np.asarray(f1, np.float64) - np.asarray(f2, np.float64))


def t_values(grid) -> np.ndarray:
    start, stop, step = grid
    return np.round(np.arange(start, stop + step / 2, step), 6)


def sweep(c: Cands, label, n_gt, t_grid, r_list, inj_list):
    rows = []
    for inj in inj_list:
        for r in r_list:
            for t in t_values(t_grid):
                keep = select(c, t, r, inj)
                rows.append({"t": float(t), "r": float(r), "injective": bool(inj),
                             "f05": float(per_s1_f05(n_gt, c.s1_idx, keep, label, c.n_s1).mean())})
    return rows


def choose(rows, injective: bool, tol: float) -> dict:
    """r = argmax of each r's best F; t = centre of the contiguous plateau (F >= Fmax - tol)
    around that r's argmax, which is steadier than a spiky argmax."""
    rs = sorted({x["r"] for x in rows if x["injective"] == injective})
    best_r = max(rs, key=lambda r: max(x["f05"] for x in rows if x["injective"] == injective and x["r"] == r))
    curve = sorted((x for x in rows if x["injective"] == injective and x["r"] == best_r), key=lambda x: x["t"])
    f = np.array([x["f05"] for x in curve])
    i = int(f.argmax())
    lo = hi = i
    while lo > 0 and f[lo - 1] >= f[i] - tol:
        lo -= 1
    while hi < len(f) - 1 and f[hi + 1] >= f[i] - tol:
        hi += 1
    mid = (lo + hi) // 2
    return {"t": curve[mid]["t"], "r": best_r, "injective": injective, "f05_at_t": float(f[mid]),
            "argmax_t": curve[i]["t"], "f05_max": float(f[i]),
            "plateau": [curve[lo]["t"], curve[hi]["t"]]}


def loss_decomposition(n_gt, gtc, s1_idx, keep, label, n_s1) -> dict:
    """1 - macroF split per S1 into: singleton emissions, blocking misses (1 - oracle F),
    model misses among candidates (oracle F - F without FPs), false positives (F w/o FP - F).
    gtc = GT matches present among the S1's candidates. The four parts sum to 1 - macroF."""
    P, n_pred, single = loss_parts(n_gt, gtc, s1_idx, keep, label, n_s1)
    return summarize_parts(P, n_pred, single, np.ones(n_s1, bool))


def loss_parts(n_gt, gtc, s1_idx, keep, label, n_s1):
    """Per-S1 loss parts (arrays); they sum to 1 - F for every S1."""
    n_gt = np.asarray(n_gt, np.float64)
    gtc = np.asarray(gtc, np.float64)
    n_pred = np.bincount(s1_idx[keep], minlength=n_s1).astype(np.float64)
    tp = np.bincount(s1_idx[keep], weights=label[keep], minlength=n_s1)
    f = _f(n_gt, n_pred, tp)
    f_or = _f(n_gt, gtc, gtc)
    f_nofp = _f(n_gt, tp, tp)
    single = n_gt == 0
    m = ~single
    P = {"singleton_emit": np.where(single, 1.0 - f, 0), "blocking_miss": np.where(m, 1.0 - f_or, 0),
         "model_miss": np.where(m, f_or - f_nofp, 0), "false_pos": np.where(m, f_nofp - f, 0),
         "total": 1.0 - f}
    return P, n_pred, single


def summarize_parts(P, n_pred, single, mask) -> dict:
    out = {k: float(v[mask].mean()) for k, v in P.items()}
    s = single & mask
    out["singleton_emit_rate"] = float((n_pred[s] > 0).mean()) if s.any() else 0.0
    return out


def selftest():
    from sklearn.metrics import fbeta_score
    # S1 0: GT {a,b}, cands {a,b,c}; S1 1: singleton; S1 2: GT {d} never retrieved; S1 3: singleton
    s1 = np.array([0, 0, 0, 1, 3])
    pool = np.array([10, 11, 12, 13, 12])
    lab = np.array([1, 1, 0, 0, 0], np.float64)
    n_gt = np.array([2, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.95], np.float32)
    c = Cands(s1, pool, p, 4)
    f = per_s1_f05(n_gt, s1, select(c, 0.5, 0.0, False), lab, 4)
    # S1 0 predicts {a,b,c}: P=2/3, R=1 -> 0.714; S1 1 empty -> 1; S1 2 empty w/ GT -> 0; S1 3 emits -> 0
    exp0 = fbeta_score([1, 1, 0], [1, 1, 1], beta=0.5)
    assert abs(f[0] - exp0) < 1e-9 and abs(f[0] - 0.7142857) < 1e-6, f
    assert f[1] == 1.0 and f[2] == 0.0 and f[3] == 0.0, f
    # injective: pool 12 is claimed by S1 0 (0.7) and S1 3 (0.95): only S1 3 keeps it
    k = select(c, 0.5, 0.0, True)
    assert k.tolist() == [True, True, False, False, True], k
    assert abs(per_s1_f05(n_gt, s1, k, lab, 4)[0] - 1.0) < 1e-12
    # relative rule: r=0.95 of max 0.9 -> keeps only p >= 0.855 in S1 0
    assert select(c, 0.5, 0.95, False).tolist() == [True, False, False, False, True]
    # decomposition sums to 1 - macroF
    gtc = np.array([2, 0, 0, 0])
    d = loss_decomposition(n_gt, gtc, s1, select(c, 0.5, 0.0, False), lab, 4)
    parts = d["singleton_emit"] + d["blocking_miss"] + d["model_miss"] + d["false_pos"]
    assert abs(parts - d["total"]) < 1e-12 and d["blocking_miss"] == 0.25, d
    # plateau choice picks the centre of the flat top
    rows = [{"t": t, "r": 0.0, "injective": True, "f05": v}
            for t, v in [(0.1, 0.5), (0.2, 0.8), (0.3, 0.8), (0.4, 0.8), (0.5, 0.6)]]
    ch = choose(rows, True, 0.001)
    assert ch["t"] == 0.3 and ch["plateau"] == [0.2, 0.4], ch
    print("threshold self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
