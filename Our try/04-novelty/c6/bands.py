#!/usr/bin/env python3
"""C6 step 3 — build the bands of THEORY.md §3 for every run and reduce them to per-frame numbers.

Per run (model x seed, plus the null map N0), for region in {full 65x65, mask_coarse}:
  M  s(x) = std of residuals over calibration half A; e_t = max_x |r_t(x)| / s(x) on half B and on test
  P  q_a(x) = conformal quantile of |r_t(x)| over all 24 calibration shots; on test: joint and pixel coverage
Everything a bootstrap needs is per frame (with its shot id), so bands/<run>.npz stays small.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
DATA = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface"
CAL, C2CACHE, PRED_CAL, PRED_TEST = DATA / "c6_cal_cache", DATA / "c2_cache", DATA / "c6_preds_cal", HERE / "preds" / "test"
OUT = HERE / "bands"
LEVELS = np.array([0.50, 0.80, 0.90, 0.95])
TEST = range(60, 68)


def conformal_k(n, a):
    """1-based rank of the split-conformal quantile; > n means an infinite band."""
    return int(np.ceil((n + 1) * a))


def split_halves(keys):
    perm = np.random.default_rng(0).permutation(len(keys))
    return [keys[i] for i in perm[:12]], [keys[i] for i in perm[12:]]


def main(runs):
    OUT.mkdir(exist_ok=True)
    region = {"full": np.ones((65, 65), bool),
              "mask": np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")["mask_coarse"].astype(bool)}
    keys = sorted(f"cal_{int(k):02d}" for k in json.loads((CAL / "index.json").read_text())["shots"])
    A, B = split_halves(keys)
    truth_cal = {k: np.load(CAL / f"{k}.npz")["psi"].astype(np.float64) for k in keys}
    truth_test = [np.load(C2CACHE / f"shot_{i:03d}.npz")["psi"].astype(np.float64) for i in TEST]
    (OUT / "split.json").write_text(json.dumps({"A": A, "B": B, "levels": LEVELS.tolist()}, indent=1))
    for run in runs:
        pc = np.load(PRED_CAL / f"{run}.npz")
        pt = np.load(PRED_TEST / f"{run}.npz")
        res = lambda ks: np.concatenate([truth_cal[k] - pc[k].astype(np.float64) for k in ks])   # noqa: E731
        sid = lambda ks: np.concatenate([np.full(len(truth_cal[k]), j) for j, k in enumerate(ks)])  # noqa: E731
        rA, rB = res(A), res(B)
        rT = np.concatenate([g - pt[f"shot_{i:04d}_psirz"].astype(np.float64) for i, g in enumerate(truth_test)])
        out = {"sid_B": sid(B), "sid_T": np.concatenate([np.full(len(g), i) for i, g in enumerate(truth_test)])}
        s = rA.std(axis=0)
        s = np.maximum(s, 1e-6 * np.median(s))
        out["s"] = s
        for rg, m in region.items():
            out[f"eB_{rg}"] = (np.abs(rB)[:, m] / s[m]).max(axis=1)
            out[f"eT_{rg}"] = (np.abs(rT)[:, m] / s[m]).max(axis=1)
            out[f"smean_{rg}"] = float(s[m].mean())
        del rA, rB
        # P: per-pixel conformal quantile over all 24 calibration shots
        absr = np.abs(res(keys)).astype(np.float32)
        n = len(absr)
        flat = absr.reshape(n, -1)
        absT = np.abs(rT).reshape(len(rT), -1)
        for j, a in enumerate(LEVELS):
            k = conformal_k(n, a)
            q = np.partition(flat, k - 1, axis=0)[k - 1] if k <= n else np.full(flat.shape[1], np.inf)
            inside = absT <= q[None, :]
            for rg, m in region.items():
                mm = m.ravel()
                out[f"P_joint_{rg}_{j}"] = inside[:, mm].all(axis=1)
                out[f"P_pix_{rg}_{j}"] = inside[:, mm].mean(axis=1)
                out[f"P_W_{rg}_{j}"] = float(q[mm].mean())
        out["n_cal_P"] = n
        np.savez_compressed(OUT / f"{run}.npz", **out)
        print(f"  {run}: n_A={len(sid(A))} n_B={len(out['sid_B'])} n_T={len(out['sid_T'])} "
              f"median e_B={np.median(out['eB_full']):.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(p.stem for p in PRED_TEST.glob("*.npz")))
