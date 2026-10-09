#!/usr/bin/env python3
"""C6 post-hoc diagnostic (NOT part of the verdict; written after analyse_c6.py was run).

Question: is the test-set undercoverage of form M a systematic shift, or the draw of 8 test shots?
Pool = 12 shots of half B + 8 test shots (scores e_t share the same s(x) from half A, so the pool is
exchangeable under the conformal assumption). (1) 2000 random splits of the pool into 12 calibration +
8 evaluation shots -> distribution of joint coverage; where the real split falls in it. (2) Leave-one-
shot-out per-shot coverage at a = 0.90 over all 20 shots. Output: diag_pool.json.
"""
import json
from pathlib import Path

import numpy as np

from analyse_c6 import ACCURATE, LEVELS, MODELS, NULL

HERE = Path(__file__).resolve().parent


def pool(run):
    z = np.load(HERE / "bands" / f"{run}.npz")
    return [z["eB_full"][z["sid_B"] == j] for j in range(12)] + [z["eT_full"][z["sid_T"] == j] for j in range(8)]


def cover(shots, cal, ev, a):
    e = np.sort(np.concatenate([shots[i] for i in cal])); n = len(e)
    k = int(np.ceil((n + 1) * a))
    Q = e[k - 1] if k <= n else np.inf
    return float(np.mean(np.concatenate([shots[i] for i in ev]) <= Q))


def main():
    rng = np.random.default_rng(1)
    splits = [rng.permutation(20) for _ in range(2000)]
    real = (list(range(12)), list(range(12, 20)))
    out = {}
    for m in list(MODELS) + ["N0"]:
        runs = [pool(r) for r in (MODELS[m] if m != "N0" else [NULL])]
        d = {}
        for a in LEVELS:
            cov = np.array([np.mean([cover(p, s[:12], s[12:], a) for p in runs]) for s in splits])
            r = float(np.mean([cover(p, *real, a) for p in runs]))
            d[f"{a:.2f}"] = {"mean": float(cov.mean()), "p05": float(np.percentile(cov, 5)),
                             "p95": float(np.percentile(cov, 95)), "real_split": r,
                             "real_split_percentile": float(np.mean(cov <= r))}
        loso = [float(np.mean([cover(p, [j for j in range(20) if j != i], [i], 0.90) for p in runs])) for i in range(20)]
        d["loso_a090"] = loso
        d["loso_frac_below_0.5"] = float(np.mean(np.array(loso) < 0.5))
        out[m] = d
    (HERE / "diag_pool.json").write_text(json.dumps(out, indent=1))
    print(f"  {'model':22s} " + "  ".join(f"a={a:.2f}: mean [p05,p95] real(pct)" for a in (0.5, 0.9)))
    for m, d in out.items():
        row = []
        for a in ("0.50", "0.90"):
            v = d[a]
            row.append(f"{v['mean']:.3f} [{v['p05']:.2f},{v['p95']:.2f}] {v['real_split']:.3f}({v['real_split_percentile']:.2f})")
        tag = " *" if m in ACCURATE else ""
        print(f"  {m:22s} {'   '.join(row)}   LOSO<0.5: {d['loso_frac_below_0.5']:.2f}{tag}")
    print("\n  LOSO per-shot coverage at 0.90 (shots 0-11 = half B, 12-19 = test 60-67):")
    for m in ACCURATE + ["N0"]:
        print(f"  {m:16s}", np.round(out[m]["loso_a090"], 2).tolist())


if __name__ == "__main__":
    main()
