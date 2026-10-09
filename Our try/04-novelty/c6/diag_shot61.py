#!/usr/bin/env python3
"""Post-hoc (NOT part of any verdict): how much of the test R2_psi of each model comes from test shot #61.

Pooled R2 as in the scorer (one scalar mean over all pixels and frames of the set). Output: diag_shot61.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "fusion equilibrium challenge" / "downloaded_huggingface"
RUNS = {"PCA+Ridge (SCORE.md)": "pca_ridge", "Linear Regression": "linear_regression", "Random Forest": "random_forest",
        "MLP (sklearn)": "mlp_sklearn", "UNet_Lite (s0)": "unet_lite_s0"}


def r2(t, p):
    t, p = np.concatenate(t), np.concatenate(p)
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def main():
    keys = sorted(f"cal_{int(k):02d}" for k in json.loads((DATA / "c6_cal_cache/index.json").read_text())["shots"])
    tc = [np.load(DATA / f"c6_cal_cache/{k}.npz")["psi"].astype(float) for k in keys]
    tt = [np.load(DATA / f"c2_cache/shot_{i:03d}.npz")["psi"].astype(float) for i in range(60, 68)]
    out = {}
    for name, run in RUNS.items():
        pc = np.load(DATA / f"c6_preds_cal/{run}.npz")
        pt = np.load(HERE / "preds" / "test" / f"{run}.npz")
        P = [pt[f"shot_{i:04d}_psirz"] for i in range(8)]
        ss = [float(((tt[i] - P[i]) ** 2).sum()) for i in range(8)]
        keep = [i for i in range(8) if i != 1]
        out[name] = {"r2_cal24": r2(tc, [pc[k] for k in keys]), "r2_test": r2(tt, P),
                     "r2_test_without_61": r2([tt[i] for i in keep], [P[i] for i in keep]),
                     "r2_shot61": r2([tt[1]], [P[1]]), "share_ssres_61": ss[1] / sum(ss)}
        print(f"  {name:22s}", {k: round(v, 3) for k, v in out[name].items()})
    (HERE / "diag_shot61.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
