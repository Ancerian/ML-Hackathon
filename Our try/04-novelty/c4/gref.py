#!/usr/bin/env python3
"""C4 step 1 — calibrate g_ref (THEORY.md §1): median GS-inconsistency of truth + 1 % white noise
(sigma = 1 % of the frame's std psi) on the 28 DIII-D shots of the C1 test, 10 frames each.
Also reports g(truth). Output: gref.json.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOV = HERE.parent
sys.path.insert(0, str(NOV))
sys.path.insert(0, str(NOV.parent / "03-deep-dives" / "D1-psi-to-scalars" / "c1"))
import sweep_core as sc                                                     # noqa: E402
from gs_residual_probe import gs_inconsistency                              # noqa: E402

_W = {}


def _init(psi, grid):
    _W.update(psi=psi, grid=grid)


def _job(i):
    R, Z, mc, mf = _W["grid"]
    p = _W["psi"][i]
    g0 = gs_inconsistency(p, R, Z, mc, mf)[0]
    rng = np.random.default_rng(1000 + i)
    g1 = gs_inconsistency(p + rng.standard_normal(p.shape) * 0.01 * np.std(p), R, Z, mc, mf)[0]
    return g0, g1


def main():
    hf = sc.PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset" / "data" / "diii_d_train"
    psi, ids = sc.load_shots(sorted(hf.glob("d3d_shot_*.parquet")), 10)
    with mp.get_context("fork").Pool(None, initializer=_init, initargs=(psi, sc.load_grid("DIII-D"))) as pool:
        r = np.array(pool.map(_job, range(len(psi))))
    ok = np.isfinite(r).all(1)
    out = {"n_frames": int(ok.sum()), "n_shots": int(len(np.unique(ids))),
           "g_truth_median": float(np.median(r[ok, 0])), "g_truth_p90": float(np.percentile(r[ok, 0], 90)),
           "g_noise1pct_median": float(np.median(r[ok, 1])),
           "g_noise1pct_p10": float(np.percentile(r[ok, 1], 10)),
           "g_ref": float(np.median(r[ok, 1]))}
    (HERE / "gref.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
