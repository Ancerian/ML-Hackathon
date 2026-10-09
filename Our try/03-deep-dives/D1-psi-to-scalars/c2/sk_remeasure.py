#!/usr/bin/env python3
"""C2 step 2 — re-measure the spectral sensitivity S(k) on 28 DIII-D shots (THEORY.md §2-3).

D1 measured S(k) at e = 0.1 on the 3 demo shots only. Here: the 3 demo + 25 HF shots of the C1
test (10 frames each), band-limited perturbations in D1's eight annuli, at e = 0.1, 0.3 and 1.0,
median |d f| / sigma_f per scalar (sigma over this sample) — D1's own estimator and functions.

Output: sk_remeasure.json  {eps: {band: [7 medians]}}, plus D1's original e = 0.1 curve.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c1"))
import sweep_core as sc                                                     # noqa: E402

EPS = [0.1, 0.3, 1.0]
BANDS = sc.d1.BANDS
_W = {}


def _init(psi, grid, scale):
    _W.update(psi=psi, grid=grid, scale=scale)


def _job(fi):
    psi, grid = _W["psi"], _W["grid"]
    base = sc.scalars(psi[fi], grid, "DIII-D")
    out = np.full((len(EPS), len(BANDS), 7), np.nan)
    if not np.isfinite(base).all():
        return fi, base, out
    for a, eps in enumerate(EPS):
        for b, (klo, khi) in enumerate(BANDS):
            rng = np.random.default_rng(10_000 * fi + 131 * b + int(eps * 1e6))
            d = sc.d1.pert_band(rng, psi.shape[1:], klo, khi) * (eps * _W["scale"])
            out[a, b] = np.abs(sc.scalars(psi[fi] + d, grid, "DIII-D") - base)
    return fi, base, out


def main():
    hf = sc.PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset" / "data" / "diii_d_train"
    psi, ids = sc.load_shots(sorted(hf.glob("d3d_shot_*.parquet")), 10)
    grid = sc.load_grid("DIII-D")
    scale = np.sqrt(((psi - psi.mean()) ** 2).sum()) / np.sqrt(len(psi))
    with mp.get_context("fork").Pool(None, initializer=_init, initargs=(psi, grid, scale)) as pool:
        res = pool.map(_job, range(len(psi)), chunksize=1)
    base = np.array([r[1] for r in sorted(res)])
    err = np.array([r[2] for r in sorted(res, key=lambda t: t[0])])        # [N, E, B, 7]
    ok = np.isfinite(base).all(1)
    sigma = np.nanstd(base[ok], axis=0)
    med = np.nanmedian(err[ok] / sigma, axis=0)                             # [E, B, 7]
    d1 = json.loads((sc.D1 / "results.json").read_text())
    d1_curve = {r["family"]: [r[f"med_{n}"] for n in d1["scalars"]]
                for r in d1["rows"] if r["family"].startswith("band-")}
    out = {"scalars": list(sc.CONS_SCALARS), "bands": BANDS, "eps": EPS,
           "n_frames": int(ok.sum()), "n_shots": int(len(np.unique(ids))),
           "median_sigma": {str(e): med[a].tolist() for a, e in enumerate(EPS)},
           "d1_demo3_e0.1": d1_curve}
    (HERE / "sk_remeasure.json").write_text(json.dumps(out, indent=1))
    names = list(sc.CONS_SCALARS)
    for a, e in enumerate(EPS):
        print(f"\n e = {e}: median |df|/sigma per band (rows) and scalar (cols)")
        print("   band     " + " ".join(f"{n:>8s}" for n in names))
        for b, (lo, hi) in enumerate(BANDS):
            print(f"   {lo:2d}-{hi:<2d}    " + " ".join(f"{v:8.3f}" for v in med[a, b]))
    print("\n D1 (3 demo shots, e = 0.1):")
    for k, v in d1_curve.items():
        print(f"   {k:10s} " + " ".join(f"{x:8.3f}" for x in v))


if __name__ == "__main__":
    main()
