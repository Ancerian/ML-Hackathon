#!/usr/bin/env python3
"""D1 — conditioning of the map psi -> derived scalars.

Thesis
------
The seven scored "consistency" scalars are differential/geometric functionals of psi:

    R_axis, Z_axis   position of a critical point (grad psi = 0)
    kappa, tri_*,
    volume           shape of a level set that passes through a SADDLE (the X-point)
    li               involves |grad psi|^2

At the X-point grad psi = 0, so the implicit-function theorem degenerates: the level set's
position is O(eps / delta) for a psi-perturbation of size eps, where delta is the local Hessian
conditioning. The map psi |-> scalars is therefore UNBOUNDED in L2 and only Lipschitz in
stronger (H1/H2) norms -- which nobody trains on. The scorer, meanwhile, measures psi in L2.

This script measures the amplification empirically, with no training whatsoever:
inject perturbations of controlled L2 norm into GROUND-TRUTH psi and watch the scalars move.

Perturbation families
---------------------
  white     spatially white noise (flat spectrum)
  band-k    noise confined to an annulus of radial wavenumbers (spectral sensitivity curve)
  pca-tail  a random direction inside the discarded tail of the equilibrium PCA basis
            (i.e. exactly the error a truncated linear decoder makes)

Normalisation (chosen to match the scorer, so results are directly interpretable)
--------------------------------------------------------------------------------
  psi error   e = ||d_psi|| / ||psi - psi_bar||   pooled over all frames  =>  R2_psi = 1 - e^2
  scalar err  |f(psi+d) - f(psi)| / sigma_f       sigma_f = population std of that scalar,
                                                  the same pooling the scorer's R2 uses.

Usage
-----
  cd "fusion equilibrium challenge/starter"
  .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/conditioning.py" --frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- wiring
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]                      # .../ТОКАМАК ПРОДЖЕКТ
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))

from common import CONS_SCALARS                                  # noqa: E402
from derive import derive_frame                                  # noqa: E402
from lcfs import extract_lcfs                                     # noqa: E402

MACHINE = "DIII-D"
N_POINTS = 512
SHOTS = ["d3d_shot_203702.parquet", "d3d_shot_203703.parquet", "d3d_shot_203704.parquet"]

# wavenumber annuli on the 65x65 grid (grid Nyquist is 32)
BANDS = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 13), (13, 18), (18, 24), (24, 32)]
EPS_GRID = [0.003, 0.01, 0.03, 0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- data
def load_psi(n_frames: int, seed: int = 0) -> np.ndarray:
    """Evenly subsample n_frames ground-truth flux maps across the three local demo shots."""
    import pyarrow.parquet as pq

    frames = []
    for name in SHOTS:
        path = STARTER / "parquet_data" / name
        if not path.exists():
            continue
        tbl = pq.ParquetFile(str(path)).read(columns=["efit_psirz"])
        arr = np.asarray(tbl.column("efit_psirz")[0].as_py(), dtype=np.float64)
        frames.append(arr)
    if not frames:
        raise SystemExit("no local parquet demo shots found")
    allf = np.concatenate(frames, axis=0)
    good = np.isfinite(allf).all(axis=(1, 2))
    allf = allf[good]
    idx = np.linspace(0, len(allf) - 1, min(n_frames, len(allf))).astype(int)
    return allf[idx]


def load_grid():
    z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    mask_coarse = z["mask_coarse"].astype(bool)
    return z["grid_R"], z["grid_Z"], mask_coarse, mask_coarse.astype(np.float64)


# --------------------------------------------------------------------------- perturbations
def _normalise(d: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def pert_white(rng, shape):
    return _normalise(rng.standard_normal(shape))


def pert_band(rng, shape, klo, khi):
    """Noise whose 2-D power spectrum is confined to the annulus klo <= |k| < khi."""
    nz, nr = shape
    kz = np.fft.fftfreq(nz) * nz
    kr = np.fft.fftfreq(nr) * nr
    KK = np.hypot(kz[:, None], kr[None, :])
    ring = (KK >= klo) & (KK < khi)
    if not ring.any():
        return pert_white(rng, shape)
    spec = np.fft.fft2(rng.standard_normal(shape)) * ring
    return _normalise(np.real(np.fft.ifft2(spec)))


def build_pca(psi: np.ndarray, n_keep: int):
    """PCA over the frame collection; returns (mean, tail basis) = the discarded directions."""
    X = psi.reshape(len(psi), -1)
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[n_keep:]


def pert_pca_tail(rng, shape, tail):
    if len(tail) == 0:
        return pert_white(rng, shape)
    c = rng.standard_normal(len(tail))
    return _normalise((c @ tail).reshape(shape))


# --------------------------------------------------------------------------- scalars
def scalars_of(psi2d, R, Z, mask_coarse, mask_f):
    c = extract_lcfs(psi2d, R, Z, MACHINE, mask_coarse, mask_f, n_points=N_POINTS)
    v = derive_frame(psi2d, R, Z, MACHINE, mask_coarse, mask_f, contour=c)
    return np.array([v[k] for k in CONS_SCALARS], dtype=float)


# --------------------------------------------------------------------------- experiment
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pca-keep", type=int, default=20)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    R, Z, mask_coarse, mask_f = load_grid()
    psi = load_psi(args.frames)
    print(f"frames: {len(psi)}  grid: {psi.shape[1:]}")

    # pooled psi scale -- exactly the denominator of the scorer's R2_psi
    psi_bar = psi.mean()
    psi_scale = np.sqrt(((psi - psi_bar) ** 2).sum())
    per_frame_scale = psi_scale / np.sqrt(len(psi))     # so a per-frame eps matches the pooled one
    print(f"pooled ||psi - psi_bar|| = {psi_scale:.4f}   per-frame = {per_frame_scale:.4f}")

    print("computing ground-truth scalars ...")
    base = np.array([scalars_of(p, R, Z, mask_coarse, mask_f) for p in psi])
    sigma = np.nanstd(base, axis=0)
    sigma[~np.isfinite(sigma) | (sigma == 0)] = np.nan
    ok = np.isfinite(base).all(axis=1)
    print(f"  usable frames: {ok.sum()}/{len(psi)}")
    for n, s in zip(CONS_SCALARS, sigma):
        print(f"    sigma[{n:8s}] = {s:.5g}")

    mu, tail = build_pca(psi, args.pca_keep)
    shape = psi.shape[1:]
    rows = []

    def run(family, maker, eps_list):
        for eps in eps_list:
            errs = []
            for fi in np.where(ok)[0]:
                for sd in range(args.seeds):
                    rng = np.random.default_rng(10_000 * fi + 97 * sd + int(eps * 1e6))
                    d = maker(rng, shape) * (eps * per_frame_scale)
                    s = scalars_of(psi[fi] + d, R, Z, mask_coarse, mask_f)
                    errs.append(np.abs(s - base[fi]) / sigma)
            E = np.array(errs)
            med = np.nanmedian(E, axis=0)
            rows.append({"family": family, "eps": eps,
                         "R2_psi": 1.0 - eps ** 2,
                         "n": int(np.isfinite(E).all(axis=1).sum()),
                         **{f"med_{n}": float(m) for n, m in zip(CONS_SCALARS, med)}})
            print(f"  {family:10s} eps={eps:<6.3f} R2psi={1-eps**2:.4f}  "
                  + "  ".join(f"{n}={m:.3f}" for n, m in zip(CONS_SCALARS, med)))

    print("\n--- amplitude sweep ---")
    run("white", pert_white, EPS_GRID)
    run("pca-tail", lambda r, s: pert_pca_tail(r, s, tail), EPS_GRID)

    print("\n--- spectral sweep (eps = 0.10, i.e. R2_psi = 0.99) ---")
    for (klo, khi) in BANDS:
        run(f"band-{klo}-{khi}", lambda r, s, a=klo, b=khi: pert_band(r, s, a, b), [0.10])

    # ---- log-log amplification exponents on the white family
    print("\n--- amplification exponents  (scalar_err ~ eps^alpha; alpha < 1 = AMPLIFICATION) ---")
    w = [r for r in rows if r["family"] == "white"]
    e = np.array([r["eps"] for r in w])
    expo = {}
    for n in CONS_SCALARS:
        y = np.array([r[f"med_{n}"] for r in w])
        m = np.isfinite(y) & (y > 0)
        if m.sum() >= 3:
            a, b = np.polyfit(np.log(e[m]), np.log(y[m]), 1)
            expo[n] = {"alpha": float(a), "log_intercept": float(b)}
            print(f"    {n:8s} alpha = {a:6.3f}   (err at eps=0.01: {np.exp(b) * 0.01 ** a:.4f} sigma)")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"scalars": CONS_SCALARS, "sigma": sigma.tolist(), "rows": rows,
         "exponents": expo, "frames": int(ok.sum()), "seeds": args.seeds,
         "pca_keep": args.pca_keep}, indent=2))
    print(f"\nwrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
