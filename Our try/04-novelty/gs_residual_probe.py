#!/usr/bin/env python3
"""Can the Grad-Shafranov residual be scored from a SUBMITTED psi ALONE?

The obstacle: GS is  Delta* psi = -mu0 R^2 p'(psi) - F F'(psi),  and a submission ships only
psi -- not p' and not FF'. So a naive residual is not computable.

The way out reuses EFIT's own structure (Lao 1985): the GS source is LINEAR in the profile
coefficients once psi is fixed. So define

    GS-inconsistency(psi) = min over admissible (p', FF')  || Delta* psi + mu0 R^2 p'(psi) + FF'(psi) ||
                            --------------------------------------------------------------------------
                                                       || Delta* psi ||

i.e. "is there ANY pair of flux functions that makes this psi an equilibrium?" For fixed psi that
is an ordinary linear least-squares problem, numpy-only, no extra submission channel.

Why it is not gameable by mere smoothness: the condition is that Delta* psi, restricted to a flux
surface, must be an AFFINE function of R^2 whose coefficients depend on psi only. That is the
flux-surface constraint itself -- a strong structural requirement a smooth-but-wrong psi fails.

This probe checks the term actually discriminates, on real ground truth.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[1] / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
from lcfs import extract_lcfs                      # noqa: E402
from derive import magnetic_axis                   # noqa: E402
from contour import grid_to_physical               # noqa: E402

MACHINE = "DIII-D"


def delta_star(psi, R, Z):
    """Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 , 2nd-order centred."""
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    d_dR = np.gradient(psi, dR, axis=1)
    d2_dR2 = np.gradient(d_dR, dR, axis=1)
    d2_dZ2 = np.gradient(np.gradient(psi, dZ, axis=0), dZ, axis=0)
    return d2_dR2 - d_dR / R[None, :] + d2_dZ2


def plasma_mask(psi, R, Z, mask_coarse, mask_f):
    """LCFS interior (point-in-polygon); NOT eroded -- only the 2-cell grid border is cleared."""
    C = extract_lcfs(psi, R, Z, MACHINE, mask_coarse, mask_f, n_points=256)
    if C is None:
        return None, None, None
    Ra, Za, iz, ir = magnetic_axis(psi, R, Z, MACHINE, mask_coarse)
    RR, ZZ = np.meshgrid(R, Z)
    # point-in-polygon via winding on the resampled contour
    from matplotlib.path import Path as MplPath
    inside = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    inside[:2] = inside[-2:] = False
    inside[:, :2] = inside[:, -2:] = False
    psi_a = psi[iz, ir]
    psi_b = float(np.mean([_bilin(psi, R, Z, p[0], p[1]) for p in C[::8]]))
    return inside, psi_a, psi_b


def _bilin(f, R, Z, r, z):
    ir = np.clip(np.searchsorted(R, r) - 1, 0, len(R) - 2)
    iz = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
    tr = (r - R[ir]) / (R[ir + 1] - R[ir]); tz = (z - Z[iz]) / (Z[iz + 1] - Z[iz])
    return ((1 - tz) * ((1 - tr) * f[iz, ir] + tr * f[iz, ir + 1])
            + tz * ((1 - tr) * f[iz + 1, ir] + tr * f[iz + 1, ir + 1]))


def gs_inconsistency(psi, R, Z, mask_coarse, mask_f, n_p=4, n_f=4):
    """Relative best-fit GS residual. 0 = psi IS an equilibrium for some (p', FF')."""
    inside, psi_a, psi_b = plasma_mask(psi, R, Z, mask_coarse, mask_f)
    if inside is None or inside.sum() < 50:
        return np.nan, 0
    ds = delta_star(psi, R, Z)
    RR, _ = np.meshgrid(R, Z)
    psin = np.clip((psi - psi_a) / (psi_b - psi_a), 0.0, 1.0)
    m = inside
    y = ds[m]
    cols = [(RR[m] ** 2) * psin[m] ** i for i in range(n_p)] + [psin[m] ** j for j in range(n_f)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, -y, rcond=None)
    r = y + A @ coef
    return float(np.linalg.norm(r) / np.linalg.norm(y)), int(m.sum())


def main():
    z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = z["grid_R"], z["grid_Z"]
    mc = z["mask_coarse"].astype(bool); mf = mc.astype(np.float64)

    import pyarrow.parquet as pq
    tbl = pq.ParquetFile(str(STARTER / "parquet_data" / "d3d_shot_203702.parquet")).read(
        columns=["efit_psirz"])
    psi_all = np.asarray(tbl.column("efit_psirz")[0].as_py(), dtype=np.float64)
    good = np.isfinite(psi_all).all(axis=(1, 2))
    psi_all = psi_all[good]
    idx = np.linspace(0, len(psi_all) - 1, 12).astype(int)

    print("Does the best-fit GS residual DISCRIMINATE?  (relative, 0 = perfect equilibrium)\n")
    print(f"{'frame':>6s} {'npix':>6s} {'GROUND TRUTH':>13s} {'+1% white':>11s} "
          f"{'+3% white':>11s} {'gauss-smoothed':>15s} {'scaled x1.05':>13s}")
    rows = []
    for k in idx:
        psi = psi_all[k]
        scale = np.std(psi)
        rng = np.random.default_rng(k)
        variants = {}
        g0, n = gs_inconsistency(psi, R, Z, mc, mf)
        variants["truth"] = g0
        for eps in (0.01, 0.03):
            p = psi + rng.standard_normal(psi.shape) * eps * scale
            variants[f"w{eps}"] = gs_inconsistency(p, R, Z, mc, mf)[0]
        # smoothing: a "plausible looking" but physically wrong psi
        from scipy.ndimage import gaussian_filter
        variants["smooth"] = gs_inconsistency(gaussian_filter(psi, 1.5), R, Z, mc, mf)[0]
        variants["scaled"] = gs_inconsistency(psi * 1.05, R, Z, mc, mf)[0]
        print(f"{k:6d} {n:6d} {variants['truth']:13.4f} {variants['w0.01']:11.4f} "
              f"{variants['w0.03']:11.4f} {variants['smooth']:15.4f} {variants['scaled']:13.4f}")
        rows.append(variants)

    print("\nmedians:")
    for key, lab in [("truth", "ground truth"), ("w0.01", "+1% white noise"),
                     ("w0.03", "+3% white noise"), ("smooth", "gaussian-smoothed"),
                     ("scaled", "uniformly scaled x1.05")]:
        v = np.nanmedian([r[key] for r in rows])
        print(f"  {lab:24s} {v:.4f}")
    print("\nNOTE: a uniform rescale of psi must NOT change the score -- GS is linear in the")
    print("      source, so (p',FF') simply rescale. That row is a built-in invariance check.")


if __name__ == "__main__":
    main()
