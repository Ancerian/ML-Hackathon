#!/usr/bin/env python3
"""Does a predicted psi have the RIGHT CRITICAL-POINT TOPOLOGY?

Why this is the Poincare question in disguise
---------------------------------------------
Magnetic field lines in a torus are a 1.5-DOF Hamiltonian system: toroidal angle = time, the
TOROIDAL flux psi_t = momentum, psi = Hamiltonian (R4). In STRICT AXISYMMETRY it is INTEGRABLE, so the
Poincare section is exactly the level sets of psi -- it carries no information beyond the
contours. The dynamical content that survives in 2D is therefore not the section itself but
the CRITICAL-POINT SKELETON of psi: which fixed points exist and of what type.

For a genuine diverted equilibrium the skeleton is tightly constrained:
    inside the LCFS   exactly ONE elliptic point (the magnetic axis, O-point), index +1
    on the separatrix the X-point(s), hyperbolic, index -1
Poincare-Hopf: the indices must sum to the Euler characteristic of the region (+1 for a disk).

A noisy predicted psi grows SPURIOUS critical points -- and it grows them preferentially where
the true gradient is small, i.e. right next to the real axis and the real X-point. Each spurious
elliptic point is a magnetic island that does not exist.

This probe measures how fast that happens, using the index computed as the winding number of
grad psi around each grid point.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[1] / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
from lcfs import extract_lcfs                                     # noqa: E402

MACHINE = "DIII-D"
# 8-neighbour loop, counter-clockwise
LOOP = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _winding(ang, iz0, iz1, ir0, ir1):
    """Winding number of grad psi around the rectangle perimeter [iz0..iz1] x [ir0..ir1]."""
    pts = ([(iz0, r) for r in range(ir0, ir1 + 1)]
           + [(r, ir1) for r in range(iz0 + 1, iz1 + 1)]
           + [(iz1, r) for r in range(ir1 - 1, ir0 - 1, -1)]
           + [(r, ir0) for r in range(iz1 - 1, iz0, -1)])
    a = np.array([ang[z_, r_] for z_, r_ in pts])
    d = np.diff(np.append(a, a[0]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return int(np.round(d.sum() / (2 * np.pi)))


def critical_points(psi, R, Z, inside):
    """Distinct critical points of psi inside `inside`, with their Poincare-Hopf index.

    A single critical point lands on SEVERAL adjacent grid cells (the axis of a real EFIT
    equilibrium shows up as a 2x2 block), so raw per-pixel winding numbers over-count by a
    factor of ~4. We therefore label connected components of the per-pixel detections and take
    the winding number around each COMPONENT's bounding box -- which is the mathematically
    correct index of whatever sits inside that box.

    index = +1 elliptic (O-point, a flux-surface centre)
    index = -1 hyperbolic (X-point, a separatrix crossing)
    """
    from scipy.ndimage import label
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    gz, gr = np.gradient(psi, dZ, dR)
    ang = np.arctan2(gz, gr)
    nz, nr = psi.shape

    raw = np.zeros_like(psi)
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            if not inside[iz, ir]:
                continue
            a = [ang[iz + dz, ir + dr] for dz, dr in LOOP]
            a.append(a[0])
            d = np.diff(a)
            d = (d + np.pi) % (2 * np.pi) - np.pi
            raw[iz, ir] = np.round(d.sum() / (2 * np.pi))

    lab, n = label(raw != 0, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        zs, rs = np.where(lab == c)
        iz0, iz1 = max(zs.min() - 1, 0), min(zs.max() + 1, nz - 1)
        ir0, ir1 = max(rs.min() - 1, 0), min(rs.max() + 1, nr - 1)
        w = _winding(ang, iz0, iz1, ir0, ir1)
        if w != 0:
            out.append({"iz": float(zs.mean()), "ir": float(rs.mean()), "index": w,
                        "npix": int(len(zs))})
    return out


def inside_lcfs(psi, R, Z, mask_coarse, mask_f, erode=2):
    from matplotlib.path import Path as MplPath
    C = extract_lcfs(psi, R, Z, MACHINE, mask_coarse, mask_f, n_points=256)
    if C is None:
        return None
    RR, ZZ = np.meshgrid(R, Z)
    ins = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    # erode so the stencil stays inside
    from scipy.ndimage import binary_erosion
    return binary_erosion(ins, np.ones((3, 3)), iterations=erode)


def main():
    z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = z["grid_R"], z["grid_Z"]
    mc = z["mask_coarse"].astype(bool); mf = mc.astype(np.float64)

    import pyarrow.parquet as pq
    a = np.asarray(pq.ParquetFile(str(STARTER / "parquet_data" / "d3d_shot_203702.parquet"))
                   .read(columns=["efit_psirz"]).column("efit_psirz")[0].as_py(), dtype=np.float64)
    psi_all = a[np.isfinite(a).all(axis=(1, 2))]
    frames = np.linspace(0, len(psi_all) - 1, 8).astype(int)

    print("Critical-point skeleton of psi inside the LCFS")
    print("a true equilibrium must have EXACTLY ONE elliptic point and ZERO interior saddles\n")
    print(f"{'noise':>8s} {'R2_psi':>9s} {'elliptic (O)':>13s} {'hyperbolic (X)':>15s} "
          f"{'index sum':>10s} {'spurious':>9s}")

    levels = [0.0, 0.001, 0.003, 0.01, 0.03, 0.10]
    for eps in levels:
        nO, nX, isum, e_meas = [], [], [], []
        for k in frames:
            psi = psi_all[k]
            rng = np.random.default_rng(1000 + k)
            d = rng.standard_normal(psi.shape) * eps * np.std(psi) if eps > 0 else np.zeros_like(psi)
            p = psi + d
            ins = inside_lcfs(p, R, Z, mc, mf)
            if ins is None or ins.sum() < 50:
                continue
            cps = critical_points(p, R, Z, ins)
            nO.append(sum(1 for c in cps if c["index"] > 0))
            nX.append(sum(1 for c in cps if c["index"] < 0))
            isum.append(sum(c["index"] for c in cps))
            e_meas.append(np.linalg.norm(d) / np.linalg.norm(psi - psi_all.mean()))
        if not nO:
            continue
        e = float(np.median(e_meas)) if e_meas else 0.0
        print(f"{eps:8.3f} {1-e**2:9.5f} {np.median(nO):13.1f} {np.median(nX):15.1f} "
              f"{np.median(isum):10.1f} {np.median(nO)+np.median(nX)-1:9.1f}")

    print("\n'spurious' = (O + X) - 1, i.e. every fixed point beyond the single real magnetic axis.")
    print("Each spurious elliptic point is a magnetic island that does not exist.")


if __name__ == "__main__":
    main()
