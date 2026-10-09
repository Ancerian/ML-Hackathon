#!/usr/bin/env python3
"""H5 -- does the DIII-D -> MAST collapse have a TOPOLOGICAL component?

The challenge's own pilot reports naive cross-machine transfer falling from SSIM 0.83 to 0.10.
The usual reading is "the values don't transfer". H5 asks something sharper:

    at a GIVEN L2 accuracy, is the TOPOLOGY of a cross-machine prediction worse than
    the topology of a within-machine prediction of the same accuracy?

If yes, the collapse is not only quantitative -- the model transfers the *values* but not the
*skeleton*, and a topological term would see the failure that R2 alone under-reports.

Proxy for "a model trained on DIII-D": a linear decoder can only express psi inside the span of
the DIII-D PCA basis. So projecting MAST ground truth onto the DIII-D basis is an UPPER BOUND on
what any DIII-D-trained linear decoder could output for MAST -- it is the best case, given
perfect knowledge of the MAST target. If even that loses the topology, no such model can keep it.

Controls: (a) DIII-D projected on its own basis, same number of components -- the within-machine
control; (b) the global sign is chosen per machine to maximise R2, exactly as the scorer does,
so the known DIII-D/MAST sign-convention difference cannot masquerade as a transfer failure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[1] / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
from lcfs import extract_lcfs                       # noqa: E402
from common import AXIS_SIGN                        # noqa: E402
from scipy.ndimage import binary_erosion, label     # noqa: E402

LOOP = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
SHOTS = {"DIII-D": ["d3d_shot_203702", "d3d_shot_203703", "d3d_shot_203704"],
         "MAST":   ["mast_shot_28348", "mast_shot_28350", "mast_shot_28351"]}
MASK = {"DIII-D": "d3d_envelope", "MAST": "mast_envelope"}


def load(machine):
    import pyarrow.parquet as pq
    fr = []
    for s in SHOTS[machine]:
        a = np.asarray(pq.ParquetFile(str(STARTER / "parquet_data" / f"{s}.parquet"))
                       .read(columns=["efit_psirz"]).column("efit_psirz")[0].as_py(), dtype=np.float64)
        fr.append(a[np.isfinite(a).all(axis=(1, 2))])
    z = np.load(STARTER / "fusion_scoring" / "masks" / f"{MASK[machine]}.npz")
    return np.concatenate(fr), z["grid_R"], z["grid_Z"], z["mask_coarse"].astype(bool)


def _wind(ang, iz0, iz1, ir0, ir1):
    pts = ([(iz0, r) for r in range(ir0, ir1 + 1)]
           + [(r, ir1) for r in range(iz0 + 1, iz1 + 1)]
           + [(iz1, r) for r in range(ir1 - 1, ir0 - 1, -1)]
           + [(r, ir0) for r in range(iz1 - 1, iz0, -1)])
    a = np.array([ang[z_, r_] for z_, r_ in pts])
    d = np.diff(np.append(a, a[0]))
    return int(np.round((((d + np.pi) % (2 * np.pi)) - np.pi).sum() / (2 * np.pi)))


def skeleton(psi, R, Z, machine, mc):
    """(n_elliptic, n_hyperbolic, index_sum) inside the LCFS, or None if extraction failed."""
    mf = mc.astype(np.float64)
    C = extract_lcfs(psi, R, Z, machine, mc, mf, n_points=256)
    if C is None:
        return None
    from matplotlib.path import Path as MplPath
    RR, ZZ = np.meshgrid(R, Z)
    ins = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    ins = binary_erosion(ins, np.ones((3, 3)), iterations=2)
    if ins.sum() < 40:
        return None
    gz, gr = np.gradient(psi, Z[1] - Z[0], R[1] - R[0])
    ang = np.arctan2(gz, gr)
    nz, nr = psi.shape
    raw = np.zeros_like(psi)
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            if not ins[iz, ir]:
                continue
            a = [ang[iz + dz, ir + dr] for dz, dr in LOOP]
            a.append(a[0])
            d = np.diff(a)
            raw[iz, ir] = np.round((((d + np.pi) % (2 * np.pi)) - np.pi).sum() / (2 * np.pi))
    lab, n = label(raw != 0, structure=np.ones((3, 3)))
    nO = nX = 0
    isum = 0
    for c in range(1, n + 1):
        zs, rs = np.where(lab == c)
        w = _wind(ang, max(zs.min() - 1, 0), min(zs.max() + 1, nz - 1),
                  max(rs.min() - 1, 0), min(rs.max() + 1, nr - 1))
        if w > 0:
            nO += 1
        elif w < 0:
            nX += 1
        isum += w
    return nO, nX, isum


def r2(pred, gt, mean):
    return 1.0 - np.sum((pred - gt) ** 2) / np.sum((gt - mean) ** 2)


def main():
    psiD, RD, ZD, mcD = load("DIII-D")
    psiM, RM, ZM, mcM = load("MAST")
    print(f"DIII-D frames={len(psiD)}   MAST frames={len(psiM)}")
    print(f"AXIS_SIGN: DIII-D={AXIS_SIGN['DIII-D']:+.0f}  MAST={AXIS_SIGN['MAST']:+.0f}  "
          "(opposite storage conventions -- neutralised below)\n")

    # ---- ground-truth skeletons: is the criterion itself machine-independent?
    print("=== 0. Ground-truth skeletons (the criterion must hold on BOTH machines) ===")
    for name, psi, R, Z, mc in [("DIII-D", psiD, RD, ZD, mcD), ("MAST", psiM, RM, ZM, mcM)]:
        idx = np.linspace(0, len(psi) - 1, 8).astype(int)
        res = [skeleton(psi[k], R, Z, name, mc) for k in idx]
        res = [r for r in res if r]
        if res:
            nO = np.median([r[0] for r in res]); nX = np.median([r[1] for r in res])
            isum = np.median([r[2] for r in res])
            print(f"  {name:7s} elliptic={nO:.1f}  hyperbolic={nX:.1f}  index sum={isum:+.1f}  "
                  f"({len(res)}/{len(idx)} frames extracted)")

    # ---- DIII-D PCA basis = what a DIII-D-trained linear decoder can express
    X = psiD.reshape(len(psiD), -1)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)

    def project(psi_set, k, sign_search):
        """Best-case output of a DIII-D-trained linear decoder, optional global sign flip."""
        Y = psi_set.reshape(len(psi_set), -1)
        best, bs = None, +1.0
        for s in ([1.0, -1.0] if sign_search else [1.0]):
            Ys = s * Y
            rec = mu + (Ys - mu) @ Vt[:k].T @ Vt[:k]
            e = np.sum((rec - Ys) ** 2)
            if best is None or e < best[0]:
                best, bs = (e, rec), s
        return best[1].reshape(psi_set.shape), bs

    idxD = np.linspace(0, len(psiD) - 1, 8).astype(int)
    idxM = np.linspace(0, len(psiM) - 1, 8).astype(int)

    print("\n=== 1. Within-machine control vs cross-machine transfer, same basis size ===")
    print(f"{'k':>4s} | {'D3D->D3D  R2psi':>16s} {'skeleton':>12s} | "
          f"{'MAST->D3D  R2psi':>16s} {'skeleton':>12s} {'sign':>5s}")
    rows = []
    for k in [5, 10, 20, 50]:
        recD, _ = project(psiD, k, False)
        recM, sM = project(psiM, k, True)

        r2D = r2(recD[idxD], psiD[idxD], psiD.mean())
        sk = [skeleton(recD[i], RD, ZD, "DIII-D", mcD) for i in idxD]
        sk = [s for s in sk if s]
        dO = np.median([s[0] for s in sk]) if sk else np.nan
        dX = np.median([s[1] for s in sk]) if sk else np.nan

        tgtM = sM * psiM
        r2M = r2(recM[idxM], tgtM[idxM], tgtM.mean())
        skm = [skeleton(recM[i], RM, ZM, "MAST", mcM) for i in idxM]
        nfail = len(idxM) - len(skm := [s for s in skm if s])
        mO = np.median([s[0] for s in skm]) if skm else np.nan
        mX = np.median([s[1] for s in skm]) if skm else np.nan

        print(f"{k:4d} | {r2D:16.5f} {f'{dO:.0f}O/{dX:.0f}X':>12s} | "
              f"{r2M:16.5f} {f'{mO:.0f}O/{mX:.0f}X' if skm else 'LCFS FAIL':>12s} {sM:+5.0f}"
              + (f"   [{nfail}/{len(idxM)} LCFS extraction failed]" if nfail else ""))
        rows.append((k, r2D, dO, dX, r2M, mO, mX, nfail))

    print("\n=== 2. The H5 question ===")
    print("  Compare the rows at COMPARABLE R2psi. If cross-machine loses the skeleton while")
    print("  within-machine keeps it at the same R2, the collapse has a topological component.")


if __name__ == "__main__":
    main()
