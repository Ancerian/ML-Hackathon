#!/usr/bin/env python3
"""Scratch: is the Euler-characteristic curve of sublevel sets a better metric
than my critical-point count?  Both measured on the same noise ladder.

chi(psi <= t) as a function of t is O(N) per level (skimage.euler_number) and was
validated as both a loss and a metric in the segmentation literature. My count-based
detector was blind until 10% noise -- the question is whether the EC curve fires earlier,
and whether it fires TOO early (i.e. becomes another high-k noise detector duplicating
the GS residual).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STARTER = REPO_ROOT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology_probe import critical_points, inside_lcfs        # noqa: E402
from skimage.measure import euler_number                        # noqa: E402

NLEV = 40


def ec_curve(psi, inside, lo, hi, nlev=NLEV):
    """chi(psi <= t) restricted to the plasma region, over nlev thresholds."""
    ts = np.linspace(lo, hi, nlev)
    out = np.zeros(nlev, dtype=int)
    for i, t in enumerate(ts):
        m = (psi <= t) & inside
        out[i] = euler_number(m, connectivity=1) if m.any() else 0
    return ts, out


def main():
    z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = z["grid_R"], z["grid_Z"]
    mc = z["mask_coarse"].astype(bool); mf = mc.astype(np.float64)

    import pyarrow.parquet as pq
    a = np.asarray(pq.ParquetFile(str(STARTER / "parquet_data" / "d3d_shot_203702.parquet"))
                   .read(columns=["efit_psirz"]).column("efit_psirz")[0].as_py(), dtype=np.float64)
    psi_all = a[np.isfinite(a).all(axis=(1, 2))]
    frames = np.linspace(0, len(psi_all) - 1, 6).astype(int)
    pooled_den = np.linalg.norm(psi_all[frames] - psi_all.mean())

    print("Euler-characteristic curve of sublevel sets  vs  critical-point count")
    print("(both restricted to the interior of the LCFS)\n")
    print(f"{'noise':>7s} {'R2_psi':>9s} {'|d chi|_1 / nlev':>17s} {'max|d chi|':>11s} "
          f"{'crit pts':>9s} {'verdict':>28s}")

    # reference curves from truth
    ref = {}
    for k in frames:
        ins = inside_lcfs(psi_all[k], R, Z, mc, mf)
        lo, hi = np.percentile(psi_all[k][ins], [1, 99])
        ref[k] = (ins, lo, hi, ec_curve(psi_all[k], ins, lo, hi)[1])

    for eps in [0.0, 0.001, 0.003, 0.01, 0.03, 0.10]:
        d1, dmax, ncp, errs = [], [], [], []
        for k in frames:
            psi = psi_all[k]
            rng = np.random.default_rng(7000 + k)
            d = rng.standard_normal(psi.shape) * eps * np.std(psi) if eps > 0 else np.zeros_like(psi)
            p = psi + d
            ins0, lo, hi, chi0 = ref[k]
            # score on the REFERENCE mask and REFERENCE thresholds: the metric must not
            # be confounded by the predicted LCFS moving.
            _, chi1 = ec_curve(p, ins0, lo, hi)
            dd = np.abs(chi1 - chi0)
            d1.append(dd.mean()); dmax.append(dd.max())
            cps = critical_points(p, R, Z, ins0)
            ncp.append(len(cps))
            errs.append(np.linalg.norm(d))
        e = float(np.sqrt(np.sum(np.square(errs))) / pooled_den)
        m1, mx, mc_ = np.median(d1), np.median(dmax), np.median(ncp)
        if eps == 0:
            verdict = "baseline"
        elif m1 > 1.0:
            verdict = "EC fires"
        elif mc_ > 1:
            verdict = "count fires, EC quiet"
        else:
            verdict = "both quiet"
        print(f"{eps:7.3f} {1-e**2:9.5f} {m1:17.3f} {mx:11.0f} {mc_:9.1f} {verdict:>28s}")

    print("\nchi of the TRUE sublevel sets (should be 1 deep inside, since psi<=t near the")
    print("axis is a disk; rising t adds the annulus and eventually the whole region):")
    k = frames[2]
    ins0, lo, hi, chi0 = ref[k]
    ts = np.linspace(lo, hi, NLEV)
    print("  t:   " + " ".join(f"{v:4.2f}" for v in ts[::5]))
    print("  chi: " + " ".join(f"{v:4d}" for v in chi0[::5]))


if __name__ == "__main__":
    main()
