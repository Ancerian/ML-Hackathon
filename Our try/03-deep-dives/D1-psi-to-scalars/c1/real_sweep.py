#!/usr/bin/env python3
"""C1 step 4 — do the alpha clusters survive a change of the shot sample?

  * DIII-D: 3 local demo shots + 25 HF training shots (fetch_hf.py), 10 frames per shot
  * MAST:   3 local demo shots, 20 frames per shot (axis_sign pinned, R9)
  * bootstrap over SHOTS (1000x): alpha CIs and partition frequencies (THEORY.md §4-5)
  * axis-curvature strata: lambda = |smallest Hessian eigenvalue| of psi at the O-point,
    frames split into terciles; alpha of R_axis / Z_axis per tercile (P1 predicts drift)

Output: real_sweep_<machine>.npz, real_sweep.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc                                                     # noqa: E402
import alpha_tools as at                                                    # noqa: E402

DEMO = sc.STARTER / "parquet_data"
HF = sc.PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset"


def axis_curvature(psi, machine):
    """|smaller Hessian eigenvalue| of phi = s*psi at the grid O-point, in flux / m^2."""
    from o_point import find_o_point
    R, Z, mc, _ = sc.load_grid(machine)
    s = sc.AXIS_SIGN[machine]
    out = []
    for p in psi:
        phi = s * p
        _, _, iz, ir, _ = find_o_point(phi, R, Z, mc)
        hR, hZ = R[1] - R[0], Z[1] - Z[0]
        frr = (phi[iz, ir + 1] - 2 * phi[iz, ir] + phi[iz, ir - 1]) / hR ** 2
        fzz = (phi[iz + 1, ir] - 2 * phi[iz, ir] + phi[iz - 1, ir]) / hZ ** 2
        frz = (phi[iz + 1, ir + 1] - phi[iz + 1, ir - 1]
               - phi[iz - 1, ir + 1] + phi[iz - 1, ir - 1]) / (4 * hR * hZ)
        ev = np.linalg.eigvalsh(np.array([[frr, frz], [frz, fzz]]))
        out.append(float(np.min(np.abs(ev))))
    return np.array(out)


def summarise(sw, fam, ids, names, n_boot):
    r = at.analyse(sw, fam, d1_eps=sc.D1_EPS)
    b = at.bootstrap(sw, fam, ids, n_boot=n_boot, d1_eps=sc.D1_EPS)
    out = {"alpha_global": r["alpha_global"].tolist(), "alpha_regime": r["alpha_regime"].tolist(),
           "alpha_global_part": r["alpha_global_part"], "alpha_regime_part": r["alpha_regime_part"],
           "slopes": r["slopes"].tolist(), "bootstrap": b}
    print(f"  [{fam}] alpha_global " + " ".join(f"{n}={a:.3f}" for n, a in zip(names, r["alpha_global"])))
    print(f"  [{fam}] alpha_regime " + " ".join(f"{n}={a:.3f}" for n, a in zip(names, r["alpha_regime"])))
    for key in ("alpha_global", "alpha_regime"):
        p = r[key + "_part"]
        bb = b[key]
        f_c1 = bb["k3_partition_freq"].get(at.C1_PARTITION, 0.0)
        f_ha = bb["best_partition_freq"].get(at.HALT_PARTITION, 0.0)
        top = list(bb["best_partition_freq"].items())[:2]
        print(f"    {key}: best k={p['k']} sil={p['silhouette']:.2f} {p['partition']}")
        print(f"      bootstrap: C1 partition (k=3) {f_c1:.1%}; H-alt (best) {f_ha:.1%}; top: {top}")
        print("      CI95: " + " ".join(f"{n}=[{lo:.2f},{hi:.2f}]" for n, (lo, hi) in zip(names, bb["ci95"])))
    return out


def run_machine(machine, files, n_per_shot, seeds, n_boot, tag):
    psi, ids = sc.load_shots(files, n_per_shot)
    print(f"\n##### {tag} ({machine}): {len(files)} shots, {len(psi)} frames")
    sw = sc.run_sweep(psi, machine, seeds=seeds)
    lam = axis_curvature(psi, machine)
    np.savez_compressed(HERE / f"real_sweep_{tag}.npz", **sw, shot_ids=ids, lam=lam)
    names = list(sw["scalars"])
    res = {"n_shots": len(files), "n_frames": int(len(psi)),
           "usable": int(np.isfinite(sw["base"]).all(axis=1).sum()),
           "files": [Path(f).name for f in files],
           "lambda_quartiles": np.percentile(lam, [0, 25, 50, 75, 100]).tolist()}
    for fam in sw["families"]:
        res[fam] = summarise(sw, fam, ids, names, n_boot)
    # curvature terciles (white noise, global alpha of the axis scalars)
    good = np.isfinite(sw["base"]).all(axis=1)
    t = np.percentile(lam[good], [33.3, 66.7])
    strata = {}
    for lab, m in (("low", lam < t[0]), ("mid", (lam >= t[0]) & (lam < t[1])), ("high", lam >= t[1])):
        fr = np.where(good & m)[0]
        r = at.analyse(sw, "white", fr, d1_eps=sc.D1_EPS)
        strata[lab] = {"lambda_median": float(np.median(lam[fr])), "n": int(len(fr)),
                       "alpha_global": r["alpha_global"].tolist(),
                       "slopes_axis": r["slopes"][:, :2].tolist()}
        print(f"  lambda {lab:4s} (median {np.median(lam[fr]):.3g}, n={len(fr)}): "
              f"R_axis {r['alpha_global'][0]:.3f}  Z_axis {r['alpha_global'][1]:.3f}")
    res["lambda_strata_white"] = strata
    return res


if __name__ == "__main__":
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    # HF dir holds the 3 demo shots (Hub layout) + the 25 fetched training shots
    d3d = sorted((HF / "data" / "diii_d_train").glob("d3d_shot_*.parquet"))
    mast = sorted(DEMO.glob("mast_shot_*.parquet"))
    out = {"DIII-D": run_machine("DIII-D", d3d, 10, 2, n_boot, "DIII-D"),
           "DIII-D_demo3": run_machine("DIII-D", sorted(DEMO.glob("d3d_shot_*.parquet")), 20, 3, n_boot, "DIII-D_demo3"),
           "MAST": run_machine("MAST", mast, 20, 3, n_boot, "MAST"),
           "C1_partition": at.C1_PARTITION, "Halt_partition": at.HALT_PARTITION}
    (HERE / "real_sweep.json").write_text(json.dumps(out, indent=1, default=str))
