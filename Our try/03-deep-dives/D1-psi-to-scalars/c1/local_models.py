#!/usr/bin/env python3
"""C1 step 2 — canonical local models with controlled geometry (THEORY.md P1-P7).

Part A  M1: an isolated paraboloid phi = -lambda/2 * r^2 on the DIII-D grid spacing, the axis
        found by the scorer's own find_o_point (grid local max + parabolic refinement).
        Absolute per-node noise eta. Tests P1: slope 1 below eta* = lambda h^2 / 2, slope 1/2
        above it (jumps of the discrete maximum), |dr| ~ 1/sqrt(lambda) in the jump regime
        (E14), and slope 1 for SMOOTH perturbations at every eta.
        Also reproduces the mechanism behind R6 (a fitted alpha that grows with lambda).

Part B  "teardrop": phi = -(x^2 + (y/k)^2) - (y/k)^3 / (3 b), x = R - R0, y = Z - Z0: one
        O-point and one lower X-point (at y = -2 b k), no GS physics, but the WHOLE scorer
        chain (LCFS bisection, contour shape, volume, li). Geometry (b, k) and the flux scale
        are controlled; all seven scalars are swept exactly as for real frames.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/c1/local_models.py"
Output: local_models.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc                                                     # noqa: E402
import alpha_tools as at                                                    # noqa: E402
from o_point import find_o_point                                            # noqa: E402

R, Z, MC, MF = sc.load_grid("DIII-D")
HR, HZ = float(R[1] - R[0]), float(Z[1] - Z[0])
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0]              # 16x range, as in E14
ETAS = np.logspace(-7, -1, 19)
N_TRIALS = 300


# ---------------------------------------------------------------------------- part A
def paraboloid(lam, rc, zc):
    RR, ZZ = np.meshgrid(R, Z)
    return -0.5 * lam * ((RR - rc) ** 2 + (ZZ - zc) ** 2)


def smooth_unit(rng):
    d = sc.d1.pert_band(rng, (len(Z), len(R)), *sc.SMOOTH_BAND)
    return d / np.sqrt(np.mean(d ** 2))                      # unit per-node RMS


def part_a():
    out = {"lambdas": LAMBDAS, "etas": ETAS.tolist(), "h": [HR, HZ], "white": {}, "smooth": {}}
    for fam in ("white", "smooth"):
        for lam in LAMBDAS:
            med = []
            for eta in ETAS:
                dR, dZ = [], []
                for t in range(N_TRIALS):
                    rng = np.random.default_rng(1000 * t + int(1e3 * lam))
                    rc = 1.69 + rng.uniform(-0.5, 0.5) * HR             # random sub-pixel centre
                    zc = 0.0 + rng.uniform(-0.5, 0.5) * HZ
                    p0 = paraboloid(lam, rc, zc)
                    Ra0, Za0, *_ = find_o_point(p0, R, Z, MC)
                    noise = (rng.standard_normal(p0.shape) if fam == "white" else smooth_unit(rng))
                    Ra, Za, *_ = find_o_point(p0 + eta * noise, R, Z, MC)
                    dR.append(abs(Ra - Ra0))
                    dZ.append(abs(Za - Za0))
                med.append([np.median(dR), np.median(dZ)])
            med = np.array(med)
            sl = at.local_slopes(ETAS, med)
            out[fam][str(lam)] = {"median_dR_dZ": med.tolist(), "slopes": sl.tolist(),
                                  "eta_star_pred": [lam * HR ** 2 / 2, lam * HZ ** 2 / 2]}
    # E14: |dr| * sqrt(lambda) in the jump regime (fixed eta well above all eta*)
    j = int(np.argmin(np.abs(ETAS - 1e-2)))
    prod = np.array([[out["white"][str(l)]["median_dR_dZ"][j][c] * np.sqrt(l) for c in (0, 1)]
                     for l in LAMBDAS])
    out["E14"] = {"eta": float(ETAS[j]), "dr_sqrt_lambda": prod.tolist(),
                  "spread_pct": (100 * (prod.max(0) - prod.min(0)) / prod.mean(0)).tolist()}
    # R6 mechanism: alpha fitted over one FIXED eta window, per lambda
    win = (ETAS >= 1e-5) & (ETAS <= 1e-2)
    out["R6_window"] = [1e-5, 1e-2]
    out["R6_alpha_R"] = {str(l): float(np.polyfit(np.log(ETAS[win]), np.log(
        np.array(out["white"][str(l)]["median_dR_dZ"])[win, 0]), 1)[0]) for l in LAMBDAS}
    return out


# ---------------------------------------------------------------------------- part B
def teardrop(b, k, r0=1.69, z0=0.25):
    RR, ZZ = np.meshgrid(R, Z)
    x, y = RR - r0, (ZZ - z0) / k
    phi = -(x ** 2 + y ** 2) - y ** 3 / (3 * b)
    return -phi                           # DIII-D ships psi with the axis at a MINIMUM (sign -1)


def part_b(seeds=2, n_offsets=12):
    shapes = [(b, k) for b in (0.40, 0.50, 0.60) for k in (1.0, 1.4)]
    frames, tags = [], []
    rng = np.random.default_rng(7)
    for b, k in shapes:
        for _ in range(n_offsets):                  # sub-pixel shifts of the whole pattern
            frames.append(teardrop(b, k, 1.69 + rng.uniform(-.5, .5) * HR,
                                   0.25 + rng.uniform(-.5, .5) * HZ))
            tags.append(f"b={b},k={k}")
    psi = np.array(frames)
    tags = np.array(tags)
    clean = np.array([sc.scalars(p, sc.load_grid("DIII-D"), "DIII-D") for p in psi])
    sw = sc.run_sweep(psi, "DIII-D", seeds=seeds)
    names = list(sw["scalars"])
    res = {"shapes": [f"b={b},k={k}" for b, k in shapes],
           "clean_scalars_median": {t: np.nanmedian(clean[tags == t], 0).tolist() for t in set(tags)},
           "usable": int(np.isfinite(sw["base"]).all(axis=1).sum()), "n": len(psi)}
    ids = np.array([list(dict.fromkeys(tags)).index(t) for t in tags])
    for fam in sw["families"]:
        r = at.analyse(sw, fam, d1_eps=sc.D1_EPS)
        per_shape = {}
        for t in dict.fromkeys(tags):
            rs = at.analyse(sw, fam, np.where((tags == t) & np.isfinite(sw["base"]).all(1))[0],
                            d1_eps=sc.D1_EPS)
            per_shape[t] = {"alpha_global": rs["alpha_global"].tolist(),
                            "part": rs["alpha_global_part"]["partition"],
                            "k3": rs["alpha_global_part"]["k3"][0]}
        res[fam] = {"alpha_global": r["alpha_global"].tolist(),
                    "alpha_regime": r["alpha_regime"].tolist(),
                    "alpha_global_part": r["alpha_global_part"],
                    "alpha_regime_part": r["alpha_regime_part"],
                    "slopes": r["slopes"].tolist(), "per_shape": per_shape}
        print(f"  teardrop [{fam}] alpha_global " +
              " ".join(f"{n}={a:.3f}" for n, a in zip(names, r["alpha_global"])))
        p = r["alpha_global_part"]
        print(f"     best k={p['k']} sil={p['silhouette']:.2f} {p['partition']} | k=3 {p['k3'][0]}")
    np.savez_compressed(HERE / "local_models_teardrop.npz", **sw, shot_ids=ids, tags=tags)
    return res


if __name__ == "__main__":
    A = part_a()
    for fam in ("white", "smooth"):
        print(f"\n== M1 paraboloid, {fam}: local slopes of |dR| (rows = eta intervals)")
        print("   eta      " + "  ".join(f"lam={l:<4g}" for l in LAMBDAS))
        for j in range(len(ETAS) - 1):
            print(f"   {ETAS[j]:.1e} " + "  ".join(f"{A[fam][str(l)]['slopes'][j][0]:8.2f}" for l in LAMBDAS))
    print("\n  eta* (pred, R):", [f"{l * HR ** 2 / 2:.1e}" for l in LAMBDAS])
    print("  E14 |dr|sqrt(lam) at eta=%.0e:" % A["E14"]["eta"], np.round(A["E14"]["dr_sqrt_lambda"], 5).tolist(),
          " spread % (R, Z):", np.round(A["E14"]["spread_pct"], 1).tolist())
    print("  R6-type alpha over eta in [1e-5, 1e-2]:", {k: round(v, 3) for k, v in A["R6_alpha_R"].items()})
    B = part_b()
    (HERE / "local_models.json").write_text(json.dumps({"M1": A, "teardrop": B}, indent=1, default=str))
