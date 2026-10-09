#!/usr/bin/env python3
"""C6 step 4 — coverage, widths, kappa*, bootstrap and the verdict of THEORY.md §4-5. Output: c6_eval.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANDS = HERE / "bands"
LEVELS = [0.50, 0.80, 0.90, 0.95]
MAIN = 2                                    # index of a = 0.90
N_BOOT = 1000
MODELS = {
    "Linear Regression": ["linear_regression"], "Ridge (CV)": ["ridge_cv"], "Random Forest": ["random_forest"],
    "MLP (sklearn)": ["mlp_sklearn"], "Simple MLP": [f"simple_mlp_s{s}" for s in range(3)],
    "Conv Decoder": [f"conv_decoder_s{s}" for s in range(3)], "UNet_Lite": [f"unet_lite_s{s}" for s in range(3)],
    "PCA+Ridge (SCORE.md)": ["pca_ridge"],
}
ACCURATE = ["MLP (sklearn)", "Simple MLP", "Conv Decoder", "UNet_Lite"]
NULL = "null_mean"
REGIONS = ["full", "mask"]


def load(run):
    z = np.load(BANDS / f"{run}.npz")
    d = {k: z[k] for k in z.files}
    for rg in REGIONS:
        d[f"B_{rg}"] = [d[f"eB_{rg}"][d["sid_B"] == j] for j in range(12)]
        d[f"T_{rg}"] = [d[f"eT_{rg}"][d["sid_T"] == j] for j in range(8)]
    return d


def m_band(d, rg, bB, bT):
    """-> per level arrays: Q, emp_J, c*, W, W* for B shots bB and test shots bT."""
    eB = np.sort(np.concatenate([d[f"B_{rg}"][j] for j in bB]))
    eT = np.sort(np.concatenate([d[f"T_{rg}"][j] for j in bT]))
    nB, nT = len(eB), len(eT)
    out = {k: np.zeros(len(LEVELS)) for k in ("Q", "emp", "c", "W", "Ws")}
    for i, a in enumerate(LEVELS):
        k = int(np.ceil((nB + 1) * a))
        Q = eB[k - 1] if k <= nB else np.inf
        need = eT[int(np.ceil(a * nT)) - 1]
        c = max(1.0, need / Q)
        out["Q"][i], out["emp"][i], out["c"][i] = Q, np.mean(eT <= Q), c
        out["W"][i] = Q * float(d[f"smean_{rg}"]); out["Ws"][i] = c * out["W"][i]
    return out


def summary(runs, null, rg, bB, bT):
    """Seed-averaged emp_J, kappa, kappa* (THEORY §4: every quantity averaged over the 3 seeds)."""
    n = m_band(null, rg, bB, bT)
    per = [m_band(r, rg, bB, bT) for r in runs]
    return {"emp": np.mean([p["emp"] for p in per], 0), "kappa": np.mean([p["W"] / n["W"] for p in per], 0),
            "kappa_s": np.mean([p["Ws"] / n["Ws"] for p in per], 0), "c": np.mean([p["c"] for p in per], 0),
            "W": np.mean([p["W"] for p in per], 0), "null_W": n["W"], "null_emp": n["emp"], "null_c": n["c"]}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    data = {m: [load(r) for r in rs] for m, rs in MODELS.items()}
    null = load(NULL)
    allB, allT = list(range(12)), list(range(8))
    rng = np.random.default_rng(0)
    reps = [(rng.integers(0, 12, 12), rng.integers(0, 8, 8)) for _ in range(N_BOOT)]
    res = {"levels": LEVELS, "regions": {}}
    for rg in REGIONS:
        R = {}
        for m, runs in data.items():
            pt = summary(runs, null, rg, allB, allT)
            bs = [summary(runs, null, rg, b, t) for b, t in reps]
            P_joint = [float(np.mean([r[f"P_joint_{rg}_{i}"].mean() for r in runs])) for i in range(len(LEVELS))]
            P_pix = [float(np.mean([r[f"P_pix_{rg}_{i}"].mean() for r in runs])) for i in range(len(LEVELS))]
            P_W = [float(np.mean([r[f"P_W_{rg}_{i}"] for r in runs])) for i in range(len(LEVELS))]
            per_shot = [float(np.mean([np.mean(r[f"T_{rg}"][j] <= m_band(r, rg, allB, allT)["Q"][MAIN]) for r in runs]))
                        for j in range(8)]
            R[m] = {
                "M_emp_J": pt["emp"].tolist(), "M_emp_J_ci": [ci([b["emp"][i] for b in bs]) for i in range(len(LEVELS))],
                "kappa": pt["kappa"].tolist(), "kappa_star": pt["kappa_s"].tolist(),
                "kappa_star_ci": [ci([b["kappa_s"][i] for b in bs]) for i in range(len(LEVELS))],
                "c_star": pt["c"].tolist(), "M_W": pt["W"].tolist(),
                "P_emp_J": P_joint, "P_emp_pix": P_pix, "P_W": P_W,
                "price_M_over_P": (pt["W"] / np.array(P_W)).tolist(),
                "M_emp_J_per_test_shot_a090": per_shot,
            }
            if m == "MLP (sklearn)":   # null numbers are identical in every model's summary
                R["_null"] = {"M_W": pt["null_W"].tolist(), "M_emp_J": pt["null_emp"].tolist(), "c_star": pt["null_c"].tolist()}
        res["regions"][rg] = R

    F = res["regions"]["full"]
    ok_cov = {m: F[m]["M_emp_J"][MAIN] >= 0.85 for m in ACCURATE}
    ok_w = {m: F[m]["kappa_star"][MAIN] <= 0.5 for m in ACCURATE}
    absurd = [m for m in ACCURATE if F[m]["kappa_star"][MAIN] >= 1.0]
    if all(ok_cov.values()) and all(ok_w.values()):
        v = "CONFIRMED"
    elif len(absurd) >= 2:
        v = "REFUTED"
    else:
        v = "REFINED"
    res["verdict"] = {"C6": v, "coverage_ok": ok_cov, "width_ok": ok_w, "absurd": absurd}
    (HERE / "c6_eval.json").write_text(json.dumps(res, indent=1, default=float))

    for rg in REGIONS:
        R = res["regions"][rg]
        print(f"\n=== region {rg}   (a = {LEVELS})")
        print(f"  {'model':22s} {'M emp_J':>27s} {'kappa*':>23s} {'c*@.9':>6s} {'P emp_J':>23s} {'W_M/W_P@.9':>10s}")
        for m in MODELS:
            v_ = R[m]
            print(f"  {m:22s} {' '.join(f'{x:.3f}' for x in v_['M_emp_J'])}   {' '.join(f'{x:.3f}' for x in v_['kappa_star'])}"
                  f"  {v_['c_star'][MAIN]:6.2f}  {' '.join(f'{x:.3f}' for x in v_['P_emp_J'])}  {v_['price_M_over_P'][MAIN]:8.2f}")
        print(f"  null: W_M {np.round(R['_null']['M_W'], 4).tolist()}  emp_J {np.round(R['_null']['M_emp_J'], 3).tolist()}  c* {np.round(R['_null']['c_star'], 3).tolist()}")
    print("\n  a=0.90 CIs (full):")
    for m in ACCURATE:
        print(f"    {m:16s} emp_J {F[m]['M_emp_J'][MAIN]:.3f} {np.round(F[m]['M_emp_J_ci'][MAIN], 3).tolist()}   "
              f"kappa* {F[m]['kappa_star'][MAIN]:.3f} {np.round(F[m]['kappa_star_ci'][MAIN], 3).tolist()}   "
              f"per-shot emp_J {np.round(F[m]['M_emp_J_per_test_shot_a090'], 2).tolist()}")
    print("\nVERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
