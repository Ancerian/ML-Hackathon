#!/usr/bin/env python3
"""C1 step 5 — collect all runs into results_c1.json and draw the figures.

Reads step1_baseline.json, real_sweep.json, local_models.json, cf_xpoint.json (all produced by
the scripts in this folder) and writes:
  results_c1.json      verdict table against THEORY.md §5
  c1_alpha_samples.png alpha (white noise, D1 estimator) per scalar and sample, with 95% CI
  c1_m1_crossover.png  local slope of the axis error vs eta / eta*  (P1, collapse over lambda)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    real, loc, cf = load("real_sweep.json"), load("local_models.json"), load("cf_xpoint.json")
    c1p, hap = real["C1_partition"], real["Halt_partition"]
    rows = {}
    for label, blob in (("DIII-D, 3 demo shots", real["DIII-D_demo3"]),
                        ("DIII-D, 28 shots", real["DIII-D"]),
                        ("MAST, 3 shots", real["MAST"]),
                        ("Cerfon-Freidberg, 12 shapes", cf)):
        for fam in ("white", "smooth", "pca-tail"):
            b = blob[fam]["bootstrap"]["alpha_global"]
            rows[f"{label} | {fam}"] = {
                "alpha": blob[fam]["alpha_global"], "ci95": b["ci95"],
                "best_partition": blob[fam]["alpha_global_part"]["partition"],
                "silhouette": blob[fam]["alpha_global_part"]["silhouette"],
                "C1_freq_k3": b["k3_partition_freq"].get(c1p, 0.0),
                "Halt_freq_best": b["best_partition_freq"].get(hap, 0.0),
                "top_partition": list(b["best_partition_freq"].items())[0]}
    for fam in ("white", "smooth", "pca-tail"):
        t = loc["teardrop"][fam]
        rows[f"teardrop (no bootstrap) | {fam}"] = {
            "alpha": t["alpha_global"], "best_partition": t["alpha_global_part"]["partition"],
            "silhouette": t["alpha_global_part"]["silhouette"]}
    m1 = loc["M1"]
    verdict = {
        "a_local_models": {
            "P1_white_crossover": "slope 1 below eta*, ~0.4-0.5 above; onset within a factor 2-3 of lambda h^2/2",
            "P1_smooth": "slope 1 up to geometric saturation",
            "E14_spread_pct_R_Z": m1["E14"]["spread_pct"],
            "R6_alpha_fixed_window": m1["R6_alpha_R"],
            "P2_tri": "predicted 1/2; observed 0.34-0.42 (real, CF, teardrop): below 1/2",
            "P4_kappa": "predicted ~1 with a 1/2 admixture; observed 0.34-0.66",
            "P6_volume": "predicted 0.8-0.95; observed 0.67-0.94",
            "P7_li": "predicted >= 1 if the LCFS is stable; observed 0.92-0.97 synthetic, 0.47 on 28 DIII-D shots",
        },
        "b_CF_C1_freq": cf["white"]["bootstrap"]["alpha_global"]["k3_partition_freq"].get(c1p, 0.0),
        "c_DIII-D28_C1_freq": real["DIII-D"]["white"]["bootstrap"]["alpha_global"]["k3_partition_freq"].get(c1p, 0.0),
        "c_MAST_C1_freq": real["MAST"]["white"]["bootstrap"]["alpha_global"]["k3_partition_freq"].get(c1p, 0.0),
        "demo3_C1_freq": real["DIII-D_demo3"]["white"]["bootstrap"]["alpha_global"]["k3_partition_freq"].get(c1p, 0.0),
        "C1": "REFUTED (THEORY.md §5: (b) and (c) give other partitions; C1 partition < 50% of bootstrap resamples)",
        "H-alt": "REFUTED (same criteria)",
    }
    (HERE / "results_c1.json").write_text(json.dumps({"verdict": verdict, "rows": rows,
                                                      "C1_partition": c1p, "Halt_partition": hap}, indent=1))

    # ---- figure 1: alpha per scalar and sample (white, D1 estimator)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    samples = [("DIII-D, 3 demo shots", "tab:gray", "o"), ("DIII-D, 28 shots", "tab:blue", "s"),
               ("MAST, 3 shots", "tab:orange", "^"), ("Cerfon-Freidberg, 12 shapes", "tab:green", "D")]
    for j, (lab, col, mk) in enumerate(samples):
        r = rows[f"{lab} | white"]
        x = np.arange(7) + (j - 1.5) * 0.17
        a = np.array(r["alpha"])
        ci = np.array(r["ci95"])
        ax.errorbar(x, a, yerr=[a - ci[:, 0], ci[:, 1] - a], fmt=mk, color=col, ms=5, capsize=2,
                    label=f"{lab}: C1 partition in {r['C1_freq_k3']:.0%} of resamples")
    for lo, hi in ((0.53, 0.55), (0.35, 0.42), (0.70, 0.70)):
        ax.axhspan(lo - 0.005, hi + 0.005, color="k", alpha=0.07)
    ax.set_xticks(range(7), NAMES)
    ax.set_ylabel(r"$\alpha$ (white noise, fit over $\varepsilon$ = 0.003-0.3)")
    ax.set_ylim(0.2, 1.05)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("C1: amplification exponents by sample (grey bands = the three E2 clusters)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "c1_alpha_samples.png")

    # ---- figure 2: P1 crossover collapse
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    etas = np.array(m1["etas"])
    hR = m1["h"][0]
    mid = np.sqrt(etas[1:] * etas[:-1])
    for lam in m1["lambdas"]:
        for fam, ls in (("white", "-"), ("smooth", ":")):
            sl = np.array(m1[fam][str(lam)]["slopes"])[:, 0]
            ax.semilogx(mid / (lam * hR ** 2 / 2), sl, ls, lw=1.2,
                        label=(fr"$\lambda$={lam:g}" if fam == "white" else None),
                        color=plt.cm.viridis(np.log2(lam) / 4))
    ax.axhline(1, color="k", lw=.6)
    ax.axhline(.5, color="k", lw=.6, ls="--")
    ax.axvline(1, color="r", lw=.6)
    ax.set_xlabel(r"$\eta\,/\,\eta^*$,  $\eta^* = \lambda h_R^2/2$")
    ax.set_ylabel(r"local slope of median $|\Delta R_{axis}|$")
    ax.set_title("P1: paraboloid, white (solid) vs smooth (dotted) perturbation", fontsize=9)
    ax.set_ylim(0.2, 1.2)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(HERE / "c1_m1_crossover.png")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
