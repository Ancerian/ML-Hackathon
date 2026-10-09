#!/usr/bin/env python3
"""sensitivity_scan.py — Systematic sensitivity analysis of S'-gate.

Evaluates model rankings, point estimates, and 95% paired bootstrap CIs
across a 2D parameter grid:
1. g_ref in {0.8x, 0.9x, 1.0x, 1.1x, 1.2x} of g_base = 0.6328
2. Gate slope tau in {0.5x, 1.0x, 2.0x} * g_ref (steep, nominal, gentle)
3. Additional gates: Hard gate, Sigmoid gate (k=10)

Uses the exact shot-level metrics from eval_{model}.json on the 8 C2 test shots.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T1_DIR = ROOT / "Our try" / "04-novelty" / "t1"

MODELS = ["UNet_Lite", "MLP (sklearn)", "PCA+Ridge", "Linear Regression"]
FILE_KEYS = {
    "UNet_Lite": "unet",
    "MLP (sklearn)": "mlp",
    "PCA+Ridge": "pca_ridge",
    "Linear Regression": "lr",
}

G_BASE = 0.6328
G_MULTS = [0.8, 0.9, 1.0, 1.1, 1.2]
SLOPE_MULTS = [0.5, 1.0, 2.0]  # steep, standard, gentle
N_BOOT = 1000
SEED = 42


def load_model_data():
    data = {}
    for m in MODELS:
        fpath = T1_DIR / f"eval_{FILE_KEYS[m]}.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        shots = d["per_shot"]
        s_arr = np.array([s["S"] for s in shots], dtype=np.float64)
        g_arr = np.array([s["bar_g"] for s in shots], dtype=np.float64)
        p_arr = np.array([s["P_topo"] for s in shots], dtype=np.float64)
        data[m] = {
            "S": s_arr,
            "g": g_arr,
            "P_topo": p_arr,
            "official_S": d["official"]["S"],
            "official_ci": d["official"]["ci_95"],
        }
    return data


def compute_gate(g_val, g_ref, tau):
    if g_val <= g_ref:
        return 1.0
    return float(np.exp(-(g_val - g_ref) / tau))


def compute_sigmoid_gate(g_val, g_ref, k=10.0):
    return float(1.0 / (1.0 + np.exp(k * (g_val - g_ref) / g_ref)))


def compute_hard_gate(g_val, g_ref):
    return 1.0 if g_val <= g_ref else 0.0


def run_scan():
    data = load_model_data()
    n_shots = 8
    rng = np.random.default_rng(SEED)
    boot_indices = [rng.integers(0, n_shots, size=n_shots) for _ in range(N_BOOT)]

    results = []

    # 1. Exponential gate grid: g_mult x slope_mult
    for g_mult in G_MULTS:
        g_ref = g_mult * G_BASE
        for slope_mult in SLOPE_MULTS:
            tau = slope_mult * g_ref
            
            # Compute shot scores for each model
            model_scores = {}
            for m in MODELS:
                s_i = data[m]["S"]
                g_i = data[m]["g"]
                p_i = data[m]["P_topo"]
                g_gate_i = np.array([compute_gate(g, g_ref, tau) for g in g_i])
                s_prime_i = s_i * g_gate_i * p_i
                
                point_s_prime = float(np.mean(s_prime_i))
                
                # Bootstrap CI
                boot_vals = [float(np.mean(s_prime_i[b])) for b in boot_indices]
                ci_low = float(np.percentile(boot_vals, 2.5))
                ci_high = float(np.percentile(boot_vals, 97.5))
                
                model_scores[m] = {
                    "shot_scores": s_prime_i,
                    "point": point_s_prime,
                    "ci": [ci_low, ci_high],
                    "boot_vals": boot_vals,
                }
            
            # Rankings
            ranked = sorted(MODELS, key=lambda m: model_scores[m]["point"], reverse=True)
            ranks = {m: i + 1 for i, m in enumerate(ranked)}
            
            # Paired difference: UNet - MLP
            diff_point = model_scores["UNet_Lite"]["point"] - model_scores["MLP (sklearn)"]["point"]
            diff_boot = np.array(model_scores["UNet_Lite"]["boot_vals"]) - np.array(model_scores["MLP (sklearn)"]["boot_vals"])
            diff_ci = [float(np.percentile(diff_boot, 2.5)), float(np.percentile(diff_boot, 97.5))]
            
            # Paired difference: UNet - PCA+Ridge
            diff_unet_pca = model_scores["UNet_Lite"]["point"] - model_scores["PCA+Ridge"]["point"]
            diff_unet_pca_boot = np.array(model_scores["UNet_Lite"]["boot_vals"]) - np.array(model_scores["PCA+Ridge"]["boot_vals"])
            diff_unet_pca_ci = [float(np.percentile(diff_unet_pca_boot, 2.5)), float(np.percentile(diff_unet_pca_boot, 97.5))]

            results.append({
                "type": "exponential",
                "g_mult": g_mult,
                "g_ref": g_ref,
                "slope_mult": slope_mult,
                "tau": tau,
                "scores": {m: {"point": model_scores[m]["point"], "ci": model_scores[m]["ci"], "rank": ranks[m]} for m in MODELS},
                "rank_order": ranked,
                "delta_unet_mlp": {
                    "point": diff_point,
                    "ci": diff_ci,
                    "significant": (diff_ci[0] > 0 or diff_ci[1] < 0),
                },
                "delta_unet_pca": {
                    "point": diff_unet_pca,
                    "ci": diff_unet_pca_ci,
                    "significant": (diff_unet_pca_ci[0] > 0 or diff_unet_pca_ci[1] < 0),
                }
            })

    # 2. Hard and Sigmoid gates at g_mult in G_MULTS
    for g_mult in G_MULTS:
        g_ref = g_mult * G_BASE
        for g_type in ["hard", "sigmoid"]:
            model_scores = {}
            for m in MODELS:
                s_i = data[m]["S"]
                g_i = data[m]["g"]
                p_i = data[m]["P_topo"]
                if g_type == "hard":
                    g_gate_i = np.array([compute_hard_gate(g, g_ref) for g in g_i])
                else:
                    g_gate_i = np.array([compute_sigmoid_gate(g, g_ref, k=10.0) for g in g_i])
                s_prime_i = s_i * g_gate_i * p_i
                
                point_s_prime = float(np.mean(s_prime_i))
                boot_vals = [float(np.mean(s_prime_i[b])) for b in boot_indices]
                ci_low = float(np.percentile(boot_vals, 2.5))
                ci_high = float(np.percentile(boot_vals, 97.5))
                
                model_scores[m] = {
                    "shot_scores": s_prime_i,
                    "point": point_s_prime,
                    "ci": [ci_low, ci_high],
                    "boot_vals": boot_vals,
                }
            
            ranked = sorted(MODELS, key=lambda m: model_scores[m]["point"], reverse=True)
            ranks = {m: i + 1 for i, m in enumerate(ranked)}
            
            diff_point = model_scores["UNet_Lite"]["point"] - model_scores["MLP (sklearn)"]["point"]
            diff_boot = np.array(model_scores["UNet_Lite"]["boot_vals"]) - np.array(model_scores["MLP (sklearn)"]["boot_vals"])
            diff_ci = [float(np.percentile(diff_boot, 2.5)), float(np.percentile(diff_boot, 97.5))]

            results.append({
                "type": g_type,
                "g_mult": g_mult,
                "g_ref": g_ref,
                "slope_mult": None,
                "tau": None,
                "scores": {m: {"point": model_scores[m]["point"], "ci": model_scores[m]["ci"], "rank": ranks[m]} for m in MODELS},
                "rank_order": ranked,
                "delta_unet_mlp": {
                    "point": diff_point,
                    "ci": diff_ci,
                    "significant": (diff_ci[0] > 0 or diff_ci[1] < 0),
                },
            })

    return results


if __name__ == "__main__":
    res = run_scan()
    print(f"Total configurations scanned: {len(res)}")
    print("\nSample Exponential Gate results:")
    for r in res[:5]:
        print(f"g_ref={r['g_ref']:.4f} ({r['g_mult']}x), slope={r['slope_mult']}x: Order: {' > '.join(r['rank_order'])}")
        print(f"  Delta(UNet - MLP) = {r['delta_unet_mlp']['point']:.4f} [CI: {r['delta_unet_mlp']['ci'][0]:.4f} .. {r['delta_unet_mlp']['ci'][1]:.4f}], Sig: {r['delta_unet_mlp']['significant']}")
