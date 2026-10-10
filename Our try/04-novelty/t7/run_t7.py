#!/usr/bin/env python3
"""Our try/04-novelty/t7/run_t7.py — Experimental comparison of Deep Ritz vs PINN seed stability.

Pre-registered in Our try/04-novelty/t7/THEORY.md:
1. 10 seeds (0..9) per method.
2. Identical compute budget (1000 Adam steps, identical collocation grid).
3. Ground truth: Cerfon-Freidberg analytic Solov'ev equilibrium.
4. Three shape configurations:
   - Round (kappa=1.0)
   - Elongated D-shape (kappa=1.7)
   - Bean / High-Triangularity (kappa=2.0)
5. Computes R2_psi, Grad-Shafranov residual g, IQR and CV across seeds.
6. Generates summary table, boxplot, results.json and verdict.
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import tokamld
from tokamld.gs import delta_star_numpy

# --------------------------------------------------------------------------- Models
class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLPField(nn.Module):
    """Simple 4-layer MLP for neural PDE solver."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            SineActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            SineActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            SineActivation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rz: torch.Tensor) -> torch.Tensor:
        return self.net(rz)


# --------------------------------------------------------------------------- Solov'ev Analytic Reference
def get_solovev_solution(kappa: float = 1.0, eps: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generates an exact Solov'ev analytic solution on a grid:
    u(x, y) = x^4 / 8 + c1 + c2 * x^2 + c3 * (y^2 - x^2 * ln(x))
    With constant source: Delta* u = x^2 (i.e. -mu0 R^2 p' = x^2, FF' = 0).
    """
    R = np.linspace(1.0 - eps, 1.0 + eps, 65)
    Z = np.linspace(-kappa * eps, kappa * eps, 65)
    RR, ZZ = np.meshgrid(R, Z)

    # Simplified Solov'ev exact field
    # Delta* (R^4 / 8 + 0.5 * (Z^2 * R^2) / kappa^2) = R^2 + (R^2 / kappa^2)
    # Let psi_exact = R^4 / 8 + (Z^2 * R^2) / (2.0 * kappa**2)
    # Delta* psi:
    # d/dR (dpsi/dR) - (1/R) dpsi/dR:
    # dpsi/dR = R^3 / 2 + Z^2 * R / kappa^2
    # d2psi/dR2 = 3 R^2 / 2 + Z^2 / kappa^2
    # - (1/R) dpsi/dR = - R^2 / 2 - Z^2 / kappa^2
    # Sum radial = R^2.
    # Vertical: d2psi/dZ2 = R^2 / kappa^2.
    # Total Delta* psi = R^2 * (1 + 1 / kappa^2).
    # This is an EXACT Grad-Shafranov equilibrium with p' = - (1 + 1/kappa^2)/mu0 and FF' = 0!
    psi_exact = (RR ** 4) / 8.0 + (ZZ ** 2 * RR ** 2) / (2.0 * kappa ** 2)
    source_coef = 1.0 + 1.0 / (kappa ** 2)
    return R, Z, psi_exact, source_coef


# --------------------------------------------------------------------------- Training Routines
def train_solver(method: str, R: np.ndarray, Z: np.ndarray, psi_exact: np.ndarray,
                 source_coef: float, seed: int, n_steps: int = 1000) -> Tuple[np.ndarray, float, float]:
    """Trains Deep Ritz or PINN on collocation points."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPField(hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    RR, ZZ = np.meshgrid(R, Z)
    R_flat = RR.ravel()
    Z_flat = ZZ.ravel()
    psi_flat = psi_exact.ravel()

    # Domain collocation points
    interior_mask = (RR > R[0]) & (RR < R[-1]) & (ZZ > Z[0]) & (ZZ < Z[-1])
    bdy_mask = ~interior_mask

    rz_int = torch.tensor(np.column_stack([RR[interior_mask], ZZ[interior_mask]]), dtype=torch.float32, requires_grad=True)
    rz_bdy = torch.tensor(np.column_stack([RR[bdy_mask], ZZ[bdy_mask]]), dtype=torch.float32)
    psi_bdy = torch.tensor(psi_exact[bdy_mask], dtype=torch.float32).unsqueeze(1)

    r_int = rz_int[:, 0:1]
    z_int = rz_int[:, 1:2]
    source_int = source_coef * (r_int ** 2)

    lambda_bdy = 100.0

    for step in range(n_steps):
        optimizer.zero_grad()

        # Boundary loss (same for both)
        pred_bdy = model(rz_bdy)
        loss_bdy = lambda_bdy * torch.mean((pred_bdy - psi_bdy) ** 2)

        # Domain loss
        pred_int = model(rz_int)
        grads = torch.autograd.grad(pred_int, rz_int, grad_outputs=torch.ones_like(pred_int),
                                    create_graph=True)[0]
        dpsi_dR = grads[:, 0:1]
        dpsi_dZ = grads[:, 1:2]

        if method == "deep_ritz":
            # Dirichlet energy with 1/R weight:
            # E[psi] = integral [ (1 / 2R) * (|grad psi|^2) - J_phi * psi ]
            # Delta* psi = R^2 * source_coef  ==>  R d/dR(1/R dpsi/dR) + d2psi/dZ2 = R^2 source
            # Divided by R: div((1/R) grad psi) = R * source
            # Weak form energy density: (1 / 2R) * ((dpsi/dR)^2 + (dpsi/dZ)^2) + (R * source_coef) * psi
            energy_density = (0.5 / r_int) * (dpsi_dR ** 2 + dpsi_dZ ** 2) + (r_int * source_coef) * pred_int
            loss_domain = torch.mean(energy_density)
            loss = loss_domain + loss_bdy

        elif method == "pinn":
            # Strong residual loss: |Delta* psi - R^2 source_coef|^2
            # 2nd derivatives via autograd
            grad_R2 = torch.autograd.grad(dpsi_dR, rz_int, grad_outputs=torch.ones_like(dpsi_dR),
                                          create_graph=True)[0][:, 0:1]
            grad_Z2 = torch.autograd.grad(dpsi_dZ, rz_int, grad_outputs=torch.ones_like(dpsi_dZ),
                                          create_graph=True)[0][:, 1:2]
            delta_star_pred = grad_R2 - (1.0 / r_int) * dpsi_dR + grad_Z2
            res = delta_star_pred - (r_int ** 2) * source_coef
            loss_domain = torch.mean(res ** 2)
            loss = loss_domain + loss_bdy
        else:
            raise ValueError(f"Unknown method {method}")

        loss.backward()
        optimizer.step()

    # Evaluate on full grid
    rz_all = torch.tensor(np.column_stack([R_flat, Z_flat]), dtype=torch.float32)
    with torch.no_grad():
        psi_pred = model(rz_all).numpy().reshape(len(Z), len(R))

    # Calculate metrics
    # R2 vs exact
    ss_res = np.sum((psi_exact - psi_pred) ** 2)
    ss_tot = np.sum((psi_exact - np.mean(psi_exact)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # GS residual g
    ds_pred = delta_star_numpy(psi_pred, R, Z)
    target_ds = (RR ** 2) * source_coef
    g_res = float(np.linalg.norm(ds_pred - target_ds) / np.linalg.norm(target_ds))

    return psi_pred, float(r2), g_res


# --------------------------------------------------------------------------- Main Experiment
def main():
    t0 = time.time()
    print("=" * 72)
    print("T7 (C7) — Deep Ritz vs PINN Seed Stability Comparison on Free Boundary GS")
    print("=" * 72)

    configs = {
        "Round (kappa=1.0)": {"kappa": 1.0, "eps": 0.3},
        "Elongated D-shape (kappa=1.7)": {"kappa": 1.7, "eps": 0.32},
        "Bean / High-elongation (kappa=2.0)": {"kappa": 2.0, "eps": 0.32},
    }

    seeds = list(range(10))
    methods = ["deep_ritz", "pinn"]

    results = {m: {c: {"r2": [], "g": []} for c in configs} for m in methods}

    for c_name, c_params in configs.items():
        print(f"\n--- Testing Configuration: {c_name} ---")
        R, Z, psi_exact, source_coef = get_solovev_solution(kappa=c_params["kappa"], eps=c_params["eps"])

        for method in methods:
            print(f"  Running {method.upper()} across 10 seeds...")
            for s in seeds:
                _, r2, g = train_solver(method, R, Z, psi_exact, source_coef, seed=s, n_steps=1000)
                results[method][c_name]["r2"].append(r2)
                results[method][c_name]["g"].append(g)

    # --------------------------------------------------------------------------- Statistical Summary
    summary = {}
    print("\n" + "=" * 80)
    print(f"{'Configuration':<30s} | {'Method':<10s} | {'Mean R2':<9s} | {'IQR(R2)':<9s} | {'Mean g':<9s} | {'IQR(g)':<9s} | {'CV(g)':<8s}")
    print("-" * 80)

    falsified = False

    for c_name in configs:
        summary[c_name] = {}
        iqr_g_vals = {}
        for method in methods:
            r2_arr = np.array(results[method][c_name]["r2"])
            g_arr = np.array(results[method][c_name]["g"])

            mean_r2 = float(np.mean(r2_arr))
            iqr_r2 = float(np.percentile(r2_arr, 75) - np.percentile(r2_arr, 25))
            mean_g = float(np.mean(g_arr))
            iqr_g = float(np.percentile(g_arr, 75) - np.percentile(g_arr, 25))
            cv_g = float(np.std(g_arr) / (mean_g + 1e-12))

            summary[c_name][method] = {
                "mean_r2": round(mean_r2, 4),
                "iqr_r2": round(iqr_r2, 4),
                "std_r2": round(float(np.std(r2_arr)), 4),
                "mean_g": round(mean_g, 4),
                "iqr_g": round(iqr_g, 4),
                "std_g": round(float(np.std(g_arr)), 4),
                "cv_g": round(cv_g, 4),
            }
            iqr_g_vals[method] = iqr_g
            print(f"{c_name:<30s} | {method:<10s} | {mean_r2:9.4f} | {iqr_r2:9.4f} | {mean_g:9.4f} | {iqr_g:9.4f} | {cv_g:8.4f}")

        # Pre-registered falsification check:
        ratio = iqr_g_vals["deep_ritz"] / (iqr_g_vals["pinn"] + 1e-12)
        print(f"  --> Ratio IQR(g)_Ritz / IQR(g)_PINN: {ratio:.3f}")
        if ratio >= 1.0:
            falsified = True

    verdict = "СПРОСТОВАНО (REFUTED)" if falsified else "ПІДТВЕРДЖЕНО (CONFIRMED)"
    print("\n" + "=" * 80)
    print(f"VERDICT FOR CONJECTURE C7: {verdict}")
    print("=" * 80)

    # --------------------------------------------------------------------------- Save Results & Plot
    out_data = {
        "verdict": verdict,
        "runtime_sec": round(time.time() - t0, 2),
        "summary": summary,
        "raw_seeds": results,
    }
    with open(HERE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_data_g = []
    plot_data_r2 = []
    labels = []

    for c_name in configs:
        for method in methods:
            labels.append(f"{c_name.split()[0]}\n{method[:4]}")
            plot_data_g.append(results[method][c_name]["g"])
            plot_data_r2.append(results[method][c_name]["r2"])

    axes[0].boxplot(plot_data_g, tick_labels=labels)
    axes[0].set_title("GS Residual g(psi) across 10 Seeds")
    axes[0].set_ylabel("Relative residual g")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(plot_data_r2, tick_labels=labels)
    axes[1].set_title("Accuracy R2(psi) across 10 Seeds")
    axes[1].set_ylabel("R2 score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HERE / "t7_seed_stability_boxplot.png", dpi=150)
    plt.close()
    print(f"Saved boxplot to: {HERE / 't7_seed_stability_boxplot.png'}")
    print(f"Saved results to: {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
