#!/usr/bin/env python3
"""Task T6: HénonNet / Symplectic Inverse Solver on Tokamap Poincaré Spectrum (D3 / c5).

Compares:
1. NLS (Nonlinear Least Squares with numerical Jacobian, baseline C5 / R13)
2. Differential Evolution (DE) on smoothed chi2
3. HénonNet (Symplectic inverse network predicting amplitudes and circular phase embeddings)

Evaluated on:
- Phase 1: Single-mode m=2 (Risk mitigation test)
- Phase 2: Dual-mode 2/1 + 3/1 (Main test)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import differential_evolution, least_squares
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]

# Tokamap grid setup
RADII = np.linspace(0.35, 1.25, 20)
N_PHASE = 8
N_IT = 150
PSI0, T0 = np.meshgrid(RADII, np.arange(N_PHASE) / N_PHASE, indexing="ij")
PSI0, T0 = PSI0.ravel(), T0.ravel()
SIGMA_NOISE = 0.03


def q_of(psi: np.ndarray) -> np.ndarray:
    return 1.0 + 3.0 * psi * psi


def wrap(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def forward_tokamap(a: np.ndarray, phi: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Computes phase-resolved excursion y in R^160."""
    a = np.asarray(a, float)
    phi = np.asarray(phi, float)
    modes = np.asarray(modes, float)
    arg_base = 2.0 * np.pi * modes[None, :]

    psi, T = PSI0.copy(), T0.copy()
    lo, hi = psi.copy(), psi.copy()

    for _ in range(N_IT):
        arg = arg_base * T[:, None] + phi[None, :]
        V = (a / (2.0 * np.pi) * np.sin(arg)).sum(axis=1)
        h = -(a / (4.0 * np.pi ** 2 * modes) * np.cos(arg)).sum(axis=1)
        P = psi - 1.0 - V
        psi_n = 0.5 * (P + np.sqrt(P * P + 4.0 * psi))
        T = np.mod(T + 1.0 / q_of(psi_n) + h / (1.0 + psi_n) ** 2, 1.0)
        psi = np.clip(psi_n, 0.0, 25.0)
        lo = np.minimum(lo, psi)
        hi = np.maximum(hi, psi)

    return hi - lo


# --------------------------------------------------------------------------- HénonNet Inverse Model
class HenonNetInverse(nn.Module):
    """Symplectic-inspired inverse model mapping excursion profile -> (log(a), cos(phi), sin(phi))."""
    def __init__(self, in_features: int = 160, n_modes: int = 2, hidden: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.lna_head = nn.Linear(hidden, n_modes)
        # Predict (cos, sin) to avoid periodic discontinuity at 0/2pi
        self.phase_head = nn.Linear(hidden, 2 * n_modes)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(y)
        lna = self.lna_head(h)
        phase_raw = self.phase_head(h).view(-1, self.n_modes, 2)
        # Normalize to unit circle
        phase_vec = F.normalize(phase_raw, p=2, dim=-1)
        phi = torch.atan2(phase_vec[..., 1], phase_vec[..., 0]) % (2.0 * np.pi)
        return lna, phi


def train_henon_inverse(n_modes: int, modes: np.ndarray, n_train: int = 2000, epochs: int = 100) -> HenonNetInverse:
    rng = np.random.default_rng(42)
    X_list, Y_lna_list, Y_phi_list = [], [], []

    for _ in range(n_train):
        a = rng.uniform(0.05, 0.25, size=n_modes)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        y = forward_tokamap(a, phi, modes)
        # Add 3% training noise
        y_noisy = y * (1.0 + SIGMA_NOISE * rng.standard_normal(y.shape))

        X_list.append(y_noisy)
        Y_lna_list.append(np.log(a))
        Y_phi_list.append(phi)

    X_t = torch.tensor(np.array(X_list), dtype=torch.float32)
    Lna_t = torch.tensor(np.array(Y_lna_list), dtype=torch.float32)
    Phi_t = torch.tensor(np.array(Y_phi_list), dtype=torch.float32)

    model = HenonNetInverse(in_features=160, n_modes=n_modes, hidden=128)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(X_t, Lna_t, Phi_t), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for bx, blna, bphi in loader:
            opt.zero_grad()
            pred_lna, pred_phi = model(bx)
            # Circular loss on phase: 1 - cos(pred - true)
            loss_amp = F.mse_loss(pred_lna, blna)
            loss_phi = torch.mean(1.0 - torch.cos(pred_phi - bphi))
            loss = loss_amp + 2.0 * loss_phi
            loss.backward()
            opt.step()

    model.eval()
    return model


# --------------------------------------------------------------------------- Solvers
def solve_nls(y_obs: np.ndarray, modes: np.ndarray, n_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Nonlinear Least Squares (TRF with finite-difference Jacobian)."""
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(0.05, 0.25, size=n_modes)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
    x0 = np.concatenate([np.log(a0), phi0])

    scale = SIGMA_NOISE * np.maximum(y_obs, 0.1 * np.median(y_obs))

    def res(x):
        a_curr = np.exp(x[:n_modes])
        phi_curr = x[n_modes:]
        y_model = forward_tokamap(a_curr, phi_curr, modes)
        return (y_model - y_obs) / scale

    lb = np.concatenate([np.full(n_modes, np.log(0.01)), np.full(n_modes, -np.inf)])
    ub = np.concatenate([np.full(n_modes, np.log(0.50)), np.full(n_modes, np.inf)])

    t0 = time.time()
    try:
        sol = least_squares(res, x0, bounds=(lb, ub), method="trf", max_nfev=60)
        a_est = np.exp(sol.x[:n_modes])
        phi_est = sol.x[n_modes:] % (2.0 * np.pi)
    except Exception:
        a_est = a0
        phi_est = phi0
    dt = time.time() - t0
    return a_est, phi_est, dt


def solve_de(y_obs: np.ndarray, modes: np.ndarray, n_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Differential Evolution on chi2 objective."""
    scale = SIGMA_NOISE * np.maximum(y_obs, 0.1 * np.median(y_obs))

    def obj(x):
        a_curr = x[:n_modes]
        phi_curr = x[n_modes:]
        y_model = forward_tokamap(a_curr, phi_curr, modes)
        return float(np.sum(((y_model - y_obs) / scale) ** 2))

    bounds = [(0.05, 0.25)] * n_modes + [(0.0, 2.0 * np.pi)] * n_modes
    t0 = time.time()
    res = differential_evolution(obj, bounds, maxiter=25, popsize=10, seed=seed)
    dt = time.time() - t0
    a_est = res.x[:n_modes]
    phi_est = res.x[n_modes:] % (2.0 * np.pi)
    return a_est, phi_est, dt


def solve_henon(model: HenonNetInverse, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """HénonNet direct amortized inference."""
    t0 = time.time()
    with torch.no_grad():
        y_t = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(0)
        lna, phi = model(y_t)
        a_est = np.exp(lna.numpy()[0])
        phi_est = phi.numpy()[0]
    dt = time.time() - t0
    return a_est, phi_est, dt


# --------------------------------------------------------------------------- Evaluation Core
def evaluate_instance(a_est: np.ndarray, phi_est: np.ndarray,
                      a_true: np.ndarray, phi_true: np.ndarray) -> tuple[bool, float, float]:
    """Evaluates success: each a within 10%, each phi within 20 deg."""
    da = np.abs(a_est - a_true) / a_true
    dp = np.abs(wrap(phi_est - phi_true))
    ok = bool(np.all(da <= 0.10) and np.all(dp <= np.deg2rad(20.0)))
    return ok, float(np.mean(da)), float(np.rad2deg(np.mean(dp)))


def run_benchmark():
    print("=" * 76)
    print("Task T6 — HénonNet vs NLS and Differential Evolution on Tokamap Spectrum")
    print("=" * 76)

    n_test = 20
    rng_test = np.random.default_rng(1234)

    results = {}

    # ----------------------------------------------------------------------- Phase 1: 1-Mode
    print("\n--- Phase 1: Single-Mode (m=2) Risk Mitigation Test ---")
    modes_1 = np.array([2.0])
    print("Training HénonNet (1-mode)...")
    henon_1 = train_henon_inverse(n_modes=1, modes=modes_1, n_train=1200, epochs=80)

    p1_stats = {"nls": {"ok": 0, "da": [], "dp": [], "time": []},
                "de":  {"ok": 0, "da": [], "dp": [], "time": []},
                "henon": {"ok": 0, "da": [], "dp": [], "time": []}}

    for i in range(n_test):
        a_true = rng_test.uniform(0.05, 0.25, size=1)
        phi_true = rng_test.uniform(0.0, 2.0 * np.pi, size=1)
        y_clean = forward_tokamap(a_true, phi_true, modes_1)
        y_obs = y_clean * (1.0 + SIGMA_NOISE * rng_test.standard_normal(y_clean.shape))

        # 1. NLS
        a_nls, p_nls, t_nls = solve_nls(y_obs, modes_1, 1, seed=i)
        ok_nls, da_nls, dp_nls = evaluate_instance(a_nls, p_nls, a_true, phi_true)
        p1_stats["nls"]["ok"] += int(ok_nls)
        p1_stats["nls"]["da"].append(da_nls)
        p1_stats["nls"]["dp"].append(dp_nls)
        p1_stats["nls"]["time"].append(t_nls)

        # 2. DE
        a_de, p_de, t_de = solve_de(y_obs, modes_1, 1, seed=i)
        ok_de, da_de, dp_de = evaluate_instance(a_de, p_de, a_true, phi_true)
        p1_stats["de"]["ok"] += int(ok_de)
        p1_stats["de"]["da"].append(da_de)
        p1_stats["de"]["dp"].append(dp_de)
        p1_stats["de"]["time"].append(t_de)

        # 3. HénonNet
        a_h, p_h, t_h = solve_henon(henon_1, y_obs)
        ok_h, da_h, dp_h = evaluate_instance(a_h, p_h, a_true, phi_true)
        p1_stats["henon"]["ok"] += int(ok_h)
        p1_stats["henon"]["da"].append(da_h)
        p1_stats["henon"]["dp"].append(dp_h)
        p1_stats["henon"]["time"].append(t_h)

    print(f"Phase 1 Results (N={n_test}):")
    print(f"  NLS:      Success={p1_stats['nls']['ok']/n_test:.1%}, mean da={np.mean(p1_stats['nls']['da']):.1%}, mean dp={np.mean(p1_stats['nls']['dp']):.1f}°, time={np.mean(p1_stats['nls']['time'])*1e3:.1f}ms")
    print(f"  DE:       Success={p1_stats['de']['ok']/n_test:.1%}, mean da={np.mean(p1_stats['de']['da']):.1%}, mean dp={np.mean(p1_stats['de']['dp']):.1f}°, time={np.mean(p1_stats['de']['time']):.2f}s")
    print(f"  HénonNet: Success={p1_stats['henon']['ok']/n_test:.1%}, mean da={np.mean(p1_stats['henon']['da']):.1%}, mean dp={np.mean(p1_stats['henon']['dp']):.1f}°, time={np.mean(p1_stats['henon']['time'])*1e3:.2f}ms")

    # ----------------------------------------------------------------------- Phase 2: Dual-Mode (2/1 + 3/1)
    print("\n--- Phase 2: Dual-Mode (2/1 + 3/1) Main Test ---")
    modes_2 = np.array([2.0, 3.0])
    print("Training HénonNet (2-mode)...")
    henon_2 = train_henon_inverse(n_modes=2, modes=modes_2, n_train=2500, epochs=100)

    p2_stats = {"nls": {"ok": 0, "da": [], "dp": [], "time": []},
                "de":  {"ok": 0, "da": [], "dp": [], "time": []},
                "henon": {"ok": 0, "da": [], "dp": [], "time": []}}

    for i in range(n_test):
        a_true = rng_test.uniform(0.05, 0.25, size=2)
        phi_true = rng_test.uniform(0.0, 2.0 * np.pi, size=2)
        y_clean = forward_tokamap(a_true, phi_true, modes_2)
        y_obs = y_clean * (1.0 + SIGMA_NOISE * rng_test.standard_normal(y_clean.shape))

        # 1. NLS
        a_nls, p_nls, t_nls = solve_nls(y_obs, modes_2, 2, seed=i)
        ok_nls, da_nls, dp_nls = evaluate_instance(a_nls, p_nls, a_true, phi_true)
        p2_stats["nls"]["ok"] += int(ok_nls)
        p2_stats["nls"]["da"].append(da_nls)
        p2_stats["nls"]["dp"].append(dp_nls)
        p2_stats["nls"]["time"].append(t_nls)

        # 2. DE
        a_de, p_de, t_de = solve_de(y_obs, modes_2, 2, seed=i)
        ok_de, da_de, dp_de = evaluate_instance(a_de, p_de, a_true, phi_true)
        p2_stats["de"]["ok"] += int(ok_de)
        p2_stats["de"]["da"].append(da_de)
        p2_stats["de"]["dp"].append(dp_de)
        p2_stats["de"]["time"].append(t_de)

        # 3. HénonNet
        a_h, p_h, t_h = solve_henon(henon_2, y_obs)
        ok_h, da_h, dp_h = evaluate_instance(a_h, p_h, a_true, phi_true)
        p2_stats["henon"]["ok"] += int(ok_h)
        p2_stats["henon"]["da"].append(da_h)
        p2_stats["henon"]["dp"].append(dp_h)
        p2_stats["henon"]["time"].append(t_h)

    print(f"\nPhase 2 Results (N={n_test}, 2/1 + 3/1):")
    s_nls = p2_stats['nls']['ok'] / n_test
    s_de = p2_stats['de']['ok'] / n_test
    s_h = p2_stats['henon']['ok'] / n_test
    print(f"  NLS:      Success={s_nls:.1%}, mean da={np.mean(p2_stats['nls']['da']):.1%}, mean dp={np.mean(p2_stats['nls']['dp']):.1f}°, time={np.mean(p2_stats['nls']['time'])*1e3:.1f}ms")
    print(f"  DE:       Success={s_de:.1%}, mean da={np.mean(p2_stats['de']['da']):.1%}, mean dp={np.mean(p2_stats['de']['dp']):.1f}°, time={np.mean(p2_stats['de']['time']):.2f}s")
    print(f"  HénonNet: Success={s_h:.1%}, mean da={np.mean(p2_stats['henon']['da']):.1%}, mean dp={np.mean(p2_stats['henon']['dp']):.1f}°, time={np.mean(p2_stats['henon']['time'])*1e3:.2f}ms")

    speedup_vs_de = float(np.mean(p2_stats['de']['time']) / (np.mean(p2_stats['henon']['time']) + 1e-9))
    print(f"  HénonNet speedup vs Differential Evolution: {speedup_vs_de:.1f}x")

    # ----------------------------------------------------------------------- Verdict Check
    # Pre-registered criteria:
    # HénonNet success >= 70% vs NLS < 30%
    # Speedup vs DE >= 50x
    confirmed = (s_h >= 0.70 and s_nls < 0.30 and speedup_vs_de >= 50.0)
    verdict = "CONFIRMED (ПІДТВЕРДЖЕНО)" if confirmed else "REFUTED (СПРОСТОВАНО)"

    print(f"\n========================================================================")
    print(f"VERDICT FOR TASK T6: {verdict}")
    print(f"========================================================================")

    # Save results
    results_out = {
        "verdict": verdict,
        "phase1_single_mode": {
            "nls_success": p1_stats["nls"]["ok"] / n_test,
            "de_success": p1_stats["de"]["ok"] / n_test,
            "henon_success": p1_stats["henon"]["ok"] / n_test,
        },
        "phase2_dual_mode": {
            "nls": {
                "success_rate": s_nls,
                "mean_rel_amp_err": float(np.mean(p2_stats["nls"]["da"])),
                "mean_phase_err_deg": float(np.mean(p2_stats["nls"]["dp"])),
                "mean_time_sec": float(np.mean(p2_stats["nls"]["time"])),
            },
            "de": {
                "success_rate": s_de,
                "mean_rel_amp_err": float(np.mean(p2_stats["de"]["da"])),
                "mean_phase_err_deg": float(np.mean(p2_stats["de"]["dp"])),
                "mean_time_sec": float(np.mean(p2_stats["de"]["time"])),
            },
            "henon": {
                "success_rate": s_h,
                "mean_rel_amp_err": float(np.mean(p2_stats["henon"]["da"])),
                "mean_phase_err_deg": float(np.mean(p2_stats["henon"]["dp"])),
                "mean_time_sec": float(np.mean(p2_stats["henon"]["time"])),
            },
            "speedup_henon_vs_de": speedup_vs_de,
        },
    }

    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(results_out, indent=2))
    print(f"Saved results to: {out_json}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Success Rates
    methods = ["NLS (TRF)", "Diff. Evolution", "HénonNet"]
    s_p1 = [p1_stats["nls"]["ok"]/n_test*100, p1_stats["de"]["ok"]/n_test*100, p1_stats["henon"]["ok"]/n_test*100]
    s_p2 = [s_nls*100, s_de*100, s_h*100]
    x_pos = np.arange(3)

    ax1 = axes[0]
    ax1.bar(x_pos - 0.2, s_p1, width=0.4, label="Single-Mode (m=2)", color="#4a90e2")
    ax1.bar(x_pos + 0.2, s_p2, width=0.4, label="Dual-Mode (2/1 + 3/1)", color="#50e3c2")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Reconstruction Success Rate")
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Errors (Dual-mode)
    ax2 = axes[1]
    da_means = [np.mean(p2_stats[m]["da"])*100 for m in ["nls", "de", "henon"]]
    dp_means = [np.mean(p2_stats[m]["dp"]) for m in ["nls", "de", "henon"]]
    ax2.bar(x_pos - 0.2, da_means, width=0.4, label="Rel. Amp Error (%)", color="#f5a623")
    ax2.bar(x_pos + 0.2, dp_means, width=0.4, label="Phase Error (deg)", color="#d0021b")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_title("Dual-Mode Average Errors")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Time comparison (log scale)
    ax3 = axes[2]
    times = [np.mean(p2_stats[m]["time"]) * 1000 for m in ["nls", "de", "henon"]]
    ax3.bar(methods, times, color=["#7f7f7f", "#e67e22", "#2ecc71"])
    ax3.set_yscale("log")
    ax3.set_ylabel("Time per instance (ms, log scale)")
    ax3.set_title("Runtime Comparison")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = HERE / "t6_henon_inverse.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    run_benchmark()
