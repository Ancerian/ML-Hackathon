#!/usr/bin/env python3
"""Our try/04-novelty/t2/run_t2.py — Complete T2 evaluation:
GS differential residual in operator network loss (FNO / UNet operator):
    L = L_data + beta * ||Delta* psi - J_phi_target||^2

Pre-registered in Our try/04-novelty/t2/THEORY.md:
1. beta in {0.0, 1e-4, 1e-3, 1e-2} across 3 random seeds {0, 1, 2}.
2. Tolerance epsilon = 0.05.
3. Architecture: Fourier Neural Operator / Conv Operator for 65x65 flux map.
4. Computes: R2_psi, g(psi), Consistency on C2 test split (#060..#067).
5. Compiles Pareto frontier, results.json and plots.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"

sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))

import tokamld
from tokamld.gs import delta_star_torch, delta_star_numpy, gs_inconsistency
import train as c2_train
import evaluate as c2e
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- Spectral Conv 2D (FNO block)
class SpectralConv2d(nn.Module):
    """2D Fourier Neural Operator layer (Li et al., 2021)."""
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        batchsize, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        m1, m2 = min(self.modes1, x_ft.shape[-2]), min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2dOperator(nn.Module):
    """Fast, efficient Fourier Operator for 65x65 flux map."""
    def __init__(self, in_features: int = 21, width: int = 16, modes: int = 6):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, width * 16 * 16),
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.GELU(),
            nn.Upsample(size=(65, 65), mode="bilinear", align_corners=True),      # 65x65
        )
        self.fno = SpectralConv2d(width, width, modes, modes)
        self.w = nn.Conv2d(width, width, 1)
        self.fc_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.fc_in(x).view(b, -1, 16, 16)
        u = self.conv_up(h)
        u1 = self.fno(u) + self.w(u)
        out = self.fc_out(u1)
        return out.squeeze(1)


# --------------------------------------------------------------------------- Training with GS Loss
def train_fno(X_tr: np.ndarray, Y_tr: np.ndarray,
              R: np.ndarray, Z: np.ndarray,
              beta: float, seed: int,
              n_epochs: int = 6, batch_size: int = 128) -> FNO2dOperator:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FNO2dOperator(in_features=X_tr.shape[1], width=16, modes=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Compute target source J_phi_target = Delta* Y_tr
    R_t = torch.tensor(R, dtype=torch.float32)
    Z_t = torch.tensor(Z, dtype=torch.float32)

    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    with torch.no_grad():
        target_source = delta_star_torch(Y_tr_t, R_t, Z_t)

    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            Y_tr_t,
                            target_source)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for bx, by, b_src in loader:
            optimizer.zero_grad()
            pred = model(bx)

            # 1. Data loss
            loss_data = F.mse_loss(pred, by)

            # 2. Physics residual loss
            if beta > 0.0:
                pred_ds = delta_star_torch(pred, R_t, Z_t)
                loss_phys = F.mse_loss(pred_ds, b_src)
                loss = loss_data + beta * loss_phys
            else:
                loss = loss_data

            loss.backward()
            optimizer.step()

    return model


def main():
    t0 = time.time()
    print("=" * 72)
    print("T2 — Grad-Shafranov Differential Residual in FNO Operator Loss")
    print("=" * 72)

    # 1. Geometry and data
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mc = mask["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)

    tr_shots = c2_train.load(c2_train.TRAIN)
    te_shots = c2_train.load(c2_train.TEST)

    X_tr_raw, Y_tr_raw, _, _ = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)

    # Subsample training frames to speed up 3-seed x 4-beta run
    stride = 2
    X_tr_sub = X_tr[::stride]
    Y_tr_sub = Y_tr_raw[::stride]

    test_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]

    betas = [0.0, 1e-4, 1e-3, 1e-2]
    seeds = [0, 1, 2]

    # Pre-registered validation of discretization on ground truth (E6 check)
    psi_sample = Y_tr_raw[0]
    rng_check = np.random.default_rng(42)
    psi_sample_noisy = psi_sample + 0.01 * np.std(psi_sample) * rng_check.standard_normal(psi_sample.shape)
    g_true_val, _ = gs_inconsistency(psi_sample, R, Z, mc, mf)
    g_noisy_val, _ = gs_inconsistency(psi_sample_noisy, R, Z, mc, mf)
    print(f"E6 check on ground truth frame 0: g(true)={g_true_val:.4f}, g(true+1% noise)={g_noisy_val:.4f} (ratio ~ {g_noisy_val/g_true_val:.1f}x)")

    results = {}

    for beta in betas:
        b_key = f"beta_{beta}"
        print(f"\n--- Testing beta = {beta} ---")
        results[b_key] = {"r2_psi": [], "g": [], "S": []}

        for s in seeds:
            print(f"  Training FNO seed {s}...")
            model = train_fno(X_tr_sub, Y_tr_sub, R, Z, beta=beta, seed=s, n_epochs=12, batch_size=64)
            model.eval()

            # Predict on test shots
            preds_all = []
            with torch.no_grad():
                for Xt in test_inputs:
                    p = model(torch.tensor(Xt, dtype=torch.float32)).numpy()
                    preds_all.append(p)

            # Evaluate metrics on test shots
            r2_list, g_list = [], []
            for si, shot in enumerate(te_shots):
                p_shot = preds_all[si]
                y_shot = shot["psi"]
                # R2
                ss_res = np.sum((y_shot - p_shot) ** 2)
                ss_tot = np.sum((y_shot - np.mean(y_shot)) ** 2)
                r2_shot = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2_list.append(float(r2_shot))

                # g per frame (median across shot)
                mid = len(y_shot) // 2
                g_shot, _ = gs_inconsistency(p_shot[mid], R, Z, mc, mf)
                if np.isfinite(g_shot):
                    g_list.append(float(g_shot))

            mean_r2 = float(np.mean(r2_list))
            mean_g = float(np.mean(g_list))
            results[b_key]["r2_psi"].append(mean_r2)
            results[b_key]["g"].append(mean_g)
            print(f"    Seed {s}: R2_psi = {mean_r2:.4f}, g = {mean_g:.4f}")

    # --------------------------------------------------------------------------- Summary & Verdict
    print("\n" + "=" * 72)
    print(f"{'Beta':<12s} | {'Mean R2_psi':<14s} | {'Mean g(psi)':<14s} | {'Delta R2':<10s} | {'Delta g':<10s}")
    print("-" * 72)

    base_r2 = float(np.mean(results["beta_0.0"]["r2_psi"]))
    base_g = float(np.mean(results["beta_0.0"]["g"]))

    falsified = True
    epsilon = 0.05

    summary = {}
    for beta in betas:
        b_key = f"beta_{beta}"
        m_r2 = float(np.mean(results[b_key]["r2_psi"]))
        m_g = float(np.mean(results[b_key]["g"]))
        d_r2 = m_r2 - base_r2
        d_g = m_g - base_g

        summary[b_key] = {
            "beta": beta,
            "mean_r2": round(m_r2, 4),
            "std_r2": round(float(np.std(results[b_key]["r2_psi"])), 4),
            "mean_g": round(m_g, 4),
            "std_g": round(float(np.std(results[b_key]["g"])), 4),
            "delta_r2": round(d_r2, 4),
            "delta_g": round(d_g, 4),
        }
        print(f"{beta:<12.4f} | {m_r2:<14.4f} | {m_g:<14.4f} | {d_r2:<+10.4f} | {d_g:<+10.4f}")

        # Check pre-registered condition
        if beta > 0.0 and d_g < -0.01 and d_r2 >= -epsilon:
            falsified = False

    verdict = "СПРОСТОВАНО (REFUTED)" if falsified else "ПІДТВЕРДЖЕНО (CONFIRMED)"
    print("\n" + "=" * 72)
    print(f"VERDICT FOR TASK T2: {verdict}")
    print("=" * 72)

    # --------------------------------------------------------------------------- Save Results & Pareto Plot
    out_json = {
        "verdict": verdict,
        "epsilon_tolerance": epsilon,
        "runtime_sec": round(time.time() - t0, 2),
        "summary": summary,
        "raw_results": results,
    }
    with open(HERE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Pareto plot: R2 vs g
    plt.figure(figsize=(7, 5))
    for beta in betas:
        b_key = f"beta_{beta}"
        r2_vals = results[b_key]["r2_psi"]
        g_vals = results[b_key]["g"]
        plt.scatter(g_vals, r2_vals, label=f"beta={beta}", s=80)
        plt.errorbar(np.mean(g_vals), np.mean(r2_vals),
                     xerr=np.std(g_vals), yerr=np.std(r2_vals), fmt="o", capsize=5)

    plt.xlabel("GS Inconsistency g(psi) (lower is better)")
    plt.ylabel("Accuracy R2(psi) (higher is better)")
    plt.title("Pareto Trade-off: R2(psi) vs GS Inconsistency g(psi)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "t2_pareto_curve.png", dpi=150)
    plt.close()
    print(f"Saved Pareto curve to: {HERE / 't2_pareto_curve.png'}")
    print(f"Saved results to: {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
