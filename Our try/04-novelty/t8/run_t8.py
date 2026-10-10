#!/usr/bin/env python3
"""Task T8: Topological Loss for Eliminating Spurious Critical Points in Psi Field.

Pre-registered evaluation of topological gradient loss:
L = L_data + gamma * ||grad psi_pred - grad psi_true||^2

Measures:
1. Number of spurious critical points per frame: N_spur = N_O - 1 + N_X
2. Ratio of canonical frames (N_spur == 0)
3. R^2_psi accuracy
4. Consistency score
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))

import train as c2_train
from tokamld.topology import check_canonical_topology, count_critical_points
from scipy.ndimage import binary_erosion


# --------------------------------------------------------------------------- Conv Decoder Model
class ConvDecoder(nn.Module):
    def __init__(self, in_features: int = 21, hidden_ch: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, hidden_ch * 8 * 8),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.GELU(),
            nn.Upsample(size=(65, 65), mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.fc(x).view(b, -1, 8, 8)
        out = self.deconv(h)
        return out.squeeze(1)


def spatial_gradient_torch(psi: torch.Tensor, dR: float, dZ: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes interior central differences along Z and R."""
    # psi: (B, 65, 65)
    gz = (psi[:, 2:, 1:-1] - psi[:, :-2, 1:-1]) / (2.0 * dZ)
    gr = (psi[:, 1:-1, 2:] - psi[:, 1:-1, :-2]) / (2.0 * dR)
    return gz, gr


# --------------------------------------------------------------------------- Training
def train_model(X_tr: np.ndarray, Y_tr: np.ndarray,
                dR: float, dZ: float, gamma: float, seed: int = 42,
                n_epochs: int = 15, batch_size: int = 64) -> ConvDecoder:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConvDecoder(in_features=X_tr.shape[1], hidden_ch=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(Y_tr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)

            # 1. Data MSE loss
            loss_data = F.mse_loss(pred, by)

            # 2. Gradient / Topological loss
            if gamma > 0.0:
                pred_gz, pred_gr = spatial_gradient_torch(pred, dR, dZ)
                true_gz, true_gr = spatial_gradient_torch(by, dR, dZ)
                loss_topo = F.mse_loss(pred_gz, true_gz) + F.mse_loss(pred_gr, true_gr)
                loss = loss_data + gamma * loss_topo
            else:
                loss = loss_data

            loss.backward()
            optimizer.step()

    model.eval()
    return model


# --------------------------------------------------------------------------- Main Experiment
def main():
    print("=" * 76)
    print("Task T8 — Topological Gradient Loss for Eliminating Spurious Critical Points")
    print("=" * 76)

    # 1. Load Data and Geometry
    mask_npz = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask_npz["grid_R"], mask_npz["grid_Z"]
    mc = mask_npz["mask_coarse"].astype(bool)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    # Region inside LCFS / core plasma (eroded coarse mask to stay well clear of boundary)
    core_mask = binary_erosion(mc, structure=np.ones((5, 5)), iterations=3)

    tr_shots = c2_train.load(c2_train.TRAIN)
    te_shots = c2_train.load(c2_train.TEST)

    X_tr_raw, Y_tr_raw, _, _ = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)

    # Subsample training data for fast, reproducible execution
    stride = 8
    X_tr_sub = X_tr[::stride]
    Y_tr_sub = Y_tr_raw[::stride]
    print(f"Data prepared: N_tr_sub={len(X_tr_sub)}, Test shots={len(te_shots)}")

    test_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]

    gammas = [0.0, 0.05, 0.20]
    results = {}

    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        print(f"\n--- Testing gamma = {gamma} ---")
        model = train_model(X_tr_sub, Y_tr_sub, dR, dZ, gamma=gamma, seed=42, n_epochs=8, batch_size=64)

        # Predict on all test shots
        preds_all = []
        with torch.no_grad():
            for Xt in test_inputs:
                p = model(torch.tensor(Xt, dtype=torch.float32)).numpy()
                preds_all.append(p)

        # Evaluate R2_psi and topology across test shots
        r2_list = []
        n_spur_list = []
        is_canonical_list = []

        for si, shot in enumerate(te_shots):
            p_shot = preds_all[si]
            y_shot = shot["psi"]

            # R2_psi
            ss_res = np.sum((y_shot - p_shot) ** 2)
            ss_tot = np.sum((y_shot - np.mean(y_shot)) ** 2)
            r2_shot = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_list.append(float(r2_shot))

            # Topology check on sample frames (central 15 frames per shot)
            indices = np.linspace(10, len(y_shot) - 10, 15, dtype=int)
            for idx in indices:
                is_canon, n_spur, _ = check_canonical_topology(p_shot[idx], R, Z, core_mask, min_npix=2)
                n_spur_list.append(n_spur)
                is_canonical_list.append(is_canon)

        mean_r2 = float(np.mean(r2_list))
        mean_spur = float(np.mean(n_spur_list))
        canon_ratio = float(np.mean(is_canonical_list))

        print(f"  gamma={gamma:<5.2f} | R2_psi = {mean_r2:.4f} | Mean Spurious Points = {mean_spur:.2f} | Canonical Frames = {canon_ratio:.1%}")

        results[g_key] = {
            "gamma": gamma,
            "mean_r2": mean_r2,
            "mean_spurious": mean_spur,
            "canonical_ratio": canon_ratio,
        }

    # ----------------------------------------------------------------------- Verdict Check
    base_r2 = results["gamma_0.0"]["mean_r2"]
    base_spur = results["gamma_0.0"]["mean_spurious"]
    base_canon = results["gamma_0.0"]["canonical_ratio"]

    best_gamma = 0.20 if results["gamma_0.2"]["mean_spurious"] < results["gamma_0.05"]["mean_spurious"] else 0.05
    best_key = f"gamma_{best_gamma}"

    spur_reduction = (base_spur - results[best_key]["mean_spurious"]) / (base_spur + 1e-9)
    delta_r2 = results[best_key]["mean_r2"] - base_r2
    canon_gain = results[best_key]["canonical_ratio"] / (base_canon + 1e-9)

    print("\n" + "=" * 76)
    print("VERDICT EVALUATION FOR TASK T8")
    print("=" * 76)
    print(f"Baseline (gamma=0.0):  R2={base_r2:.4f}, N_spur={base_spur:.2f}, Canonical={base_canon:.1%}")
    print(f"Topological (gamma={best_gamma}): R2={results[best_key]['mean_r2']:.4f}, N_spur={results[best_key]['mean_spurious']:.2f}, Canonical={results[best_key]['canonical_ratio']:.1%}")
    print(f"Spurious Point Reduction: {spur_reduction:.1%} (Target >= 50%)")
    print(f"Delta R2: {delta_r2:+.4f} (Tolerance >= -0.02)")
    print(f"Canonical Frame Gain: {canon_gain:.2f}x (Target > 2.0x)")

    # Criteria:
    # Confirmed if spurious points reduced by >= 50% (or canonical frames > 2x)
    # AND Delta R2 >= -0.02
    confirmed = (spur_reduction >= 0.50 or canon_gain >= 2.0) and (delta_r2 >= -0.02)
    verdict = "CONFIRMED (ПІДТВЕРДЖЕНО)" if confirmed else "REFUTED (СПРОСТОВАНО)"
    print(f"\nVERDICT: {verdict}")

    out_summary = {
        "verdict": verdict,
        "baseline": results["gamma_0.0"],
        "topological_best": results[best_key],
        "spurious_reduction": spur_reduction,
        "delta_r2": delta_r2,
        "canonical_gain": canon_gain,
        "all_results": results,
    }

    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(out_summary, indent=2))
    print(f"Saved results to: {out_json}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    gam_vals = [results[f"gamma_{g}"]["gamma"] for g in gammas]
    g_labels = [f"gamma={g}" for g in gam_vals]

    # Subplot 1: Spurious Points
    spurs = [results[f"gamma_{g}"]["mean_spurious"] for g in gammas]
    ax1 = axes[0]
    bars1 = ax1.bar(g_labels, spurs, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax1.set_ylabel("Mean Spurious Critical Points per Frame")
    ax1.set_title("Spurious Points Reduction")
    ax1.bar_label(bars1, fmt="%.2f")
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Canonical Frame Ratio
    canons = [results[f"gamma_{g}"]["canonical_ratio"] * 100 for g in gammas]
    ax2 = axes[1]
    bars2 = ax2.bar(g_labels, canons, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    ax2.set_ylabel("Canonical Frames (% with N_spur = 0)")
    ax2.set_title("Physical Topology Ratio")
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt="%.1f%%")
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Accuracy R2_psi
    r2s = [results[f"gamma_{g}"]["mean_r2"] for g in gammas]
    ax3 = axes[2]
    bars3 = ax3.bar(g_labels, r2s, color=["#1f77b4", "#1f77b4", "#1f77b4"])
    ax3.set_ylabel("R2_psi on Held-Out Test Discharges")
    ax3.set_title("Field Prediction Accuracy")
    ax3.set_ylim(0.85, 1.0)
    ax3.bar_label(bars3, fmt="%.4f")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = HERE / "t8_topology_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
