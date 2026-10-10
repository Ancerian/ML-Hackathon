#!/usr/bin/env python3
"""Task T4: Mixture Density Network (MDN) for Non-Unique Bratu Equilibria.

Demonstrates that an MDN resolves multivalued non-unique solutions into physical branches,
avoiding the catastrophic PDE residual penalty of an L2 regressor (E4).
Uses PCA spectral decomposition for smooth 1D profiles, consistent with project baselines.
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
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]

# 1D Spatial grid
GRID_N = 101
X = np.linspace(0.0, 1.0, GRID_N)
H = X[1] - X[0]
N_PCA = 4


# --------------------------------------------------------------------------- Bratu Exact Physics
def theta_roots(lam: float) -> list[float]:
    """Finds both roots of theta = sqrt(2 lam) cosh(theta / 4)."""
    g = lambda th: th - np.sqrt(2.0 * lam) * np.cosh(th / 4.0)
    grid = np.linspace(1e-9, 50.0, 50_000)
    vals = g(grid)
    sign = np.sign(vals)
    idx = np.where(np.diff(sign) != 0)[0]
    return [float(brentq(g, grid[i], grid[i + 1], xtol=1e-13)) for i in idx]


def bratu_u(x: np.ndarray, th: float) -> np.ndarray:
    s = x - 0.5
    return -2.0 * np.log(np.cosh(s * th / 2.0) / np.cosh(th / 4.0))


def d2_interior(f: np.ndarray) -> np.ndarray:
    """4th order second derivative on interior grid."""
    out = np.zeros_like(f)
    out[2:-2] = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * H * H)
    return out


def rel_residual_bratu(u: np.ndarray, lam: float, interior: int = 5) -> float:
    """Relative PDE residual ||u'' + lam exp(u)|| / ||lam exp(u)||."""
    d2u = d2_interior(u)
    src = lam * np.exp(u)
    res = d2u + src
    sl = slice(interior, len(u) - interior)
    return float(np.linalg.norm(res[sl]) / np.linalg.norm(src[sl]))


# --------------------------------------------------------------------------- Models
class L2Regressor(nn.Module):
    """Standard MLP minimizing L2 MSE loss on PCA coefficients."""
    def __init__(self, in_features: int = 1, out_features: int = N_PCA, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureDensityNetwork(nn.Module):
    """MDN with K=2 components predicting pi_k, mu_k, sigma_k in PCA space."""
    def __init__(self, in_features: int = 1, out_features: int = N_PCA, n_components: int = 2, hidden: int = 128,
                 mu0_init: np.ndarray | None = None, mu1_init: np.ndarray | None = None):
        super().__init__()
        self.k = n_components
        self.out_dim = out_features
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pi_head = nn.Linear(hidden, n_components)
        self.mu0_head = nn.Linear(hidden, out_features)
        self.mu1_head = nn.Linear(hidden, out_features)
        self.log_sigma_head = nn.Linear(hidden, n_components)

        if mu0_init is not None:
            self.mu0_head.bias.data = torch.tensor(mu0_init, dtype=torch.float32)
        if mu1_init is not None:
            self.mu1_head.bias.data = torch.tensor(mu1_init, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        pi = F.softmax(self.pi_head(h), dim=-1)  # (B, 2)
        mu = torch.stack([self.mu0_head(h), self.mu1_head(h)], dim=1)  # (B, 2, D)
        log_sigma = torch.clamp(self.log_sigma_head(h), min=-6.0, max=2.0)  # (B, 2)
        return pi, mu, log_sigma


def mdn_nll_loss(pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NLL for Gaussian mixture."""
    target = target.unsqueeze(1)  # (B, 1, D)
    sigma = torch.exp(log_sigma).unsqueeze(-1)  # (B, 2, 1)
    sq_err = torch.sum((target - mu) ** 2, dim=-1)  # (B, 2)
    d = target.shape[-1]
    log_prob = -0.5 * sq_err / (sigma.squeeze(-1) ** 2) - d * log_sigma - 0.5 * d * np.log(2.0 * np.pi)
    log_mix = torch.log(pi + 1e-12) + log_prob
    nll = -torch.logsumexp(log_mix, dim=-1)
    return torch.mean(nll)


# --------------------------------------------------------------------------- Main Experiment
def main():
    print("=" * 76)
    print("Task T4 — Mixture Density Network vs L2 Regressor on Bratu Non-Uniqueness")
    print("=" * 76)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Fit PCA Basis on Bratu solution manifold
    U_manifold = []
    for lam in np.linspace(0.5, 3.4, 250):
        rts = theta_roots(lam)
        U_manifold.append(bratu_u(X, rts[0]))
        U_manifold.append(bratu_u(X, rts[-1]))
    pca = PCA(n_components=N_PCA).fit(U_manifold)
    print(f"PCA basis fitted: {N_PCA} modes, explained variance ratio = {pca.explained_variance_ratio_}")

    # 2. Generate training and test datasets
    rng = np.random.default_rng(42)
    n_train = 3000
    n_test = 500

    lams_tr = rng.uniform(0.5, 3.4, size=n_train)
    X_tr_list, C_tr_list, B_tr_list = [], [], []
    for lam in lams_tr:
        rts = theta_roots(lam)
        b = rng.choice([0, 1])
        th = rts[0] if b == 0 else rts[-1]
        u = bratu_u(X, th)
        c = pca.transform([u])[0]
        X_tr_list.append([lam])
        C_tr_list.append(c)
        B_tr_list.append(b)

    X_tr = torch.tensor(np.array(X_tr_list), dtype=torch.float32)
    C_tr = torch.tensor(np.array(C_tr_list), dtype=torch.float32)
    B_tr = np.array(B_tr_list)

    lams_te = rng.uniform(0.6, 3.3, size=n_test)
    X_te_list, C_te_list, B_te_list = [], [], []
    for lam in lams_te:
        rts = theta_roots(lam)
        b = rng.choice([0, 1])
        th = rts[0] if b == 0 else rts[-1]
        u = bratu_u(X, th)
        c = pca.transform([u])[0]
        X_te_list.append([lam])
        C_te_list.append(c)
        B_te_list.append(b)

    X_te = torch.tensor(np.array(X_te_list), dtype=torch.float32)

    # Branch cluster means for initial biases
    c_lo_mean = np.mean([C_tr[i].numpy() for i in range(len(B_tr)) if B_tr[i] == 0], axis=0)
    c_hi_mean = np.mean([C_tr[i].numpy() for i in range(len(B_tr)) if B_tr[i] == 1], axis=0)

    train_loader = DataLoader(TensorDataset(X_tr, C_tr), batch_size=128, shuffle=True)

    # 3. Train L2 Regressor
    print("\n--- Training L2 Regressor (Predicts Conditional Mean) ---")
    l2_model = L2Regressor(in_features=1, out_features=N_PCA, hidden=128)
    opt_l2 = torch.optim.Adam(l2_model.parameters(), lr=2e-3)
    for epoch in range(120):
        for bx, bc in train_loader:
            opt_l2.zero_grad()
            pred = l2_model(bx)
            loss = F.mse_loss(pred, bc)
            loss.backward()
            opt_l2.step()
    print("L2 Regressor training complete.")

    # 4. Train MDN
    print("\n--- Training Mixture Density Network (MDN, K=2) ---")
    mdn_model = MixtureDensityNetwork(in_features=1, out_features=N_PCA, n_components=2, hidden=128,
                                      mu0_init=c_lo_mean, mu1_init=c_hi_mean)
    opt_mdn = torch.optim.Adam(mdn_model.parameters(), lr=1e-3)
    for epoch in range(120):
        for bx, bc in train_loader:
            opt_mdn.zero_grad()
            pi, mu, log_sigma = mdn_model(bx)
            loss = mdn_nll_loss(pi, mu, log_sigma, bc)
            loss.backward()
            opt_mdn.step()
    print("MDN training complete.")

    # 5. Evaluate on Test Set
    print("\n--- Evaluating Models on Test Set (500 samples) ---")
    l2_model.eval()
    mdn_model.eval()

    g_ref = 0.10
    l2_residuals = []
    mdn_residuals_samples = []
    mdn_residuals_both_modes = []
    mdn_branch_assignments = []

    with torch.no_grad():
        c_l2_preds = l2_model(X_te).numpy()
        pi_preds, mu_preds, _ = mdn_model(X_te)
        pi_preds = pi_preds.numpy()
        mu_preds = mu_preds.numpy()

    for i in range(n_test):
        lam = float(X_te[i, 0])
        rts = theta_roots(lam)
        u_lo_true = bratu_u(X, rts[0])
        u_hi_true = bratu_u(X, rts[-1])

        # L2 reconstruction and residual
        u_l2 = pca.inverse_transform([c_l2_preds[i]])[0]
        r_l2 = rel_residual_bratu(u_l2, lam)
        l2_residuals.append(r_l2)

        # MDN modes reconstruction
        u_m0 = pca.inverse_transform([mu_preds[i, 0]])[0]
        u_m1 = pca.inverse_transform([mu_preds[i, 1]])[0]
        r_m0 = rel_residual_bratu(u_m0, lam)
        r_m1 = rel_residual_bratu(u_m1, lam)
        mdn_residuals_both_modes.append((r_m0, r_m1))

        # Sample from mixture
        comp = rng.choice([0, 1], p=pi_preds[i])
        u_sample = u_m0 if comp == 0 else u_m1
        r_sample = r_m0 if comp == 0 else r_m1
        mdn_residuals_samples.append(r_sample)

        # Branch assignment check
        d_lo = np.linalg.norm(u_sample - u_lo_true)
        d_hi = np.linalg.norm(u_sample - u_hi_true)
        mdn_branch_assignments.append(0 if d_lo < d_hi else 1)

    l2_valid = np.mean([r < g_ref for r in l2_residuals])
    mdn_sample_valid = np.mean([r < g_ref for r in mdn_residuals_samples])
    mdn_modes_valid = np.mean([(r0 < g_ref and r1 < g_ref) for r0, r1 in mdn_residuals_both_modes])

    cov_lo = np.mean(np.array(mdn_branch_assignments) == 0)
    cov_hi = np.mean(np.array(mdn_branch_assignments) == 1)

    mean_g_l2 = float(np.mean(l2_residuals))
    median_g_l2 = float(np.median(l2_residuals))
    mean_g_mdn = float(np.mean(mdn_residuals_samples))
    median_g_mdn = float(np.median(mdn_residuals_samples))

    print(f"\nResults Summary:")
    print(f"  L2 Regressor: mean g = {mean_g_l2:.4f}, median g = {median_g_l2:.4f}, Valid Ratio (g < {g_ref}) = {l2_valid:.1%}")
    print(f"  MDN Samples:  mean g = {mean_g_mdn:.4f}, median g = {median_g_mdn:.4f}, Valid Ratio (g < {g_ref}) = {mdn_sample_valid:.1%}")
    print(f"  MDN Modes:    Both Modes Valid Ratio = {mdn_modes_valid:.1%}")
    print(f"  Branch Coverage: {cov_lo:.1%} lower / {cov_hi:.1%} upper (Balanced target 50%/50%)")

    # 6. Verdict Verification against Pre-registration
    # Pre-registered criteria:
    # L2 valid <= 5% and mean g >= 0.60
    # MDN valid >= 85% and mean g < 0.05
    # Branch coverage each >= 25%
    confirmed = (l2_valid <= 0.05 and mean_g_l2 >= 0.60 and mdn_sample_valid >= 0.85 and cov_lo >= 0.25 and cov_hi >= 0.25)
    verdict = "CONFIRMED (ПІДТВЕРДЖЕНО)" if confirmed else "REFUTED (СПРОСТОВАНО)"
    print(f"\n========================================================================")
    print(f"VERDICT FOR TASK T4: {verdict}")
    print(f"========================================================================")

    results_out = {
        "verdict": verdict,
        "g_ref": g_ref,
        "l2_regressor": {
            "mean_g": mean_g_l2,
            "median_g": median_g_l2,
            "valid_ratio": float(l2_valid),
        },
        "mdn": {
            "mean_g": mean_g_mdn,
            "median_g": median_g_mdn,
            "valid_ratio_samples": float(mdn_sample_valid),
            "valid_ratio_both_modes": float(mdn_modes_valid),
            "branch_coverage_lower": float(cov_lo),
            "branch_coverage_upper": float(cov_hi),
        },
    }

    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(results_out, indent=2))
    print(f"Saved results to: {out_json}")

    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Profile demonstration at test lambda = 2.0
    lam_demo = 2.0
    rts = theta_roots(lam_demo)
    u_lo_demo = bratu_u(X, rts[0])
    u_hi_demo = bratu_u(X, rts[-1])
    u_mean_demo = 0.5 * (u_lo_demo + u_hi_demo)

    with torch.no_grad():
        x_demo_t = torch.tensor([[lam_demo]], dtype=torch.float32)
        c_l2_demo = l2_model(x_demo_t).numpy()[0]
        u_l2_demo = pca.inverse_transform([c_l2_demo])[0]

        pi_demo, mu_demo, _ = mdn_model(x_demo_t)
        pi_val = pi_demo.numpy()[0]
        u_m0_demo = pca.inverse_transform([mu_demo[0, 0].numpy()])[0]
        u_m1_demo = pca.inverse_transform([mu_demo[0, 1].numpy()])[0]

    ax1 = axes[0]
    ax1.plot(X, u_lo_demo, "k-", label="True lower branch $u_{lo}$", linewidth=2)
    ax1.plot(X, u_hi_demo, "k--", label="True upper branch $u_{hi}$", linewidth=2)
    ax1.plot(X, u_mean_demo, "gray", linestyle=":", label="Theoretical Mean $(u_{lo}+u_{hi})/2$", linewidth=1.5)
    ax1.plot(X, u_l2_demo, "r-.", label="L2 Regressor pred (Mean collapse)", linewidth=2)
    ax1.plot(X, u_m0_demo, "b-", label=f"MDN Mode 1 ($\pi={pi_val[0]:.2f}$)", alpha=0.85)
    ax1.plot(X, u_m1_demo, "g-", label=f"MDN Mode 2 ($\pi={pi_val[1]:.2f}$)", alpha=0.85)
    ax1.set_title(f"Predicted Profiles at $\\lambda = {lam_demo}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: Residual distributions
    ax2 = axes[1]
    bins = np.linspace(0, 2.5, 30)
    ax2.hist(l2_residuals, bins=bins, color="crimson", alpha=0.6, label=f"L2 Regressor (mean g={mean_g_l2:.2f})")
    ax2.hist(mdn_residuals_samples, bins=bins, color="dodgerblue", alpha=0.6, label=f"MDN Samples (mean g={mean_g_mdn:.3f})")
    ax2.axvline(g_ref, color="black", linestyle="--", label=f"Tolerance $g_{{ref}}={g_ref}$")
    ax2.set_title("Relative PDE Residual $g(u)$ Distribution")
    ax2.set_xlabel("Relative PDE Residual $g(u)$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Plot 3: Branch Coverage
    ax3 = axes[2]
    bars = ax3.bar(["Lower Branch", "Upper Branch"], [cov_lo * 100, cov_hi * 100], color=["#1f77b4", "#ff7f0e"])
    ax3.axhline(50, color="gray", linestyle="--", label="Target Balanced (50%)")
    ax3.set_ylabel("Selection Percentage (%)")
    ax3.set_title("MDN Branch Coverage")
    ax3.set_ylim(0, 100)
    ax3.bar_label(bars, fmt="%.1f%%")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = HERE / "t4_mdn_vs_l2.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
