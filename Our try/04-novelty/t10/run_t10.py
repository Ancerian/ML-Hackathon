#!/usr/bin/env python3
"""Task T10: Evaluating and Enhancing Model Robustness to Diagnostic Degradation.

Pre-registered in THEORY.md:
1. Degradation model:
   (a) Channel dropout (p in [0.0, 0.30])
   (b) Integrator drift (alpha_drift = 0.20)
   (c) Combined
2. PyTorch augmentations (random mask + drift) vs Unaugmented Baseline.
   No leakage: Calibration on independent validation shots (36..39), test on 60..67.
3. Degradation curves: S, S', g(psi), and Simultaneous Conformal Coverage.
4. Conformal breakdown threshold determination.
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
from tokamld.conformal import SimultaneousConformalBands
from tokamld.gs import gs_inconsistency, gate, DEFAULT_G_REF
from tokamld.topology import inside_lcfs, count_critical_points


# --------------------------------------------------------------------------- Model Architecture
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


# --------------------------------------------------------------------------- Training
def train_model(X_tr: np.ndarray, Y_tr: np.ndarray,
                augment: bool = False, seed: int = 42,
                n_epochs: int = 12, batch_size: int = 64) -> ConvDecoder:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_tr.shape[1]
    model = ConvDecoder(in_features=in_dim, hidden_ch=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(Y_tr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        for bx, by in loader:
            optimizer.zero_grad()

            if augment:
                bx_aug = bx.clone()
                b_sz = bx_aug.shape[0]
                # 1. Random channel dropout (up to 30%, i.e. 0 to 6 channels)
                for i in range(b_sz):
                    if torch.rand(1).item() < 0.75:
                        k = torch.randint(1, 7, (1,)).item()
                        drop_idx = torch.randperm(in_dim)[:k]
                        bx_aug[i, drop_idx] = 0.0
                # 2. Random synthetic drift
                if torch.rand(1).item() < 0.60:
                    drift_scale = torch.empty((b_sz, in_dim), device=bx.device).uniform_(-0.20, 0.20)
                    bx_aug = bx_aug + drift_scale
                pred = model(bx_aug)
            else:
                pred = model(bx)

            loss = F.mse_loss(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# --------------------------------------------------------------------------- Degradation Simulation
def apply_degradation(X_clean: np.ndarray, p_drop: float = 0.0,
                      alpha_drift: float = 0.0, seed: int = 42) -> np.ndarray:
    """Applies channel dropout and linear integrator drift to shot features."""
    T, D = X_clean.shape
    X_deg = X_clean.copy()
    rng = np.random.default_rng(seed)

    # 1. Channel Dropout
    if p_drop > 0.0:
        k = int(round(p_drop * D))
        if k > 0:
            drop_indices = rng.choice(D, size=k, replace=False)
            X_deg[:, drop_indices] = 0.0

    # 2. Integrator Drift (linear accumulation over shot duration)
    if alpha_drift > 0.0:
        xi = rng.normal(0.0, 1.0, size=D)
        t_norm = np.linspace(0.0, 1.0, T)[:, None]
        drift = alpha_drift * t_norm * xi[None, :]
        X_deg += drift

    return X_deg


# --------------------------------------------------------------------------- Evaluation
def evaluate_condition(model: ConvDecoder, band: SimultaneousConformalBands,
                       test_shots: list[dict], test_inputs_clean: list[np.ndarray],
                       p_drop: float, alpha_drift: float,
                       R: np.ndarray, Z: np.ndarray,
                       mask_coarse: np.ndarray, mask_f: np.ndarray) -> dict:
    """Evaluates model performance and conformal coverage under degradation."""
    preds_all = []
    y_true_all = []
    r2_shots = []

    # Predict degraded shots
    with torch.no_grad():
        for si, (shot, X_clean) in enumerate(zip(test_shots, test_inputs_clean)):
            # Unique deterministic degradation seed per shot
            deg_seed = 1000 + si * 73 + int(p_drop * 1000)
            X_deg = apply_degradation(X_clean, p_drop=p_drop, alpha_drift=alpha_drift, seed=deg_seed)

            pred = model(torch.tensor(X_deg, dtype=torch.float32)).numpy()
            y_true = shot["psi"]
            preds_all.append(pred)
            y_true_all.append(y_true)

            # R2 per shot
            ss_res = np.sum((y_true - pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_s = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_shots.append(float(r2_s))

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(preds_all)
    mean_r2 = float(np.mean(r2_shots))

    # 1. Conformal Coverage (Simultaneous max studentized residual <= c*)
    res = np.abs(y_true_concat - y_pred_concat)
    max_rel = np.max(res[:, mask_coarse] / band.scale_map[mask_coarse], axis=1)
    cov_M = float(np.mean(max_rel <= band.c_star))

    # 2. Physics & Topological Metrics (on 15 representative frames per shot)
    g_list = []
    p_topo_list = []
    for si, (shot, pred) in enumerate(zip(test_shots, preds_all)):
        T = len(shot["psi"])
        indices = np.linspace(5, T - 5, min(15, T - 10), dtype=int)
        for idx in indices:
            # GS residual
            g_val, _ = gs_inconsistency(pred[idx], R, Z, mask_coarse, mask_f)
            g_list.append(g_val if np.isfinite(g_val) else 1.0)

            # Topology
            ins = inside_lcfs(pred[idx], R, Z, mask_coarse, mask_f)
            if ins is not None and ins.sum() >= 50:
                n_O, n_X = count_critical_points(pred[idx], R, Z, ins, min_npix=3)
                n_spur = abs(n_O - 1) + n_X
                p_topo_list.append(float(np.exp(-0.5 * n_spur)))
            else:
                p_topo_list.append(0.0)

    median_g = float(np.median(g_list)) if g_list else 1.0
    mean_p_topo = float(np.mean(p_topo_list)) if p_topo_list else 0.0

    # 3. Composite score approximations:
    # S = 0.55 * max(0, R2) + 0.15 * R2_qb + 0.10 * (1 - D_LCFS) + 0.20 * Consistency
    # Typical baseline values: R2_qb ~ 0.0, D_LCFS ~ 0.10, Consistency correlates with R2
    cons_approx = max(0.0, mean_r2 * 0.25)
    lcfs_term = max(0.0, 0.90 - 0.5 * max(0.0, 1.0 - mean_r2))
    S = 0.55 * max(0.0, mean_r2) + 0.10 * lcfs_term + 0.20 * cons_approx

    # S' = S * G_GS(g) * P_topo
    g_gs = gate(median_g, score=1.0, g_ref=DEFAULT_G_REF)
    S_prime = float(S * g_gs * mean_p_topo)

    return {
        "p_drop": p_drop,
        "alpha_drift": alpha_drift,
        "r2_psi": mean_r2,
        "S": float(S),
        "S_prime": S_prime,
        "g_median": median_g,
        "G_GS": g_gs,
        "p_topo": mean_p_topo,
        "conformal_cov_M": cov_M,
    }


# --------------------------------------------------------------------------- Main Pipeline
def main():
    print("=" * 76)
    print("Task T10 — Model Robustness and Conformal Breakdown under Diagnostic Degradation")
    print("=" * 76)

    # 1. Load Data
    print("Loading data splits from cache...")
    mask_npz = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask_npz["grid_R"], mask_npz["grid_Z"]
    mc = mask_npz["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)

    tr_shots = c2_train.load(range(0, 36))
    val_shots = c2_train.load(range(36, 40))
    te_shots = c2_train.load(range(60, 68))

    X_tr_raw, Y_tr_raw, _, _ = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)

    stride = 6
    X_tr_sub = X_tr[::stride]
    Y_tr_sub = Y_tr_raw[::stride]
    print(f"Dataset: N_train={len(X_tr_sub)}, N_val_shots={len(val_shots)}, N_test_shots={len(te_shots)}")

    # Inputs for validation and test
    val_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in val_shots]
    test_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]

    val_y_true = np.concatenate([s["psi"] for s in val_shots])

    # 2. Train Models
    print("\nTraining Baseline ConvDecoder (clean only)...")
    t0 = time.time()
    model_base = train_model(X_tr_sub, Y_tr_sub, augment=False, seed=42, n_epochs=12, batch_size=64)
    print(f"  Baseline trained in {time.time() - t0:.1f} s")

    print("\nTraining Robust ConvDecoder (random mask + drift augmentations)...")
    t0 = time.time()
    model_robust = train_model(X_tr_sub, Y_tr_sub, augment=True, seed=42, n_epochs=12, batch_size=64)
    print(f"  Robust model trained in {time.time() - t0:.1f} s")

    # 3. Calibrate Simultaneous Conformal Bands (on clean calibration shots 36..39)
    print("\nCalibrating Simultaneous Conformal Prediction Bands on independent validation shots...")
    with torch.no_grad():
        val_pred_base = np.concatenate([model_base(torch.tensor(X, dtype=torch.float32)).numpy() for X in val_inputs])
        val_pred_robust = np.concatenate([model_robust(torch.tensor(X, dtype=torch.float32)).numpy() for X in val_inputs])

    band_base = SimultaneousConformalBands(alpha=0.90).fit(val_y_true, val_pred_base, mask=mc)
    band_robust = SimultaneousConformalBands(alpha=0.90).fit(val_y_true, val_pred_robust, mask=mc)
    print(f"  Baseline Band: c* = {band_base.c_star:.4f}, mean scale = {np.mean(band_base.scale_map[mc]):.4f}")
    print(f"  Robust Band:   c* = {band_robust.c_star:.4f}, mean scale = {np.mean(band_robust.scale_map[mc]):.4f}")

    # 4. Degradation Grid Sweep
    p_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    drift_val = 0.20

    results = {
        "dropout_only": {"baseline": [], "robust": []},
        "combined": {"baseline": [], "robust": []},
    }

    print("\n--- Evaluating Dropout Only Sweep (p in [0.0, 0.30]) ---")
    print(f"{'p':<5s} | {'Base R2':<8s} {'Base S':<7s} {'Base S\'':<8s} {'Base g':<7s} {'Base Cov':<8s} | {'Rob R2':<8s} {'Rob S':<7s} {'Rob S\'':<8s} {'Rob g':<7s} {'Rob Cov':<8s}")
    print("-" * 105)

    for p in p_grid:
        res_b = evaluate_condition(model_base, band_base, te_shots, test_inputs,
                                   p_drop=p, alpha_drift=0.0, R=R, Z=Z, mask_coarse=mc, mask_f=mf)
        res_r = evaluate_condition(model_robust, band_robust, te_shots, test_inputs,
                                   p_drop=p, alpha_drift=0.0, R=R, Z=Z, mask_coarse=mc, mask_f=mf)
        results["dropout_only"]["baseline"].append(res_b)
        results["dropout_only"]["robust"].append(res_r)

        print(f"{p:<5.2f} | {res_b['r2_psi']:<8.4f} {res_b['S']:<7.4f} {res_b['S_prime']:<8.4f} {res_b['g_median']:<7.4f} {res_b['conformal_cov_M']:<8.1%} | "
              f"{res_r['r2_psi']:<8.4f} {res_r['S']:<7.4f} {res_r['S_prime']:<8.4f} {res_r['g_median']:<7.4f} {res_r['conformal_cov_M']:<8.1%}")

    print("\n--- Evaluating Combined Degradation Sweep (p + drift alpha=0.20) ---")
    print(f"{'p':<5s} | {'Base R2':<8s} {'Base S':<7s} {'Base S\'':<8s} {'Base g':<7s} {'Base Cov':<8s} | {'Rob R2':<8s} {'Rob S':<7s} {'Rob S\'':<8s} {'Rob g':<7s} {'Rob Cov':<8s}")
    print("-" * 105)

    for p in p_grid:
        res_b = evaluate_condition(model_base, band_base, te_shots, test_inputs,
                                   p_drop=p, alpha_drift=drift_val, R=R, Z=Z, mask_coarse=mc, mask_f=mf)
        res_r = evaluate_condition(model_robust, band_robust, te_shots, test_inputs,
                                   p_drop=p, alpha_drift=drift_val, R=R, Z=Z, mask_coarse=mc, mask_f=mf)
        results["combined"]["baseline"].append(res_b)
        results["combined"]["robust"].append(res_r)

        print(f"{p:<5.2f} | {res_b['r2_psi']:<8.4f} {res_b['S']:<7.4f} {res_b['S_prime']:<8.4f} {res_b['g_median']:<7.4f} {res_b['conformal_cov_M']:<8.1%} | "
              f"{res_r['r2_psi']:<8.4f} {res_r['S']:<7.4f} {res_r['S_prime']:<8.4f} {res_r['g_median']:<7.4f} {res_r['conformal_cov_M']:<8.1%}")

    # 5. Conformal Breakdown Threshold Analysis
    def find_breakdown(res_list: list[dict], threshold: float = 0.85) -> float:
        for item in res_list:
            if item["conformal_cov_M"] < threshold:
                return item["p_drop"]
        return 1.0

    p_star_base = find_breakdown(results["dropout_only"]["baseline"], 0.85)
    p_star_rob = find_breakdown(results["dropout_only"]["robust"], 0.85)
    p_crit_base = find_breakdown(results["dropout_only"]["baseline"], 0.70)
    p_crit_rob = find_breakdown(results["dropout_only"]["robust"], 0.70)

    # 6. Verdict Evaluation
    print("\n" + "=" * 76)
    print("VERDICT EVALUATION FOR TASK T10")
    print("=" * 76)
    clean_base_S = results["dropout_only"]["baseline"][0]["S"]
    clean_rob_S = results["dropout_only"]["robust"][0]["S"]
    delta_clean = clean_rob_S - clean_base_S

    deg_base_S = results["dropout_only"]["baseline"][4]["S"]  # p = 0.20
    deg_rob_S = results["dropout_only"]["robust"][4]["S"]    # p = 0.20
    gain_deg = deg_rob_S - deg_base_S

    print(f"Clean Performance: Baseline S={clean_base_S:.4f}, Robust S={clean_rob_S:.4f} (Delta: {delta_clean:+.4f})")
    print(f"Degraded Performance (p=0.20): Baseline S={deg_base_S:.4f}, Robust S={deg_rob_S:.4f} (Gain: {gain_deg:+.4f})")
    print(f"Conformal Breakdown Threshold (Cov < 85%): Baseline p*={p_star_base:.2f}, Robust p*={p_star_rob:.2f}")
    print(f"Critical Conformal Breakdown (Cov < 70%):  Baseline p_crit={p_crit_base:.2f}, Robust p_crit={p_crit_rob:.2f}")

    conformal_breaks = (p_star_base <= 0.30)
    robust_advantage = (gain_deg >= 0.05) and (delta_clean >= -0.02)
    threshold_shifted = (p_star_rob > p_star_base) or (p_crit_rob > p_crit_base)

    confirmed = conformal_breaks and robust_advantage
    verdict = "CONFIRMED (ПІДТВЕРДЖЕНО)" if confirmed else "REFUTED (СПРОСТОВАНО)"
    print(f"\nVERDICT: {verdict}")

    # 7. Save JSON Summary
    summary = {
        "verdict": verdict,
        "conformal_breaks": conformal_breaks,
        "breakdown_threshold_p_star_baseline": p_star_base,
        "breakdown_threshold_p_star_robust": p_star_rob,
        "critical_threshold_p_crit_baseline": p_crit_base,
        "critical_threshold_p_crit_robust": p_crit_rob,
        "clean_delta_S": delta_clean,
        "degraded_gain_S_at_p0.20": gain_deg,
        "results": results,
    }
    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to: {out_json}")

    # 8. Visualizations
    # Figure 1: Robustness Curves (S, S', g, R2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    b_res = results["dropout_only"]["baseline"]
    r_res = results["dropout_only"]["robust"]
    ps = [x["p_drop"] * 100 for x in b_res]

    # Subplot (0, 0): Composite S
    ax = axes[0, 0]
    ax.plot(ps, [x["S"] for x in b_res], "r-o", label="Baseline (unaugmented)")
    ax.plot(ps, [x["S"] for x in r_res], "g-s", label="Robust (augmented)")
    ax.set_ylabel("Composite Score S")
    ax.set_xlabel("Missing Channels (%)")
    ax.set_title("Composite Metric S vs. Channel Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot (0, 1): Calibrated S'
    ax = axes[0, 1]
    ax.plot(ps, [x["S_prime"] for x in b_res], "r-o", label="Baseline")
    ax.plot(ps, [x["S_prime"] for x in r_res], "g-s", label="Robust")
    ax.set_ylabel("Calibrated Score S'")
    ax.set_xlabel("Missing Channels (%)")
    ax.set_title("Physics/Topology-Gated S' vs. Channel Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot (1, 0): Grad-Shafranov Inconsistency g
    ax = axes[1, 0]
    ax.plot(ps, [x["g_median"] for x in b_res], "r-o", label="Baseline")
    ax.plot(ps, [x["g_median"] for x in r_res], "g-s", label="Robust")
    ax.axhline(DEFAULT_G_REF, color="k", linestyle="--", label="g_ref (0.6328)")
    ax.set_ylabel("Median GS Inconsistency g(ψ)")
    ax.set_xlabel("Missing Channels (%)")
    ax.set_title("Physical Residual Inconsistency vs. Channel Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot (1, 1): Field R2_psi
    ax = axes[1, 1]
    ax.plot(ps, [x["r2_psi"] for x in b_res], "r-o", label="Baseline")
    ax.plot(ps, [x["r2_psi"] for x in r_res], "g-s", label="Robust")
    ax.set_ylabel("Field R²ψ")
    ax.set_xlabel("Missing Channels (%)")
    ax.set_title("Field Reconstruction R² vs. Channel Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig_path1 = HERE / "t10_robustness_curves.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved robustness curves to: {fig_path1}")

    # Figure 2: Conformal Coverage Breakdown
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ps, [x["conformal_cov_M"] * 100 for x in b_res], "r-o", linewidth=2, label="Baseline (Calibrated on clean)")
    ax.plot(ps, [x["conformal_cov_M"] * 100 for x in r_res], "g-s", linewidth=2, label="Robust Model")
    ax.axhline(90.0, color="blue", linestyle="--", label="Nominal Target (90%)")
    ax.axhline(85.0, color="orange", linestyle=":", label="Breakdown Threshold (85%)")
    ax.axhline(70.0, color="red", linestyle=":", label="Critical Threshold (70%)")

    ax.set_xlabel("Missing Diagnostic Channels (%)", fontsize=12)
    ax.set_ylabel("Simultaneous 2D Conformal Coverage (%)", fontsize=12)
    ax.set_title("Breakdown of Conformal Coverage under Diagnostic Degradation", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig_path2 = HERE / "t10_conformal_breakdown.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved conformal breakdown plot to: {fig_path2}")


if __name__ == "__main__":
    main()
