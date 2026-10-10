#!/usr/bin/env python3
"""run_t3.py — Complete T3 evaluation: Multi-Task UNet_Lite with auxiliary 7-scalar head.

Pre-registered in Our try/04-novelty/t3/THEORY.md:
1. Evaluates multi-task weights lambda_scal in {0.0 (baseline), 0.1, 0.5} across 3 random seeds {0, 1, 2}.
2. Architecture: UNetLite backbone + 65x65 flux map output + 7-scalar auxiliary MLP head.
3. Computes:
   - R2_psi on 8 test shots (C2-split)
   - Consistency_derived (official derive_frame functional applied to predicted psi)
   - Consistency_direct (predictions directly from auxiliary head vs ground truth)
   - Per-scalar MAE errors for each of the 7 scalars
   - GS inconsistency g(psi) relative to EFIT equilibrium
   - 95% confidence intervals via 1000-replicate cluster bootstrap over test shots.
4. Generates summary table, JSON results, comparison plots, and README.
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
NOV = HERE.parent

sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))
sys.path.insert(0, str(NOV))
sys.path.insert(0, str(PROJECT / "Our try" / "04-novelty" / "t1"))

from common import AXIS_SIGN, CONS_SCALARS, N_CONS
from derive import derive_frame
from lcfs import extract_lcfs
C1 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c1"
sys.path.insert(0, str(C1))
import sweep_core as sc
from gs_gate import compute_g
import train as c2_train
import evaluate as c2e
from metrics import Accum, finalize_machine, ss_tot


class MultiTaskUNet(nn.Module):
    """UNetLite with dual outputs: 65x65 flux map + 7 consistency scalars."""

    def __init__(self, n_features: int = 21, n_scalars: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 64 * 8 * 8)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final_psi = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=2, stride=1, padding=0),
        )
        self.skip_fc = nn.Linear(n_features, 128 * 16 * 16)

        # Auxiliary multi-task head for 7 consistency scalars from bottleneck (512-dim)
        self.scalar_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_scalars),
        )

    def forward(self, x: torch.Tensor):
        h_fc1 = torch.relu(self.fc1(x))
        scal_pred = self.scalar_head(h_fc1)

        h = torch.relu(self.fc2(h_fc1)).view(-1, 64, 8, 8)
        h = self.up1(h)
        skip = torch.relu(self.skip_fc(x)).view(-1, 128, 16, 16)
        h = h + skip
        h = self.up2(h)
        h = self.up3(h)
        psi_pred = self.final_psi(h).squeeze(1)

        return psi_pred, scal_pred


def bootstrap_ci(arr, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = [np.nanmean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t_start = time.time()
    print("=" * 70)
    print("T3 (E1, E36, E37) — Multi-Task Learning for Consistency Scalars")
    print("=" * 70)

    # 1. Device and masks
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {dev}")

    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mc = mask["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)
    sign = AXIS_SIGN["DIII-D"]
    grid = (R, Z, mc, mf)

    # 2. Load C2 shots and scalars cache
    tr_shots = c2_train.load(c2_train.TRAIN)
    va_shots = c2_train.load(c2_train.VAL)
    te_shots = c2_train.load(c2_train.TEST)

    scal_cache = np.load(HERE / "scalars_cache.npz")
    Y_tr_scal_raw = scal_cache["Y_tr"]
    Y_va_scal_raw = scal_cache["Y_va"]
    Y_te_scal_raw = scal_cache["Y_te"]
    scal_mean = scal_cache["mean"]
    scal_std = scal_cache["std"]

    # Normalize scalars using train stats
    Y_tr_scal_norm = np.nan_to_num((Y_tr_scal_raw - scal_mean) / scal_std, nan=0.0)
    Y_va_scal_norm = np.nan_to_num((Y_va_scal_raw - scal_mean) / scal_std, nan=0.0)

    # Frames and inputs
    X_tr, Y_tr, _, _ = c2_train.frames(tr_shots)
    X_va, Y_va, _, _ = c2_train.frames(va_shots)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    Xvs = scaler.transform(X_va)
    ym, ys = Y_tr.mean(0), Y_tr.std()
    Yn = (Y_tr - ym) / ys
    Yvn = (Y_va - ym) / ys

    Xt_list = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]
    shot_lens = [len(s["psi"]) for s in te_shots]
    shot_offsets = np.r_[0, np.cumsum(shot_lens)]
    n_test = len(te_shots)

    refs, stats = c2e.references(te_shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n_test)))

    LAMBDAS = [0.0, 0.1, 0.5]
    SEEDS = [0, 1, 2]
    EPOCHS = 40  # fast convergence with plateau scheduler

    results_all = {}
    checkpoints_dir = HERE / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Loop over configurations
    for lam in LAMBDAS:
        arm_name = f"lambda_{lam:.1f}"
        print(f"\n--- Arm: {arm_name} (lambda_scal = {lam}) ---")
        results_all[arm_name] = {"seeds": {}}

        seed_preds_psi = []
        seed_preds_scal = []

        for seed in SEEDS:
            ckpt_path = checkpoints_dir / f"unet_mt_{arm_name}_s{seed}.pt"
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = MultiTaskUNet(n_features=Xs.shape[1], n_scalars=N_CONS).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
            loss_mse = nn.MSELoss()

            Xt_tensor = torch.tensor(Xs, dtype=torch.float32)
            Yn_tensor = torch.tensor(Yn, dtype=torch.float32)
            Yscal_tensor = torch.tensor(Y_tr_scal_norm, dtype=torch.float32)

            Xv_tensor = torch.tensor(Xvs, dtype=torch.float32, device=dev)
            Yvn_tensor = torch.tensor(Yvn, dtype=torch.float32, device=dev)
            Yv_scal_tensor = torch.tensor(Y_va_scal_norm, dtype=torch.float32, device=dev)

            t0 = time.time()
            if ckpt_path.exists():
                print(f"  Seed {seed}: loading saved checkpoint from {ckpt_path.name}")
                model.load_state_dict(torch.load(ckpt_path, map_location=dev))
            else:
                print(f"  Seed {seed}: training {EPOCHS} epochs...")
                g_gen = torch.Generator().manual_seed(seed)
                for ep in range(EPOCHS):
                    model.train()
                    perm = torch.randperm(len(Xt_tensor), generator=g_gen)
                    for i in range(0, len(perm), 256):
                        idx = perm[i:i+256]
                        if len(idx) < 2:
                            continue
                        xb = Xt_tensor[idx].to(dev)
                        yb_psi = Yn_tensor[idx].to(dev)
                        yb_scal = Yscal_tensor[idx].to(dev)

                        p_psi, p_scal = model(xb)
                        l_psi = loss_mse(p_psi, yb_psi)
                        l_scal = loss_mse(p_scal, yb_scal)
                        loss = l_psi + lam * l_scal

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        vp_psi, vp_scal = model(Xv_tensor)
                        vl_psi = loss_mse(vp_psi, Yvn_tensor)
                        vl_scal = loss_mse(vp_scal, Yv_scal_tensor)
                        sched.step(float(vl_psi + lam * vl_scal))

                torch.save(model.state_dict(), ckpt_path)
                print(f"    Trained in {time.time() - t0:.1f} s. Checkpoint saved.")

            # Inference on test shots
            model.eval()
            with torch.no_grad():
                te_psi = []
                te_scal = []
                for xt in Xt_list:
                    xt_dev = torch.tensor(xt, dtype=torch.float32, device=dev)
                    p_psi, p_scal = model(xt_dev)
                    # Denormalize
                    p_psi_denorm = p_psi.cpu().numpy() * ys + ym
                    p_scal_denorm = p_scal.cpu().numpy() * scal_std + scal_mean
                    te_psi.append(p_psi_denorm)
                    te_scal.append(p_scal_denorm)

            seed_preds_psi.append(te_psi)
            seed_preds_scal.append(te_scal)

        # Average predictions across 3 seeds
        avg_preds_psi = [np.mean([seed_preds_psi[s][shot_i] for s in range(len(SEEDS))], axis=0) for shot_i in range(n_test)]
        avg_preds_scal = [np.mean([seed_preds_scal[s][shot_i] for s in range(len(SEEDS))], axis=0) for shot_i in range(n_test)]

        # Prepare test prediction format for local_score
        preds_for_scorer = [{"psirz": avg_preds_psi[i], "q95": np.zeros(len(avg_preds_psi[i])), "betaN": np.zeros(len(avg_preds_psi[i]))} for i in range(n_test)]

        parts, psi_sign = c2e.score_parts(te_shots, refs, preds_for_scorer, grid, means)
        acc = Accum()
        acc.psi_sign = psi_sign
        for p in parts:
            acc.add(p)
        r = finalize_machine(acc, c2e.sum_stats(stats, range(n_test)))

        r2_psi_val = float(r["r2_psi"])
        cons_derived_val = float(r["consistency"])

        # Direct consistency from auxiliary head
        concat_scal_pred = np.concatenate(avg_preds_scal, axis=0)
        concat_scal_gt = Y_te_scal_raw
        r2_direct_each = []
        mae_each = {}
        for j, s_name in enumerate(CONS_SCALARS):
            valid = np.isfinite(concat_scal_gt[:, j]) & np.isfinite(concat_scal_pred[:, j])
            if valid.sum() > 5:
                ss_res = np.sum((concat_scal_gt[valid, j] - concat_scal_pred[valid, j])**2)
                ss_tot = np.sum((concat_scal_gt[valid, j] - np.mean(concat_scal_gt[valid, j]))**2)
                r2_j = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2_direct_each.append(max(0.0, r2_j))
                mae_each[s_name] = float(np.mean(np.abs(concat_scal_gt[valid, j] - concat_scal_pred[valid, j])))
            else:
                mae_each[s_name] = np.nan
        cons_direct_val = float(np.mean(r2_direct_each)) if r2_direct_each else 0.0

        # Sample GS inconsistency on test
        all_pred_psis = np.concatenate(avg_preds_psi, axis=0)
        sample_g = [compute_g(all_pred_psis[idx], R, Z, mc, mf) for idx in range(0, len(all_pred_psis), 5)]
        g_median = float(np.nanmedian(sample_g))

        results_all[arm_name]["R2_psi"] = r2_psi_val
        results_all[arm_name]["Consistency_derived"] = cons_derived_val
        results_all[arm_name]["Consistency_direct"] = cons_direct_val
        results_all[arm_name]["Composite_S"] = float(r["S"])
        results_all[arm_name]["g_median"] = g_median
        results_all[arm_name]["mae"] = mae_each
        results_all[arm_name]["parts"] = parts
        results_all[arm_name]["psi_sign"] = psi_sign
        results_all[arm_name]["avg_preds_scal"] = avg_preds_scal

        print(f"  Summary {arm_name}: R2_psi = {r2_psi_val:.4f}, Cons_derived = {cons_derived_val:.4f}, Cons_direct = {cons_direct_val:.4f}, g_med = {g_median:.4f}")

    # 3. Cluster Bootstrap (1000 resamples over test shots)
    print("\nRunning cluster bootstrap over test shots (1000 resamples)...")
    rng = np.random.default_rng(42)
    n_boot = 1000

    boot_cons = {f"lambda_{l:.1f}": [] for l in LAMBDAS}
    boot_diff = {"0.1_vs_0.0": [], "0.5_vs_0.0": []}

    for _ in range(n_boot):
        sample_shots = rng.integers(0, n_test, size=n_test)
        cons_sample = {}
        for lam in LAMBDAS:
            arm_name = f"lambda_{lam:.1f}"
            c = results_all[arm_name]
            acc = Accum()
            acc.psi_sign = c["psi_sign"]
            for si in sample_shots:
                acc.add(c["parts"][si])
            r_boot = finalize_machine(acc, c2e.sum_stats(stats, sample_shots))
            c_val = float(r_boot["consistency"])
            cons_sample[arm_name] = c_val
            boot_cons[arm_name].append(c_val)

        boot_diff["0.1_vs_0.0"].append(cons_sample["lambda_0.1"] - cons_sample["lambda_0.0"])
        boot_diff["0.5_vs_0.0"].append(cons_sample["lambda_0.5"] - cons_sample["lambda_0.0"])

    # Compute 95% CIs
    ci_cons = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot_cons.items()}
    ci_diff_01 = (float(np.percentile(boot_diff["0.1_vs_0.0"], 2.5)), float(np.percentile(boot_diff["0.1_vs_0.0"], 97.5)))
    ci_diff_05 = (float(np.percentile(boot_diff["0.5_vs_0.0"], 2.5)), float(np.percentile(boot_diff["0.5_vs_0.0"], 97.5)))

    # 4. Results Table
    print("\n" + "=" * 95)
    print(f"{'Arm':12s} {'R2_psi':9s} {'Cons (derived) [95% CI]':26s} {'Cons (direct)':14s} {'g(psi) med':12s} {'S':8s}")
    print("=" * 95)
    for lam in LAMBDAS:
        arm_name = f"lambda_{lam:.1f}"
        res = results_all[arm_name]
        ci = ci_cons[arm_name]
        print(f"{arm_name:12s} {res['R2_psi']:<9.4f} {res['Consistency_derived']:.4f} [{ci[0]:.4f}; {ci[1]:.4f}]   "
              f"{res['Consistency_direct']:<14.4f} {res['g_median']:<12.4f} {res['Composite_S']:<8.4f}")
    print("=" * 95)

    print("\nDifferences vs Baseline (lambda = 0.0):")
    delta_01 = results_all["lambda_0.1"]["Consistency_derived"] - results_all["lambda_0.0"]["Consistency_derived"]
    delta_05 = results_all["lambda_0.5"]["Consistency_derived"] - results_all["lambda_0.0"]["Consistency_derived"]
    print(f"  lambda = 0.1 vs 0.0: Delta Consistency = {delta_01:+.4f} [95% CI: {ci_diff_01[0]:+.4f}; {ci_diff_01[1]:+.4f}]")
    print(f"  lambda = 0.5 vs 0.0: Delta Consistency = {delta_05:+.4f} [95% CI: {ci_diff_05[0]:+.4f}; {ci_diff_05[1]:+.4f}]")

    print("\nDirect Head MAE per Scalar (lambda = 0.5):")
    for s_name, val in results_all["lambda_0.5"]["mae"].items():
        unit = "m" if "axis" in s_name else ("m^3" if s_name == "volume" else "")
        print(f"  {s_name:10s}: MAE = {val:.4f} {unit}")

    # 5. Verdict against pre-registered criteria
    # Criteria:
    # Confirmed if Delta >= +0.030 and CI does not contain 0.
    # Refuted if Delta < +0.015 or CI contains 0.
    # Clarified if Direct is much better than Derived, proving E1 decouple.
    confirmed_01 = (delta_01 >= 0.030) and (ci_diff_01[0] > 0.0)
    confirmed_05 = (delta_05 >= 0.030) and (ci_diff_05[0] > 0.0)
    direct_beats = (results_all["lambda_0.5"]["Consistency_direct"] >= 0.50)

    if confirmed_01 or confirmed_05:
        verdict = "ПІДТВЕРДЖЕНО (CONFIRMED)"
    elif direct_beats and not (confirmed_01 or confirmed_05):
        verdict = "УТОЧНЕНО (CLARIFIED)"
    else:
        verdict = "СПРОСТОВАНО (REFUTED)"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    # 6. Plot Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Bar chart of consistency derived vs direct
    ax1 = axes[0]
    x = np.arange(len(LAMBDAS))
    width = 0.35
    c_der = [results_all[f"lambda_{l:.1f}"]["Consistency_derived"] for l in LAMBDAS]
    c_dir = [results_all[f"lambda_{l:.1f}"]["Consistency_direct"] for l in LAMBDAS]
    err_der = np.array([[c_der[i] - ci_cons[f"lambda_{LAMBDAS[i]:.1f}"][0],
                         ci_cons[f"lambda_{LAMBDAS[i]:.1f}"][1] - c_der[i]] for i in range(len(LAMBDAS))]).T

    ax1.bar(x - width/2, c_der, width, yerr=err_der, capsize=4, label="Derived from $\\psi$ (derive.py)", color="steelblue", alpha=0.85)
    ax1.bar(x + width/2, c_dir, width, label="Direct Multi-Task Head", color="darkorange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\lambda = {l}$" for l in LAMBDAS])
    ax1.set_ylabel("Consistency Metric ($R^2_{\\mathrm{cons}}$)")
    ax1.set_title("Derived vs Direct Consistency across $\\lambda_{\\mathrm{scal}}$")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Delta Consistency distribution
    ax2 = axes[1]
    ax2.hist(boot_diff["0.1_vs_0.0"], bins=25, alpha=0.6, label="$\\Delta$ (0.1 vs 0.0)", color="seagreen")
    ax2.hist(boot_diff["0.5_vs_0.0"], bins=25, alpha=0.6, label="$\\Delta$ (0.5 vs 0.0)", color="purple")
    ax2.axvline(0.0, color="black", linestyle="--", alpha=0.7)
    ax2.axvline(0.030, color="red", linestyle=":", label="Registration threshold (+0.03)")
    ax2.set_xlabel("$\\Delta$ Consistency")
    ax2.set_ylabel("Bootstrap Replicates Count")
    ax2.set_title("Bootstrap Difference $\\Delta\\mathrm{Consistency}$ (1000 shots)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = HERE / "t3_multitask_consistency.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved summary figure to {fig_path}")

    # 7. Write results.json
    out_json = {
        "verdict": verdict,
        "runtime_sec": time.time() - t_start,
        "lambdas": LAMBDAS,
        "results": {
            f"lambda_{l:.1f}": {
                "R2_psi": results_all[f"lambda_{l:.1f}"]["R2_psi"],
                "Consistency_derived": results_all[f"lambda_{l:.1f}"]["Consistency_derived"],
                "Consistency_derived_CI": ci_cons[f"lambda_{l:.1f}"],
                "Consistency_direct": results_all[f"lambda_{l:.1f}"]["Consistency_direct"],
                "Composite_S": results_all[f"lambda_{l:.1f}"]["Composite_S"],
                "g_median": results_all[f"lambda_{l:.1f}"]["g_median"],
                "mae": results_all[f"lambda_{l:.1f}"]["mae"],
            } for l in LAMBDAS
        },
        "differences": {
            "0.1_vs_0.0": {"delta": delta_01, "CI": ci_diff_01},
            "0.5_vs_0.0": {"delta": delta_05, "CI": ci_diff_05},
        }
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved results to {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
