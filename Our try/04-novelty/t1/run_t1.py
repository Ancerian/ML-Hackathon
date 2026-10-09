#!/usr/bin/env python3
"""run_t1.py — Complete T1 evaluation: GS gate and penalty against baselines.

Implements pre-registered hypothesis checks from Our try/04-novelty/t1/THEORY.md:
1. Evaluates 4 baselines on 8 C2-split test shots (#60..#67):
   - Linear Regression (PCA-based)
   - Ridge (CV) / PCA+Ridge (PCA-based)
   - MLP (sklearn) (PCA-based, high S)
   - UNet_Lite (direct map-output CNN, high S, high GS inconsistency)
2. Computes official composite scores S (R2_psi, R2_qb, LCFS, Consistency).
3. Computes relative Grad-Shafranov inconsistency g(psi) for every frame.
4. Applies three pre-registered gates/penalties:
   - Variant A: Hard gate S'_A = S * 1[g < g_ref]
   - Variant B: Sigmoid gate S'_B = S / (1 + exp(k * (g - g_ref)/g_ref)) (k=10)
   - Variant C: Log/Exp penalty S'_C = S * exp(-beta * max(0, g - g_ref)/g_ref) (beta=1)
5. Performs 1000-replicate cluster bootstrap over test shots to calculate:
   - 95% confidence intervals for S and S'
   - Kendall rank correlation tau(S, S')
   - Robust swap rates (fraction of bootstrap replicates where pair order reverses)
6. Compiles summary table, JSON results, and generates figure.
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
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
sys.path.insert(0, str(HERE))

import train as c2_train
import evaluate as c2e
from experiments import get_sklearn_models
from experiments_torch import UNetLite
from gs_gate import gate, compute_g, DEFAULT_G_REF
from metrics import Accum, finalize_machine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor


def bootstrap_ci(arr, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = [np.nanmean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def kendall_tau(ranked_a, ranked_b):
    pa = {m: i for i, m in enumerate(ranked_a)}
    pb = {m: i for i, m in enumerate(ranked_b)}
    c = d = 0
    for x, y in combinations(ranked_a, 2):
        s1 = np.sign(pa[x] - pa[y])
        s2 = np.sign(pb[x] - pb[y])
        c += (s1 == s2)
        d += (s1 != s2)
    return float((c - d) / (c + d)) if (c + d) > 0 else 1.0


def main():
    t_start = time.time()
    print("=" * 70)
    print("T1 (R12, E38) — Grad-Shafranov Gate & Penalty Evaluation")
    print("=" * 70)

    # 1. Load masks and geometry
    mask_path = STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))
    R, Z, mc, mf = grid

    # 2. Load C2 split shots
    print("Loading C2 train, val, and test shots...")
    tr_shots = c2_train.load(c2_train.TRAIN)
    va_shots = c2_train.load(c2_train.VAL)
    te_shots = c2_train.load(c2_train.TEST)
    n_test = len(te_shots)

    refs, stats = c2e.references(te_shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n_test)))

    # Frame indexing across test shots
    shot_lens = [len(s["psi"]) for s in te_shots]
    shot_offsets = np.r_[0, np.cumsum(shot_lens)]
    total_test_frames = shot_offsets[-1]
    print(f"Total test frames across {n_test} shots: {total_test_frames}")

    # 3. Fit scaler, PCA, and models
    print("\nFitting feature scaler and PCA on training shots...")
    X_tr, Y_tr, q_tr, b_tr = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    pca = PCA(n_components=50, random_state=42).fit(Y_tr.reshape(len(Y_tr), -1))
    Y_tr_pca = pca.transform(Y_tr.reshape(len(Y_tr), -1))

    # Scalar heads (q95, betaN) via Ridge
    ridge_q = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, q_tr)
    ridge_b = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, b_tr)

    test_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]
    test_heads = [(ridge_q.predict(Xt), ridge_b.predict(Xt)) for Xt in test_inputs]

    models = {}
    print("\nTraining / Preparing Models:")

    # Model 1: Linear Regression
    print("  1. Training Linear Regression (PCA)...")
    lr = LinearRegression().fit(X_tr_s, Y_tr_pca)
    lr_preds = []
    for i, s in enumerate(te_shots):
        psi_pred = pca.inverse_transform(lr.predict(test_inputs[i])).reshape(len(s["psi"]), 65, 65)
        lr_preds.append({"psirz": psi_pred, "q95": test_heads[i][0], "betaN": test_heads[i][1]})
    models["Linear Regression"] = lr_preds

    # Model 2: Ridge (CV) / PCA+Ridge
    print("  2. Training Ridge (CV) (PCA)...")
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, Y_tr_pca)
    ridge_preds = []
    for i, s in enumerate(te_shots):
        psi_pred = pca.inverse_transform(ridge.predict(test_inputs[i])).reshape(len(s["psi"]), 65, 65)
        ridge_preds.append({"psirz": psi_pred, "q95": test_heads[i][0], "betaN": test_heads[i][1]})
    models["PCA+Ridge"] = ridge_preds

    # Model 3: MLP (sklearn)
    print("  3. Training MLP (sklearn) (PCA)...")
    mlp_sk = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=42).fit(X_tr_s, Y_tr_pca)
    mlp_preds = []
    for i, s in enumerate(te_shots):
        psi_pred = pca.inverse_transform(mlp_sk.predict(test_inputs[i])).reshape(len(s["psi"]), 65, 65)
        mlp_preds.append({"psirz": psi_pred, "q95": test_heads[i][0], "betaN": test_heads[i][1]})
    models["MLP (sklearn)"] = mlp_preds

    # Model 4: UNet_Lite
    print("  4. Training / Loading UNet_Lite (direct map output, 50 epochs)...")
    unet_cache_path = HERE / "unet_test_preds.npz"
    if unet_cache_path.exists():
        print("     Loaded precomputed UNet_Lite test predictions from cache.")
        z_unet = np.load(unet_cache_path)
        unet_preds_list = [z_unet[f"pred_{i}"] for i in range(len(te_shots))]
        ys, ym = z_unet["ys"], z_unet["ym"]
    else:
        X_va, Y_va, _, _ = c2_train.frames(va_shots)
        Xvs = scaler.transform(X_va)
        ym, ys = Y_tr.mean(0), Y_tr.std()
        Yn = (Y_tr - ym) / ys
        Yvn = (Y_va - ym) / ys
        w_ones = np.ones((65, 65), dtype=np.float32)
        unet_preds_list, _ = c2_train.unet(X_tr_s, Yn, Xvs, Yvn, w_ones, seed=42, Xt_list=test_inputs)
        np.savez_compressed(unet_cache_path, ys=ys, ym=ym, **{f"pred_{i}": p for i, p in enumerate(unet_preds_list)})
    
    unet_preds = []
    for i, s in enumerate(te_shots):
        psi_pred = unet_preds_list[i] * ys + ym
        unet_preds.append({"psirz": psi_pred, "q95": test_heads[i][0], "betaN": test_heads[i][1]})
    models["UNet_Lite"] = unet_preds

    # 4. Compute scorer parts and Grad-Shafranov inconsistency g per frame
    print("\nComputing official scores and GS inconsistency g(psi)...")
    g_ref = DEFAULT_G_REF
    eval_cache = {}

    for name, preds in models.items():
        parts, psi_sign = c2e.score_parts(te_shots, refs, preds, grid, means)
        # Compute g for each frame across all test shots
        all_psis = np.concatenate([p["psirz"] for p in preds])
        g_vals = []
        for frame_idx in range(len(all_psis)):
            g_vals.append(compute_g(all_psis[frame_idx], R, Z, mc, mf))
        g_vals = np.array(g_vals)
        eval_cache[name] = {
            "parts": parts,
            "sign": psi_sign,
            "g": g_vals,
            "g_median": float(np.nanmedian(g_vals))
        }
        print(f"  {name:18s}: median g = {eval_cache[name]['g_median']:.4f}")

    # Helper function to evaluate shot subset
    def score_subset(m_name, shot_indices):
        c = eval_cache[m_name]
        acc = Accum()
        acc.psi_sign = c["sign"]
        for si in shot_indices:
            acc.add(c["parts"][si])
        r = finalize_machine(acc, c2e.sum_stats(stats, shot_indices))
        s_orig = float(r["S"])

        # Frame indices for this shot subset
        frame_idx = np.concatenate([np.arange(shot_offsets[si], shot_offsets[si+1]) for si in shot_indices])
        g_sub = c["g"][frame_idx]
        g_shot_mean = float(np.nanmedian(g_sub))

        # Apply gates/penalties to S using frame-median g
        s_hard = gate(None, s_orig, g_ref=g_ref, mode="hard", precomputed_g=g_shot_mean)
        s_sigmoid = gate(None, s_orig, g_ref=g_ref, mode="sigmoid", k=10.0, precomputed_g=g_shot_mean)
        s_log = gate(None, s_orig, g_ref=g_ref, mode="log", beta=1.0, precomputed_g=g_shot_mean)

        return {
            "S": s_orig,
            "g": g_shot_mean,
            "S_hard": s_hard,
            "S_sigmoid": s_sigmoid,
            "S_log": s_log
        }

    # Point evaluation on all 8 test shots
    all_shots_idx = list(range(n_test))
    point_results = {}
    for name in models:
        point_results[name] = score_subset(name, all_shots_idx)

    # 5. Cluster Bootstrap (1000 replicates over shots)
    print("\nRunning cluster bootstrap over test shots (1000 replicates)...")
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_samples = {name: {"S": [], "S_hard": [], "S_sigmoid": [], "S_log": []} for name in models}

    for _ in range(n_boot):
        sample_shots = rng.integers(0, n_test, size=n_test)
        for name in models:
            res_b = score_subset(name, sample_shots)
            for k in ["S", "S_hard", "S_sigmoid", "S_log"]:
                boot_samples[name][k].append(res_b[k])

    # 6. Rank analysis and robust swaps
    model_names = list(models.keys())
    modes = [("Original S", "S"), ("Hard Gate S'_A", "S_hard"),
             ("Sigmoid Gate S'_B", "S_sigmoid"), ("Log Penalty S'_C", "S_log")]

    ranking_summary = {}
    robust_swaps = {m_key: [] for _, m_key in modes if m_key != "S"}

    ranked_s = sorted(model_names, key=lambda m: point_results[m]["S"], reverse=True)
    for mode_title, mode_key in modes:
        # Point rank (descending order of score)
        sorted_models = sorted(model_names, key=lambda m: point_results[m][mode_key], reverse=True)
        ranking_summary[mode_key] = {
            "title": mode_title,
            "point_rank": sorted_models,
            "tau_vs_S": kendall_tau(ranked_s, sorted_models) if mode_key != "S" else 1.0,
            "scores": {m: point_results[m][mode_key] for m in model_names},
            "ci": {m: (float(np.percentile(boot_samples[m][mode_key], 2.5)),
                       float(np.percentile(boot_samples[m][mode_key], 97.5))) for m in model_names}
        }

    # Check pairwise rank reversals in bootstrap
    for _, mode_key in modes[1:]:
        for a, b in combinations(model_names, 2):
            for i, j in ((a, b), (b, a)):
                # Original rank i > j, but gated rank j > i
                reversals = [(boot_samples[i]["S"][b_idx] > boot_samples[j]["S"][b_idx]) and
                             (boot_samples[j][mode_key][b_idx] > boot_samples[i][mode_key][b_idx])
                             for b_idx in range(n_boot)]
                rate = float(np.mean(reversals))
                if rate >= 0.5:
                    swap_info = {
                        "S_above": i,
                        "Sprime_above": j,
                        "rate": rate,
                        "robust": rate >= 0.95
                    }
                    robust_swaps[mode_key].append(swap_info)

    # 7. Print Results Table
    print("\n" + "=" * 90)
    print(f"{'Model':20s} {'Median g':10s} {'Original S [95% CI]':24s} {'Hard Gate S_A':16s} {'Sigmoid S_B':14s} {'Log S_C':14s}")
    print("=" * 90)
    for m in model_names:
        s_ci = ranking_summary["S"]["ci"][m]
        print(f"{m:20s} {eval_cache[m]['g_median']:<10.4f} "
              f"{point_results[m]['S']:.4f} [{s_ci[0]:.4f}; {s_ci[1]:.4f}]   "
              f"{point_results[m]['S_hard']:<16.4f} "
              f"{point_results[m]['S_sigmoid']:<14.4f} "
              f"{point_results[m]['S_log']:<14.4f}")
    print("=" * 90)

    print("\nRankings:")
    print("  Original S       :", " > ".join(ranking_summary["S"]["point_rank"]))
    print("  Hard Gate S'_A   :", " > ".join(ranking_summary["S_hard"]["point_rank"]), f"(tau = {ranking_summary['S_hard']['tau_vs_S']:.3f})")
    print("  Sigmoid Gate S'_B:", " > ".join(ranking_summary["S_sigmoid"]["point_rank"]), f"(tau = {ranking_summary['S_sigmoid']['tau_vs_S']:.3f})")
    print("  Log Penalty S'_C :", " > ".join(ranking_summary["S_log"]["point_rank"]), f"(tau = {ranking_summary['S_log']['tau_vs_S']:.3f})")

    print("\nPairwise Rank Reversals (Bootstrap Replicates Rate >= 95% is Robust):")
    for mode_title, mode_key in modes[1:]:
        print(f"\n  --- {mode_title} ---")
        swaps = robust_swaps[mode_key]
        if not swaps:
            print("    No majority rank swaps found.")
        for s in swaps:
            status = "[ROBUST >= 95%]" if s["robust"] else "[NOT ROBUST < 95%]"
            print(f"    S: {s['S_above']} > {s['Sprime_above']}  -->  S': {s['Sprime_above']} > {s['S_above']} ({s['rate']:.1%}) {status}")

    # 8. Verdict against registered criteria
    has_robust_hard = any(s["robust"] for s in robust_swaps["S_hard"])
    has_robust_sigmoid = any(s["robust"] for s in robust_swaps["S_sigmoid"])
    has_robust_log = any(s["robust"] for s in robust_swaps["S_log"])

    verdict_text = "ПІДТВЕРДЖЕНО (CONFIRMED)" if (has_robust_hard or has_robust_sigmoid or has_robust_log) else "СПРОСТОВАНО (REFUTED)"
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict_text}")
    print("=" * 70)

    # 9. Plot figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of S vs S' across models
    x = np.arange(len(model_names))
    width = 0.2
    ax = axes[0]
    ax.bar(x - 1.5*width, [point_results[m]["S"] for m in model_names], width, label="Original S", color="steelblue", alpha=0.85)
    ax.bar(x - 0.5*width, [point_results[m]["S_hard"] for m in model_names], width, label="Hard Gate S'_A", color="crimson", alpha=0.85)
    ax.bar(x + 0.5*width, [point_results[m]["S_sigmoid"] for m in model_names], width, label="Sigmoid S'_B", color="forestgreen", alpha=0.85)
    ax.bar(x + 1.5*width, [point_results[m]["S_log"] for m in model_names], width, label="Log Penalty S'_C", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Model Scores Under GS Gating")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Inconsistency vs Score scatter
    ax2 = axes[1]
    for m in model_names:
        med_g = eval_cache[m]["g_median"]
        s_val = point_results[m]["S"]
        ax2.scatter(med_g, s_val, s=120, label=m)
        ax2.annotate(m, (med_g, s_val), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax2.axvline(g_ref, color="red", linestyle="--", label=f"$g_{{ref}} = {g_ref:.3f}$ (Truth + 1% noise)")
    ax2.set_xlabel("Grad-Shafranov Inconsistency $g(\\psi)$")
    ax2.set_ylabel("Original Score $S$")
    ax2.set_title("E38 Separation: Inconsistency vs Original Score")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig_path = HERE / "t1_gate_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved plot to {fig_path}")

    # 10. Save results to JSON
    results_json = {
        "verdict": verdict_text,
        "g_ref": g_ref,
        "runtime_sec": time.time() - t_start,
        "rankings": ranking_summary,
        "robust_swaps": {k: [s for s in v if s["robust"]] for k, v in robust_swaps.items()},
        "all_swaps": robust_swaps,
        "models": {
            m: {
                "median_g": eval_cache[m]["g_median"],
                "S": point_results[m]["S"],
                "S_hard": point_results[m]["S_hard"],
                "S_sigmoid": point_results[m]["S_sigmoid"],
                "S_log": point_results[m]["S_log"],
            } for m in model_names
        }
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
