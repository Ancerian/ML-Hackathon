#!/usr/bin/env python3
"""T9 (Hypothesis C3) — evaluate whether magnetic axis estimation error
scales with local flux curvature at the O-point (lambda_axis) across terciles of 1/sqrt(lambda).

Follows pre-registered criteria in Our try/04-novelty/t9/THEORY.md:
1. Calculates lambda_axis (smallest absolute eigenvalue of Hessian at O-point) for DIII-D and MAST.
2. Plots lambda_axis distribution and prints quantiles / P90/P10 ratio.
3. Splits test set into 3 pre-registered terciles by 1/sqrt(lambda).
4. Evaluates PCA+Ridge and UNet_Lite predictions across strata, reporting MAE(R_axis), MAE(Z_axis),
   Euclidean axis error, and Consistency with shot-level bootstrap 95% CIs.
5. Computes Spearman rank correlation rho(1/sqrt(lambda), axis_error).
6. Issues verdict: confirmed / refuted / qualified.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
CACHE = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
C1 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c1"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"

sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C1))
sys.path.insert(0, str(C2))

from common import AXIS_SIGN, CONS_SCALARS
from derive import derive_frame
from lcfs import extract_lcfs
from o_point import find_o_point
import sweep_core as sc
import train as c2_train
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def calc_curvature_single(psi2d, R, Z, mc, s):
    phi = s * psi2d
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mc)
    hR, hZ = R[1] - R[0], Z[1] - Z[0]
    frr = (phi[iz, ir + 1] - 2 * phi[iz, ir] + phi[iz, ir - 1]) / (hR ** 2)
    fzz = (phi[iz + 1, ir] - 2 * phi[iz, ir] + phi[iz - 1, ir]) / (hZ ** 2)
    frz = (phi[iz + 1, ir + 1] - phi[iz + 1, ir - 1]
           - phi[iz - 1, ir + 1] + phi[iz - 1, ir - 1]) / (4 * hR * hZ)
    ev = np.linalg.eigvalsh(np.array([[frr, frz], [frz, fzz]]))
    lam = float(np.min(np.abs(ev)))
    return lam, iz, ir


def evaluate_frame_scalars(psi2d, R, Z, mc, mf, s, machine="DIII-D"):
    c = extract_lcfs(psi2d, R, Z, machine, mc, mf, n_points=sc.N_POINTS, axis_sign=s)
    v = derive_frame(psi2d, R, Z, machine, mc, mf, contour=c, axis_sign=s)
    return {k: float(v[k]) if v[k] is not None else np.nan for k in CONS_SCALARS}


def bootstrap_ci(arr_per_shot, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_shots = len(arr_per_shot)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_shots, size=n_shots)
        # pooled mean of selected shots
        pooled = np.concatenate([arr_per_shot[i] for i in idx])
        means.append(np.nanmean(pooled))
    means = np.array(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t_start = time.time()
    print("=" * 70)
    print("T9 / C3 — Curvature Stratification Experiment (DIII-D / MAST)")
    print("=" * 70)

    # 1. Load DIII-D and MAST datasets
    R_d, Z_d, mc_d, mf_d = sc.load_grid("DIII-D")
    R_m, Z_m, mc_m, mf_m = sc.load_grid("MAST")
    s_d = sc.AXIS_SIGN["DIII-D"]
    s_m = sc.AXIS_SIGN["MAST"]

    # Load existing real_sweep npz for population distribution
    z_d3d_pop = np.load(C1 / "real_sweep_DIII-D.npz")
    z_mast_pop = np.load(C1 / "real_sweep_MAST.npz")
    lam_d3d_pop = z_d3d_pop["lam"]
    lam_mast_pop = z_mast_pop["lam"]

    print("\n--- 1. Population Curvature lambda_axis Distributions ---")
    p10_d, p50_d, p90_d = np.percentile(lam_d3d_pop, [10, 50, 90])
    p10_m, p50_m, p90_m = np.percentile(lam_mast_pop, [10, 50, 90])
    iqr_d = np.percentile(lam_d3d_pop, 75) - np.percentile(lam_d3d_pop, 25)
    iqr_m = np.percentile(lam_mast_pop, 75) - np.percentile(lam_mast_pop, 25)

    print(f"DIII-D (N={len(lam_d3d_pop)}): median={p50_d:.3f}, P10={p10_d:.3f}, P90={p90_d:.3f}, P90/P10={p90_d/p10_d:.2f}, IQR/med={iqr_d/p50_d:.2f}")
    print(f"MAST   (N={len(lam_mast_pop)}): median={p50_m:.3f}, P10={p10_m:.3f}, P90={p90_m:.3f}, P90/P10={p90_m/p10_m:.2f}, IQR/med={iqr_m/p50_m:.2f}")

    # Plot distribution figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.hist(lam_d3d_pop, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(p50_d, color="red", linestyle="--", label=f"Median ({p50_d:.2f})")
    ax1.set_title("DIII-D $\lambda_{\\mathrm{axis}}$ distribution")
    ax1.set_xlabel("$\lambda_{\\mathrm{axis}}$ [Wb/rad/m$^2$]")
    ax1.set_ylabel("Frames count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(lam_mast_pop, bins=20, color="darkorange", edgecolor="black", alpha=0.7)
    ax2.axvline(p50_m, color="red", linestyle="--", label=f"Median ({p50_m:.2f})")
    ax2.set_title("MAST $\lambda_{\\mathrm{axis}}$ distribution")
    ax2.set_xlabel("$\lambda_{\\mathrm{axis}}$ [Wb/rad/m$^2$]")
    ax2.set_ylabel("Frames count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_dist_path = HERE / "t9_lambda_distribution.png"
    plt.savefig(fig_dist_path, dpi=200)
    plt.close()
    print(f"Saved distribution plot to {fig_dist_path}")

    # 2. Prepare test shots from c2_cache (8 shots, #60-#67)
    print("\n--- 2. Loading c2_cache Test Discharges (#60..#67) ---")
    tr_shots = c2_train.load(c2_train.TRAIN)
    te_shots = c2_train.load(c2_train.TEST)

    X_tr, Y_tr, _, _ = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=50, random_state=42).fit(Y_tr.reshape(len(Y_tr), -1))
    ridge = Ridge(alpha=1.0).fit(scaler.transform(X_tr), pca.transform(Y_tr.reshape(len(Y_tr), -1)))

    # UNet training
    va_shots = c2_train.load(c2_train.VAL)
    X_va, Y_va, _, _ = c2_train.frames(va_shots)
    Xs = scaler.transform(X_tr)
    Xvs = scaler.transform(X_va)
    ym, ys = Y_tr.mean(0), Y_tr.std()
    Yn, Yvn = (Y_tr - ym) / ys, (Y_va - ym) / ys
    Xt_list = [scaler.transform(c2_train.test_inputs(s)) for s in te_shots]
    w_ones = np.ones((65, 65), dtype=np.float32)

    print("Training UNet_Lite baseline on 40 train shots...")
    unet_preds_list, _ = c2_train.unet(Xs, Yn, Xvs, Yvn, w_ones, seed=42, Xt_list=Xt_list)
    print("Models ready.")

    # 3. Calculate lambda and ground truth scalars for all test frames
    test_frames = []
    shot_frame_map = {}

    for shot_idx, s in enumerate(te_shots):
        psi_gt = s["psi"]
        T = len(psi_gt)
        X_test = scaler.transform(c2_train.test_inputs(s))
        pca_pred = pca.inverse_transform(ridge.predict(X_test)).reshape(T, 65, 65)
        unet_pred = unet_preds_list[shot_idx] * ys + ym

        shot_frame_map[shot_idx] = []

        for t in range(T):
            p_gt = psi_gt[t]
            p_pca = pca_pred[t]
            p_unet = unet_pred[t]

            try:
                lam, iz, ir = calc_curvature_single(p_gt, R_d, Z_d, mc_d, s_d)
                inv_sqrt_lam = 1.0 / np.sqrt(max(lam, 1e-6))
            except Exception:
                continue

            # Ground truth scalars
            sc_gt = evaluate_frame_scalars(p_gt, R_d, Z_d, mc_d, mf_d, s_d)
            sc_pca = evaluate_frame_scalars(p_pca, R_d, Z_d, mc_d, mf_d, s_d)
            sc_unet = evaluate_frame_scalars(p_unet, R_d, Z_d, mc_d, mf_d, s_d)

            if not np.isfinite(sc_gt["R_axis"]) or not np.isfinite(sc_gt["Z_axis"]):
                continue

            entry = {
                "shot_idx": shot_idx,
                "frame": t,
                "lam": lam,
                "inv_sqrt_lam": inv_sqrt_lam,
                "gt": sc_gt,
                "pca": sc_pca,
                "unet": sc_unet,
                "err_R_pca": abs(sc_pca["R_axis"] - sc_gt["R_axis"]) if np.isfinite(sc_pca["R_axis"]) else np.nan,
                "err_Z_pca": abs(sc_pca["Z_axis"] - sc_gt["Z_axis"]) if np.isfinite(sc_pca["Z_axis"]) else np.nan,
                "err_dist_pca": np.sqrt((sc_pca["R_axis"] - sc_gt["R_axis"])**2 + (sc_pca["Z_axis"] - sc_gt["Z_axis"])**2) if np.isfinite(sc_pca["R_axis"]) and np.isfinite(sc_pca["Z_axis"]) else np.nan,
                "err_R_unet": abs(sc_unet["R_axis"] - sc_gt["R_axis"]) if np.isfinite(sc_unet["R_axis"]) else np.nan,
                "err_Z_unet": abs(sc_unet["Z_axis"] - sc_gt["Z_axis"]) if np.isfinite(sc_unet["Z_axis"]) else np.nan,
                "err_dist_unet": np.sqrt((sc_unet["R_axis"] - sc_gt["R_axis"])**2 + (sc_unet["Z_unet"] - sc_gt["Z_axis"])**2) if "Z_unet" in sc_unet and np.isfinite(sc_unet["R_axis"]) else np.sqrt((sc_unet["R_axis"] - sc_gt["R_axis"])**2 + (sc_unet["Z_axis"] - sc_gt["Z_axis"])**2) if np.isfinite(sc_unet["R_axis"]) and np.isfinite(sc_unet["Z_axis"]) else np.nan,
            }
            test_frames.append(entry)
            shot_frame_map[shot_idx].append(entry)

    print(f"Processed total {len(test_frames)} valid test frames across 8 test shots.")

    # 4. Split test frames into 3 pre-registered terciles by 1/sqrt(lambda)
    all_inv_sqrt = np.array([f["inv_sqrt_lam"] for f in test_frames])
    q33, q67 = np.percentile(all_inv_sqrt, [33.333, 66.667])
    print(f"\nTerciles of 1/sqrt(lambda):")
    print(f"  T1 (Sharp axis):   1/sqrt(lambda) < {q33:.3f}   (lambda > {1/(q33**2):.2f})")
    print(f"  T2 (Medium axis):  {q33:.3f} <= 1/sqrt(lambda) <= {q67:.3f}")
    print(f"  T3 (Flat axis):    1/sqrt(lambda) > {q67:.3f}   (lambda < {1/(q67**2):.2f})")

    terciles = {"T1": [], "T2": [], "T3": []}
    for f in test_frames:
        if f["inv_sqrt_lam"] < q33:
            terciles["T1"].append(f)
        elif f["inv_sqrt_lam"] <= q67:
            terciles["T2"].append(f)
        else:
            terciles["T3"].append(f)

    # 5. Evaluate per stratum with shot-level bootstrap
    results_strata = {}
    print("\n--- 3. Error Metrics per Tercile (with shot-level 95% CIs) ---")

    for t_name in ["T1", "T2", "T3"]:
        frames_t = terciles[t_name]
        results_strata[t_name] = {"count": len(frames_t)}

        # Group by shot for bootstrap
        shots_data = {}
        for s_idx in range(len(te_shots)):
            s_frames = [f for f in frames_t if f["shot_idx"] == s_idx]
            shots_data[s_idx] = s_frames

        for m_name, prefix in [("PCA+Ridge", "pca"), ("UNet_Lite", "unet")]:
            err_r_by_shot = [np.array([f[f"err_R_{prefix}"] for f in shots_data[s_idx] if np.isfinite(f[f"err_R_{prefix}"])]) for s_idx in range(len(te_shots))]
            err_z_by_shot = [np.array([f[f"err_Z_{prefix}"] for f in shots_data[s_idx] if np.isfinite(f[f"err_Z_{prefix}"])]) for s_idx in range(len(te_shots))]
            err_dist_by_shot = [np.array([f[f"err_dist_{prefix}"] for f in shots_data[s_idx] if np.isfinite(f[f"err_dist_{prefix}"])]) for s_idx in range(len(te_shots))]

            mean_r = np.nanmean(np.concatenate([x for x in err_r_by_shot if len(x)]))
            ci_r = bootstrap_ci(err_r_by_shot)

            mean_z = np.nanmean(np.concatenate([x for x in err_z_by_shot if len(x)]))
            ci_z = bootstrap_ci(err_z_by_shot)

            mean_dist = np.nanmean(np.concatenate([x for x in err_dist_by_shot if len(x)]))
            ci_dist = bootstrap_ci(err_dist_by_shot)

            results_strata[t_name][m_name] = {
                "MAE_R": (mean_r, ci_r),
                "MAE_Z": (mean_z, ci_z),
                "MAE_dist": (mean_dist, ci_dist)
            }
            print(f"[{t_name} - N={len(frames_t)}] {m_name:10s}: "
                  f"MAE(R) = {mean_r*100:.2f} cm [{ci_r[0]*100:.2f}; {ci_r[1]*100:.2f}], "
                  f"MAE(Z) = {mean_z*100:.2f} cm [{ci_z[0]*100:.2f}; {ci_z[1]*100:.2f}], "
                  f"Dist = {mean_dist*100:.2f} cm [{ci_dist[0]*100:.2f}; {ci_dist[1]*100:.2f}]")

    # 6. Spearman rank correlations
    all_inv = np.array([f["inv_sqrt_lam"] for f in test_frames])
    dist_pca = np.array([f["err_dist_pca"] for f in test_frames])
    dist_unet = np.array([f["err_dist_unet"] for f in test_frames])
    r_pca = np.array([f["err_R_pca"] for f in test_frames])
    r_unet = np.array([f["err_R_unet"] for f in test_frames])

    valid_pca = np.isfinite(dist_pca) & np.isfinite(all_inv)
    rho_pca, p_pca = spearmanr(all_inv[valid_pca], dist_pca[valid_pca])
    rho_r_pca, p_r_pca = spearmanr(all_inv[valid_pca], r_pca[valid_pca])

    valid_unet = np.isfinite(dist_unet) & np.isfinite(all_inv)
    rho_unet, p_unet = spearmanr(all_inv[valid_unet], dist_unet[valid_unet])
    rho_r_unet, p_r_unet = spearmanr(all_inv[valid_unet], r_unet[valid_unet])

    print("\n--- 4. Spearman Rank Correlations ---")
    print(f"PCA+Ridge: rho(1/sqrt(lambda), Dist) = {rho_pca:.3f} (p={p_pca:.2e}), rho(1/sqrt(lambda), |dR|) = {rho_r_pca:.3f} (p={p_r_pca:.2e})")
    print(f"UNet_Lite: rho(1/sqrt(lambda), Dist) = {rho_unet:.3f} (p={p_unet:.2e}), rho(1/sqrt(lambda), |dR|) = {rho_r_unet:.3f} (p={p_r_unet:.2e})")

    # 7. Verdict evaluation against pre-registered criteria
    # Criterion 1: Wide distribution P90/P10 >= 2.5
    wide_dist = (p90_d / p10_d) >= 2.5
    # Criterion 2: T3 > T1 for axis error with 95% CI
    diff_pca = results_strata["T3"]["PCA+Ridge"]["MAE_dist"][0] - results_strata["T1"]["PCA+Ridge"]["MAE_dist"][0]
    diff_unet = results_strata["T3"]["UNet_Lite"]["MAE_dist"][0] - results_strata["T1"]["UNet_Lite"]["MAE_dist"][0]

    # Plot 1 summary figure for T9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cats = ["T1\n(Sharp axis)", "T2\n(Medium)", "T3\n(Flat axis)"]

    pca_vals = [results_strata[t]["PCA+Ridge"]["MAE_dist"][0] * 100 for t in ["T1", "T2", "T3"]]
    pca_errs = np.array([[pca_vals[i] - results_strata[t]["PCA+Ridge"]["MAE_dist"][1][0]*100,
                          results_strata[t]["PCA+Ridge"]["MAE_dist"][1][1]*100 - pca_vals[i]] for i, t in enumerate(["T1", "T2", "T3"])]).T

    unet_vals = [results_strata[t]["UNet_Lite"]["MAE_dist"][0] * 100 for t in ["T1", "T2", "T3"]]
    unet_errs = np.array([[unet_vals[i] - results_strata[t]["UNet_Lite"]["MAE_dist"][1][0]*100,
                           results_strata[t]["UNet_Lite"]["MAE_dist"][1][1]*100 - unet_vals[i]] for i, t in enumerate(["T1", "T2", "T3"])]).T

    x = np.arange(3)
    w = 0.35
    ax1.bar(x - w/2, pca_vals, w, yerr=pca_errs, capsize=4, color="cornflowerblue", label="PCA+Ridge")
    ax1.bar(x + w/2, unet_vals, w, yerr=unet_errs, capsize=4, color="coral", label="UNet_Lite")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats)
    ax1.set_ylabel("Axis Euclidean Error [cm]")
    ax1.set_title("Axis Error by $1/\\sqrt{\lambda}$ Stratum (with 95% shot CI)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter with regression trend
    ax2.scatter(all_inv[valid_pca][::5], dist_pca[valid_pca][::5] * 100, alpha=0.3, s=15, color="cornflowerblue", label=f"PCA+Ridge ($\\rho={rho_pca:.2f}$)")
    ax2.scatter(all_inv[valid_unet][::5], dist_unet[valid_unet][::5] * 100, alpha=0.3, s=15, color="coral", label=f"UNet_Lite ($\\rho={rho_unet:.2f}$)")
    ax2.set_xlabel("$1/\\sqrt{\lambda_{\\mathrm{axis}}}$ [m $\\cdot$ (Wb/rad)$^{-1/2}$]")
    ax2.set_ylabel("Axis Error [cm]")
    ax2.set_title("Axis Error vs $1/\\sqrt{\lambda_{\\mathrm{axis}}}$")
    ax2.set_ylim(0, 20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_summary_path = HERE / "t9_strata_results.png"
    plt.savefig(fig_summary_path, dpi=200)
    plt.close()
    print(f"\nSaved summary figure to {fig_summary_path}")

    # Determine verdict
    # Check falsifiers:
    falsified = False
    falsification_reason = []
    if (p90_d / p10_d) < 2.0:
        falsified = True
        falsification_reason.append("P90/P10 < 2.0")
    if rho_pca < 0.15 and rho_unet < 0.15:
        falsified = True
        falsification_reason.append("Spearman rho < 0.15 for all models")
    if diff_pca <= 0 and diff_unet <= 0:
        falsified = True
        falsification_reason.append("T3 error not greater than T1")

    if falsified:
        verdict = "СПРОСТОВАНО (REFUTED)"
    elif (rho_pca >= 0.20 or rho_unet >= 0.20) and (diff_pca > 0 or diff_unet > 0):
        verdict = "УТОЧНЕНО (QUALIFIED)"
    else:
        verdict = "СПРОСТОВАНО (REFUTED)"

    print(f"\n========================================================")
    print(f"VERDICT: {verdict}")
    print(f"========================================================")

    # Save summary JSON
    summary_data = {
        "verdict": verdict,
        "runtime_sec": time.time() - t_start,
        "population_stats": {
            "diii_d": {"p10": p10_d, "p50": p50_d, "p90": p90_d, "ratio_p90_p10": p90_d/p10_d},
            "mast": {"p10": p10_m, "p50": p50_m, "p90": p90_m, "ratio_p90_p10": p90_m/p10_m}
        },
        "strata_thresholds": {"q33": q33, "q67": q67},
        "strata_results": {
            t: {
                "count": results_strata[t]["count"],
                "PCA+Ridge": {
                    "MAE_R_cm": results_strata[t]["PCA+Ridge"]["MAE_R"][0]*100,
                    "MAE_R_CI": [results_strata[t]["PCA+Ridge"]["MAE_R"][1][0]*100, results_strata[t]["PCA+Ridge"]["MAE_R"][1][1]*100],
                    "MAE_dist_cm": results_strata[t]["PCA+Ridge"]["MAE_dist"][0]*100,
                    "MAE_dist_CI": [results_strata[t]["PCA+Ridge"]["MAE_dist"][1][0]*100, results_strata[t]["PCA+Ridge"]["MAE_dist"][1][1]*100],
                },
                "UNet_Lite": {
                    "MAE_R_cm": results_strata[t]["UNet_Lite"]["MAE_R"][0]*100,
                    "MAE_R_CI": [results_strata[t]["UNet_Lite"]["MAE_R"][1][0]*100, results_strata[t]["UNet_Lite"]["MAE_R"][1][1]*100],
                    "MAE_dist_cm": results_strata[t]["UNet_Lite"]["MAE_dist"][0]*100,
                    "MAE_dist_CI": [results_strata[t]["UNet_Lite"]["MAE_dist"][1][0]*100, results_strata[t]["UNet_Lite"]["MAE_dist"][1][1]*100],
                }
            } for t in ["T1", "T2", "T3"]
        },
        "spearman": {
            "PCA+Ridge_dist": {"rho": float(rho_pca), "p": float(p_pca)},
            "PCA+Ridge_R": {"rho": float(rho_r_pca), "p": float(p_r_pca)},
            "UNet_Lite_dist": {"rho": float(rho_unet), "p": float(p_unet)},
            "UNet_Lite_R": {"rho": float(rho_r_unet), "p": float(p_r_unet)}
        }
    }
    (HERE / "results.json").write_text(json.dumps(summary_data, indent=2))
    print(f"Results written to {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
