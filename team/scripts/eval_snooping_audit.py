#!/usr/bin/env python3
"""eval_snooping_audit.py — Audits test snooping and evaluates models on non-test shots.

1. Checks local file availability of shots 36..59.
2. Identifies usage history of shots 36..39 (used as VAL in C2) and 40..59 (not present locally).
3. Evaluates the 4 baseline models on shots 36..39 (1039 frames) without tuning:
   - Linear Regression
   - PCA+Ridge
   - MLP (sklearn)
   - UNet_Lite
4. Computes official S, physical residual g(psi), topology P_topo, and diagnostic S'-gate.
5. Compares performance metrics side-by-side: shots 36..39 vs 60..67.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STARTER = ROOT / "fusion equilibrium challenge" / "starter"
C2 = ROOT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
T3 = ROOT / "Our try" / "04-novelty" / "t3"
CACHE = ROOT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"

sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))
sys.path.insert(0, str(T3))
sys.path.insert(0, str(ROOT / "src"))

import train as c2_train
import evaluate as c2e
import local_score as ls
from common import PSI_SIGNS
from metrics import Accum, finalize_machine
from run_t3 import MultiTaskUNet
from tokamld.gs import gs_inconsistency, DEFAULT_G_REF
from tokamld.topology import inside_lcfs, count_critical_points


def audit_shot_availability():
    """Checks which shots 36..59 exist on disk."""
    available = []
    missing = []
    for s_id in range(36, 60):
        p = CACHE / f"shot_{s_id:03d}.npz"
        if p.exists():
            available.append(s_id)
        else:
            missing.append(s_id)
    return available, missing


def score_shot_set(shots, preds, grid):
    R, Z, mc, mf = grid
    n_shots = len(shots)
    refs, stats = c2e.references(shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n_shots)))

    # Sign determination
    tot = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = ls.psi_residuals(s_["psi"], p["psirz"], means[0])
        for sg in PSI_SIGNS:
            tot[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: tot[sg])

    # Score each shot
    parts = []
    shot_s = []
    shot_s_prime = []
    shot_g = []
    shot_p_topo = []

    for i in range(n_shots):
        s_ = shots[i]
        ref = refs[i]
        p = preds[i]
        part = ls.score_shot(s_, ref, p, R, Z, mc, mf, psi_sign, means)
        
        psi_pred = np.asarray(p["psirz"], dtype=np.float64) * psi_sign
        T = psi_pred.shape[0]
        
        g_vals = []
        p_frame_vals = []
        for k in range(T):
            g_val, _ = gs_inconsistency(psi_pred[k], R, Z, mc, mf)
            g_vals.append(g_val if np.isfinite(g_val) else 1.0)
            
            ins = inside_lcfs(psi_pred[k], R, Z, mc, mf)
            if ins is None or ins.sum() < 50:
                p_frame_vals.append(0.0)
            else:
                n_O, n_X = count_critical_points(psi_pred[k], R, Z, ins, min_npix=3)
                p_frame_vals.append(float(np.exp(-0.5 * (abs(n_O - 1) + n_X))))
        
        bar_g = float(np.nanmedian(g_vals)) if g_vals else 1.0
        g_gs = 1.0 if bar_g <= DEFAULT_G_REF else float(np.exp(-(bar_g - DEFAULT_G_REF) / DEFAULT_G_REF))
        p_topo = float(np.mean(p_frame_vals)) if p_frame_vals else 0.0
        
        part["bar_g"] = bar_g
        part["G_GS"] = g_gs
        part["P_topo"] = p_topo
        parts.append(part)

        acc_i = Accum()
        acc_i.psi_sign = psi_sign
        acc_i.add(part)
        res_i = finalize_machine(acc_i, stats[i])
        s_i = float(res_i["S"])
        shot_s.append(s_i)
        shot_s_prime.append(s_i * g_gs * p_topo)
        shot_g.append(bar_g)
        shot_p_topo.append(p_topo)

    # Global S
    acc_all = Accum()
    acc_all.psi_sign = psi_sign
    for p in parts:
        acc_all.add(p)
    res_all = finalize_machine(acc_all, c2e.sum_stats(stats, range(n_shots)))

    # Bootstrap CIs (1000 resamples)
    rng = np.random.default_rng(42)
    boot_s = []
    boot_sp = []
    for _ in range(1000):
        b = rng.integers(0, n_shots, size=n_shots)
        acc_b = Accum()
        acc_b.psi_sign = psi_sign
        for idx in b:
            acc_b.add(parts[idx])
        res_b = finalize_machine(acc_b, c2e.sum_stats(stats, b))
        boot_s.append(float(res_b["S"]))
        boot_sp.append(float(np.mean([shot_s_prime[idx] for idx in b])))

    return {
        "official": {
            "S": float(res_all["S"]),
            "ci_95": [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
            "r2_psi": float(res_all["r2_psi"]),
            "r2_qb": float(res_all["r2_qb"]),
            "dlcfs": float(res_all["dlcfs"]),
            "consistency": float(res_all["consistency"]),
        },
        "extended_s_prime": {
            "S_prime": float(np.mean(shot_s_prime)),
            "ci_95": [float(np.percentile(boot_sp, 2.5)), float(np.percentile(boot_sp, 97.5))],
            "mean_G_GS": float(np.mean([p["G_GS"] for p in parts])),
            "mean_P_topo": float(np.mean(shot_p_topo)),
            "median_g": float(np.median(shot_g)),
        },
        "per_shot": shot_s,
        "per_shot_sp": shot_s_prime,
    }


def main():
    print("=" * 70)
    print("TEST SNOOPING CONTROL & EVALUATION ON SHOTS #036..#039")
    print("=" * 70)

    avail, missing = audit_shot_availability()
    print(f"Shots #036..#059 audit:")
    print(f"  Available locally: {avail} (count: {len(avail)})")
    print(f"  Missing locally:   {missing[:5]}..{missing[-1]} (count: {len(missing)})")

    # Load geometry
    mask_path = STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))

    # Load data
    tr_shots = c2_train.load(c2_train.TRAIN)
    va_shots = c2_train.load(c2_train.VAL)

    X_tr, Y_tr, q_tr, b_tr = c2_train.frames(tr_shots)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    pca = PCA(n_components=50, random_state=42).fit(Y_tr.reshape(len(Y_tr), -1))
    Y_tr_pca = pca.transform(Y_tr.reshape(len(Y_tr), -1))

    # Scalar heads via Ridge
    ridge_q = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, q_tr)
    ridge_b = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, b_tr)

    val_inputs = [scaler.transform(c2_train.test_inputs(s)) for s in va_shots]
    val_heads = [(ridge_q.predict(Xt), ridge_b.predict(Xt)) for Xt in val_inputs]

    models = {}

    # 1. Linear Regression
    print("Fitting Linear Regression (PCA)...")
    lr = LinearRegression().fit(X_tr_s, Y_tr_pca)
    lr_preds = []
    for i, s in enumerate(va_shots):
        psi_pred = pca.inverse_transform(lr.predict(val_inputs[i])).reshape(len(s["psi"]), 65, 65)
        lr_preds.append({"psirz": psi_pred, "q95": val_heads[i][0], "betaN": val_heads[i][1]})
    models["Linear Regression"] = lr_preds

    # 2. PCA+Ridge
    print("Fitting PCA+Ridge...")
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_tr_s, Y_tr_pca)
    ridge_preds = []
    for i, s in enumerate(va_shots):
        psi_pred = pca.inverse_transform(ridge.predict(val_inputs[i])).reshape(len(s["psi"]), 65, 65)
        ridge_preds.append({"psirz": psi_pred, "q95": val_heads[i][0], "betaN": val_heads[i][1]})
    models["PCA+Ridge"] = ridge_preds

    # 3. MLP (sklearn)
    print("Fitting MLP (sklearn)...")
    mlp_sk = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=42).fit(X_tr_s, Y_tr_pca)
    mlp_preds = []
    for i, s in enumerate(va_shots):
        psi_pred = pca.inverse_transform(mlp_sk.predict(val_inputs[i])).reshape(len(s["psi"]), 65, 65)
        mlp_preds.append({"psirz": psi_pred, "q95": val_heads[i][0], "betaN": val_heads[i][1]})
    models["MLP (sklearn)"] = mlp_preds

    # 4. UNet_Lite
    print("Running UNet_Lite inference on validation shots...")
    ym, ys = Y_tr.mean(0), Y_tr.std()
    ckpt_path = T3 / "checkpoints" / "unet_mt_lambda_0.0_s0.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet_model = MultiTaskUNet(n_features=X_tr.shape[1], n_scalars=7)
    unet_model.load_state_dict(ckpt)
    unet_model.eval()

    unet_preds = []
    with torch.no_grad():
        for i, s in enumerate(va_shots):
            psi_norm = unet_model(torch.tensor(val_inputs[i], dtype=torch.float32))[0].numpy()
            psi_pred = psi_norm * ys + ym
            unet_preds.append({"psirz": psi_pred, "q95": val_heads[i][0], "betaN": val_heads[i][1]})
    models["UNet_Lite"] = unet_preds

    # Evaluate on shots 36..39
    results_val = {}
    print("\nScoring models on shots #036..#039 (1039 frames)...")
    for name, p in models.items():
        t0 = time.time()
        res = score_shot_set(va_shots, p, grid)
        results_val[name] = res
        print(f"  {name:18s}: S = {res['official']['S']:.4f} [CI: {res['official']['ci_95'][0]:.4f}..{res['official']['ci_95'][1]:.4f}], S'-gate = {res['extended_s_prime']['S_prime']:.4f}, g = {res['extended_s_prime']['median_g']:.4f} ({time.time()-t0:.1f}s)")

    # Save results to json for permanent reference
    out_path = HERE.parent / "notes" / "val_shots_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_val, f, indent=2)
    print(f"\nSaved validation shot evaluation to {out_path}")


if __name__ == "__main__":
    main()
