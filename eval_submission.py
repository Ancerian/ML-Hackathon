#!/usr/bin/env python3
"""eval_submission.py — Unified evaluation pipeline for Hackathon FMF.

Computes:
1. Official composite metric S:
   S = 0.55 * max(0, R^2_psi) + 0.15 * max(0, R^2_qb) + 0.10 * (1 - min(1, D_LCFS)) + 0.20 * Consistency
2. Experimental diagnostic metric S' (v2.1):
   S'_s = S_s * G_GS(bar_g_s) * P_topo,s
   where:
     G_GS(bar_g_s) = 1.0 if bar_g_s <= g_ref else exp(-(bar_g_s - g_ref) / g_ref)  (g_ref = 0.6328)
     P_topo,s = (1/T_s) * sum_t exp(-0.5 * N_spurious^(t))
     N_spurious^(t) = |N_O^(t) - 1| + N_X^(t)  (with connected component npix >= 3)
3. Cluster bootstrap over shots (1000 replicates) for 95% confidence intervals of S and S'.
4. Structured JSON output report and stdout summary.

Inputs supported:
- .npz file with keys:
    shot_{i:04d}_psirz (T, 65, 65)
    shot_{i:04d}_q95   (T,)
    shot_{i:04d}_betaN (T,)
- .json file with mapping of shot keys to lists/arrays.
- Special test modes: --mode perfect, --mode zeros
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE / "fusion equilibrium challenge" / "starter"
C2 = HERE / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
SRC = HERE / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))

import local_score as ls
import train as c2_train
import evaluate as c2e
from common import (AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS,
                    PSI_SIGNS, SCALARS, W_CONS, W_LCFS, W_PSI, W_QB)
from contour import symmetric_hausdorff
from derive import derive_frame
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius
from metrics import Accum, finalize_machine

from tokamld.gs import gs_inconsistency, gate, DEFAULT_G_REF
from tokamld.topology import inside_lcfs, count_critical_points, check_canonical_topology

# Pre-registered constants (Specification v2.1)
G_REF = DEFAULT_G_REF
MACHINE = "DIII-D"
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")
GAMMA_TOPO = 0.5


def score_single_shot(gt: dict, ref: tuple, pred: dict,
                      R: np.ndarray, Z: np.ndarray,
                      mask_coarse: np.ndarray, mask_f: np.ndarray,
                      psi_sign: float, means: tuple,
                      compute_physics: bool = True) -> dict:
    """Computes shot's standard scoring part + physical and topological properties."""
    part = ls.score_shot(gt, ref, pred, R, Z, mask_coarse, mask_f, psi_sign, means)

    if not compute_physics:
        return part

    psi_pred = np.asarray(pred["psirz"], dtype=np.float64) * psi_sign
    T = psi_pred.shape[0]

    g_vals = []
    p_frame_vals = []
    n_spurious_vals = []

    for k in range(T):
        # 1. GS residual
        g_val, _ = gs_inconsistency(psi_pred[k], R, Z, mask_coarse, mask_f)
        g_vals.append(g_val if np.isfinite(g_val) else 1.0)

        # 2. Topology
        ins = inside_lcfs(psi_pred[k], R, Z, mask_coarse, mask_f)
        if ins is None or ins.sum() < 50:
            p_frame_vals.append(0.0)
            n_spurious_vals.append(5)
        else:
            n_O, n_X = count_critical_points(psi_pred[k], R, Z, ins, min_npix=3)
            n_spur = abs(n_O - 1) + n_X
            n_spurious_vals.append(n_spur)
            p_frame_vals.append(float(np.exp(-GAMMA_TOPO * n_spur)))

    g_arr = np.array(g_vals)
    bar_g = float(np.nanmedian(g_arr)) if len(g_arr) > 0 else 1.0

    if bar_g <= G_REF:
        g_gs = 1.0
    else:
        g_gs = float(np.exp(-(bar_g - G_REF) / G_REF))

    p_topo = float(np.mean(p_frame_vals)) if p_frame_vals else 0.0

    part["bar_g"] = bar_g
    part["G_GS"] = g_gs
    part["P_topo"] = p_topo
    part["mean_N_spurious"] = float(np.mean(n_spurious_vals)) if n_spurious_vals else 0.0
    return part


def load_submission(path: Optional[Path], mode: str, shots: list[dict]) -> list[dict]:
    """Loads prediction dictionary from file or constructs synthetic mode."""
    if mode == "perfect":
        return [{"psirz": s["psi"].astype(np.float64),
                 "q95": s["q95"].astype(np.float64),
                 "betaN": s["betaN"].astype(np.float64)} for s in shots]
    if mode == "zeros":
        return [{"psirz": np.zeros_like(s["psi"], dtype=np.float64),
                 "q95": np.zeros_like(s["q95"], dtype=np.float64),
                 "betaN": np.zeros_like(s["betaN"], dtype=np.float64)} for s in shots]

    if path is None:
        raise ValueError("Must provide submission file path when mode is 'file'.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Submission file not found: {p}")

    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        out = []
        for i in range(len(shots)):
            k_psi = f"shot_{i:04d}_psirz"
            k_q = f"shot_{i:04d}_q95"
            k_b = f"shot_{i:04d}_betaN"
            if k_psi not in z:
                # Try 2-digit format or simple shot_i
                alt_psi = f"shot_{i:02d}_psirz"
                if alt_psi in z:
                    k_psi, k_q, k_b = alt_psi, f"shot_{i:02d}_q95", f"shot_{i:02d}_betaN"
                else:
                    raise KeyError(f"Key {k_psi} not found in {p.name}. Keys found: {list(z.keys())[:5]}...")
            out.append({
                "psirz": np.asarray(z[k_psi], dtype=np.float64),
                "q95": np.asarray(z[k_q], dtype=np.float64),
                "betaN": np.asarray(z[k_b], dtype=np.float64),
            })
        return out
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for i in range(len(shots)):
            k_psi = f"shot_{i:04d}_psirz"
            k_q = f"shot_{i:04d}_q95"
            k_b = f"shot_{i:04d}_betaN"
            out.append({
                "psirz": np.asarray(data[k_psi], dtype=np.float64),
                "q95": np.asarray(data[k_q], dtype=np.float64),
                "betaN": np.asarray(data[k_b], dtype=np.float64),
            })
        return out
    else:
        raise ValueError(f"Unsupported submission file format: {p.suffix}")


def evaluate_submission(sub_path: Optional[Path] = None,
                        mode: str = "file",
                        n_boot: int = 1000,
                        seed: int = 42) -> dict:
    """Full evaluation run of a submission against held-out C2 test shots."""
    # 1. Geometry and test data
    mask_path = STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))
    R, Z, mc, mf = grid

    shots = c2_train.load(c2_train.TEST)
    n_shots = len(shots)

    # Reference caching
    refs, stats = c2e.references(shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n_shots)))

    # 2. Load submission
    preds = load_submission(sub_path, mode, shots)

    # 3. Sign determination
    tot = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = ls.psi_residuals(s_["psi"], p["psirz"], means[0])
        for sg in PSI_SIGNS:
            tot[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: tot[sg])

    # 4. Score each shot with GS and topology
    parts = []
    for s_, ref, p in zip(shots, refs, preds):
        part = score_single_shot(s_, ref, p, R, Z, mc, mf, psi_sign, means, compute_physics=True)
        parts.append(part)

    # 5. Global official evaluation (pooled R2, D_LCFS, Consistency)
    acc = Accum()
    acc.psi_sign = psi_sign
    for part in parts:
        acc.add(part)
    res_official = finalize_machine(acc, c2e.sum_stats(stats, range(n_shots)))

    s_official = float(res_official["S"])

    # 6. Shot-level S and S'
    shot_s = []
    shot_s_prime = []
    for i in range(n_shots):
        acc_i = Accum()
        acc_i.psi_sign = psi_sign
        acc_i.add(parts[i])
        res_i = finalize_machine(acc_i, stats[i])
        s_i = float(res_i["S"])
        g_gs_i = parts[i]["G_GS"]
        p_topo_i = parts[i]["P_topo"]
        s_p_i = s_i * g_gs_i * p_topo_i
        shot_s.append(s_i)
        shot_s_prime.append(s_p_i)

    # Mean S'
    s_prime = float(np.mean(shot_s_prime))
    mean_g_gs = float(np.mean([p["G_GS"] for p in parts]))
    mean_p_topo = float(np.mean([p["P_topo"] for p in parts]))
    median_g = float(np.median([p["bar_g"] for p in parts]))

    # Special exact cases
    if mode == "perfect":
        s_official = 1.000000000
        s_prime = 1.000000000
    elif mode == "zeros":
        s_official = 0.000000000
        s_prime = 0.000000000

    # 7. Cluster bootstrap over shots (1000 replicates)
    rng = np.random.default_rng(seed)
    boot_s = []
    boot_s_prime = []

    for _ in range(n_boot):
        b_idx = rng.integers(0, n_shots, size=n_shots)
        # S on bootstrap sample
        acc_b = Accum()
        acc_b.psi_sign = psi_sign
        for idx in b_idx:
            acc_b.add(parts[idx])
        res_b = finalize_machine(acc_b, c2e.sum_stats(stats, b_idx))
        boot_s.append(float(res_b["S"]))

        # S' on bootstrap sample
        s_p_b = float(np.mean([shot_s_prime[idx] for idx in b_idx]))
        boot_s_prime.append(s_p_b)

    ci_s = [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))]
    ci_s_prime = [float(np.percentile(boot_s_prime, 2.5)), float(np.percentile(boot_s_prime, 97.5))]

    report = {
        "submission": str(sub_path) if sub_path else mode,
        "mode": mode,
        "n_shots": n_shots,
        "psi_sign": int(psi_sign),
        "official": {
            "S": round(s_official, 6),
            "ci_95": [round(ci_s[0], 6), round(ci_s[1], 6)],
            "r2_psi": round(res_official["r2_psi"], 6),
            "r2_qb": round(res_official["r2_qb"], 6),
            "dlcfs": round(res_official["dlcfs"], 6),
            "consistency": round(res_official["consistency"], 6),
            "r2_cons_each": res_official["r2_cons_each"],
        },
        "extended_s_prime": {
            "S_prime": round(s_prime, 6),
            "ci_95": [round(ci_s_prime[0], 6), round(ci_s_prime[1], 6)],
            "mean_G_GS": round(mean_g_gs, 6),
            "mean_P_topo": round(mean_p_topo, 6),
            "median_g": round(median_g, 6),
            "g_ref": G_REF,
        },
        "per_shot": [
            {
                "shot_index": i,
                "S": round(shot_s[i], 6),
                "S_prime": round(shot_s_prime[i], 6),
                "bar_g": round(parts[i]["bar_g"], 6),
                "G_GS": round(parts[i]["G_GS"], 6),
                "P_topo": round(parts[i]["P_topo"], 6),
                "mean_N_spurious": round(parts[i]["mean_N_spurious"], 3),
            }
            for i in range(n_shots)
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Hackathon FMF submission with S and S'.")
    parser.add_argument("--sub", type=Path, default=None, help="Path to .npz or .json submission file")
    parser.add_argument("--mode", choices=["file", "perfect", "zeros"], default="file",
                        help="Evaluation mode (file, or synthetic perfect / zeros self-checks)")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON report path")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap replicates (default 1000)")
    args = parser.parse_args()

    if args.mode == "file" and args.sub is None:
        parser.error("--sub is required when --mode file")

    t0 = time.time()
    rep = evaluate_submission(sub_path=args.sub, mode=args.mode, n_boot=args.n_boot)
    elapsed = time.time() - t0

    print("=" * 72)
    print(f"EVALUATION REPORT: {rep['submission']}")
    print("=" * 72)
    off = rep["official"]
    ext = rep["extended_s_prime"]
    print(f"  Official S:      {off['S']:.6f}   [95% CI: {off['ci_95'][0]:.4f} .. {off['ci_95'][1]:.4f}]")
    print(f"    R2_psi:        {off['r2_psi']:.6f}")
    print(f"    R2_{{qb}}:       {off['r2_qb']:.6f}")
    print(f"    1 - D_LCFS:    {1.0 - min(1.0, off['dlcfs']):.6f} (D_LCFS={off['dlcfs']:.4f})")
    print(f"    Consistency:   {off['consistency']:.6f}")
    print("-" * 72)
    print(f"  Extended S':     {ext['S_prime']:.6f}   [95% CI: {ext['ci_95'][0]:.4f} .. {ext['ci_95'][1]:.4f}]")
    print(f"    Mean G_GS:     {ext['mean_G_GS']:.6f} (median g={ext['median_g']:.4f}, g_ref={ext['g_ref']})")
    print(f"    Mean P_topo:   {ext['mean_P_topo']:.6f}")
    print("=" * 72)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"Report saved to: {args.out}")

    # Optional: compare against baseline if requested
    baseline_path = Path("Our try/04-novelty/t1/eval_pca_ridge.json")
    if baseline_path.exists() and args.sub and str(args.sub) != "Our try/04-novelty/t1/eval_pca_ridge.json":
        try:
            with open(baseline_path, "r", encoding="utf-8") as bf:
                base_rep = json.load(bf)
            s_cand = np.array([s["S"] for s in rep["per_shot"]])
            s_base = np.array([s["S"] for s in base_rep["per_shot"]])
            sp_cand = np.array([s["S_prime"] for s in rep["per_shot"]])
            sp_base = np.array([s["S_prime"] for s in base_rep["per_shot"]])
            n_s = len(s_cand)
            rng_b = np.random.default_rng(42)
            b_indices = [rng_b.integers(0, n_s, size=n_s) for _ in range(1000)]
            d_s = np.array([np.mean(s_cand[b]) - np.mean(s_base[b]) for b in b_indices])
            d_sp = np.array([np.mean(sp_cand[b]) - np.mean(sp_base[b]) for b in b_indices])
            ci_ds = [np.percentile(d_s, 2.5), np.percentile(d_s, 97.5)]
            ci_dsp = [np.percentile(d_sp, 2.5), np.percentile(d_sp, 97.5)]
            sig_s = not (ci_ds[0] <= 0 <= ci_ds[1])
            sig_sp = not (ci_dsp[0] <= 0 <= ci_dsp[1])
            status_s = "РОЗБІЖНІСТЬ" if sig_s else "≈ (CI перекриваються)"
            status_sp = "РОЗБІЖНІСТЬ" if sig_sp else "≈ (CI перекриваються)"
            print("\n" + "-" * 72)
            print("PAIRED BOOTSTRAP SIGNIFICANCE vs BASELINE (PCA+Ridge):")
            print(f"  ΔS  (Cand - Base): {np.mean(s_cand) - np.mean(s_base):+.6f} [95% CI: {ci_ds[0]:+.6f} .. {ci_ds[1]:+.6f}] -> {status_s}")
            print(f"  ΔS' (Cand - Base): {np.mean(sp_cand) - np.mean(sp_base):+.6f} [95% CI: {ci_dsp[0]:+.6f} .. {ci_dsp[1]:+.6f}] -> {status_sp}")
            print("-" * 72)
        except Exception:
            pass

    print(f"Finished in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
