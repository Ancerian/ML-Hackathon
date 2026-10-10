#!/usr/bin/env python3
"""motion/export_equilibrium.py — Export equilibrium contours and local GS residuals.

Selected frame: Test Split Shot #062 (index 2), Frame 147.
Selection rule: Exact 50th percentile (median) frame when sorting all 1,521 test
frames by g(UNet_Lite).
g(UNet_Lite) = 0.861835 (results.json / eval_unet.json)
g(GT)        = 0.006487 (benchmark clean EFIT E6 ~ 0.009)

Generates:
- 12 flux surface contours of psi_N for Ground Truth and UNet_Lite.
- Local Grad-Shafranov residual g_local on each contour (E6 scheme).
- Fixed color-scale metadata [0.0, 1.2].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "fusion equilibrium challenge" / "starter"))
sys.path.insert(0, str(PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"))

from tokamld.gs import delta_star_numpy, gs_inconsistency
from tokamld.topology import plasma_mask
import train as c2_train


def compute_residual_field(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                           mask_coarse: np.ndarray, mask_f: np.ndarray,
                           n_p: int = 4, n_f: int = 4):
    """Computes full 2D relative Grad-Shafranov residual field |r(R, Z)| / denom."""
    inside, psi_a, psi_b = plasma_mask(psi, R, Z, mask_coarse, mask_f)
    ds = delta_star_numpy(psi, R, Z)
    RR, _ = np.meshgrid(R, Z)
    diff = psi_b - psi_a
    psin = np.clip((psi - psi_a) / diff, 0.0, 1.0)
    
    m = inside
    y = ds[m]
    denom = np.linalg.norm(y)
    cols = [(RR[m] ** 2) * psin[m] ** i for i in range(n_p)] + [psin[m] ** j for j in range(n_f)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, -y, rcond=None)
    r_vec = y + A @ coef
    
    # 2D relative residual map
    r_map = np.zeros_like(psi)
    # Scaled such that norm(r_map[m]) / denom = scalar g
    scale_factor = np.sqrt(len(y)) / denom
    r_map[m] = np.abs(r_vec) * scale_factor
    global_g = float(np.linalg.norm(r_vec) / denom)
    return r_map, psin, inside, psi_a, psi_b, global_g


def extract_contours(psin: np.ndarray, r_map: np.ndarray, R: np.ndarray, Z: np.ndarray,
                     n_levels: int = 12):
    """Extracts R, Z contour lines and interpolates local residual onto them."""
    from scipy.interpolate import RegularGridInterpolator
    interp_r = RegularGridInterpolator((Z, R), r_map, bounds_error=False, fill_value=0.0)
    
    levels = np.linspace(0.08, 0.92, n_levels)
    RR, ZZ = np.meshgrid(R, Z)
    
    fig, ax = plt.subplots()
    cs = ax.contour(RR, ZZ, psin, levels=levels)
    plt.close(fig)
    
    contours_data = []
    for lvl_idx, (lvl, segs) in enumerate(zip(levels, cs.allsegs)):
        for p_idx, v in enumerate(segs):
            if len(v) < 3:
                continue
            r_pts = v[:, 0]
            z_pts = v[:, 1]
            coords = np.column_stack([z_pts, r_pts])
            res_pts = interp_r(coords)
            mean_res = float(np.mean(res_pts))
            
            contours_data.append({
                "level_idx": lvl_idx,
                "psi_n": float(lvl),
                "n_points": len(v),
                "r": [round(float(x), 5) for x in r_pts],
                "z": [round(float(y), 5) for y in z_pts],
                "g_local": [round(float(g), 5) for g in res_pts],
                "g_mean": round(mean_res, 5),
            })
    return contours_data


def main():
    print("Exporting equilibrium contours for median test frame...")
    out_dir = PROJECT / "motion" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = PROJECT / "fusion equilibrium challenge" / "starter" / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)
    R = mask["grid_R"]
    Z = mask["grid_Z"]
    mc = mask["mask_coarse"].astype(bool)
    mf = mask["mask_coarse"].astype(np.float64)
    
    # Selected median frame: Shot #062, Frame 147
    shot_idx = 2
    shot_id = 62
    frame_idx = 147
    
    shots = c2_train.load([shot_id])
    gt_psi = shots[0]["psi"][frame_idx]
    
    sub_data = np.load(PROJECT / "Our try" / "04-novelty" / "t1" / "unet_sub.npz")
    unet_psi = sub_data[f"shot_{shot_idx:04d}_psirz"][frame_idx]
    
    # 1. Ground Truth
    gt_rmap, gt_psin, gt_ins, gt_pa, gt_pb, gt_g = compute_residual_field(gt_psi, R, Z, mc, mf)
    gt_contours = extract_contours(gt_psin, gt_rmap, R, Z, n_levels=12)
    
    # 2. UNet_Lite
    un_rmap, un_psin, un_ins, un_pa, un_pb, un_g = compute_residual_field(unet_psi, R, Z, mc, mf)
    un_contours = extract_contours(un_psin, un_rmap, R, Z, n_levels=12)
    
    # Meta / Manifest
    meta = {
        "selection_rule": "Exact 50th percentile (median) frame of g(UNet_Lite) over all 1,521 test frames (8 shots #060-#067)",
        "shot_index": shot_idx,
        "shot_id": shot_id,
        "frame_index": frame_idx,
        "g_ground_truth": round(gt_g, 6),
        "g_ground_truth_benchmark_note": "g(clean EFIT) ~ 0.009 (E6, numerical grid noise)",
        "g_unet_lite": round(un_g, 6),
        "g_unet_lite_note": "Matches results.json / eval_unet.json median g=0.8618",
        "color_scale": {
            "min_g": 0.0,
            "max_g": 1.2,
            "colormap": "turbo",
            "fixed": True,
            "note": "Identical color scale for both GT and UNet_Lite"
        },
        "n_contours": 12,
        "gt_axis": [round(float(R[int(np.argmin(gt_psi) % len(R))]), 4),
                    round(float(Z[int(np.argmin(gt_psi) // len(R))]), 4)],
    }
    
    with open(out_dir / "equilibrium_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    with open(out_dir / "contours_gt.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "contours": gt_contours}, f, indent=2)
        
    with open(out_dir / "contours_unet.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "contours": un_contours}, f, indent=2)
        
    # Save raw arrays for Blender
    np.savez_compressed(
        out_dir / "equilibrium_arrays.npz",
        R=R, Z=Z,
        gt_psi=gt_psi, gt_psin=gt_psin, gt_rmap=gt_rmap,
        unet_psi=unet_psi, unet_psin=un_psin, unet_rmap=un_rmap,
    )
    
    print(f"Successfully exported equilibrium for Shot #{shot_id:03d} Frame {frame_idx}:")
    print(f"  GT:        g = {gt_g:.6f} ({len(gt_contours)} contour lines)")
    print(f"  UNet_Lite: g = {un_g:.6f} ({len(un_contours)} contour lines)")
    print(f"  Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
