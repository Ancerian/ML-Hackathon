#!/usr/bin/env python3
"""motion/generate_highres_and_contact_sheets.py — High-res Poincare tracing, FTLE analysis, and contact sheets.

Performs:
1. FTLE analysis across Section B amplitudes: evaluates ratio k vs single-mode regular baseline.
   Records honest verdict: "стохастичність ймовірна (S≈1.0–1.1), кількісно не підтверджена".
2. High-res tracing for S >= 0.9 (120 lines x 300 turns) saved to motion/data/poincare_highres_chaos.json.
3. Generates motion/data/poincare_contact_sheet.png (all 15 sweep frames in a 3x5 grid).
4. Generates motion/data/contours_compare.png (2D panel GT vs UNet_Lite on #062 frame 147 with fixed colormap).
5. Updates motion/data/poincare_sweep.json with corrected single-mode S="н/д (одна мода)", units, and mm island widths.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tokamak-3d-viz" / "src"))

from tokamld.tracer.bfield import MagneticField2D
from tokamld.tracer.perturbation import HelicalPerturbation, ResonantMode
from tokamld.tracer.tracer import trace_poincare_cartesian, compute_ftle_cartesian
from tokviz.equilibrium import load_shot, select_frame, Equilibrium


def setup_equilibrium():
    shot_path = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset" / "data" / "diii_d_train" / "d3d_shot_00000a10ac.parquet"
    shot = load_shot(str(shot_path))
    idx = select_frame(shot)
    eq = Equilibrium(shot.psirz[idx], shot.grid_R, shot.grid_Z, machine="DIII-D")
    eq.find_axis()
    eq.set_boundary_from_contour(shot.lcfs_r[idx, :int(shot.lcfs_n[idx])], shot.lcfs_z[idx, :int(shot.lcfs_n[idx])])
    eq.calibrate_F(shot.q95[idx])
    bfield = MagneticField2D.from_tokviz_equilibrium(eq)
    return eq, bfield, shot, idx


def run_ftle_audit(bfield):
    print("\n>>> 1. FTLE Quantitative Audit across Section B...")
    t_turns = 50
    dphi = 2.0 * np.pi / 60.0
    
    r_arr = np.linspace(bfield.r_axis + 0.01, bfield.r_axis + 0.55, 200)
    psn_arr = np.array([float(bfield.psi_n_at(jnp.array(r), jnp.array(bfield.z_axis))) for r in r_arr])
    
    # Regular reference: single mode 2/1, A=1e-3
    pert_reg = HelicalPerturbation([ResonantMode(m=2, n=1, amp=1.0e-3, phase=0.0)])
    
    # Overlap region psi_n ~ 0.82
    r_test = float(np.interp(0.82, psn_arr, r_arr))
    s_test = jnp.array([r_test, bfield.z_axis], dtype=jnp.float64)
    ftle_reg = compute_ftle_cartesian(s_test, t_turns, dphi, bfield, pert_reg)
    
    amps_audit = [1.0e-3, 3.0e-3, 4.8e-3, 5.5e-3, 6.0e-3]
    ftle_results = {}
    for a in amps_audit:
        pert_b = HelicalPerturbation([
            ResonantMode(m=2, n=1, amp=a, phase=0.0),
            ResonantMode(m=3, n=1, amp=a, phase=0.0)
        ])
        ftle_b = compute_ftle_cartesian(s_test, t_turns, dphi, bfield, pert_b)
        k = ftle_b / max(ftle_reg, 1e-12)
        s_chir = float(0.457559 * np.sqrt(a / 1.0e-3))
        ftle_results[f"{a:.1e}"] = {
            "amp": float(a),
            "chirikov_S": round(s_chir, 4),
            "ftle_regular": round(float(ftle_reg), 6),
            "ftle_val": round(float(ftle_b), 6),
            "ratio_k": round(float(k), 3),
            "criterion_k2_met": bool(k >= 2.0)
        }
        print(f"  A={a:.1e}: S={s_chir:.3f}, FTLE={ftle_b:.4e} rad^-1, ratio k={k:.2f}")
        
    return ftle_results


def run_highres_tracing(bfield):
    print("\n>>> 2. High-Resolution Tracing for S >= 0.9 (120 lines x 300 turns)...")
    out_file = PROJECT / "motion" / "data" / "poincare_highres_chaos.json"
    
    r_axis = float(bfield.r_axis)
    z_axis = float(bfield.z_axis)
    
    r_seeds = np.linspace(r_axis + 0.03, r_axis + 0.54, 120)
    z_seeds = np.full_like(r_seeds, z_axis)
    states_0 = jnp.column_stack([r_seeds, z_seeds])
    
    n_turns = 300
    dphi = 2.0 * np.pi / 60.0
    
    # 4 high-amplitude cases: S >= 0.9
    high_amps = [4.0e-3, 4.8e-3, 5.5e-3, 6.0e-3]
    highres_frames = []
    
    t0_all = time.perf_counter()
    for idx_h, a in enumerate(high_amps):
        s_chir = float(0.457559 * np.sqrt(a / 1.0e-3))
        w1_psin = float(0.0735895 * np.sqrt(a / 1.0e-3))
        w1_mm = float(w1_psin / 2.7116 * 1000.0)
        
        t0 = time.perf_counter()
        pert = HelicalPerturbation([
            ResonantMode(m=2, n=1, amp=float(a), phase=0.0),
            ResonantMode(m=3, n=1, amp=float(a), phase=0.0)
        ])
        punct, alive = trace_poincare_cartesian(states_0, n_turns=n_turns, dphi=dphi, bfield=bfield, pert=pert)
        punct_np = np.asarray(punct)
        alive_np = np.asarray(alive)
        dt = time.perf_counter() - t0
        print(f"  High-res frame {idx_h+1}/{len(high_amps)}: A={a:.1e}, S={s_chir:.3f} traced in {dt:.2f} s")
        
        lines_data = []
        for line_idx in range(len(r_seeds)):
            pts = punct_np[line_idx, alive_np[line_idx]]
            if len(pts) > 0:
                sub = pts[::2]  # Subsample every 2 turns for JSON size
                lines_data.append({
                    "seed_id": int(line_idx),
                    "r_seed": round(float(r_seeds[line_idx]), 4),
                    "points": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in sub]
                })
                
        highres_frames.append({
            "amp": float(a),
            "chirikov_S": round(s_chir, 4),
            "island_width_2_1_psin": round(w1_psin, 4),
            "island_width_2_1_mm": round(w1_mm, 2),
            "n_lines": len(lines_data),
            "n_turns": n_turns,
            "calc_time_sec": round(dt, 2),
            "lines": lines_data
        })
        
    total_time = time.perf_counter() - t0_all
    print(f"High-res tracing completed in {total_time:.2f} s")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Високороздільні перерізи Пуанкаре для режимів перекриття S >= 0.9 (120 траєкторій, 300 обертів)",
            "total_calc_time_sec": round(total_time, 2),
            "frames": highres_frames
        }, f)
    print(f"Saved high-res data to: {out_file}")
    return total_time


def update_poincare_json_and_make_contact_sheet():
    print("\n>>> 3. Updating poincare_sweep.json and rendering contact sheet...")
    sweep_path = PROJECT / "motion" / "data" / "poincare_sweep.json"
    with open(sweep_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    frames = data["frames"]
    meta = data["meta"]
    
    # Update labels and metrics per user request
    grad_psi_n = 2.7116  # dpsi_N / dR at midplane in m^-1 (from T12 THEORY.md)
    for fr in frames:
        a = fr["amp"]
        w1_psin = float(0.0735895 * np.sqrt(a / 1.0e-3)) if a > 0 else 0.0
        w1_mm = float(w1_psin / grad_psi_n * 1000.0) if a > 0 else 0.0
        fr["amp_units"] = "fraction of poloidal flux range |psi_bdy - psi_axis|"
        fr["island_width_2_1_psin"] = round(w1_psin, 4)
        fr["island_width_2_1_mm"] = round(w1_mm, 2)
        
        if fr["section"] == "A_single_mode_2_1":
            fr["chirikov_S_display"] = "н/д (одна мода)"
            fr["chirikov_S"] = "н/д (одна мода)"
            fr["note"] = "Єдина мода 2/1 ніколи не дає динамічного хаосу (теорема E20)"
        else:
            s_val = fr["chirikov_S"]
            if s_val >= 1.0:
                fr["note"] = "стохастичність ймовірна (S≈1.0–1.1), кількісно не підтверджена"
            else:
                fr["note"] = "Острови розділені інваріантними торами"
                
    meta["single_mode_chirikov"] = "н/д (одна мода)"
    meta["stochasticity_verdict"] = "стохастичність ймовірна (S≈1.0–1.1), кількісно не підтверджена"
    meta["section_b_range_explanation"] = (
        "Діапазон амплітуд секції Б розширено до A=6.0e-3 (порівняно з номіналом 1.0e-3 у tokamak-3d-viz/README.md) "
        "виключно як штучне вакуумне збурення для досягнення зони перекриття резонансів Чирикова S >= 1.0."
    )
    
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f, indent=2)
        
    # --- Generate Poincare Contact Sheet (3x5 grid) ---
    fig, axes = plt.subplots(3, 5, figsize=(22, 14), facecolor="#0e1117")
    fig.suptitle(
        "DIII-D #203702: Свіп перерізів Пуанкаре за амплітудою збурення\n"
        "(Секція А: мода 2/1, S = н/д | Секція Б: моди 2/1 + 3/1; вакуумні моди, не симуляція плазми)",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )
    
    wall_r = [1.01, 1.01, 1.38, 2.38, 2.38, 1.70, 1.01]
    wall_z = [-1.30, 1.25, 1.38, 0.80, -0.90, -1.35, -1.30]
    
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, 34))
    
    for idx, (ax, fr) in enumerate(zip(axes.flat, frames)):
        ax.set_facecolor("#161b22")
        ax.plot(wall_r, wall_z, color="#484f58", lw=1.2, ls="--", alpha=0.7)
        
        for l_idx, line in enumerate(fr["lines"]):
            pts = np.array(line["points"])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=0.35, color=colors[l_idx % len(colors)], alpha=0.75, rasterized=True)
                
        # Titles
        sec_label = "Секція А (2/1)" if "A" in fr["section"] else "Секція Б (2/1+3/1)"
        s_disp = fr["chirikov_S_display"] if "chirikov_S_display" in fr else f"S = {fr['chirikov_S']:.3f}"
        w_mm = fr["island_width_2_1_mm"]
        w_psi = fr["island_width_2_1_psin"]
        
        title_text = f"#{idx+1}: {sec_label}\n$A = {fr['amp']:.2e}$ | {s_disp}\n$w_{{2/1}} = {w_psi:.3f}\\,\\psi_N$ ({w_mm:.1f} мм)"
        
        title_color = "#58a6ff" if "A" in fr["section"] else ("#f85149" if (isinstance(fr["chirikov_S"], (int, float)) and fr["chirikov_S"] >= 1.0) else "#e3b341")
        ax.set_title(title_text, color=title_color, fontsize=9.5, fontweight="semibold", pad=6)
        
        ax.set_xlim(1.0, 2.45)
        ax.set_ylim(-1.4, 1.4)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.grid(color="#30363d", lw=0.4, alpha=0.5)
        
        if idx % 5 == 0:
            ax.set_ylabel("Z [м]", color="#8b949e", fontsize=9)
        if idx >= 10:
            ax.set_xlabel("R [м]", color="#8b949e", fontsize=9)
            
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    sheet_out = PROJECT / "motion" / "data" / "poincare_contact_sheet.png"
    plt.savefig(sheet_out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Generated Poincare Contact Sheet: {sheet_out}")
    return sheet_out


def generate_contours_compare_png():
    print("\n>>> 4. Generating contours_compare.png (GT vs UNet_Lite, DIII-D #062 Frame 147)...")
    arr_path = PROJECT / "motion" / "data" / "equilibrium_arrays.npz"
    arrs = np.load(arr_path)
    R = arrs["R"]
    Z = arrs["Z"]
    RR, ZZ = np.meshgrid(R, Z)
    
    gt_psi = arrs["gt_psi"]
    gt_rmap = arrs["gt_rmap"]
    gt_psin = arrs["gt_psin"]
    
    unet_psi = arrs["unet_psi"]
    unet_rmap = arrs["unet_rmap"]
    unet_psin = arrs["unet_psin"]
    
    norm = Normalize(vmin=0.0, vmax=1.2)
    cmap = plt.cm.turbo
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5), facecolor="#0e1117")
    fig.suptitle(
        "DIII-D розряд #062, кадр 147: Медіанний за $g(\\psi)$ кадр тестового спліту (#060–#067)\n"
        "Порівняння контурів полоїдального потоку $\\psi_N$ та локальної нев'язки Ґреда–Шафранова $g_{\\mathrm{local}}$ (E6)",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )
    
    levels = np.linspace(0.08, 0.92, 12)
    
    # 1. Ground Truth Panel
    ax1.set_facecolor("#161b22")
    c1 = ax1.pcolormesh(RR, ZZ, gt_rmap, cmap=cmap, norm=norm, shading="auto", alpha=0.88, rasterized=True)
    cs1 = ax1.contour(RR, ZZ, gt_psin, levels=levels, colors="white", linewidths=1.0, alpha=0.8)
    ax1.clabel(cs1, fmt="%.2f", colors="white", fontsize=7.5)
    ax1.set_title(
        "Істинне поле (Ground Truth, EFIT)\n"
        "$g(\\mathrm{GT}) = 0.0065$ (чистий EFIT [E6] ~ 0.009, шум сітки $65\\times 65$)\n"
        "Гладкі похідні, фізична рівновага збережена",
        color="#58a6ff", fontsize=11, fontweight="semibold", pad=8
    )
    ax1.set_xlim(R.min(), R.max())
    ax1.set_ylim(Z.min(), Z.max())
    ax1.set_xlabel("R [м]", color="#8b949e", fontsize=10)
    ax1.set_ylabel("Z [м]", color="#8b949e", fontsize=10)
    ax1.tick_params(colors="#8b949e")
    ax1.grid(color="#30363d", lw=0.5, alpha=0.5)
    
    # 2. UNet_Lite Panel
    ax2.set_facecolor("#161b22")
    c2 = ax2.pcolormesh(RR, ZZ, unet_rmap, cmap=cmap, norm=norm, shading="auto", alpha=0.88, rasterized=True)
    cs2 = ax2.contour(RR, ZZ, unet_psin, levels=levels, colors="white", linewidths=1.0, alpha=0.8)
    ax2.clabel(cs2, fmt="%.2f", colors="white", fontsize=7.5)
    ax2.set_title(
        "Передбачення UNet_Lite\n"
        "$g(\\mathrm{UNet}) = 0.8618$ (медіанний кадр тесту; $R^2_\\psi = 0.977$)\n"
        "Високочастотний шум других похідних оператора $\\Delta^*$ [E22]",
        color="#f85149", fontsize=11, fontweight="semibold", pad=8
    )
    ax2.set_xlim(R.min(), R.max())
    ax2.set_ylim(Z.min(), Z.max())
    ax2.set_xlabel("R [м]", color="#8b949e", fontsize=10)
    ax2.tick_params(colors="#8b949e")
    ax2.grid(color="#30363d", lw=0.5, alpha=0.5)
    
    # Unified Colorbar
    cbar_ax = fig.add_axes([0.18, 0.08, 0.64, 0.025])
    cb = fig.colorbar(c1, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Локальна відносна нев'язка Ґреда–Шафранова $g_{\\mathrm{local}}$ (фіксована шкала [0.0 .. 1.2])",
                 color="white", fontsize=10, labelpad=6)
    cb.ax.tick_params(colors="white", labelsize=9)
    
    # Explanation note
    fig.text(
        0.5, 0.015,
        "Примітка: Медіана лідерборду 0.8627 обчислюється як середня з медіан окремих розрядів mean(bar_g_s), "
        "тоді як 0.8618 — це пряма глобальна медіана за всіма 1 521 індивідуальними кадрами тесту.",
        color="#8b949e", fontsize=8.5, ha="center"
    )
    
    plt.subplots_adjust(bottom=0.18, top=0.88, left=0.08, right=0.95, wspace=0.16)
    compare_out = PROJECT / "motion" / "data" / "contours_compare.png"
    plt.savefig(compare_out, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Generated Contours Comparison: {compare_out}")
    return compare_out


def main():
    print("=" * 70)
    print("  HIGH-RES POINCARE, FTLE AUDIT & CONTACT SHEET GENERATION")
    print("=" * 70)
    
    eq, bfield, shot, idx = setup_equilibrium()
    
    # 1. FTLE audit
    ftle_results = run_ftle_audit(bfield)
    
    # 2. High-res tracing for S >= 0.9
    calc_time = run_highres_tracing(bfield)
    
    # 3. Update Poincare JSON and make contact sheet
    sheet_path = update_poincare_json_and_make_contact_sheet()
    
    # 4. Generate Contours comparison
    compare_path = generate_contours_compare_png()
    
    print("\n" + "=" * 70)
    print("ALL ARTIFACTS GENERATED SUCCESSFULLY:")
    print(f"  Poincare Contact Sheet: {sheet_path}")
    print(f"  Contours Comparison:    {compare_path}")
    print(f"  High-res JSON:          motion/data/poincare_highres_chaos.json (took {calc_time:.2f} s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
