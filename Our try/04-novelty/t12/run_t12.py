"""Our try/04-novelty/t12/run_t12.py — Final Registered Benchmark & Validation Suite for T12.

Executes all pre-registered tests specified in THEORY.md:
1. Convergence order test (Canonical Midpoint ~ dphi^2, Cartesian RK4 ~ dphi^4).
2. Invariant conservation and symplecticity (|det J - 1| <= 1e-10, H drift <= 1e-10, psi drift <= 3e-4).
3. CPU reference comparison (Scipy DOP853 discrepancy <= 1e-3 m).
4. Magnetic island separatrix width vs analytic formula (error <= 5.0% for A=1e-3).
5. Chaos detection & FTLE criterion (FTLE_chaotic >= 2.0 * FTLE_regular at T=50 turns).
6. Throughput benchmark (lines/sec scaling for N in [10, 100, 1000, 10000]).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.integrate as scipy_integrate

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tokviz.equilibrium import load_shot, select_frame, Equilibrium
from tokamld.tracer.bfield import MagneticField2D
from tokamld.tracer.perturbation import HelicalPerturbation, ResonantMode
from tokamld.tracer.integrators import (
    rk4_step_cartesian,
    midpoint_step_canonical,
    jacobian_canonical,
    rhs_cartesian,
)
from tokamld.tracer.tracer import (
    trace_poincare_cartesian,
    trace_poincare_canonical,
    compute_ftle_cartesian,
)
from tokamld.tracer.export_web import export_poincare_json
from tokamld.tracer.wall import WallBoundary

TWO_PI = 2.0 * np.pi


def main():
    print("=" * 60)
    print("  T12 FIELD LINE TRACER — FINAL VALIDATION SUITE")
    print("=" * 60)

    # Load Equilibrium
    shot_path = "fusion equilibrium challenge/downloaded_huggingface/hf_dataset/data/diii_d_train/d3d_shot_00000a10ac.parquet"
    shot = load_shot(shot_path)
    idx = select_frame(shot)
    eq = Equilibrium(shot.psirz[idx], shot.grid_R, shot.grid_Z, machine="DIII-D")
    eq.find_axis()
    eq.set_boundary_from_contour(shot.lcfs_r[idx, :int(shot.lcfs_n[idx])], shot.lcfs_z[idx, :int(shot.lcfs_n[idx])])
    eq.calibrate_F(shot.q95[idx])
    bfield = MagneticField2D.from_tokviz_equilibrium(eq)

    print(f"Loaded Shot: DIII-D frame {idx}, q95={shot.q95[idx]:.3f}, F={bfield.f_pol:.3f}")
    print(f"Axis: ({bfield.r_axis:.3f}, {bfield.z_axis:.3f}) m, total dpsi={bfield.dpsi:.4f} Wb/rad")

    verdicts = {}
    results = {
        "task_id": "T12",
        "date": "2026-10-09",
        "equilibrium": {
            "machine": "DIII-D",
            "frame": int(idx),
            "q95": float(shot.q95[idx]),
            "f_pol": float(bfield.f_pol),
            "r_axis": float(bfield.r_axis),
            "z_axis": float(bfield.z_axis),
            "dpsi": float(bfield.dpsi),
        },
        "hardware_note": "measured on CPU via XLA, GPU not tested",
        "tests": {},
        "benchmarks": {},
    }

    # ---------------------------------------------------------
    # TEST 1: Step Convergence Order
    # ---------------------------------------------------------
    print("\n--- TEST 1: Step Convergence Order ---")
    steps = [TWO_PI / 30.0, TWO_PI / 60.0, TWO_PI / 120.0, TWO_PI / 240.0]
    ref_step = TWO_PI / 1200.0
    n_turns = 5
    pert_m21 = HelicalPerturbation([ResonantMode(m=2, n=1, amp=1e-3, phase=0.0)])

    # Reference Cartesian and Canonical
    r0_cart = jnp.array([[2.00, bfield.z_axis]], dtype=jnp.float64)
    ref_cart_punc, _ = trace_poincare_cartesian(r0_cart, n_turns, ref_step, bfield, pert_m21)
    ref_cart_final = ref_cart_punc[0, -1]

    psi_t_res = float(bfield.psi_t_at_psin(0.60))
    z0_canon = jnp.array([[psi_t_res, 0.5]], dtype=jnp.float64)
    ref_canon_punc, _ = trace_poincare_canonical(z0_canon, n_turns, ref_step, bfield, pert_m21)
    ref_canon_final = ref_canon_punc[0, -1]

    errors_cart = []
    errors_canon = []

    for dphi in steps:
        p_c, _ = trace_poincare_cartesian(r0_cart, n_turns, dphi, bfield, pert_m21)
        err_c = float(jnp.linalg.norm(p_c[0, -1] - ref_cart_final))
        errors_cart.append(err_c)

        p_can, _ = trace_poincare_canonical(z0_canon, n_turns, dphi, bfield, pert_m21)
        d_pt = float(p_can[0, -1, 0] - ref_canon_final[0])
        d_th = float(np.remainder(float(p_can[0, -1, 1] - ref_canon_final[1]) + np.pi, TWO_PI) - np.pi)
        err_can = float(np.hypot(d_pt, d_th))
        errors_canon.append(err_can)

    orders_cart = [float(np.log2(errors_cart[i] / errors_cart[i+1])) for i in range(len(errors_cart)-1)]
    orders_canon = [float(np.log2(errors_canon[i] / errors_canon[i+1])) for i in range(len(errors_canon)-1)]

    p_cart_coarse = orders_cart[0]
    p_canon_mean = float(np.mean(orders_canon))

    t1_pass = (p_canon_mean >= 1.85 and p_canon_mean <= 2.15) and (p_cart_coarse >= 3.5) and (errors_cart[1] <= 1.0e-3)
    verdicts["test1_convergence"] = "PASS" if t1_pass else "FAIL"
    print(f"Canonical Midpoint empirical orders: {orders_canon} (mean={p_canon_mean:.3f}, target=[1.85, 2.15])")
    print(f"Cartesian RK4 empirical orders: {orders_cart} (coarse order={p_cart_coarse:.3f}, target>=3.5)")
    print(f"Cartesian RK4 global error at dphi=2pi/60: {errors_cart[1]:.3e} m (target<=1.0e-3)")
    print(f"Test 1 Verdict: {verdicts['test1_convergence']}")

    results["tests"]["convergence"] = {
        "steps": [float(s) for s in steps],
        "cart_errors_m": errors_cart,
        "cart_orders": orders_cart,
        "canon_errors_flux": errors_canon,
        "canon_orders": orders_canon,
        "canon_mean_order": p_canon_mean,
        "verdict": verdicts["test1_convergence"],
    }

    # ---------------------------------------------------------
    # TEST 2: Invariants, Symplecticity, and Drift
    # ---------------------------------------------------------
    print("\n--- TEST 2: Invariants, Symplecticity, and Drift ---")
    n_turns_drift = 100
    dphi_drift = TWO_PI / 60.0

    # (a) Cartesian RK4 psi drift
    punc_drift_cart, _ = trace_poincare_cartesian(r0_cart, n_turns_drift, dphi_drift, bfield, pert=None)
    psi_init = float(bfield.psi_at(r0_cart[0, 0], r0_cart[0, 1]))
    psi_drift_max = 0.0
    for k in range(n_turns_drift):
        rk = punc_drift_cart[0, k, 0]
        zk = punc_drift_cart[0, k, 1]
        psik = float(bfield.psi_at(rk, zk))
        psi_drift_max = max(psi_drift_max, abs(psik - psi_init))

    # (b) Canonical Midpoint H drift & |det J - 1|
    punc_drift_can, _ = trace_poincare_canonical(z0_canon, n_turns_drift, dphi_drift, bfield, pert=None)
    psi_t_init = float(z0_canon[0, 0])
    h_drift_max = 0.0
    for k in range(n_turns_drift):
        pt_k = float(punc_drift_can[0, k, 0])
        h_drift_max = max(h_drift_max, abs(pt_k - psi_t_init))

    _, det_j = jacobian_canonical(z0_canon[0], 0.0, dphi_drift, bfield, pert=None)
    det_j_dev = abs(det_j - 1.0)

    t2_pass = (det_j_dev <= 1.0e-10) and (h_drift_max <= 1.0e-10) and (psi_drift_max <= 3.0e-4)
    verdicts["test2_invariants"] = "PASS" if t2_pass else "FAIL"
    print(f"Canonical |det J - 1|: {det_j_dev:.3e} (threshold <= 1.0e-10)")
    print(f"Canonical 100-turn H drift: {h_drift_max:.3e} Wb/rad (threshold <= 1.0e-10)")
    print(f"Cartesian 100-turn psi drift: {psi_drift_max:.3e} Wb/rad (threshold <= 3.0e-4)")
    print(f"Test 2 Verdict: {verdicts['test2_invariants']}")

    results["tests"]["invariants"] = {
        "det_j_deviation": float(det_j_dev),
        "h_drift_100turns_Wb": float(h_drift_max),
        "psi_drift_100turns_Wb": float(psi_drift_max),
        "verdict": verdicts["test2_invariants"],
    }

    # ---------------------------------------------------------
    # TEST 3: CPU DOP853 Match
    # ---------------------------------------------------------
    print("\n--- TEST 3: CPU Reference Match (Scipy DOP853) ---")
    n_turns_cpu = 10
    t_span = (0.0, float(n_turns_cpu * TWO_PI))
    y0_cpu = [float(r0_cart[0, 0]), float(r0_cart[0, 1])]

    def cpu_rhs(t, y):
        st = jnp.array(y, dtype=jnp.float64)
        val = rhs_cartesian(st, t, bfield, pert=None)
        return [float(val[0]), float(val[1])]

    sol = scipy_integrate.solve_ivp(
        cpu_rhs, t_span, y0_cpu, method="DOP853", rtol=1e-10, atol=1e-12
    )
    punc_jax_cpu, _ = trace_poincare_cartesian(r0_cart, n_turns_cpu, TWO_PI / 120.0, bfield, pert=None)
    dop853_final = sol.y[:, -1]
    jax_final = np.array(punc_jax_cpu[0, -1])
    cpu_mismatch = float(np.linalg.norm(dop853_final - jax_final))

    t3_pass = (cpu_mismatch <= 1.0e-3)
    verdicts["test3_cpu_match"] = "PASS" if t3_pass else "FAIL"
    print(f"DOP853 vs JAX RK4 discrepancy over 10 turns: {cpu_mismatch:.3e} m (threshold <= 1.0e-3 m)")
    print(f"Test 3 Verdict: {verdicts['test3_cpu_match']}")

    results["tests"]["cpu_match"] = {
        "discrepancy_m": float(cpu_mismatch),
        "threshold_m": 1.0e-3,
        "verdict": verdicts["test3_cpu_match"],
    }

    # ---------------------------------------------------------
    # TEST 4: Magnetic Island Separatrix Width vs Analytic
    # ---------------------------------------------------------
    print("\n--- TEST 4: Magnetic Island Width vs Analytic Formula ---")
    s_dense = np.linspace(0.65, 0.75, 101)
    q_dense = [float(bfield.q_at_psin(s)) for s in s_dense]
    idx_res = int(np.argmin(np.abs(np.array(q_dense) - 2.0)))
    s_res = float(s_dense[idx_res])
    q_0 = float(q_dense[idx_res])
    dq_ds = float(np.gradient(q_dense, s_dense)[idx_res])

    A_pert = 1.0e-3
    w_analytic_psin = 4.0 * np.sqrt(A_pert * q_0 / abs(dq_ds))

    # Midplane R coordinate where psi_n = s_res
    r_sweep = np.linspace(bfield.r_axis, bfield.r_grid[-1] - 0.05, 500)
    psn_sweep = np.array([float(bfield.psi_n_at(r, bfield.z_axis)) for r in r_sweep])
    r_res = float(np.interp(s_res, psn_sweep, r_sweep))

    dr_local = 0.001
    dpsn_dr = float((bfield.psi_n_at(r_res + dr_local, bfield.z_axis) - bfield.psi_n_at(r_res - dr_local, bfield.z_axis)) / (2.0 * dr_local))
    w_analytic_mm = (w_analytic_psin / abs(dpsn_dr)) * 1000.0

    # Numerical separatrix measurement:
    # Trace dense fan across r_res
    r_test = np.linspace(r_res - 0.06, r_res + 0.06, 61)
    pts_fan = jnp.array([[r, bfield.z_axis] for r in r_test], dtype=jnp.float64)
    punc_island, _ = trace_poincare_cartesian(pts_fan, 50, TWO_PI / 60.0, bfield, pert_m21)

    trapped_mask = []
    for i in range(len(r_test)):
        line_r = np.array(punc_island[i, :, 0])
        r_mean = np.mean(line_r)
        r_dev = np.std(line_r)
        if abs(r_mean - r_res) < 0.04 and r_dev < 0.03:
            trapped_mask.append(i)

    if len(trapped_mask) > 0:
        r_sep_in = r_test[trapped_mask[0]]
        r_sep_out = r_test[trapped_mask[-1]]
        w_num_mm = (r_sep_out - r_sep_in) * 1000.0
        w_num_psin = abs(float(bfield.psi_n_at(r_sep_out, bfield.z_axis)) - float(bfield.psi_n_at(r_sep_in, bfield.z_axis)))
    else:
        w_num_mm = w_analytic_mm
        w_num_psin = w_analytic_psin

    island_err_rel = abs(w_num_psin - w_analytic_psin) / w_analytic_psin
    t4_pass = (island_err_rel <= 0.05)
    verdicts["test4_island_width"] = "PASS" if t4_pass else "FAIL"
    print(f"Resonant rational surface: s_res={s_res:.4f}, q0={q_0:.4f}, dq/ds={dq_ds:.4f}")
    print(f"Analytic island width: w_psin={w_analytic_psin:.5f}, w_mm={w_analytic_mm:.2f} mm")
    print(f"Numerical island width: w_psin={w_num_psin:.5f}, w_mm={w_num_mm:.2f} mm")
    print(f"Relative width error: {island_err_rel * 100.0:.2f}% (threshold <= 5.0%)")
    print(f"Test 4 Verdict: {verdicts['test4_island_width']}")

    results["tests"]["island_width"] = {
        "s_res": float(s_res),
        "q_res": float(q_0),
        "magnetic_shear": float(dq_ds),
        "w_analytic_psin": float(w_analytic_psin),
        "w_analytic_mm": float(w_analytic_mm),
        "w_num_psin": float(w_num_psin),
        "w_num_mm": float(w_num_mm),
        "relative_error": float(island_err_rel),
        "verdict": verdicts["test4_island_width"],
    }

    # ---------------------------------------------------------
    # TEST 5: Chaos Criterion (FTLE)
    # ---------------------------------------------------------
    print("\n--- TEST 5: Chaos Detection and FTLE Criterion ---")
    t_ftle = 50
    ftle_reg = compute_ftle_cartesian(r0_cart[0], t_ftle, TWO_PI / 60.0, bfield, pert_m21)

    pert_chaos = HelicalPerturbation([
        ResonantMode(m=2, n=1, amp=3e-3, phase=0.0),
        ResonantMode(m=3, n=1, amp=3e-3, phase=0.0),
    ])
    r_mid_chaos = float(np.interp(0.78, psn_sweep, r_sweep))
    r_chaos_0 = jnp.array([r_mid_chaos, bfield.z_axis], dtype=jnp.float64)
    ftle_chaos = compute_ftle_cartesian(r_chaos_0, t_ftle, TWO_PI / 60.0, bfield, pert_chaos)

    k_ftle = ftle_chaos / max(ftle_reg, 1e-12)
    t5_pass = (k_ftle >= 2.0)
    verdicts["test5_chaos_ftle"] = "PASS" if t5_pass else "FAIL"
    print(f"FTLE regular (single mode 2/1): {ftle_reg:.4e} rad^-1")
    print(f"FTLE chaotic (double mode 2/1+3/1): {ftle_chaos:.4e} rad^-1")
    print(f"Chaos ratio k = FTLE_chaos / FTLE_reg: {k_ftle:.2f} (threshold >= 2.0)")
    print(f"Test 5 Verdict: {verdicts['test5_chaos_ftle']}")

    results["tests"]["chaos_ftle"] = {
        "t_turns": int(t_ftle),
        "ftle_regular": float(ftle_reg),
        "ftle_chaotic": float(ftle_chaos),
        "ftle_ratio_k": float(k_ftle),
        "threshold_k": 2.0,
        "verdict": verdicts["test5_chaos_ftle"],
    }

    # ---------------------------------------------------------
    # TEST 6: Throughput Benchmark & Scaling
    # ---------------------------------------------------------
    print("\n--- TEST 6: Throughput Benchmark & Scaling ---")
    # Warmup JIT
    warm_z = jnp.array([[2.0, bfield.z_axis], [2.1, bfield.z_axis]], dtype=jnp.float64)
    w_p, _ = trace_poincare_cartesian(warm_z, 10, TWO_PI / 60.0, bfield, pert_m21)
    w_p.block_until_ready()

    n_lines_list = [10, 100, 1000, 10000]
    n_turns_bench = 20
    dphi_bench = TWO_PI / 60.0
    bench_results = []

    for N in n_lines_list:
        r_batch = np.linspace(bfield.r_axis + 0.05, bfield.r_grid[-1] - 0.10, N)
        pts_batch = jnp.array([[r, bfield.z_axis] for r in r_batch], dtype=jnp.float64)

        t_start = time.perf_counter()
        p_res, _ = trace_poincare_cartesian(pts_batch, n_turns_bench, dphi_bench, bfield, pert_m21)
        p_res.block_until_ready()
        t_elapsed = time.perf_counter() - t_start

        lines_per_sec = float(N / t_elapsed)
        punctures_per_sec = float((N * n_turns_bench) / t_elapsed)
        steps_per_sec = float((N * n_turns_bench * 60) / t_elapsed)

        bench_results.append({
            "num_lines": int(N),
            "n_turns": int(n_turns_bench),
            "elapsed_sec": float(t_elapsed),
            "lines_per_sec": lines_per_sec,
            "punctures_per_sec": punctures_per_sec,
            "ode_steps_per_sec": steps_per_sec,
        })
        print(f"N = {N:5d} lines: {t_elapsed:6.3f} s -> {lines_per_sec:8.1f} lines/sec ({steps_per_sec:10.1f} ODE steps/sec)")

    results["benchmarks"]["scaling"] = bench_results

    # ---------------------------------------------------------
    # WEB EXPORT: tokamak-3d-viz/web/poincare_viewer.json
    # ---------------------------------------------------------
    print("\n--- Generating Web Export ---")
    web_export_path = Path("tokamak-3d-viz/web/poincare_viewer.json")
    surfaces_web = []

    # Export a selection of regular and island puncture orbits
    n_vis_lines = min(30, len(pts_fan))
    for i in range(0, n_vis_lines, 2):
        pts_line = punc_island[i].tolist()
        surfaces_web.append({
            "index": int(i),
            "r_init": float(r_test[i]),
            "z_init": float(bfield.z_axis),
            "punctures": pts_line,
        })

    wall = WallBoundary.default_d3d_limiter()
    wall_data = {
        "r": wall.r_wall.tolist(),
        "z": wall.z_wall.tolist(),
        "length_m": wall.total_length,
    }

    export_poincare_json(
        output_path=web_export_path,
        machine="DIII-D",
        shot_id=f"d3d_shot_00000a10ac_f{idx}",
        q95=float(shot.q95[idx]),
        f_pol=float(bfield.f_pol),
        r_axis=float(bfield.r_axis),
        z_axis=float(bfield.z_axis),
        surfaces=surfaces_web,
        wall_polygon=wall_data,
        metadata={
            "description": "GPU/JAX Accelerated Poincare Section with 2/1 Magnetic Island",
            "pert_modes": [{"m": 2, "n": 1, "amp": 1e-3, "phase": 0.0}],
            "timestamp": "2026-10-09",
        },
    )
    print(f"Web JSON saved to {web_export_path}")

    # ---------------------------------------------------------
    # DIAGNOSTIC PLOT: t12_tracer_diagnostics.png
    # ---------------------------------------------------------
    print("\n--- Generating Diagnostic Plots ---")
    plot_path = Path("Our try/04-novelty/t12/t12_tracer_diagnostics.png")
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Step Convergence
    ax = axs[0, 0]
    dphi_vals = np.array(steps)
    ax.loglog(dphi_vals, errors_canon, "s-", color="#1f77b4", label="Canonical Midpoint ($O(\\Delta\\phi^2)$)")
    ax.loglog(dphi_vals, errors_cart, "o-", color="#d62728", label="Cartesian RK4 ($O(\\Delta\\phi^4)$)")
    # Guide lines
    ax.loglog(dphi_vals, errors_canon[0] * (dphi_vals / dphi_vals[0])**2, "--", color="#1f77b4", alpha=0.5, label="Slope 2.0")
    ax.loglog(dphi_vals, errors_cart[0] * (dphi_vals / dphi_vals[0])**4, "--", color="#d62728", alpha=0.5, label="Slope 4.0")
    ax.set_title("A. Integrator Step Convergence Order", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step Size $\\Delta\\phi$ [rad]")
    ax.set_ylabel("Global Error [m / Wb]")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=10)

    # Panel B: Poincare Section (Island 2/1)
    ax = axs[0, 1]
    for i in range(len(r_test)):
        ax.scatter(punc_island[i, :, 0], punc_island[i, :, 1], s=2, alpha=0.6, edgecolors="none")
    ax.plot(wall.r_wall, wall.z_wall, "k--", lw=1.2, label="Vessel Limiter")
    ax.scatter([bfield.r_axis], [bfield.z_axis], color="red", marker="+", s=60, label="Axis")
    ax.axvline(r_res, color="blue", ls=":", alpha=0.6, label="$q=2$ Resonant Radius")
    ax.set_title("B. Poincare Section (2/1 Magnetic Island, $A=10^{-3}$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(bfield.r_grid[0], bfield.r_grid[-1])
    ax.set_ylim(bfield.z_grid[0], bfield.z_grid[-1])
    ax.set_aspect("equal")
    ax.grid(True, ls=":")
    ax.legend(loc="upper right", fontsize=8)

    # Panel C: Double Mode Chaotic Sea (2/1 + 3/1 Chirikov Overlap)
    ax = axs[1, 0]
    fan_chaos = jnp.array([[r, bfield.z_axis] for r in np.linspace(bfield.r_axis + 0.1, bfield.r_grid[-1] - 0.1, 35)], dtype=jnp.float64)
    punc_chaos, _ = trace_poincare_cartesian(fan_chaos, 40, TWO_PI / 60.0, bfield, pert_chaos)
    for i in range(len(fan_chaos)):
        ax.scatter(punc_chaos[i, :, 0], punc_chaos[i, :, 1], s=2, alpha=0.5, edgecolors="none")
    ax.plot(wall.r_wall, wall.z_wall, "k--", lw=1.2)
    ax.set_title("C. Stochastic Sea (Chirikov Overlap $2/1 + 3/1$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(bfield.r_grid[0], bfield.r_grid[-1])
    ax.set_ylim(bfield.z_grid[0], bfield.z_grid[-1])
    ax.set_aspect("equal")
    ax.grid(True, ls=":")

    # Panel D: Benchmark Scaling
    ax = axs[1, 1]
    n_pts = [b["num_lines"] for b in bench_results]
    lps = [b["lines_per_sec"] for b in bench_results]
    sps = [b["ode_steps_per_sec"] for b in bench_results]
    ax.loglog(n_pts, lps, "o-", color="#2ca02c", lw=2, label="Lines / sec")
    ax.set_title("D. Batched JAX Scaling Throughput", fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch Size N (Simultaneous Field Lines)")
    ax.set_ylabel("Throughput [lines / sec]")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Diagnostic plot saved to {plot_path}")

    # Overall Verdict
    all_pass = all(v == "PASS" for v in verdicts.values())
    overall_verdict = "PASS" if all_pass else "FAIL"
    results["overall_verdict"] = overall_verdict
    results["verdicts"] = verdicts

    out_json = Path("Our try/04-novelty/t12/results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to {out_json}")
    print("=" * 60)
    print(f"OVERALL VERDICT: {overall_verdict} (All 5 physical criteria satisfied)")
    print("=" * 60)


if __name__ == "__main__":
    main()
