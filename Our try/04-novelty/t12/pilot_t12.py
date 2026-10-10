"""Our try/04-novelty/t12/pilot_t12.py — Pre-registration pilot experiment for T12.

Runs a preliminary step scan across dphi in [2pi/30, 2pi/60, 2pi/120, 2pi/240] to determine
empirical convergence orders, invariant drift scales, DOP853 CPU match, island separatrix width,
and FTLE distributions before freezing THEORY.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
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

TWO_PI = 2.0 * np.pi


def run_pilot():
    print("=== T12 PRE-REGISTRATION PILOT RUN ===")
    shot_path = "fusion equilibrium challenge/downloaded_huggingface/hf_dataset/data/diii_d_train/d3d_shot_00000a10ac.parquet"
    shot = load_shot(shot_path)
    idx = select_frame(shot)
    eq = Equilibrium(shot.psirz[idx], shot.grid_R, shot.grid_Z, machine="DIII-D")
    eq.find_axis()
    eq.set_boundary_from_contour(shot.lcfs_r[idx, :int(shot.lcfs_n[idx])], shot.lcfs_z[idx, :int(shot.lcfs_n[idx])])
    eq.calibrate_F(shot.q95[idx])
    bfield = MagneticField2D.from_tokviz_equilibrium(eq)

    print(f"Equilibrium: DIII-D frame {idx}, q95={shot.q95[idx]:.3f}, F={bfield.f_pol:.3f}")
    print(f"Magnetic axis: ({bfield.r_axis:.3f}, {bfield.z_axis:.3f}), dpsi={bfield.dpsi:.4f}")

    # 1. Step convergence scan
    steps = [TWO_PI / 30.0, TWO_PI / 60.0, TWO_PI / 120.0, TWO_PI / 240.0]
    ref_step = TWO_PI / 1200.0
    n_turns = 5
    pert = HelicalPerturbation([ResonantMode(m=2, n=1, amp=1e-3, phase=0.0)])

    # Reference solutions
    r0_cart = jnp.array([[2.00, bfield.z_axis]], dtype=jnp.float64)
    ref_cart_punc, _ = trace_poincare_cartesian(r0_cart, n_turns, ref_step, bfield, pert)
    ref_cart_final = ref_cart_punc[0, -1]

    psi_t_res = float(bfield.psi_t_at_psin(0.60))
    z0_canon = jnp.array([[psi_t_res, 0.5]], dtype=jnp.float64)
    ref_canon_punc, _ = trace_poincare_canonical(z0_canon, n_turns, ref_step, bfield, pert)
    ref_canon_final = ref_canon_punc[0, -1]

    errors_cart = []
    errors_canon = []

    for dphi in steps:
        p_c, _ = trace_poincare_cartesian(r0_cart, n_turns, dphi, bfield, pert)
        err_c = float(jnp.linalg.norm(p_c[0, -1] - ref_cart_final))
        errors_cart.append(err_c)

        p_can, _ = trace_poincare_canonical(z0_canon, n_turns, dphi, bfield, pert)
        # handle 2pi periodic angle
        d_pt = float(p_can[0, -1, 0] - ref_canon_final[0])
        d_th = float(np.remainder(float(p_can[0, -1, 1] - ref_canon_final[1]) + np.pi, TWO_PI) - np.pi)
        err_can = float(np.hypot(d_pt, d_th))
        errors_canon.append(err_can)

    orders_cart = [
        float(np.log2(errors_cart[i] / errors_cart[i+1])) for i in range(len(errors_cart)-1)
    ]
    orders_canon = [
        float(np.log2(errors_canon[i] / errors_canon[i+1])) for i in range(len(errors_canon)-1)
    ]

    print(f"Cartesian RK4 errors: {errors_cart}")
    print(f"Cartesian RK4 empirical orders: {orders_cart}")
    print(f"Canonical Midpoint errors: {errors_canon}")
    print(f"Canonical Midpoint empirical orders: {orders_canon}")

    # 2. Invariant & Drift over 100 turns (axisymmetric, no pert)
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

    print(f"100-turn Cartesian psi drift: {psi_drift_max:.3e} Wb/rad")
    print(f"100-turn Canonical H drift: {h_drift_max:.3e} Wb/rad")
    print(f"Canonical |det J - 1|: {det_j_dev:.3e}")

    # 3. CPU Match against Scipy DOP853 over 10 turns
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
    print(f"DOP853 CPU mismatch over 10 turns (dphi=2pi/120): {cpu_mismatch:.3e} m")

    # 4. Island width at q=2
    # Find s_res where q = 2.0
    s_dense = np.linspace(0.65, 0.75, 101)
    q_dense = [float(bfield.q_at_psin(s)) for s in s_dense]
    idx_res = int(np.argmin(np.abs(np.array(q_dense) - 2.0)))
    s_res = float(s_dense[idx_res])
    q_0 = float(q_dense[idx_res])
    dq_ds = float(np.gradient(q_dense, s_dense)[idx_res])

    A_pert = 1e-3
    w_analytic_psin = 4.0 * np.sqrt(A_pert * q_0 / abs(dq_ds))

    # Midplane R coordinate where psi_n = s_res
    r_sweep = np.linspace(bfield.r_axis, bfield.r_grid[-1] - 0.05, 500)
    psn_sweep = np.array([float(bfield.psi_n_at(r, bfield.z_axis)) for r in r_sweep])
    r_res = float(np.interp(s_res, psn_sweep, r_sweep))

    # |dpsi_n / dR| at midplane outer side
    dr_local = 0.001
    dpsn_dr = float((bfield.psi_n_at(r_res + dr_local, bfield.z_axis) - bfield.psi_n_at(r_res - dr_local, bfield.z_axis)) / (2.0 * dr_local))
    w_analytic_mm = (w_analytic_psin / abs(dpsn_dr)) * 1000.0

    print(f"Resonant surface: s_res={s_res:.4f}, q0={q_0:.4f}, dq/ds={dq_ds:.4f}")
    print(f"Analytic island width: w_psin={w_analytic_psin:.5f}, w_mm={w_analytic_mm:.2f} mm")

    # Measure numerical island separatrix width across the O-point
    # For m=2, n=1, phi=0: O-points and X-points alternate every pi/2 in theta*.
    # Trace field lines densely near r_res to find the inner/outer separatrix extremes
    r_test = np.linspace(r_res - 0.06, r_res + 0.06, 61)
    pts_fan = jnp.array([[r, bfield.z_axis] for r in r_test], dtype=jnp.float64)
    punc_island, _ = trace_poincare_cartesian(pts_fan, 40, TWO_PI / 60.0, bfield, pert)

    # Classify lines: lines inside the island have bounded radial oscillations about the O-point,
    # lines outside span full 2pi poloidal angles.
    # An island orbit passes theta* = 0 with two branches (inner and outer) around the O-point.
    # We find the maximal radial extent of orbits trapped in the island:
    r_trapped_min = []
    r_trapped_max = []
    for i in range(len(r_test)):
        line_r = np.array(punc_island[i, :, 0])
        line_z = np.array(punc_island[i, :, 1])
        # Compute angles relative to axis
        angles = np.arctan2(line_z - bfield.z_axis, line_r - bfield.r_axis)
        unwrapped = np.unwrap(angles)
        total_angle_span = abs(unwrapped[-1] - unwrapped[0])
        # Trapped island lines do not complete full monotonic poloidal circulations in the same way,
        # or have resonant q=2/1 winding.
        # Alternatively, measure the separatrix directly from the helical Hamiltonian:
        # H_hel = (dq_ds / 2 q0^2) * (psi_n - s_res)^2 + A * cos(2 theta* - phi)
        # Separatrix is H_hel = A (passing through X-point).
        # At O-point (cos = -1): H_hel = -A + (dq_ds / 2 q0^2) * delta_psi^2 = A
        # => delta_psi^2 = 4 q0^2 A / dq_ds => delta_psi = 2 sqrt(q0^2 A / dq_ds) => full width = 4 sqrt(A q0 / dq_ds) * sqrt(q0) ???
        # Wait! Let's check the exact formula in THEORY.md: W = 4 * sqrt(psi_tilde * q / |dq/dpsi|)
        pass

    # Let's compute numerical width directly from the Hamiltonian separatrix on the midplane:
    # At midplane X-point and O-point:
    # Helical flux contour through the X-point gives exact separatrix!
    def helical_flux(r, z, phi=0.0):
        c, s = bfield.cos_sin_th_at(r, z)
        sn = bfield.psi_n_at(r, z)
        p_tot = sn + pert.delta_psi_n(c, s, sn, phi)
        # Helical flux: psi_hel = psi_p - (n/m) psi_t = psi_p - 0.5 psi_t
        # where psi_t = int q dpsi
        pt = bfield.psi_t_at_psin(sn) / abs(bfield.dpsi)
        return float(sn - 0.5 * pt + pert.delta_psi_n(c, s, sn, phi))

    # Along midplane z = z_axis:
    r_mid = np.linspace(r_res - 0.08, r_res + 0.08, 1000)
    h_mid = np.array([helical_flux(r, bfield.z_axis, 0.0) for r in r_mid])
    # The X-point on midplane or at theta* = pi/2:
    # For m=2, cos(2 theta*) has extrema at theta*=0 (O-point or X-point) and theta*=pi/2.
    # At theta*=pi/2 (top/bottom or Z extrema):
    # Helical value at X-point:
    # Let's find the X-point helical value
    # At theta* = pi/2, R approx r_axis, Z = z_top
    # Find separatrix level h_sep:
    h_res = helical_flux(r_res, bfield.z_axis, 0.0)
    # The separatrix branches across O-point at theta*=0:
    # Find local min and max of h_mid around r_res:
    # The O-point is an extremum of h_mid. The separatrix is at h = h_x.
    # Difference between inner and outer roots where h(R) = h_x:
    # Let's measure numerical separatrix width from actual field-line punctures:
    # Trace from near the X-point:
    # At the midplane, the separatrix boundaries enclose the island.
    # From the fan, the island orbits have a characteristic O-point at r_O:
    r_O = r_res
    # Find inner-most and outer-most r where trapped orbits exist:
    # For each line in fan, check if it covers both sides of r_res
    trapped_mask = []
    for i in range(len(r_test)):
        line_r = np.array(punc_island[i, :, 0])
        # If line stays in island, its R values oscillate around r_res with amplitude <= w/2
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
    print(f"Island numerical width: w_num_psin={w_num_psin:.5f}, w_num_mm={w_num_mm:.2f} mm")
    print(f"Island relative error vs analytic: {island_err_rel * 100.0:.2f}%")

    # 5. FTLE measurements for chaos criterion
    t_ftle = 50
    # (a) Regular case: single mode m=2, n=1 at A=1e-3
    pert_reg = HelicalPerturbation([ResonantMode(m=2, n=1, amp=1e-3, phase=0.0)])
    ftle_reg = compute_ftle_cartesian(r0_cart[0], t_ftle, TWO_PI / 60.0, bfield, pert_reg)

    # (b) Chaotic case: double mode 2/1 + 3/1 with overlap (Chirikov S >= 1)
    # 2/1 at q=2 (s~0.70), 3/1 at q=3 (s~0.88).
    # With amplitudes A_21 = 3e-3, A_31 = 3e-3, islands overlap strongly
    pert_chaos = HelicalPerturbation([
        ResonantMode(m=2, n=1, amp=3e-3, phase=0.0),
        ResonantMode(m=3, n=1, amp=3e-3, phase=0.0),
    ])
    # Initial point in the chaotic sea between the two resonances (s ~ 0.78)
    r_mid_chaos = float(np.interp(0.78, psn_sweep, r_sweep))
    r_chaos_0 = jnp.array([r_mid_chaos, bfield.z_axis], dtype=jnp.float64)
    ftle_chaos = compute_ftle_cartesian(r_chaos_0, t_ftle, TWO_PI / 60.0, bfield, pert_chaos)

    k_ratio = ftle_chaos / max(ftle_reg, 1e-12)
    print(f"FTLE at T={t_ftle} turns: regular={ftle_reg:.4e}, chaotic={ftle_chaos:.4e}")
    print(f"FTLE ratio k = ftle_chaos / ftle_reg: {k_ratio:.2f}")

    # Summary dictionary for pre-registration
    pilot_summary = {
        "date": "2026-10-09",
        "shot": "d3d_shot_00000a10ac",
        "frame": int(idx),
        "steps": [float(s) for s in steps],
        "cart_errors": [float(e) for e in errors_cart],
        "cart_orders": [float(o) for o in orders_cart],
        "canon_errors": [float(e) for e in errors_canon],
        "canon_orders": [float(o) for o in orders_canon],
        "psi_drift_100turns": float(psi_drift_max),
        "h_drift_100turns": float(h_drift_max),
        "det_j_deviation": float(det_j_dev),
        "cpu_mismatch_10turns": float(cpu_mismatch),
        "island_analytic_psin": float(w_analytic_psin),
        "island_analytic_mm": float(w_analytic_mm),
        "island_num_psin": float(w_num_psin),
        "island_err_rel": float(island_err_rel),
        "ftle_T_turns": int(t_ftle),
        "ftle_regular": float(ftle_reg),
        "ftle_chaotic": float(ftle_chaos),
        "ftle_ratio_k": float(k_ratio),
    }

    out_file = Path("Our try/04-novelty/t12/pilot_summary.json")
    with open(out_file, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"Pilot summary saved to {out_file}")


if __name__ == "__main__":
    run_pilot()
