#!/usr/bin/env python3
"""Bake the JET 1975 plasma physics (phase F4, "physics inside").

Runs OUTSIDE Blender (numpy/scipy/scikit-image/matplotlib/PyYAML).

    python src/run_bake_jet_physics.py                       # default
    python src/run_bake_jet_physics.py --island 3/1 --amp 1e-3   # demo island

Writes
  data/bake_jet1975/physics/        npz + manifest.json in the run_bake.py style
      (psi_n_grid, flux_surfaces, lcfs, fieldlines, punctures, q_profile)
  data/bake_jet1975/physics.json    every number (equilibrium, QA, ripple)
  data/bake_jet1975/ripple.png      our own ripple plot vs the source values

The physics bake goes in a SUB-directory because data/bake_jet1975/manifest.json
belongs to the machine bake (run_bake_machine.py) and must not be overwritten.

Equilibrium: Solov'ev / Cerfon-Freidberg, fixed boundary = the G2 D
(R0 2.96, a 1.25, b 2.10, delta 0.3476), B0 = 2.77 T, I_p = 3.8 MA,
beta_p = 0.9 (Table I.3-1 p.83).  No perturbation unless --island is given.
See docs/JET-EQUILIBRIUM.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config                                            # noqa: E402
from tokviz.equilibrium import TWO_PI                                # noqa: E402
from tokviz.export import Bake, _jsonable                            # noqa: E402

from tokviz import surfaces as S                                     # noqa: E402
from tokviz.machine import equilibrium_analytic as EA                # noqa: E402
from tokviz.machine import ripple as RP                              # noqa: E402


# ---------------------------------------------------------------------------
def outboard_R_of_psin(eq, s, z=0.0):
    """Outboard-midplane R where psi_n = s (bisection; psi_n grows outward)."""
    lo, hi = eq.r_axis, eq.sol.R0 + eq.sol.a + 0.05
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(eq.psi_n(mid, z)) < s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def psin_of_q(x, q, qt):
    if not (q.min() <= qt <= q.max()):
        return None
    return float(np.interp(qt, q, x))          # q is monotone increasing


def trace_drift(fl, eq, r0, z0, n_turns):
    phi = np.linspace(0.0, TWO_PI * n_turns, 40 * n_turns + 1)
    R, Z, alive = fl.trace(r0, z0, phi, rtol=1e-10, atol=1e-12)
    psi = eq.psi_at(R.ravel(), Z.ravel()).reshape(R.shape)
    return np.abs(psi - psi[:, :1]).max(axis=1) / abs(eq.psi_bdy - eq.psi_axis), alive


# ---------------------------------------------------------------------------
def ripple_study(design, log):
    n_tf = int(design["n_tf"][0])
    Ic_b, Ic_e = design["Ic_basic"][0], design["Ic_extended"][0]
    c = RP.winding_centreline()
    ph0 = RP.kernel_tf_phi0_rad()          # absolute azimuth of coil 0 (kernel)
    loops1 = RP.tf_filaments(c, n_coils=n_tf, current_total=Ic_b, phi0=ph0)
    loops9 = RP.tf_filaments(c, n_coils=n_tf, current_total=Ic_b, n_rad=3, n_tor=3, phi0=ph0)
    loopsE = RP.tf_filaments(c, n_coils=n_tf, current_total=Ic_e, phi0=ph0)

    R = np.linspace(1.72, 4.55, 284)
    d1, e1, Bmax1, Bmin1 = RP.ripple(loops1, R, 0.0, n_coils=n_tf, phi0=ph0)
    d9, e9, _, _ = RP.ripple(loops9, R, 0.0, n_coils=n_tf, phi0=ph0)

    # B on axis (R0, coil-to-coil average) for basic and extended currents
    R0 = design["R0"][0]
    B_R0 = {}
    for lab, L in (("basic", loops1), ("extended", loopsE)):
        _, _, bmx, bmn = RP.ripple(L, np.array([R0]), 0.0, n_coils=n_tf, phi0=ph0)
        B_R0[lab] = float(0.5 * (bmx[0] + bmn[0]))

    # the source's reference radius: Re(plasma) = 4.21 m (Table I.3-1 p.83)
    Rref = 4.21
    ref = {}
    for lab, L in (("1_filament", loops1), ("3x3_filaments", loops9)):
        d, e, _, _ = RP.ripple(L, np.array([Rref]), 0.0, n_coils=n_tf, phi0=ph0)
        ref[lab] = {"delta_pct": float(d[0] * 100), "eps_src_pct": float(e[0] * 100)}

    # Table IV.2-4 (p.332): 30 boundary points with the source's eps
    tab = RP.load_csv("plasma_boundary", cols=3)
    _, et1, _, _ = RP.ripple(loops1, tab[:, 0], tab[:, 1], n_coils=n_tf, phi0=ph0)
    _, et9, _, _ = RP.ripple(loops9, tab[:, 0], tab[:, 1], n_coils=n_tf, phi0=ph0)

    # sensitivity to the digitized centreline (+-2 cm) -- the coil shape error
    sens = {}
    for dh in (-0.02, +0.02):
        L = RP.tf_filaments(RP.winding_centreline(0.186 + dh), n_coils=n_tf,
                            current_total=Ic_b, phi0=ph0)
        _, e, _, _ = RP.ripple(L, np.array([Rref]), 0.0, n_coils=n_tf, phi0=ph0)
        sens[f"{dh*100:+.0f}cm"] = float(e[0] * 100)

    fig = RP.FIG_IV24_EPS_PCT
    e_fig_ours = np.interp(fig[:, 0], R, e1) * 100
    table_rows = [{"R_m": float(r), "eps_src_fig_pct": float(v), "eps_ours_pct": float(o),
                   "delta_ours_pct": float(o / 2), "ratio": float(o / v)}
                  for (r, v), o in zip(fig, e_fig_ours)]
    log("      ripple (source convention eps = 2(Bmax-Bmin)/(Bmax+Bmin)):")
    for row in table_rows:
        log(f"        R={row['R_m']:.2f}  Fig.IV.2-4 {row['eps_src_fig_pct']:6.2f}%  "
            f"ours {row['eps_ours_pct']:6.2f}%  ratio {row['ratio']:.2f}")
    log(f"      at Re(plasma)=4.21 m: eps {ref['1_filament']['eps_src_pct']:.2f}% (1 fil), "
        f"{ref['3x3_filaments']['eps_src_pct']:.2f}% (3x3) vs 3.6 % (Table I.3-1)")
    log(f"      B(R0) = {B_R0['basic']:.4f} T (basic 41 MA), {B_R0['extended']:.4f} T (51 MA)")

    out = {
        "model": {"n_coils": n_tf, "tf_phi0_deg": float(np.rad2deg(ph0)), "centreline": "bore contour (tf_coil_inner.csv) offset "
                  "0.186 m outward (half of 0.3735/0.371 m leg thickness, p.325)",
                  "filaments": "1 per coil (baseline); 3x3 over a 0.30 x 0.26 m pack (sensitivity)",
                  "Ic_basic_A": Ic_b, "Ic_extended_A": Ic_e},
        "conventions": {"delta": "(Bmax-Bmin)/(Bmax+Bmin)",
                        "eps_src": "2(Bmax-Bmin)/(Bmax+Bmin) -- Fig. IV.2-4 caption p.330; "
                                   "Table I.3-1 '3.6 %' and Table IV.2-4 use it"},
        "B_R0_T": B_R0, "B_R0_design_T": {"basic": 2.77, "extended": 3.45},
        "reference_radius_m": Rref,
        "at_reference_radius": ref, "source_value_pct": 3.6,
        "centreline_sensitivity_eps_pct_at_ref": sens,
        "fig_IV_2_4_comparison": table_rows,
        "table_IV_2_4_comparison": {
            "n_points": int(len(tab)),
            "ratio_1fil": {"mean": float(np.mean(et1 * 100 / tab[:, 2])),
                           "min": float(np.min(et1 * 100 / tab[:, 2])),
                           "max": float(np.max(et1 * 100 / tab[:, 2]))},
            "ratio_3x3": {"mean": float(np.mean(et9 * 100 / tab[:, 2])),
                          "min": float(np.min(et9 * 100 / tab[:, 2])),
                          "max": float(np.max(et9 * 100 / tab[:, 2]))},
            "first_point": {"R_m": float(tab[0, 0]), "Z_m": float(tab[0, 1]),
                            "eps_src_pct": float(tab[0, 2]),
                            "eps_ours_1fil_pct": float(et1[0] * 100),
                            "eps_ours_3x3_pct": float(et9[0] * 100)}},
        "midplane": {"R_m": R, "delta_1fil": d1, "eps_1fil": e1, "delta_3x3": d9,
                     "Bmax_T": Bmax1, "Bmin_T": Bmin1},
        "axis_vs_edge": {"delta_at_R0": float(np.interp(R0, R, d1)),
                         "delta_at_4p21": float(np.interp(4.21, R, d1))},
    }
    return out, tab, et1, et9


def plot_ripple(rip, tab, et1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    m = rip["midplane"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=GRID, lw=0.8, which="both")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=MUTED)
    ax.semilogy(m["R_m"], np.asarray(m["delta_1fil"]) * 100, color=BLUE, lw=2,
                label="ours: δ = (Bmax−Bmin)/(Bmax+Bmin), 32 filaments")
    ax.semilogy(m["R_m"], np.asarray(m["eps_1fil"]) * 100, color=BLUE, lw=2, ls="--",
                label="ours in source convention ε = 2δ")
    f = RP.FIG_IV24_EPS_PCT
    ax.semilogy(f[:, 0], f[:, 1], "o", ms=8, mfc=ORANGE, mec="#fcfcfb", mew=2,
                label="source ε, Fig. IV.2-4 p.330 (digitized by us)")
    ax.semilogy([4.21], [3.6], "D", ms=9, mfc="none", mec=ORANGE, mew=2,
                label="source ε = 3.6 % (Table I.3-1 p.83), R = Re(plasma)")
    for x, lab, ha, dx in ((2.96, "R0", "left", 0.02), (4.21, "plasma edge", "right", -0.03)):
        ax.axvline(x, color=MUTED, lw=1, ls=":")
        ax.text(x + dx, 30, lab, color=MUTED, fontsize=8, ha=ha)
    ax.set_ylim(1e-6, 60)
    ax.set_xlim(1.7, 4.55)
    ax.set_xlabel("R on the midplane (m)", color=INK)
    ax.set_ylabel("ripple (%)", color=INK)
    ax.set_title("JET 1975 TF ripple, midplane", color=INK, loc="left", fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK)

    bx.plot(tab[:, 1], tab[:, 2], "o", ms=8, mfc=ORANGE, mec="#fcfcfb", mew=2,
            label="source ε, Table IV.2-4 p.332")
    bx.plot(tab[:, 1], et1 * 100, color=BLUE, lw=2, label="ours ε, same (R, Z) points")
    bx.set_xlabel("Z of the boundary point (m), upper-outboard quadrant", color=INK)
    bx.set_ylabel("ε (%)", color=INK)
    bx.set_title("Along the D boundary", color=INK, loc="left", fontsize=11)
    bx.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.BAKE_DIR / "bake_jet1975"))
    ap.add_argument("--method", default="cf7", choices=["cf7", "lsq"])
    ap.add_argument("--beta-p", type=float, default=EA.JET_BETAP)
    ap.add_argument("--n-pol", type=int, default=160)
    ap.add_argument("--n-tor", type=int, default=192)
    ap.add_argument("--n-punctures", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--fieldline-turns", type=int, default=8)
    ap.add_argument("--island", default="", help="demo m/n island, e.g. 3/1 (off by default)")
    ap.add_argument("--amp", type=float, default=1.0e-3,
                    help="island demo amplitude, fraction of the flux range")
    ap.add_argument("--skip-ripple", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    log = lambda *a: print(*a, flush=True)
    root = Path(args.out)
    bake = Bake(root / "physics")
    design = EA.load_jet_design()
    R0, a, b = design["R0"][0], design["a"][0], design["b"][0]
    B0, Ip = design["B0"][0], design["Ip"][0]

    # ------------------------------------------------------------ equilibrium
    log(f"[1/7] Solov'ev ({args.method}) for R0={R0} a={a} b={b} delta={EA.JET_DELTA} "
        f"B0={B0} T Ip={Ip/1e6} MA beta_p={args.beta_p}")
    eq, sol, info = EA.build_equilibrium(R0, a, b, EA.JET_DELTA, B0, Ip,
                                         beta_p=args.beta_p, method=args.method)
    log(f"      A={info['A']:.5f} psi0={info['psi0_Wb_per_rad']:.4f} Wb/rad  "
        f"boundary residual max {info['boundary_residual_max_m']*1e3:.3f} mm")
    log(f"      Ip={info['Ip_A']/1e6:.4f} MA (area {info['Ip_area_A']/1e6:.4f})  "
        f"beta_p={info['beta_p']:.4f} beta_t={info['beta_t']*100:.3f}%  "
        f"li(1)={info['li1']:.4f} li(3)={info['li3']:.4f}")
    log(f"      axis ({eq.r_axis:.4f}, {eq.z_axis:.2e})  Shafranov shift "
        f"{eq.r_axis - R0:.4f} m")

    # ------------------------------------------------------------- q profile
    log("[2/7] q profile")
    lv = np.r_[np.linspace(0.01, 0.99, 99), 1.0]
    x, q = eq.q_profile(lv)
    dq = np.gradient(q, x)
    q0 = eq.q_axis()
    q95 = float(np.interp(0.95, x, q))
    qa = float(q[-1])
    Bphi_R0 = float(sol.F_of_psi(sol.psi(R0, 0.0)) / R0)
    # design q(a): footnote formula of Table I.3-1 (p.83), k = 1.1
    eps_ = a / R0
    q_design_formula = 1.1 * (1 + (b / a) ** 2) / 2 * eps_ ** 2 / (1 - eps_ ** 2) ** 2 \
        * design["Ic_basic"][0] / Ip
    q_cyl_round = TWO_PI * a ** 2 * B0 / (EA.MU0 * R0 * 2.6e6)
    q_star = TWO_PI * a ** 2 * B0 * (1 + (b / a) ** 2) / 2 / (EA.MU0 * R0 * Ip)
    log(f"      q0={q0:.4f}  q95={q95:.4f}  q(1)={qa:.4f}  q(a)/q0={qa/q0:.3f}")
    log(f"      design: q(a)=6 (p.83, formula gives {q_design_formula:.3f}); "
        f"round plasma q(a)=2.8 at 2.6 MA (cylinder formula gives {q_cyl_round:.3f}); "
        f"q* (cyl, elongated) = {q_star:.3f}")
    log(f"      B_phi(R0) with plasma = {Bphi_R0:.4f} T (vacuum {B0} T)")

    # ------------------------------------------------------------------ QA
    log("[3/7] QA: psi invariance and traced q")
    fl = EA.FieldLinesFpsi(eq, None)
    qa_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    r_qa = np.array([outboard_R_of_psin(eq, s) for s in qa_levels])
    drift, _ = trace_drift(fl, eq, r_qa, np.zeros_like(r_qa), n_turns=100)
    log(f"      max |dpsi|/dpsi_tot over 100 turns = {drift.max():.3e}")
    qt = [EA.q_traced(fl, r, 0.0, n_turns=200) for r in r_qa]
    qc = [eq.q_at(s) for s in qa_levels]
    q_err = [abs(t / c - 1) for t, c in zip(qt, qc)]
    for s, t, c in zip(qa_levels, qt, qc):
        log(f"      psi_n={s:.2f}: q contour {c:.5f}  traced {t:.5f}  "
            f"rel {abs(t/c-1)*100:.4f}%")

    # --------------------------------------------------------- perturbation
    pert, island = None, None
    if args.island:
        from tokviz.perturbation import Mode, Perturbation, island_width_psin
        m_s, _, n_s = args.island.partition("/")
        md = Mode(m=int(m_s), n=int(n_s), amp=args.amp)
        s_res = psin_of_q(x, q, md.q_res)
        if s_res is None:
            log(f"      island {args.island}: q={md.q_res:.3f} not in [{q.min():.2f}, "
                f"{q.max():.2f}] -- skipped (q0 > 2 here, so 2/1 is never resonant)")
        else:
            md.psin_res = s_res
            pert = Perturbation(eq, [md], tsf=eq.theta_star_field())
            epsA = pert.amplitude_at(md)
            W = island_width_psin(epsA, md.q_res, float(np.interp(s_res, x, dq)))
            island = {"m": md.m, "n": md.n, "psi_n_res": s_res, "amp_fraction": args.amp,
                      "eps_effective": epsA, "width_psin": W,
                      "formula": "W = 4 sqrt(eps q / |dq/dpsi_n|) (corrected 2026-09-29)"}
            log(f"      DEMO island {md.m}/{md.n} at psi_n={s_res:.4f}, W={W:.4f}")
    flp = EA.FieldLinesFpsi(eq, pert)

    # ----------------------------------------------------------- field lines
    log(f"[4/7] field lines ({args.fieldline_turns} toroidal turns)")
    golden = (np.sqrt(5) - 1) / 2
    wanted = [("rational", 3.0), ("rational", 3.5), ("rational", 4.0), ("rational", 5.0),
              ("irrational", 3.0 + golden), ("irrational", 4.0 + golden),
              ("irrational", 2.9 + 0.1 * golden)]
    kinds, qs, sn_seed = [], [], []
    for kind, qq in wanted:
        s = psin_of_q(x, q, qq)
        if s is not None and s < 0.995:
            kinds.append(kind); qs.append(qq); sn_seed.append(s)
    r_c = np.array([outboard_R_of_psin(eq, s) for s in sn_seed])
    z_c = np.zeros_like(r_c)
    xyz, alive_c = flp.trace_xyz(r_c, z_c, n_turns=args.fieldline_turns, pts_per_turn=160,
                                 rtol=1e-9, atol=1e-11)
    bake.save("fieldlines", xyz=xyz[alive_c].astype(np.float32),
              psi_n0=np.asarray(sn_seed)[alive_c].astype(np.float32),
              q=np.asarray(qs)[alive_c].astype(np.float32),
              kind=np.asarray(kinds)[alive_c])
    # closure check for rational lines: after m toroidal turns the line returns
    closure = []
    for i, (kind, qq) in enumerate(zip(kinds, qs)):
        frac = qq % 1.0
        mt = int(round(qq)) if frac == 0 else (int(round(2 * qq)) if abs(frac - 0.5) < 1e-9 else None)
        if kind == "rational" and mt and mt <= args.fieldline_turns:
            j = mt * 160
            closure.append({"q": qq, "turns": mt,
                            "gap_m": float(np.linalg.norm(xyz[i, j] - xyz[i, 0]))})
    for c in closure:
        log(f"      q={c['q']}: returns after {c['turns']} turns to within {c['gap_m']*1e3:.2f} mm")

    # ------------------------------------------------------------ punctures
    log(f"[5/7] Poincare section: {args.n_seeds} seeds x {args.n_punctures}")
    s_seed = np.linspace(0.05, 0.98, args.n_seeds)
    r0 = np.array([outboard_R_of_psin(eq, s) for s in s_seed])
    R, Z, alive = flp.poincare(r0, np.zeros_like(r0), n_punctures=args.n_punctures,
                               rtol=1e-9, atol=1e-11)
    sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
    seed_id = np.repeat(np.arange(r0.size)[:, None], R.shape[1], axis=1)
    bake.save("punctures",
              points=S.punctures_xyz(R[alive], Z[alive]).astype(np.float32),
              psi_n=sn[alive].ravel().astype(np.float32),
              seed_id=seed_id[alive].ravel().astype(np.int32),
              R=R[alive].ravel().astype(np.float32), Z=Z[alive].ravel().astype(np.float32),
              shape=np.asarray(R[alive].shape, dtype=np.int32),
              label_id=np.zeros(int(alive.sum()) * R.shape[1], dtype=np.int32),
              label_names=np.asarray(["regular", "island", "chaotic", "escaped"]))

    # --------------------------------------------------------------- geometry
    log("[6/7] flux surfaces, LCFS, grids")
    levels = np.round(np.arange(0.2, 1.0001, 0.1), 3)
    fs = S.flux_surface_set(eq, levels, n_pol=args.n_pol, n_tor=args.n_tor)
    bake.save_meshes("flux_surfaces", fs)
    lcfs = S.resample_closed(sol.boundary(2000), args.n_pol)
    v, f = S.revolve(lcfs, n_tor=args.n_tor)
    bake.save("lcfs", verts=v.astype(np.float32), faces=f, profile=lcfs.astype(np.float32))
    bake.save("psi_n_grid", psi_n=eq.psi_n_grid().astype(np.float32),
              R=eq.R.astype(np.float32), Z=eq.Z.astype(np.float32),
              R_scaled=eq.R.astype(np.float32), Z_scaled=eq.Z.astype(np.float32))
    bake.save("q_profile", psi_n=x.astype(np.float32), q=q.astype(np.float32),
              dq_dpsin=dq.astype(np.float32))
    log(f"      {len(fs)} flux surfaces at psi_n = {list(fs.keys())}")

    # ------------------------------------------------------------------ ripple
    rip = None
    if not args.skip_ripple:
        log("[7/7] TF ripple (Biot-Savart, 32 coils)")
        rip, tab, et1, _ = ripple_study(design, log)
        plot_ripple(rip, tab, et1, root / "ripple.png")

    # ---------------------------------------------------------------- manifest
    eq_block = {
        "model": "Solov'ev, Cerfon-Freidberg basis (Phys. Plasmas 17, 032502, 2010)",
        "R0": R0, "a": a, "b": b, "kappa": b / a, "delta": EA.JET_DELTA,
        "r_axis": eq.r_axis, "z_axis": eq.z_axis, "psi_axis": eq.psi_axis,
        "psi_bdy": eq.psi_bdy, "axis_sign": eq.axis_sign,
        "B0_vacuum_T": B0, "B_phi_R0_T": Bphi_R0, "F0_Tm": sol.F0,
        "q0": q0, "q95": q95, "q_edge": qa, "q_edge_over_q0": qa / q0,
        "q_design_Table_I31": 6.0, "q_design_formula_k1p1": q_design_formula,
        "q_round_plasma_cyl_2p6MA": q_cyl_round, "q_star_cyl": q_star,
        **{k: v for k, v in info.items()},
    }
    qa_block = {"psi_invariant_drift_100_turns": float(drift.max()),
                "q_traced_vs_contour": [{"psi_n": s, "q_contour": c, "q_traced": t}
                                        for s, c, t in zip(qa_levels, qc, qt)],
                "q_traced_max_rel_err": float(max(q_err)),
                "rational_line_closure": closure,
                "integrator": "scipy solve_ivp DOP853, rtol 1e-9..1e-10"}
    bake.set("provenance", {"machine": "JET 1975 design (EUR 5516e)",
                            "shot_file": "analytic Solov'ev equilibrium (no shot)",
                            "frame_index": 0, "time_ms": 0.0,
                            "sources": "Table I.3-1 p.83; Fig. IV.2-4 p.330; "
                                       "Table IV.2-4 p.332; Fig. IV.2-1 p.325"})
    bake.set("equilibrium", eq_block)
    bake.set("geometry", {"scale": 1.0, "R0_scaled": R0, "units": "m",
                          "note": "true size, same frame as data/bake_jet1975 machine meshes",
                          "flux_surface_levels": [float(k) for k in fs]})
    bake.set("perturbation", {"kind": "none (axisymmetric)"} if island is None else
             {"kind": "DEMO prescribed vacuum-like resonant perturbation", **island,
              "caveat": "put in by hand; not a JET 1975 prediction"})
    bake.set("qa", qa_block)
    path = bake.finish()

    phys = {"equilibrium": eq_block, "qa": qa_block, "perturbation": island,
            "ripple": {k: v for k, v in (rip or {}).items() if k != "midplane"},
            "ripple_midplane_sampled": None if rip is None else {
                "R_m": rip["midplane"]["R_m"][::8], "delta_pct": rip["midplane"]["delta_1fil"][::8] * 100,
                "eps_src_pct": rip["midplane"]["eps_1fil"][::8] * 100}}
    (root / "physics.json").write_text(json.dumps(_jsonable(phys), indent=2, ensure_ascii=False))
    log(f"\ndone in {time.time()-t_start:.1f}s -> {path}, {root/'physics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
