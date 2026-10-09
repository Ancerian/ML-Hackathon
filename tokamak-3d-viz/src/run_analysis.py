#!/usr/bin/env python3
"""Quantitative analysis of the Poincare sections.

Measures, at several perturbation amplitudes:

  * the rotation number profile nu(psi_n) = 1/q, straight off the field lines;
  * orbit classification into regular / island / chaotic;
  * the island width the section ACTUALLY shows, against the pendulum
    prediction W = 4 sqrt(eps q / |dq/dpsi_N|);
  * the chaotic fraction, against the Chirikov overlap parameter -- a test of
    a heuristic that is routinely quoted as if it were a theorem.

    python src/run_analysis.py --out data/analysis
    python src/run_analysis.py --out data/analysis --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokviz import config
from tokviz.equilibrium import load_shot, select_frame, Equilibrium
from tokviz.fieldline import FieldLines
from tokviz.perturbation import (Mode, Perturbation, resonant_surfaces,
                                 island_width_psin, chirikov)
from tokviz import analysis as A

RATIONALS = [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1), (7, 2), (4, 1), (5, 1)]


def build_equilibrium(shot_path, q95_target=3.7):
    shot = load_shot(shot_path)
    k = select_frame(shot, q95_target)
    eq = Equilibrium(shot.psirz[k], shot.grid_R, shot.grid_Z, machine=shot.source)
    eq.find_axis()
    n = int(shot.lcfs_n[k])
    eq.set_boundary_from_contour(shot.lcfs_r[k][:n], shot.lcfs_z[k][:n])
    eq.calibrate_F(float(shot.q95[k]))
    return shot, k, eq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default="data/analysis")
    ap.add_argument("--modes", default="2/1,3/1")
    ap.add_argument("--amps", default="0,2.5e-4,5e-4,1e-3,2e-3,4e-3")
    ap.add_argument("--n-seeds", type=int, default=40,
                    help="UNIFORM seeds; the unbiased sample")
    ap.add_argument("--n-refine", type=int, default=0,
                    help="extra seeds packed around each resonance")
    ap.add_argument("--refine-span", type=float, default=1.5,
                    help="refinement half-width, in units of the widest "
                         "predicted island")
    ap.add_argument("--n-turns", type=int, default=120)
    ap.add_argument("--pts-per-turn", type=int, default=30)
    ap.add_argument("--n-punctures", type=int, default=200)
    ap.add_argument("--ref-amp", type=float, default=1e-3,
                    help="amplitude used for the section figure")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.amps = "0,1e-3,4e-3"
        args.n_seeds, args.n_turns, args.pts_per_turn = 18, 40, 20
        args.n_punctures = 80

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    amps = [float(a) for a in args.amps.split(",")]
    t0 = time.time()

    shot, k, eq = build_equilibrium(args.shot)
    print(f"{shot.source} frame {k}  t={shot.times[k]:.0f} ms  "
          f"q95={shot.q95[k]:.4f}  B_phi(axis)={eq.F/eq.r_axis:.4f} T", flush=True)

    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    dq = np.gradient(q, x)
    tsf = eq.theta_star_field()

    mspec = []
    for tok in args.modes.split(","):
        m_s, _, n_s = tok.strip().partition("/")
        mspec.append((int(m_s), int(n_s)))
    amps_raw = [a for a in amps if a > 0] or [1e-3]

    # --- seeding -----------------------------------------------------------
    # Uniform seeds give the UNBIASED statistics.  Refinement seeds are packed
    # around the resonances to resolve the island widths -- and are excluded
    # from the chaos fraction, because the resonances are exactly where chaos
    # lives and counting them would inflate it.
    r_uni = np.linspace(eq.r_axis + 0.03, eq.r_axis + 0.555, args.n_seeds)
    r_extra = []
    if args.n_refine > 0:
        amp_max = max(amps_raw)
        for (m, n) in mspec:
            roots = resonant_surfaces(x, q, m / n)
            if not roots:
                continue
            s_res = roots[0]
            dqds = float(np.interp(s_res, x, dq))
            g = s_res ** (0.5 * m)
            W = island_width_psin(amp_max * g, m / n, dqds)
            half = args.refine_span * max(W, 0.02)
            # keep the band inside the plasma: seeds beyond the LCFS leave the
            # grid and are wasted
            lo = max(s_res - half, 0.06)
            hi = min(s_res + half, 0.955)
            # invert psi_n(R) on the midplane to place seeds evenly in psi_n
            rr = np.linspace(eq.r_axis + 0.03, eq.r_axis + 0.60, 600)
            ss = eq.psi_n(rr, np.full_like(rr, eq.z_axis))
            keep = np.isfinite(ss)
            targets = np.linspace(lo, hi, args.n_refine)
            r_extra.append(np.interp(targets, ss[keep], rr[keep]))
    r_seeds = np.concatenate([r_uni] + r_extra) if r_extra else r_uni
    is_uniform = np.concatenate(
        [np.ones(r_uni.size, bool)] + [np.zeros(a.size, bool) for a in r_extra]
    ) if r_extra else np.ones(r_uni.size, bool)
    order = np.argsort(r_seeds)
    r_seeds = r_seeds[order]
    is_uniform = is_uniform[order]
    z_seeds = np.full_like(r_seeds, eq.z_axis)
    psi_n0 = eq.psi_n(r_seeds, z_seeds)
    print(f"seeds: {int(is_uniform.sum())} uniform + "
          f"{int((~is_uniform).sum())} refinement = {r_seeds.size}", flush=True)

    # The chaos threshold is CALIBRATED on the amp = 0 case, where the system
    # is provably integrable, rather than guessed.  The weighted-Birkhoff
    # convergence there is limited by integrator tolerance (~6 digits), not by
    # dynamics, so anything at or above that floor cannot be chaos.
    amps = sorted(set(amps))
    min_digits = 5.0
    calib = None
    if amps and amps[0] == 0.0:
        fl0 = FieldLines(eq, None)
        th0, sn0, phi0, al0 = A.trace_theta(fl0, r_seeds, z_seeds,
                                            n_turns=args.n_turns,
                                            pts_per_turn=args.pts_per_turn)
        d0, _ = A.rotation_number_digits(th0, phi0)
        floor = float(np.min(d0[al0])) if al0.any() else 5.0
        min_digits = max(2.0, floor - 1.0)
        calib = dict(integrable_min_digits=floor,
                     integrable_median_digits=float(np.median(d0[al0])),
                     chaos_threshold_digits=min_digits)
        print(f"calibration on the integrable case: min={floor:.2f} digits, "
              f"median={np.median(d0[al0]):.2f} -> chaos threshold "
              f"{min_digits:.2f} digits", flush=True)

    results = []
    ref_pack = None

    for amp in amps:
        modes = []
        for (m, n) in mspec:
            roots = resonant_surfaces(x, q, m / n)
            if not roots:
                continue
            md = Mode(m=m, n=n, amp=amp)
            md.psin_res = roots[0]
            modes.append(md)

        pert = Perturbation(eq, modes, tsf=tsf) if amp > 0 else None
        fl = FieldLines(eq, pert)

        pred = []
        for md in modes:
            eps = pert.amplitude_at(md) if pert else 0.0
            dqds = float(np.interp(md.psin_res, x, dq))
            W = island_width_psin(eps, md.q_res, dqds) if pert else 0.0
            pred.append(dict(m=md.m, n=md.n, q_res=md.q_res,
                             psi_n_res=md.psin_res, eps=eps,
                             dq_dpsin=dqds, W_predicted=W))
        S = (chirikov(pred[0]["W_predicted"], pred[0]["psi_n_res"],
                      pred[1]["W_predicted"], pred[1]["psi_n_res"])
             if len(pred) >= 2 else None)

        ta = time.time()
        th, sn, phi, alive = A.trace_theta(fl, r_seeds, z_seeds,
                                           n_turns=args.n_turns,
                                           pts_per_turn=args.pts_per_turn)
        cls = A.classify(th, sn, phi, alive, min_digits=min_digits,
                         rationals=RATIONALS, psi_n0=psi_n0,
                         nu_profile=(x, 1.0 / q))
        chaotic_mask = np.array([l == "chaotic" for l in cls["labels"]])
        uni_alive = alive & is_uniform
        cls["chaos_fraction_uniform"] = float(
            (chaotic_mask & uni_alive).sum() / max(uni_alive.sum(), 1))
        dt = time.time() - ta

        for p in pred:
            mw = A.measured_island_width(sn, cls, p["q_res"])
            p["W_measured"] = mw["width"] if mw else None
            p["measured_span"] = mw

        plat = A.plateau_report(cls["nu"], psi_n0, RATIONALS)

        print(f"\namp={amp:.2e}  S={S if S is None else round(S,3)}  ({dt:.1f}s)")
        print(f"  regular={cls['n_regular']}  island={cls['n_island']}  "
              f"chaotic={cls['n_chaotic']}  escaped={cls['n_escaped']}")
        print(f"  chaos_frac: all={cls['chaos_fraction']:.3f}  "
              f"UNIFORM(unbiased)={cls['chaos_fraction_uniform']:.3f}")
        for p in pred:
            wm = "n/a" if p["W_measured"] is None else f"{p['W_measured']:.4f}"
            print(f"  m/n={p['m']}/{p['n']}  W_pred={p['W_predicted']:.4f}  W_meas={wm}")
        if plat:
            print("  nu plateaus: " + ", ".join(
                f"{d['m']}/{d['n']}(n={d['n_orbits']},span={d['psi_n_span']:.3f})"
                for d in plat))

        results.append(dict(amp=amp, chirikov_S=S, modes=pred,
                            bands=[dict(m=b["m"], n=b["n"], q=b["q"],
                                        n_members=len(b["members"]),
                                        kind=b.get("kind"),
                                        nu_mean=b.get("nu_mean"),
                                        nu_spread=b.get("nu_spread"))
                                   for b in cls["bands"]],
                            median_digits=float(np.median(cls["digits"][alive]))
                            if alive.any() else None,
                            n_regular=cls["n_regular"], n_island=cls["n_island"],
                            n_chaotic=cls["n_chaotic"], n_escaped=cls["n_escaped"],
                            chaos_fraction=cls["chaos_fraction"],
                            chaos_fraction_uniform=cls["chaos_fraction_uniform"],
                            plateaus=plat,
                            trace_seconds=dt))

        if abs(amp - args.ref_amp) < 1e-12:
            Rp, Zp, alivep = fl.poincare(r_seeds, z_seeds,
                                         n_punctures=args.n_punctures,
                                         rtol=1e-9, atol=1e-11)
            snp = eq.psi_n(Rp.ravel(), Zp.ravel()).reshape(Rp.shape)
            ref_pack = dict(R=Rp, Z=Zp, psin=snp, alive=alivep,
                            labels=cls["labels"], nu=cls["nu"], amp=amp,
                            pred=pred, S=S)

    payload = dict(
        provenance=dict(machine=shot.source, shot=Path(args.shot).name,
                        frame=int(k), time_ms=float(shot.times[k]),
                        q95=float(shot.q95[k]),
                        B_phi_axis=float(eq.F / eq.r_axis)),
        settings=dict(n_seeds=args.n_seeds, n_turns=args.n_turns,
                      pts_per_turn=args.pts_per_turn,
                      modes=args.modes, amps=amps,
                      chaos_indicator="weighted Birkhoff convergence digits",
                      chaos_threshold_digits=min_digits),
        calibration=calib,
        q_profile=dict(psi_n=x.tolist(), q=q.tolist()),
        seeds_psi_n=psi_n0.tolist(),
        results=results,
        total_seconds=time.time() - t0,
    )
    (out / "analysis.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(out / "seeds.npz", psi_n0=psi_n0, r=r_seeds, z=z_seeds,
                        is_uniform=is_uniform)
    if ref_pack is not None:
        np.savez_compressed(out / "reference_section.npz",
                            R=ref_pack["R"], Z=ref_pack["Z"],
                            psin=ref_pack["psin"], alive=ref_pack["alive"],
                            nu=ref_pack["nu"],
                            labels=np.array([str(s) for s in ref_pack["labels"]]))

    try:
        make_figure(out, payload, ref_pack, eq, x, q, psi_n0)
    except Exception as e:
        print(f"[figure] skipped: {type(e).__name__}: {e}")
    print(f"\ndone in {time.time()-t0:.1f}s -> {out}/analysis.json")
    return 0


def make_figure(out, payload, ref, eq, x, q, psi_n0):
    COL = {"regular": "#4a7fd4", "island": "#f2a33c",
           "chaotic": "#d94f4f", "escaped": "#777777"}
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.22)

    # -- A: classified section ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    if ref is not None:
        for lab in ("regular", "island", "chaotic"):
            sel = np.array([l == lab for l in ref["labels"]]) & ref["alive"]
            if not sel.any():
                continue
            ax.plot(ref["R"][sel].ravel(), ref["Z"][sel].ravel(), ".",
                    ms=0.7, color=COL[lab], label=f"{lab} ({int(sel.sum())})",
                    rasterized=True)
        ax.plot(eq.r_axis, eq.z_axis, "k+", ms=10)
        ax.legend(fontsize=9, markerscale=12, loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"A. Poincare section, classified (amp = {ref['amp']:.1e})"
                 if ref else "A. section")

    # -- B: rotation number profile ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    res = payload["results"]
    ref_res = min(res, key=lambda r: abs(r["amp"] - (ref["amp"] if ref else 1e-3)))
    if ref is not None:
        for lab in ("regular", "island", "chaotic"):
            sel = np.array([l == lab for l in ref["labels"]])
            if sel.any():
                ax.plot(psi_n0[sel], ref["nu"][sel], "o", ms=5,
                        color=COL[lab], label=lab)
    ax.plot(x, 1.0 / q, "k-", lw=1.0, alpha=0.6, label=r"$1/q$ from equilibrium")
    for (m, n) in [(2, 1), (3, 1), (5, 2), (3, 2)]:
        ax.axhline(n / m, color="#999", ls=":", lw=0.8)
        ax.text(0.015, n / m, f"{n}/{m}", fontsize=8, color="#666", va="bottom")
    ax.set_xlabel(r"$\psi_N$ of the seed"); ax.set_ylabel(r"$\nu = 1/q$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title("B. Rotation number: plateaus mark the resonances")

    # -- C: island width, measured vs predicted ----------------------------
    ax = fig.add_subplot(gs[1, 0])
    for i, mk in enumerate(["2/1", "3/1"]):
        A_, Wp, Wm = [], [], []
        for r in res:
            for p in r["modes"]:
                if f"{p['m']}/{p['n']}" != mk or r["amp"] <= 0:
                    continue
                A_.append(r["amp"]); Wp.append(p["W_predicted"])
                Wm.append(p["W_measured"] if p["W_measured"] else np.nan)
        c = ["#2a6ebb", "#bb5a2a"][i]
        ax.plot(A_, Wp, "-", color=c, label=f"m/n={mk} predicted")
        ax.plot(A_, Wm, "o--", color=c, mfc="w", label=f"m/n={mk} measured")
    if any(np.isfinite(v) and v > 0 for v in ax.get_lines()[0].get_xdata()) if ax.get_lines() else False:
        ax.set_xscale("log")
        ys = np.concatenate([l.get_ydata() for l in ax.get_lines()]) if ax.get_lines() else np.array([])
        if ys.size and np.nanmax(ys) > 0:
            ax.set_yscale("log")
    ax.set_xlabel("perturbation amplitude (fraction of flux range)")
    ax.set_ylabel(r"island width in $\psi_N$")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")
    ax.set_title(r"C. Island width: pendulum $W=4\sqrt{\epsilon q/|dq/d\psi_N|}$")

    # -- D: chaos fraction vs Chirikov -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    S = [r["chirikov_S"] for r in res if r["chirikov_S"] is not None]
    F = [r.get("chaos_fraction_uniform", r["chaos_fraction"])
         for r in res if r["chirikov_S"] is not None]
    if S:
        ax.plot(S, F, "o-", color="#d94f4f")
    for r in res:
        if r["chirikov_S"] is None:
            continue
        ax.annotate(f"{r['amp']:.0e}", (r["chirikov_S"], r["chaos_fraction"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_ylabel("chaotic fraction (uniform seeds only)")
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.text(1.02, 0.92, "Chirikov S = 1\n(heuristic)", fontsize=9,
            transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("Chirikov overlap parameter S")
    ax.set_ylabel("fraction of orbits classified chaotic")
    ax.grid(alpha=0.2)
    ax.set_title("D. Does stochasticity actually set in at S = 1?")

    p = payload["provenance"]
    fig.suptitle(f"Poincare analysis — {p['machine']} shot "
                 f"{p['shot'].split('_')[-1].split('.')[0]}, t = {p['time_ms']:.0f} ms, "
                 f"q95 = {p['q95']:.3f}   |   prescribed vacuum perturbation; "
                 f"a vacuum section OVERESTIMATES stochasticity",
                 fontsize=12)
    fig.savefig(Path(out) / "poincare_analysis.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {Path(out)/'poincare_analysis.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
