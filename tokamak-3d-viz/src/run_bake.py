#!/usr/bin/env python3
"""Bake the physics for the Blender scene.

Runs OUTSIDE Blender, in a venv with numpy/scipy/pyarrow/scikit-image.
Writes a directory of .npz plus a manifest.json that Blender then loads.

    python src/run_bake.py --out data/bake01

Every number the render annotates comes from manifest.json, so the image and
the document can never drift apart.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config
from tokviz.equilibrium import load_shot, select_frame, Equilibrium, TWO_PI
from tokviz.fieldline import FieldLines, invariant_drift, q_from_tracing
from tokviz.perturbation import (Mode, Perturbation, resonant_surfaces,
                                 island_width_psin, chirikov)
from tokviz.islands import island_seeds, classify_fixed_points
from tokviz import analysis as ANA
from tokviz import surfaces as S
from tokviz.export import Bake


def parse_modes(spec: str, amp: float):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        mn, _, a = tok.partition(":")
        m_s, _, n_s = mn.partition("/")
        out.append(Mode(m=int(m_s), n=int(n_s), amp=float(a) if a else amp))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default=str(config.BAKE_DIR / "bake01"))
    ap.add_argument("--q95-target", type=float, default=3.7)
    ap.add_argument("--frame", type=int, default=None, help="override frame index")
    ap.add_argument("--modes", default="2/1,3/1")
    ap.add_argument("--amp", type=float, default=1.0e-3,
                    help="perturbation amplitude as a fraction of the flux range")
    ap.add_argument("--scale", type=float, default=config.GEOMETRY_SCALE)
    ap.add_argument("--n-punctures", type=int, default=240)
    ap.add_argument("--n-seeds", type=int, default=48)
    ap.add_argument("--n-surfaces", type=int, default=14)
    ap.add_argument("--n-pol", type=int, default=160)
    ap.add_argument("--n-tor", type=int, default=192)
    ap.add_argument("--fieldline-turns", type=int, default=5)
    ap.add_argument("--skip-trace", action="store_true",
                    help="geometry only; skip the expensive tracing")
    args = ap.parse_args(argv)

    t_start = time.time()
    bake = Bake(args.out)
    log = lambda *a: print(*a, flush=True)

    # ---------------------------------------------------------------- data
    log(f"[1/7] loading {args.shot}")
    shot = load_shot(args.shot)
    k = args.frame if args.frame is not None else select_frame(shot, args.q95_target)
    log(f"      {shot.source} frame {k}  t={shot.times[k]:.1f} ms  "
        f"q95={shot.q95[k]:.4f} betaN={shot.beta_n[k]:.3f} li={shot.li[k]:.3f}")

    eq = Equilibrium(shot.psirz[k], shot.grid_R, shot.grid_Z, machine=shot.source)
    ra, za, psi_a = eq.find_axis()
    nb = int(shot.lcfs_n[k])
    lcfs = np.column_stack([shot.lcfs_r[k][:nb], shot.lcfs_z[k][:nb]])
    psi_b = eq.set_boundary_from_contour(lcfs[:, 0], lcfs[:, 1])
    F = eq.calibrate_F(float(shot.q95[k]))
    log(f"      axis ({ra:.5f}, {za:.5f})  EFIT ({shot.r_axis[k]:.5f}, {shot.z_axis[k]:.5f})"
        f"  dR={abs(ra-shot.r_axis[k])*1e3:.3f} mm dZ={abs(za-shot.z_axis[k])*1e3:.3f} mm")
    log(f"      F={F:.5f} T m -> B_phi(axis)={F/ra:.4f} T")

    # ------------------------------------------------------------ q profile
    log("[2/7] q profile")
    lv = np.linspace(0.03, 0.99, 140)
    x, q = eq.q_profile(lv)
    dq = np.gradient(q, x)
    log(f"      q0~{q[0]:.3f} (psi_n={x[0]:.2f})  q95={np.interp(0.95,x,q):.4f}  "
        f"q_edge={q[-1]:.3f}")

    # -------------------------------------------------------------- modes
    log("[3/7] resonant surfaces and island widths")
    modes = parse_modes(args.modes, args.amp)
    tsf = eq.theta_star_field()
    mode_info = []
    for md in modes:
        roots = resonant_surfaces(x, q, md.q_res)
        if not roots:
            log(f"      m/n={md.m}/{md.n}: q={md.q_res:.3f} NOT in profile -- skipped")
            continue
        md.psin_res = roots[0]
        mode_info.append(md)
    modes = mode_info
    if not modes:
        raise SystemExit("no resonant modes inside the plasma")

    pert = Perturbation(eq, modes, tsf=tsf)
    for md in modes:
        eps = pert.amplitude_at(md)
        dqds = float(np.interp(md.psin_res, x, dq))
        md.width_psin = island_width_psin(eps, md.q_res, dqds)
        md.eps = eps
        log(f"      m/n={md.m}/{md.n} q={md.q_res:.3f} psi_n={md.psin_res:.4f} "
            f"eps={eps:.3e} dq/dpsi_n={dqds:.3f} W={md.width_psin:.4f}")
    S_chir = None
    if len(modes) >= 2:
        a, b = modes[0], modes[1]
        S_chir = chirikov(a.width_psin, a.psin_res, b.width_psin, b.psin_res)
        log(f"      Chirikov S={S_chir:.3f} (heuristic, not a theorem)")

    fl = FieldLines(eq, pert)

    # -------------------------------------------------------- QA: integrator
    log("[4/7] QA: invariant drift on the unperturbed field")
    drift, _ = invariant_drift(eq, ra + np.array([0.15, 0.35, 0.50]),
                               np.full(3, za), n_turns=60)
    log(f"      max |dpsi|/dpsi_tot over 60 turns = {drift.max():.3e}")

    # ------------------------------------------------------------- tracing
    punct = None
    if not args.skip_trace:
        log(f"[5/7] Poincare: {args.n_seeds} radial seeds + island seeds "
            f"x {args.n_punctures} punctures")
        r_rad = np.linspace(ra + 0.03, ra + 0.555, args.n_seeds)
        z_rad = np.full_like(r_rad, za)
        r_isl, z_isl = [], []
        for md in modes:
            ri, zi = island_seeds(eq, tsf, fl, md)
            r_isl.append(ri); z_isl.append(zi)
        r0 = np.concatenate([r_rad] + [a for a in r_isl if a.size])
        z0 = np.concatenate([z_rad] + [a for a in z_isl if a.size])
        t0 = time.time()
        R, Z, alive = fl.poincare(r0, z0, n_punctures=args.n_punctures,
                                  rtol=1e-9, atol=1e-11)
        log(f"      {r0.size} seeds traced in {time.time()-t0:.1f}s, "
            f"alive {int(alive.sum())}/{r0.size}")
        sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
        seed_id = np.repeat(np.arange(r0.size)[:, None], R.shape[1], axis=1)

        # Classify each orbit so the render can colour punctures by orbit type.
        # Needs its own dense trace: at q < 2 a puncture advances the poloidal
        # angle by more than pi, so the rotation number cannot be recovered from
        # the section alone.
        log("[5a ] classifying orbits (regular / island / chaotic)")
        tc = time.time()
        th_c, sn_c, phi_c, alive_c = ANA.trace_theta(
            fl, r0, z0, n_turns=90, pts_per_turn=26)
        cls = ANA.classify(th_c, sn_c, phi_c, alive_c, min_digits=4.0,
                           rationals=[(md.m, md.n) for md in modes]
                                     + [(1, 1), (3, 2), (5, 2), (7, 2)],
                           psi_n0=eq.psi_n(r0, z0), nu_profile=(x, 1.0 / q))
        LABEL_ID = {"regular": 0, "island": 1, "chaotic": 2, "escaped": 3}
        lab_id = np.array([LABEL_ID[str(l)] for l in cls["labels"]], dtype=np.int32)
        log(f"      regular={cls['n_regular']} island={cls['n_island']} "
            f"chaotic={cls['n_chaotic']} ({time.time()-tc:.0f}s)")
        pts = S.punctures_xyz(R[alive], Z[alive], 0.0, scale=args.scale)
        bake.save("punctures",
                  points=pts.astype(np.float32),
                  psi_n=sn[alive].ravel().astype(np.float32),
                  seed_id=seed_id[alive].ravel().astype(np.int32),
                  R=R[alive].ravel().astype(np.float32),
                  Z=Z[alive].ravel().astype(np.float32),
                  shape=np.asarray(R[alive].shape, dtype=np.int32),
                  label_id=np.repeat(lab_id[alive][:, None],
                                     R.shape[1], axis=1).ravel(),
                  label_names=np.asarray(["regular", "island", "chaotic", "escaped"]),
                  nu=np.repeat(cls["nu"][alive][:, None],
                               R.shape[1], axis=1).ravel().astype(np.float32))
        punct = (R, Z, alive, sn)

        log(f"[5b ] field-line curves, {args.fieldline_turns} toroidal turns")
        r_c = np.concatenate([np.linspace(ra + 0.08, ra + 0.50, 8)]
                             + [a[:2] for a in r_isl if a.size])
        z_c = np.concatenate([np.full(8, za)] + [a[:2] for a in z_isl if a.size])
        xyz, alive_c = fl.trace_xyz(r_c, z_c, n_turns=args.fieldline_turns,
                                    pts_per_turn=160, rtol=1e-9, atol=1e-11)
        bake.save("fieldlines",
                  xyz=(xyz[alive_c] * args.scale).astype(np.float32),
                  psi_n0=eq.psi_n(r_c[alive_c], z_c[alive_c]).astype(np.float32))
        log(f"      {int(alive_c.sum())} curves x {xyz.shape[1]} points")
    else:
        log("[5/7] tracing skipped (--skip-trace)")

    # ------------------------------------------------------------ geometry
    log("[6/7] geometry")
    levels = np.linspace(0.12, 0.97, args.n_surfaces)
    fs = S.flux_surface_set(eq, levels, n_pol=args.n_pol, n_tor=args.n_tor,
                            scale=args.scale)
    bake.save_meshes("flux_surfaces", fs)
    log(f"      {len(fs)} flux surfaces ({args.n_pol}x{args.n_tor})")

    lcfs_rs = S.resample_closed(lcfs, args.n_pol)
    v, f = S.revolve(lcfs_rs, n_tor=args.n_tor, scale=args.scale)
    bake.save("lcfs", verts=v.astype(np.float32), faces=f.astype(np.int32),
              profile=(lcfs_rs * args.scale).astype(np.float32))

    # first wall / vessel: offset outside the LCFS
    wall = S.offset_closed(lcfs_rs, 0.10)
    vw, vf = S.revolve(wall, n_tor=args.n_tor, scale=args.scale)
    bake.save("vessel", verts=vw.astype(np.float32), faces=vf.astype(np.int32),
              profile=(wall * args.scale).astype(np.float32))

    coil_meshes = {nm: S.coil_torus(c, n_tor=64, scale=args.scale)
                   for nm, c in shot.coils.items()}
    bake.save_meshes("pf_coils", coil_meshes)
    log(f"      {len(coil_meshes)} PF coils from the measured coil table")

    # TF coil bore: offset the vessel profile outward so the coils hug the
    # vessel, instead of an independent D that can end up far too tall.
    tf_prof = S.offset_closed(wall, 0.32) * args.scale
    bake.save("tf_profile", profile=tf_prof.astype(np.float32))

    # psi_n on the (R, Z) grid -- the volume shader samples this as a float
    # image, which is exact because an axisymmetric psi depends only on (R, Z).
    bake.save("psi_n_grid",
              psi_n=eq.psi_n_grid().astype(np.float32),
              R=eq.R.astype(np.float32), Z=eq.Z.astype(np.float32),
              R_scaled=(eq.R * args.scale).astype(np.float32),
              Z_scaled=(eq.Z * args.scale).astype(np.float32))

    bake.save("q_profile", psi_n=x.astype(np.float32), q=q.astype(np.float32),
              dq_dpsin=dq.astype(np.float32))

    # ------------------------------------------------------------ manifest
    log("[7/7] manifest")
    bake.set("provenance", {
        "shot_file": str(Path(args.shot).name),
        "machine": shot.source,
        "frame_index": int(k),
        "time_ms": float(shot.times[k]),
        "efit_q95": float(shot.q95[k]),
        "efit_beta_n": float(shot.beta_n[k]),
        "efit_li": float(shot.li[k]),
        "efit_r_axis": float(shot.r_axis[k]),
        "efit_z_axis": float(shot.z_axis[k]),
        "grid": [int(eq.Z.size), int(eq.R.size)],
        "R_range": [float(eq.R[0]), float(eq.R[-1])],
        "Z_range": [float(eq.Z[0]), float(eq.Z[-1])],
        "data_licence": "CC BY 4.0, Sophelio + General Atomics (not redistributed)",
    })
    bake.set("equilibrium", {
        "r_axis": ra, "z_axis": za, "psi_axis": psi_a, "psi_bdy": psi_b,
        "axis_sign": eq.axis_sign,
        "F_calibrated": F, "B_phi_on_axis": F / ra,
        "F_method": "solved so q(psi_n=0.95) equals the dataset's efit_q95; "
                    "the dataset ships no fpol",
        "q0_approx": float(q[0]), "q_edge": float(q[-1]),
    })
    bake.set("geometry", {
        "scale": float(args.scale),
        "scale_note": "uniform similarity scale; preserves aspect ratio, so "
                      "q(psi) and the field-line topology are invariant",
        "R0_scaled": ra * args.scale,
        "n_pol": int(args.n_pol), "n_tor": int(args.n_tor),
    })
    bake.set("perturbation", {
        "kind": "prescribed vacuum-like resonant perturbation on psi",
        "divergence_free": "exact: B = curl(psi_total grad phi)",
        "angle": "straight-field-line theta*, so mode m resonates at q=m/n",
        "caveat": "a vacuum section OVERESTIMATES stochasticity; the plasma "
                  "screens resonant components",
        "modes": [{"m": md.m, "n": md.n, "q_res": md.q_res,
                   "amp_fraction": md.amp, "phase": md.phase,
                   "envelope": md.envelope,
                   "psi_n_res": md.psin_res,
                   "eps_effective": float(md.eps),
                   "island_width_psin": float(md.width_psin)} for md in modes],
        "chirikov_S": S_chir,
        "chirikov_note": "S >= 1 is a heuristic for stochasticity onset, not a theorem",
    })
    bake.set("orbit_classification", {
        "method": "weighted Birkhoff convergence + rotation-number locking",
        "n_regular": int(cls["n_regular"]) if not args.skip_trace else None,
        "n_island": int(cls["n_island"]) if not args.skip_trace else None,
        "n_chaotic": int(cls["n_chaotic"]) if not args.skip_trace else None,
        "label_ids": {"regular": 0, "island": 1, "chaotic": 2, "escaped": 3},
    } if not args.skip_trace else {"skipped": True})
    bake.set("qa", {
        "axis_dR_mm": abs(ra - float(shot.r_axis[k])) * 1e3,
        "axis_dZ_mm": abs(za - float(shot.z_axis[k])) * 1e3,
        "invariant_drift_60_turns": float(drift.max()),
        "integrator": "scipy solve_ivp DOP853, rtol 1e-9, atol 1e-11",
    })
    path = bake.finish()
    log(f"\ndone in {time.time()-t_start:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
