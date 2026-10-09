#!/usr/bin/env python3
"""Test the parent project's conjecture C5: is the multi-mode INVERSE problem
(recover the (m, n) perturbation spectrum from a Poincare observable) stable
enough to be a hackathon task?

The pilot in `Our try/03-deep-dives/D3-poincare-inverse/` could not answer this.
Its single-mode version was trivially easy (SNR 57 at a 3% amplitude change) and
its multi-mode generalisation was untestable, because generalising the Tokamap
by substitution broke symplecticity (their R8: max|det J - 1| = 1.7). The stated
blocker was "a valid multi-mode symplectic stand is needed".

This stand cannot have that failure: the field is

    B = curl( psi_total grad phi ),
    psi_total = psi(R,Z) + sum_k A_k g_k(psi_n) cos(m_k theta* - n_k phi + a_k)

which is divergence-free EXACTLY, for any number of modes and any amplitudes,
because it is a curl.  Caveat kept in view: the FIELD preserves flux exactly,
but the INTEGRATOR (DOP853) is not symplectic; its measured drift is ~1e-7 over
200 toroidal turns.

Observable
----------
The phase-averaged radial profile of psi_n excursion -- the 3-D analogue of the
observable in the pilot's `identifiability.py`, where averaging over initial
phases dropped the noise floor by 540x.  For each radial level, orbits are
seeded at several straight-field-line angles and their excursions averaged.

    python src/run_inverse.py --out data/inverse
    python src/run_inverse.py --out data/inverse --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config
from tokviz.equilibrium import load_shot, select_frame, Equilibrium, TWO_PI
from tokviz.fieldline import FieldLines
from tokviz.perturbation import Mode, Perturbation, resonant_surfaces
from tokviz.islands import point_at
from run_analysis import build_equilibrium


def build_seeds(eq, tsf, levels, n_phase):
    """Seeds at several straight-field-line angles on each radial level."""
    rs, zs, lvl = [], [], []
    for s in levels:
        for j in range(n_phase):
            th = TWO_PI * j / n_phase
            p = point_at(eq, tsf, float(s), th)
            if p is None:
                continue
            rs.append(p[0]); zs.append(p[1]); lvl.append(float(s))
    return np.asarray(rs), np.asarray(zs), np.asarray(lvl)


def observable(eq, tsf, modes, amps, r0, z0, lvl, levels, n_turns, pts_per_turn,
               phases=None):
    """Phase-averaged psi_n excursion per radial level, given the spectrum."""
    for i, (md, a) in enumerate(zip(modes, amps)):
        md.amp = float(a)
        if phases is not None:
            md.phase = float(phases[i])
    pert = Perturbation(eq, modes, tsf=tsf) if np.any(np.asarray(amps) > 0) else None
    fl = FieldLines(eq, pert)
    phi = np.linspace(0.0, TWO_PI * n_turns, int(n_turns * pts_per_turn) + 1)
    R, Z, alive = fl.trace(r0, z0, phi, rtol=1e-8, atol=1e-10)
    sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
    exc = sn.max(axis=1) - sn.min(axis=1)
    exc = np.where(alive, exc, np.nan)
    # Two reductions of the SAME traces, so comparing them is free:
    #  * averaged  -- mean excursion per radial level (the pilot's observable,
    #                 phase-averaged to kill the reseed noise floor)
    #  * resolved  -- the full (level, seed-phase) map, which keeps the
    #                 poloidal phase information that averaging throws away
    avg = np.array([np.nanmean(exc[lvl == s]) if np.isfinite(exc[lvl == s]).any()
                    else 0.0 for s in levels])
    return np.nan_to_num(avg, nan=0.0), np.nan_to_num(exc, nan=0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default="data/inverse")
    ap.add_argument("--modes", default="2/1,3/1,4/1")
    ap.add_argument("--amp0", type=float, default=1.0e-3,
                    help="linearisation point: amplitude of every mode")
    ap.add_argument("--n-levels", type=int, default=26)
    ap.add_argument("--n-phase", type=int, default=6)
    ap.add_argument("--n-turns", type=int, default=30)
    ap.add_argument("--pts-per-turn", type=int, default=12)
    ap.add_argument("--rel-step", type=float, default=0.15,
                    help="finite-difference step, as a fraction of amp0")
    ap.add_argument("--noise", default="0,0.01,0.03,0.10",
                    help="relative observation noise levels to test")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--fit-phases", action="store_true",
                    help="treat mode phases as unknowns too, as C5 asks")
    ap.add_argument("--phase-step", type=float, default=0.30,
                    help="finite-difference step for phases, radians")
    ap.add_argument("--phase-scale", type=float, default=np.pi / 4,
                    help="reference phase change used to non-dimensionalise")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_levels, args.n_phase, args.n_turns = 12, 4, 20
        args.modes = "2/1,3/1"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shot, k, eq = build_equilibrium(args.shot)
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    tsf = eq.theta_star_field()
    print(f"{shot.source} frame {k}  q95={shot.q95[k]:.4f}", flush=True)

    modes, res = [], []
    for tok in args.modes.split(","):
        m_s, _, n_s = tok.strip().partition("/")
        m, n = int(m_s), int(n_s)
        roots = resonant_surfaces(x, q, m / n)
        if not roots:
            print(f"  m/n={m}/{n}: q={m/n:.3f} not in profile -- skipped")
            continue
        md = Mode(m=m, n=n, amp=args.amp0); md.psin_res = roots[0]
        modes.append(md); res.append(roots[0])
        print(f"  m/n={m}/{n}  q={m/n:.3f}  psi_n_res={roots[0]:.4f}")
    K = len(modes)
    if K < 2:
        raise SystemExit("need at least two resonant modes")

    # radial levels: cover the resonances generously
    lo = max(min(res) - 0.18, 0.08)
    hi = min(max(res) + 0.10, 0.955)
    levels = np.linspace(lo, hi, args.n_levels)
    r0, z0, lvl = build_seeds(eq, tsf, levels, args.n_phase)
    print(f"observable: {len(levels)} levels x {args.n_phase} phases = {r0.size} orbits, "
          f"{args.n_turns} turns", flush=True)

    A0 = np.full(K, args.amp0)
    P0 = np.zeros(K)
    rng0 = np.random.default_rng(7)
    if args.fit_phases:
        P0 = rng0.uniform(0.0, TWO_PI, K)     # a generic, non-degenerate point
    ta = time.time()
    y0, y0r = observable(eq, tsf, modes, A0, r0, z0, lvl, levels,
                         args.n_turns, args.pts_per_turn, phases=P0)
    print(f"baseline observable in {time.time()-ta:.0f}s "
          f"(averaged: {y0.size} values, resolved: {y0r.size})", flush=True)

    # ---- Jacobian by central differences --------------------------------
    n_par = 2 * K if args.fit_phases else K
    J = np.zeros((len(levels), n_par))
    Jr = np.zeros((y0r.size, n_par))
    labels = []
    h = args.rel_step * args.amp0
    for j in range(K):
        Ap = A0.copy(); Ap[j] += h
        Am = A0.copy(); Am[j] -= h
        yp, ypr = observable(eq, tsf, modes, Ap, r0, z0, lvl, levels,
                             args.n_turns, args.pts_per_turn, phases=P0)
        ym, ymr = observable(eq, tsf, modes, Am, r0, z0, lvl, levels,
                             args.n_turns, args.pts_per_turn, phases=P0)
        J[:, j] = (yp - ym) / (2 * h)
        Jr[:, j] = (ypr - ymr) / (2 * h)
        labels.append(f"A[{modes[j].m}/{modes[j].n}]")
        print(f"  d(obs)/dA[{modes[j].m}/{modes[j].n}] done "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    if args.fit_phases:
        hp = args.phase_step
        for j in range(K):
            Pp = P0.copy(); Pp[j] += hp
            Pm = P0.copy(); Pm[j] -= hp
            yp, ypr = observable(eq, tsf, modes, A0, r0, z0, lvl, levels,
                                 args.n_turns, args.pts_per_turn, phases=Pp)
            ym, ymr = observable(eq, tsf, modes, A0, r0, z0, lvl, levels,
                                 args.n_turns, args.pts_per_turn, phases=Pm)
            J[:, K + j] = (yp - ym) / (2 * hp)
            Jr[:, K + j] = (ypr - ymr) / (2 * hp)
            labels.append(f"phase[{modes[j].m}/{modes[j].n}]")
            print(f"  d(obs)/dphase[{modes[j].m}/{modes[j].n}] done "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # Non-dimensionalise so the condition number is meaningful: relative change
    # in the observable per relative change in amplitude, and per a reference
    # phase change of phase_scale radians.  Mixing units in a condition number
    # is always a choice -- this one is stated rather than hidden.
    col_scale = np.concatenate([A0, np.full(K, args.phase_scale)]) if args.fit_phases else A0

    def nondim(Jmat, y):
        d = np.where(np.abs(y)[:, None] > 0, np.abs(y)[:, None], np.nan)
        return np.nan_to_num(Jmat * (col_scale[None, :] / d))

    Jn = nondim(J, y0)
    Jrn = nondim(Jr, y0r)

    # Control for the obvious confound: the phase-resolved observable has
    # n_phase times more values than the averaged one, so part of any advantage
    # is data volume rather than phase content.  Subsample it to the SAME number
    # of rows and re-measure.
    rng_sub = np.random.default_rng(11)
    idx = rng_sub.choice(Jrn.shape[0], size=min(Jn.shape[0], Jrn.shape[0]),
                         replace=False)
    Jrn_sub = Jrn[idx]

    report = {}
    for tag, M in (("averaged", Jn), ("phase-resolved", Jrn),
                   ("phase-resolved-subsampled", Jrn_sub)):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
        print(f"\n[{tag}] singular values: {np.array2string(S, precision=4)}")
        rel = S / S[0]
        print(f"[{tag}] condition number: {cond:.1f}")
        print(f"[{tag}] spectrum relative to the leading direction: "
              f"{np.array2string(rel, precision=3)}")
        for i in range(len(S)):
            dom = np.argsort(-np.abs(Vt[i]))[:3]
            comp = ", ".join(f"{labels[j]}:{Vt[i, j]:+.3f}" for j in dom)
            print(f"  direction {i} (sv={S[i]:.4f}): {comp}")
        report[tag] = dict(singular_values=S.tolist(), condition_number=cond,
                           relative_spectrum=(S / S[0]).tolist(),
                           n_observations=int(M.shape[0]),
                           right_singular_vectors=Vt.tolist())
    S = np.linalg.svd(Jn, compute_uv=False)
    cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")

    # ---- synthetic inversion with noise ---------------------------------
    inv = {}
    for tag, M in (("averaged", Jn), ("phase-resolved", Jrn),
                   ("phase-resolved-subsampled", Jrn_sub)):
        rng = np.random.default_rng(0)
        rows = []
        print(f"\n[{tag}] synthetic recovery:")
        for nl in [float(v) for v in args.noise.split(",")]:
            errs, aerr, perr = [], [], []
            for _ in range(args.n_trials):
                dtrue = rng.normal(0.0, 0.25, n_par)
                dy = M @ dtrue + rng.normal(0.0, nl, M.shape[0])
                dhat, *_ = np.linalg.lstsq(M, dy, rcond=None)
                errs.append(np.linalg.norm(dhat - dtrue) / np.linalg.norm(dtrue))
                aerr.append(np.linalg.norm(dhat[:K] - dtrue[:K])
                            / max(np.linalg.norm(dtrue[:K]), 1e-12))
                if args.fit_phases:
                    perr.append(np.linalg.norm(dhat[K:] - dtrue[K:])
                                / max(np.linalg.norm(dtrue[K:]), 1e-12))
            row = dict(noise=nl, median_rel_error=float(np.median(errs)),
                       p90_rel_error=float(np.percentile(errs, 90)),
                       median_amp_error=float(np.median(aerr)),
                       median_phase_error=float(np.median(perr)) if perr else None)
            rows.append(row)
            extra = (f"   amp {np.median(aerr)*100:5.1f}%"
                     f"   phase {np.median(perr)*100:6.1f}%" if perr
                     else f"   amp {np.median(aerr)*100:5.1f}%")
            print(f"  noise {nl*100:5.1f}%  ->  all {np.median(errs)*100:6.1f}%{extra}")
        inv[tag] = rows

    payload = dict(
        provenance=dict(machine=shot.source, shot=Path(args.shot).name,
                        frame=int(k), q95=float(shot.q95[k])),
        settings=dict(modes=args.modes, amp0=args.amp0, n_levels=args.n_levels,
                      n_phase=args.n_phase, n_turns=args.n_turns,
                      rel_step=args.rel_step, n_trials=args.n_trials,
                      fit_phases=bool(args.fit_phases),
                      phase_scale=float(args.phase_scale)),
        parameter_labels=labels,
        modes=[dict(m=md.m, n=md.n, q=md.q_res, psi_n_res=md.psin_res) for md in modes],
        levels=levels.tolist(), baseline_observable=y0.tolist(),
        jacobian_nondim=Jn.tolist(),
        jacobian_resolved_nondim=Jrn.tolist(),
        subsample_rows=idx.tolist(),
        conditioning=report,
        singular_values=S.tolist(), condition_number=cond,
        inversion=inv, total_seconds=time.time() - t0)
    (out / "inverse.json").write_text(json.dumps(payload, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}/inverse.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
