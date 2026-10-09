#!/usr/bin/env python3
"""VC6: is the inverse problem posed on the WALL FOOTPRINT harder than the one
posed on radial excursions?

Observable
----------
Per seed, three smooth numbers: the connection length in toroidal turns, and
the strike position on the wall as (cos 2pi s, sin 2pi s).

A raw 2-D strike HISTOGRAM would be the natural thing to hand a team, but it is
useless for a finite-difference Jacobian: bin counts jump discontinuously as the
parameters move.  The per-seed quantities vary smoothly, and the histogram can
always be formed from them afterwards.

    python src/run_footprint.py --out data/footprint
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
from tokviz.equilibrium import TWO_PI
from tokviz.fieldline import FieldLines
from tokviz.perturbation import Mode, Perturbation, resonant_surfaces
from tokviz.surfaces import offset_closed, resample_closed
from tokviz.footprint import Wall, trace_to_wall, footprint_map
from run_analysis import build_equilibrium


def sol_seeds(eq, wall, levels=(1.005, 1.02, 1.04), n_theta=60):
    """Seeds just outside the separatrix, all around poloidally.

    psi_n > 1 has no closed contour, so seeds are placed by radial search along
    rays from the magnetic axis rather than by contour extraction.
    """
    out = []
    for lv in levels:
        for th in np.linspace(0.0, TWO_PI, n_theta, endpoint=False):
            s = np.linspace(0.05, 0.95, 400)
            R = eq.r_axis + s * np.cos(th)
            Z = eq.z_axis + s * np.sin(th)
            sn = eq.psi_n(R, Z)
            idx = np.where(np.isfinite(sn) & (sn >= lv))[0]
            if idx.size:
                out.append((R[idx[0]], Z[idx[0]]))
    P = np.array(out)
    keep = wall.inside(P[:, 0], P[:, 1])
    return P[keep, 0], P[keep, 1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default="data/footprint")
    ap.add_argument("--modes", default="2/1,3/1")
    ap.add_argument("--amp0", type=float, default=3.0e-3)
    ap.add_argument("--taper", type=float, default=0.60,
                    help="perturbation taper width outside the separatrix; the "
                         "island studies use 0.05, which suppresses the field "
                         "exactly where a footprint forms")
    ap.add_argument("--wall-offset", type=float, default=0.12)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--rel-step", type=float, default=0.15)
    ap.add_argument("--phase-step", type=float, default=0.30)
    ap.add_argument("--phase-scale", type=float, default=np.pi / 4)
    ap.add_argument("--noise", default="0,0.01,0.03,0.10")
    ap.add_argument("--n-trials", type=int, default=40)
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shot, k, eq = build_equilibrium(args.shot)
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    tsf = eq.theta_star_field()
    nb = int(shot.lcfs_n[k])
    lcfs = np.column_stack([shot.lcfs_r[k][:nb], shot.lcfs_z[k][:nb]])
    wall = Wall(offset_closed(resample_closed(lcfs, 160), args.wall_offset))
    r0, z0 = sol_seeds(eq, wall)
    print(f"{shot.source} frame {k}; wall perimeter {wall.length:.3f} m; "
          f"{r0.size} SOL seeds", flush=True)

    mspec = []
    for tok in args.modes.split(","):
        a, _, b = tok.strip().partition("/")
        mspec.append((int(a), int(b)))
    modes = []
    for (m, n) in mspec:
        roots = resonant_surfaces(x, q, m / n)
        if not roots:
            continue
        md = Mode(m=m, n=n, amp=args.amp0, taper=args.taper)
        md.psin_res = roots[0]
        modes.append(md)
    K = len(modes)
    print(f"  modes: " + ", ".join(f"{md.m}/{md.n}@{md.psin_res:.3f}" for md in modes))

    def observable(amps, phases):
        for md, a, p in zip(modes, amps, phases):
            md.amp = float(a); md.phase = float(p)
        fl = FieldLines(eq, Perturbation(eq, modes, tsf=tsf))
        res = trace_to_wall(fl, wall, r0, z0, max_turns=args.max_turns,
                            pts_per_turn=40)
        s = np.nan_to_num(res["s"], nan=0.0)
        t = np.nan_to_num(res["turns"], nan=float(args.max_turns))
        return np.concatenate([t, np.cos(TWO_PI * s), np.sin(TWO_PI * s)]), res

    A0 = np.full(K, args.amp0)
    rng0 = np.random.default_rng(7)
    P0 = rng0.uniform(0.0, TWO_PI, K)
    y0, res0 = observable(A0, P0)
    print(f"  baseline: struck {int(res0['struck'].sum())}/{r0.size}, "
          f"median connection {np.median(res0['turns']):.2f} turns, "
          f"observable length {y0.size} ({time.time()-t0:.0f}s)", flush=True)

    n_par = 2 * K
    J = np.zeros((y0.size, n_par))
    labels = []
    h = args.rel_step * args.amp0
    for j in range(K):
        Ap = A0.copy(); Ap[j] += h
        Am = A0.copy(); Am[j] -= h
        J[:, j] = (observable(Ap, P0)[0] - observable(Am, P0)[0]) / (2 * h)
        labels.append(f"A[{modes[j].m}/{modes[j].n}]")
    for j in range(K):
        Pp = P0.copy(); Pp[j] += args.phase_step
        Pm = P0.copy(); Pm[j] -= args.phase_step
        J[:, K + j] = (observable(A0, Pp)[0] - observable(A0, Pm)[0]) / (2 * args.phase_step)
        labels.append(f"phase[{modes[j].m}/{modes[j].n}]")
    print(f"  Jacobian done ({time.time()-t0:.0f}s)", flush=True)

    col = np.concatenate([A0, np.full(K, args.phase_scale)])
    den = np.where(np.abs(y0)[:, None] > 1e-12, np.abs(y0)[:, None], np.nan)
    Jn = np.nan_to_num(J * (col[None, :] / den))
    U, S, Vt = np.linalg.svd(Jn, full_matrices=False)
    cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    print(f"\nsingular values: {np.array2string(S, precision=4)}")
    print(f"condition number: {cond:.1f}")
    print(f"spectrum relative to the leading direction: "
          f"{np.array2string(S/S[0], precision=3)}")
    for i in range(len(S)):
        dom = np.argsort(-np.abs(Vt[i]))[:2]
        print(f"  direction {i} (sv={S[i]:.4f}): " +
              ", ".join(f"{labels[j]}:{Vt[i, j]:+.3f}" for j in dom))

    rng = np.random.default_rng(0)
    rows = []
    print("\nsynthetic recovery:")
    for nl in [float(v) for v in args.noise.split(",")]:
        errs, aerr, perr = [], [], []
        for _ in range(args.n_trials):
            dt = rng.normal(0.0, 0.25, n_par)
            dy = Jn @ dt + rng.normal(0.0, nl, Jn.shape[0])
            dh, *_ = np.linalg.lstsq(Jn, dy, rcond=None)
            errs.append(np.linalg.norm(dh - dt) / np.linalg.norm(dt))
            aerr.append(np.linalg.norm(dh[:K] - dt[:K]) / max(np.linalg.norm(dt[:K]), 1e-12))
            perr.append(np.linalg.norm(dh[K:] - dt[K:]) / max(np.linalg.norm(dt[K:]), 1e-12))
        rows.append(dict(noise=nl, median_rel_error=float(np.median(errs)),
                         median_amp_error=float(np.median(aerr)),
                         median_phase_error=float(np.median(perr))))
        print(f"  noise {nl*100:5.1f}%  ->  all {np.median(errs)*100:6.1f}%"
              f"   amp {np.median(aerr)*100:5.1f}%   phase {np.median(perr)*100:6.1f}%")

    H = footprint_map(res0)
    (out / "footprint.json").write_text(json.dumps(dict(
        provenance=dict(machine=shot.source, frame=int(k)),
        settings=dict(modes=args.modes, amp0=args.amp0, taper=args.taper,
                      wall_offset=args.wall_offset, max_turns=args.max_turns,
                      n_seeds=int(r0.size), n_observations=int(y0.size)),
        parameter_labels=labels,
        singular_values=S.tolist(), condition_number=cond,
        relative_spectrum=(S / S[0]).tolist(),
        struck=int(res0["struck"].sum()),
        median_connection_turns=float(np.median(res0["turns"])),
        footprint_hist=H.tolist(),
        inversion=rows, total_seconds=time.time() - t0), indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}/footprint.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
