#!/usr/bin/env python3
"""VC2: do the residual ~1.3% spurious poloidal harmonics make visible islands?

The perturbation contains only m/n = 2/1 and 3/1.  Any island chain found at
another resonance is either

  * a genuine high-order BEAT of the driven modes, or
  * an artefact of the residual spectral pollution in the stand.

These cannot be told apart by which resonance appears -- almost any (m, n) is
reachable by some integer combination a(2,1) + b(3,1), including (1,1) as
2(2,1) - (3,1).  They are told apart by SIZE:

  * spurious m = 1 at 1.3% of the driven amplitude gives, since W ~ sqrt(amp),
    an island about sqrt(0.013) = 11% of the driven width, i.e. W ~ 0.008
    (0.012 with the pre-2026-09-29 formula, which overestimated W by sqrt(q));
  * a third-order beat at amp = 1e-3 is smaller by orders of magnitude.

So the test is quantitative: dense seeding at the candidate resonances, and a
width threshold.

    python src/run_parasitic.py --out data/parasitic
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
from tokviz.perturbation import Mode, Perturbation, resonant_surfaces, island_width_psin
from tokviz import analysis as A
from run_analysis import build_equilibrium

#: (m, n, is_driven, reachable_order)
#: order = |a| + |b| for the lowest integer combination a(2,1) + b(3,1) = (m, n)
CANDIDATES = [
    (2, 1, True, 1),     # driven
    (3, 1, True, 1),     # driven
    (5, 2, False, 2),    # (2,1)+(3,1) -- legitimate second-order beat
    (1, 1, False, 3),    # 2(2,1)-(3,1) -- third order, expected negligible
    (3, 2, False, 4),    # 3(2,1)-(3,1) -- fourth order
    (7, 2, False, 3),    # 2(2,1)+(3,1)
    (4, 1, False, None), # not reachable at low order
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default="data/parasitic")
    ap.add_argument("--amp", type=float, default=1.0e-3)
    ap.add_argument("--n-per-band", type=int, default=44)
    ap.add_argument("--band", type=float, default=0.045,
                    help="half-width in psi_n of the dense band at each resonance")
    ap.add_argument("--n-turns", type=int, default=150)
    ap.add_argument("--width-threshold", type=float, default=0.004,
                    help="island width below which we call it not visible")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shot, k, eq = build_equilibrium(args.shot)
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 160))
    dq = np.gradient(q, x)
    tsf = eq.theta_star_field()
    print(f"{shot.source} frame {k}  q in [{q[0]:.3f}, {q[-1]:.3f}]", flush=True)

    driven = []
    for m, n, isd, _ in CANDIDATES:
        if not isd:
            continue
        md = Mode(m=m, n=n, amp=args.amp)
        md.psin_res = resonant_surfaces(x, q, m / n)[0]
        driven.append(md)
    pert = Perturbation(eq, driven, tsf=tsf)
    fl = FieldLines(eq, pert)

    W_driven = []
    for md in driven:
        eps = pert.amplitude_at(md)
        W_driven.append(island_width_psin(eps, md.q_res,
                                          float(np.interp(md.psin_res, x, dq))))
    print(f"driven: " + ", ".join(
        f"{md.m}/{md.n} at psi_n={md.psin_res:.4f} W={w:.4f}"
        for md, w in zip(driven, W_driven)))
    W_ref = float(np.mean(W_driven))
    print(f"expected width of a 1.3% spurious chain: "
          f"{W_ref*np.sqrt(0.013):.4f}  (threshold {args.width_threshold})\n")

    # dense seeding in a band around every candidate resonance
    targets, rs, zs, tag = [], [], [], []
    rr = np.linspace(eq.r_axis + 0.02, eq.r_axis + 0.60, 900)
    ss = eq.psi_n(rr, np.full_like(rr, eq.z_axis))
    keep = np.isfinite(ss) & (np.diff(np.r_[ss[0] - 1, ss]) > 0)
    for m, n, isd, order in CANDIDATES:
        roots = resonant_surfaces(x, q, m / n)
        if not roots:
            print(f"  {m}/{n}: q={m/n:.3f} outside the profile -- skipped")
            continue
        s_res = roots[0]
        lo = max(s_res - args.band, 0.05); hi = min(s_res + args.band, 0.955)
        tg = np.linspace(lo, hi, args.n_per_band)
        rs.append(np.interp(tg, ss[keep], rr[keep]))
        tag += [f"{m}/{n}"] * args.n_per_band
        targets.append(dict(m=m, n=n, q=m / n, psi_n_res=s_res,
                            driven=bool(isd), beat_order=order))
    r0 = np.concatenate(rs); z0 = np.full_like(r0, eq.z_axis)
    tag = np.array(tag)
    psi_n0 = eq.psi_n(r0, z0)
    print(f"tracing {r0.size} orbits x {args.n_turns} turns ...", flush=True)

    th, sn, phi, alive = A.trace_theta(fl, r0, z0, n_turns=args.n_turns,
                                       pts_per_turn=26)
    d0, _ = A.rotation_number_digits(th, phi)
    cls = A.classify(th, sn, phi, alive, min_digits=4.0,
                     rationals=[(t["m"], t["n"]) for t in targets],
                     psi_n0=psi_n0, nu_profile=(x, 1.0 / q))
    print(f"traced in {time.time()-t0:.0f}s; "
          f"regular={cls['n_regular']} island={cls['n_island']} "
          f"chaotic={cls['n_chaotic']}\n", flush=True)

    print(f"{'m/n':>6} {'q':>6} {'psi_n':>7} {'driven':>7} {'beat':>5} "
          f"{'W_meas':>8} {'verdict':>12}")
    for t in targets:
        mw = A.measured_island_width(sn, cls, t["q"])
        w = mw["width"] if mw else 0.0
        t["W_measured"] = float(w)
        if t["driven"]:
            verdict = "driven"
        elif w <= args.width_threshold:
            verdict = "not visible"
        elif t["beat_order"] == 2:
            verdict = "beat (real)"
        else:
            verdict = "SPURIOUS?"
        t["verdict"] = verdict
        bo = "-" if t["beat_order"] is None else str(t["beat_order"])
        print(f"{t['m']}/{t['n']:<4} {t['q']:>6.3f} {t['psi_n_res']:>7.4f} "
              f"{str(t['driven']):>7} {bo:>5} {w:>8.4f} {verdict:>12}")

    spurious = [t for t in targets
                if not t["driven"] and t["W_measured"] > args.width_threshold
                and t["beat_order"] != 2]
    print()
    if spurious:
        print("VC2 REFUTED: visible chains at " +
              ", ".join(f"{t['m']}/{t['n']} (W={t['W_measured']:.4f})" for t in spurious))
    else:
        print(f"VC2 HOLDS at amp={args.amp:.0e}: no undriven chain above "
              f"W={args.width_threshold} except legitimate second-order beats.")

    (out / "parasitic.json").write_text(json.dumps(dict(
        provenance=dict(machine=shot.source, frame=int(k), amp=args.amp),
        settings=dict(n_per_band=args.n_per_band, band=args.band,
                      n_turns=args.n_turns, width_threshold=args.width_threshold),
        driven_widths=W_driven, expected_spurious_width=W_ref * float(np.sqrt(0.013)),
        targets=targets,
        counts=dict(regular=cls["n_regular"], island=cls["n_island"],
                    chaotic=cls["n_chaotic"]),
        verdict="refuted" if spurious else "holds",
        total_seconds=time.time() - t0), indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}/parasitic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
