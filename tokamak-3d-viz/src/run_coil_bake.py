#!/usr/bin/env python3
"""Bake a Poincare section driven by the COIL backend, for side-by-side render.

Produces a bake directory compatible with the Blender scene builder: the
geometry is copied from an existing analytic bake, and only the punctures and
their orbit classification are recomputed with the coil field.

    python src/run_coil_bake.py --from data/bake01 --out data/bake_coil
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config
from tokviz.equilibrium import TWO_PI
from tokviz.fieldline import FieldLines
from tokviz.rmp_coils import icoil_array
from tokviz.coilfield import CoilPerturbation
from tokviz import analysis as ANA
from tokviz import surfaces as S
from run_analysis import build_equilibrium

LABEL_ID = {"regular": 0, "island": 1, "chaotic": 2, "escaped": 3}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--from", dest="src", default="data/bake01")
    ap.add_argument("--out", default="data/bake_coil")
    ap.add_argument("--current", type=float, default=3.0e3)
    ap.add_argument("--n-only", type=int, default=0,
                    help="keep only this toroidal harmonic (0 = full comb)")
    ap.add_argument("--n-seeds", type=int, default=44)
    ap.add_argument("--n-punctures", type=int, default=140)
    ap.add_argument("--class-turns", type=int, default=60)
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("flux_surfaces.npz", "lcfs.npz", "vessel.npz", "pf_coils.npz",
              "tf_profile.npz", "q_profile.npz", "psi_n_grid.npz",
              "fieldlines.npz"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    man = json.loads((src / "manifest.json").read_text())
    scale = float(man["geometry"]["scale"])

    t0 = time.time()
    shot, k, eq = build_equilibrium(args.shot)
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    print(f"building the coil field ...", flush=True)
    cp = CoilPerturbation(eq, icoil_array(n_toroidal=1, current=1.0))
    if args.n_only:
        cp.n_keep = (int(args.n_only),)
    cp.gain = args.current
    div = cp.divergence_report()
    print(f"  div rel = {div['relative_rms']:.2e}   ({time.time()-t0:.0f}s)", flush=True)

    fl = FieldLines(eq, cp)
    r0 = np.linspace(eq.r_axis + 0.03, eq.r_axis + 0.555, args.n_seeds)
    z0 = np.full_like(r0, eq.z_axis)

    print(f"tracing {args.n_seeds} seeds x {args.n_punctures} punctures ...", flush=True)
    ta = time.time()
    R, Z, alive = fl.poincare(r0, z0, n_punctures=args.n_punctures,
                              rtol=1e-8, atol=1e-10)
    print(f"  {time.time()-ta:.0f}s, alive {int(alive.sum())}/{r0.size}", flush=True)

    print("classifying ...", flush=True)
    tc = time.time()
    th, sn_c, phi_c, al_c = ANA.trace_theta(fl, r0, z0, n_turns=args.class_turns,
                                            pts_per_turn=22)
    cls = ANA.classify(th, sn_c, phi_c, al_c, min_digits=4.0,
                       rationals=[(1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (7, 2)],
                       psi_n0=eq.psi_n(r0, z0), nu_profile=(x, 1.0 / q))
    print(f"  regular={cls['n_regular']} island={cls['n_island']} "
          f"chaotic={cls['n_chaotic']}  ({time.time()-tc:.0f}s)", flush=True)

    lab = np.array([LABEL_ID[str(l)] for l in cls["labels"]], dtype=np.int32)
    sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
    pts = S.punctures_xyz(R[alive], Z[alive], 0.0, scale=scale)
    np.savez_compressed(
        out / "punctures.npz",
        points=pts.astype(np.float32),
        psi_n=sn[alive].ravel().astype(np.float32),
        seed_id=np.repeat(np.arange(r0.size)[:, None], R.shape[1], axis=1)[alive].ravel().astype(np.int32),
        R=R[alive].ravel().astype(np.float32), Z=Z[alive].ravel().astype(np.float32),
        shape=np.asarray(R[alive].shape, dtype=np.int32),
        label_id=np.repeat(lab[alive][:, None], R.shape[1], axis=1).ravel(),
        label_names=np.asarray(["regular", "island", "chaotic", "escaped"]),
        nu=np.repeat(cls["nu"][alive][:, None], R.shape[1], axis=1).ravel().astype(np.float32))

    man["perturbation"] = dict(
        kind="coil array via Biot-Savart, held as a vector potential",
        current_A=args.current, n_only=args.n_only or "full comb",
        divergence_relative_rms=div["relative_rms"],
        caveat="chaos statistics from this backend are NOT trusted; see docs/COIL-BACKEND.md")
    man["orbit_classification"] = dict(
        n_regular=cls["n_regular"], n_island=cls["n_island"],
        n_chaotic=cls["n_chaotic"], label_ids=LABEL_ID)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
