#!/usr/bin/env python3
"""C5 stand B: tokamak-3d-viz curl field, DIII-D 203702 frame 75, modes 2/1, 5/2, 3/1 (THEORY.md).

3 instances x 6 starts (nonlinear fits) + 12x12 phase-pair scans for instance 0.
Same sampling as tokamak-3d-viz/docs/INVERSE.md: 22 levels x 6 phases, 28 turns, 12 pts/turn.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/c5/run_viz.py"   # ~5 h
"""
from __future__ import annotations
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VIZ = HERE.parents[3] / "tokamak-3d-viz" / "src"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(VIZ))
import fit_common as fc                                                   # noqa: E402

PRIOR, BOUNDS = (1e-3, 3e-3), (2e-4, 1e-2)
N_INST, N_START, MAX_NFEV = 3, 6, 20
N_GRID = 12
N_TURNS = 28
if os.environ.get("C5_SMOKE"):          # code test only: tiny budget, results discarded
    N_INST, N_START, MAX_NFEV, N_GRID, N_TURNS = 1, 1, 1, 3, 2
_S = {}


def init():
    os.chdir(VIZ.parent)
    from tokviz import config
    from tokviz.perturbation import Mode, resonant_surfaces
    from run_analysis import build_equilibrium
    from run_inverse import build_seeds
    _, _, eq = build_equilibrium(str(config.DATA_DIR / config.DEFAULT_SHOT))
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140)); tsf = eq.theta_star_field()
    modes = []
    for m, n in [(2, 1), (5, 2), (3, 1)]:
        md = Mode(m=m, n=n, amp=1e-3); md.psin_res = resonant_surfaces(x, q, m / n)[0]
        modes.append(md)
    res = [md.psin_res for md in modes]
    levels = np.linspace(max(min(res) - 0.18, 0.08), min(max(res) + 0.10, 0.955), 22)
    r0, z0, lvl = build_seeds(eq, tsf, levels, 6)
    _S.update(eq=eq, tsf=tsf, modes=modes, levels=levels, r0=r0, z0=z0, lvl=lvl)


def forward(a, p):
    from run_inverse import observable
    S = _S
    _, yr = observable(S["eq"], S["tsf"], S["modes"], np.asarray(a), S["r0"], S["z0"], S["lvl"],
                       S["levels"], N_TURNS, 12, phases=np.asarray(p))
    return yr


def truth(seed):
    return fc.draw(np.random.default_rng(seed), *PRIOR)


def fit_job(args):
    seed, start = args
    a, p = truth(seed)
    return fc.run_one(forward, a, p, seed, 5000 + 100 * seed + start, PRIOR, BOUNDS, MAX_NFEV)


def scan_job(args):
    pair, i, j = args
    a, p = truth(0)
    g = 2 * np.pi * np.arange(N_GRID) / N_GRID
    pp = p.copy(); pp[pair[0]] = g[i]; pp[pair[1]] = g[j]
    return pair, i, j, forward(a, pp)


def main():
    t0 = time.time()
    tag = "_smoke" if os.environ.get("C5_SMOKE") else ""
    log = (HERE / f"results_viz_fits{tag}.jsonl").open("a")
    jobs = [(s, k) for s in range(N_INST) for k in range(N_START)]
    recs = []
    with Pool(9, initializer=init) as pool:
        for r in pool.imap_unordered(fit_job, jobs):
            recs.append(r); log.write(json.dumps(r) + "\n"); log.flush()
            print(f"[{time.time()-t0:6.0f}s] inst {r['seed']} start {r['start']}: within={r['within']}"
                  f" chi2={r['chi2']:.1f} (true {r['chi2_true']:.1f}) nfev={r['nfev']}"
                  f" {r['seconds']/60:.0f} min", flush=True)
    summ = fc.summarise(recs)

    # phase-pair scans on noise-free data, instance 0 (THEORY §4)
    init_local = not _S
    if init_local:
        init()
    a, p = truth(0); y0 = forward(a, p)
    scale = fc.SIGMA * np.maximum(y0, 0.1 * np.median(y0))
    scans = {}
    sjobs = [(pair, i, j) for pair in ((0, 1), (1, 2)) for i in range(N_GRID) for j in range(N_GRID)]
    with Pool(10, initializer=init) as pool:
        for pair, i, j, y in pool.imap_unordered(scan_job, sjobs, chunksize=2):
            scans.setdefault(str(pair), np.zeros((N_GRID, N_GRID)))[i, j] = np.sum(((y - y0) / scale) ** 2)
    g = 2 * np.pi * np.arange(N_GRID) / N_GRID
    scan_out, spurious_scan = {}, False
    for key, C in scans.items():
        pair = eval(key)
        ti = np.abs(fc.wrap(g - p[pair[0]])) / (2 * np.pi / N_GRID)
        tj = np.abs(fc.wrap(g - p[pair[1]])) / (2 * np.pi / N_GRID)
        minima = []
        for i in range(N_GRID):
            for j in range(N_GRID):
                nb = [C[(i + di) % N_GRID, (j + dj) % N_GRID] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                      if (di, dj) != (0, 0)]
                if C[i, j] < min(nb):
                    far = max(ti[i], tj[j]) > 2
                    sp = bool(far and C[i, j] <= y0.size)
                    spurious_scan |= sp
                    minima.append(dict(i=i, j=j, chi2_0=float(C[i, j]), far_from_truth=bool(far),
                                       spurious=bool(sp)))
        scan_out[key] = dict(grid=C.tolist(), local_minima=minima)
        print(f"scan {key}: minima {[(m['i'], m['j'], round(m['chi2_0'], 1), m['far_from_truth']) for m in minima]}")
    summ["instances"][0]["spurious_scan"] = bool(spurious_scan)
    if spurious_scan:
        summ["instances"][0]["spurious"] = True
        summ["spurious_rate"] = sum(i["spurious"] for i in summ["instances"]) / summ["n_instances"]
    out = dict(summary={k: v for k, v in summ.items() if k != "instances"}, instances=summ["instances"],
               records=recs, scans=scan_out, n_obs=int(y0.size), total_seconds=time.time() - t0)
    (HERE / f"results_viz{tag}.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done in {(time.time()-t0)/3600:.2f} h: success {summ['success_rate']:.3f} "
          f"spurious {summ['spurious_rate']:.3f} tech-fail {summ['tech_fail_rate']:.3f}")


if __name__ == "__main__":
    main()
