#!/usr/bin/env python3
"""C5 stand A: 40 instances x 16 starts on the corrected multi-mode Tokamap (THEORY.md).

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/c5/run_tokamap.py"
"""
from __future__ import annotations
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_common as fc                                                   # noqa: E402
from stand_tokamap import observable                                     # noqa: E402

PRIOR, BOUNDS = (0.05, 0.25), (0.01, 0.6)
N_INST, N_START, MAX_NFEV = 40, 16, 60


def job(args):
    seed, start, sigma = args
    a, p = fc.draw(np.random.default_rng(seed), *PRIOR)
    return fc.run_one(observable, a, p, seed, 5000 + 100 * seed + start, PRIOR, BOUNDS,
                      MAX_NFEV, sigma=sigma)


def main():
    out = {}
    for sigma in (0.03, 0.10):
        jobs = [(s, k, sigma) for s in range(N_INST) for k in range(N_START)]
        with Pool(10) as pool:
            recs = pool.map(job, jobs, chunksize=4)
        summ = fc.summarise(recs)
        out[f"sigma_{sigma}"] = dict(summary={k: v for k, v in summ.items() if k != "instances"},
                                     instances=summ["instances"], records=recs)
        print(f"sigma={sigma}: success {summ['success_rate']:.3f}  spurious {summ['spurious_rate']:.3f}"
              f"  tech-fail {summ['tech_fail_rate']:.3f}  median max amp err {summ['median_amp_err']:.3f}"
              f"  median max phase err {summ['median_phase_err_deg']:.1f} deg", flush=True)
    (HERE / "results_tokamap.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
