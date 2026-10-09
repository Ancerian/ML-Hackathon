#!/usr/bin/env python3
"""E14 / R6 — reproducible curvature test (restores the lost 23.09 synthetic test, N5).

Runs part A of local_models.py only: a paraboloid of curvature lambda (16x range) on the DIII-D
grid, axis found by the scorer's find_o_point, white vs smooth per-node noise eta.
Prints  |dR| * sqrt(lambda)  at eta = 1e-2 (E14: constant if eta >> eta* = lambda h^2 / 2)
and the alpha fitted over one fixed eta window per lambda (the R6 mechanism).

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/c1/e14_curvature.py"
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_models as lm                                                   # noqa: E402

if __name__ == "__main__":
    A = lm.part_a()
    prod = np.array(A["E14"]["dr_sqrt_lambda"])
    print(f"eta = {A['E14']['eta']:.0e};  eta*_R = lambda * {lm.HR ** 2 / 2:.2e},"
          f"  eta*_Z = lambda * {lm.HZ ** 2 / 2:.2e}")
    for lam, (pr, pz) in zip(lm.LAMBDAS, prod):
        print(f"  lambda = {lam:4g}:  |dR| sqrt(lambda) = {pr:.5f}   |dZ| sqrt(lambda) = {pz:.5f}")
    sp = A["E14"]["spread_pct"]
    print(f"  spread (max-min)/mean: R {sp[0]:.1f} %  (= +-{sp[0] / 2:.1f} %),  Z {sp[1]:.1f} %")
    print("  R6: alpha over eta in [1e-5, 1e-2]:",
          ", ".join(f"lambda={k}: {v:.3f}" for k, v in A["R6_alpha_R"].items()))
