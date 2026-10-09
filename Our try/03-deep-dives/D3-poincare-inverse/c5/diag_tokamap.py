#!/usr/bin/env python3
"""POST-HOC diagnostic for stand A (not part of the pre-registered verdict; THEORY.md is frozen).
Question: is the 0 % success a code bug, an optimiser failure, or a property of the landscape?"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_common as fc
from stand_tokamap import observable
from run_tokamap import PRIOR, BOUNDS
out = {}
rows = []
for seed in range(8):
    a, p = fc.draw(np.random.default_rng(seed), *PRIOR)
    y = fc.noisy(observable(a, p), seed); res = fc.make_residual(observable, y)
    xt = np.r_[np.log(a), p]; c_true = np.sum(res(xt) ** 2)
    lnb = (np.log(BOUNDS[0]), np.log(BOUNDS[1]))
    r = dict(seed=seed, chi2_true=float(c_true))
    for tag, dx in (("from_truth", np.zeros(6)), ("near_5pct_10deg", np.r_[np.full(3, 0.05), np.full(3, np.deg2rad(10))]),
                    ("near_20pct_45deg", np.r_[np.full(3, 0.2), np.full(3, np.deg2rad(45))])):
        sol, _ = fc.fit(res, xt + dx, lnb, 60)
        ok, da, dp = fc.within(np.exp(sol.x[:3]), sol.x[3:], a, p)
        r[tag] = dict(chi2_ratio=float(2 * sol.cost / c_true), within=ok, max_amp_err=float(da.max()),
                      max_phase_err_deg=float(np.rad2deg(dp).max()), status=int(sol.status), nfev=int(sol.nfev))
    rows.append(r)
    print(seed, {k: (round(v["chi2_ratio"], 2), v["within"]) for k, v in r.items() if isinstance(v, dict)})
out["local_fits"] = rows
# landscape roughness: noise-free chi2 along phi_1 through truth, instance 0
a, p = fc.draw(np.random.default_rng(0), *PRIOR); y0 = observable(a, p)
scale = fc.SIGMA * np.maximum(y0, 0.1 * np.median(y0))
for name, idx, grid in (("phi1_fine", 3, np.deg2rad(np.linspace(-20, 20, 81))),
                        ("phi1_full", 3, np.deg2rad(np.linspace(-180, 180, 145))),
                        ("lna1_fine", 0, np.linspace(-0.2, 0.2, 81))):
    c = []
    for d in grid:
        aa, pp = a.copy(), p.copy()
        if idx == 0: aa[0] *= np.exp(d)
        else: pp[0] += d
        c.append(float(np.sum(((observable(aa, pp) - y0) / scale) ** 2)))
    c = np.array(c)
    nmin = int(np.sum((c[1:-1] < c[:-2]) & (c[1:-1] < c[2:])))
    out[name] = dict(grid=grid.tolist(), chi2_0=c.tolist(), n_local_minima=nmin)
    print(f"{name}: local minima {nmin}; chi2_0 range {c.min():.1f}..{c.max():.1f}; at +-1 step {c[len(c)//2-1]:.1f}, {c[len(c)//2+1]:.1f}")
(HERE / "diag_tokamap.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
