#!/usr/bin/env python3
"""D2 -- what an L2-trained regressor does when the target is MULTIVALUED.

The claim under test
--------------------
Every ML paper on equilibrium reconstruction assumes the map (diagnostics) -> psi is a
FUNCTION. Ham & Farrell (Nucl. Fusion 64, 034001, 2024) and Pentland et al. (arXiv:2503.05674,
MAST-U) show it can be a multivalued RELATION: distinct equilibria with identical measurements.

If so, a regressor minimising E||f(x) - y||^2 converges to the CONDITIONAL MEAN of the
branches -- and the mean of two solutions of a NONLINEAR PDE is not a solution. The damage is
therefore invisible to an L2 metric and visible only in the PDE RESIDUAL. That is the whole
argument for scoring a residual, and this script measures it.

Design: convex control vs non-convex test
-----------------------------------------
Following the structure that the superconductivity review forced on us (see
02-field-map/D-superconductivity.md): the honest experiment is not "are these the same
problem" but "does the failure mode differ between a CONVEX free-boundary problem and a
NON-CONVEX one".

  CONTROL  (convex, unique)      Bean critical state, 1D slab, constant Jc.
                                 Prigozhin 1996 Thm 2: maximal monotone => UNIQUE solution.
                                 Observable H_a -> field profile H(.) is single-valued.

  TEST     (non-convex, multiple) 1D Bratu / Gelfand:  -u'' = lam * exp(u),  u(0)=u(1)=0.
                                 Exactly TWO solutions for lam < lam* = 3.5138307...,
                                 one at lam*, none beyond -- the canonical fold bifurcation.
                                 Bartolucci et al. (arXiv:2106.04331) place the plasma
                                 free-boundary problem in exactly this Gelfand / mean-field
                                 vortex / semilinear-bifurcation family.

SCOPE, STATED HONESTLY: this is a MECHANISM SURROGATE, not Grad-Shafranov. It isolates the
mechanism (fold bifurcation -> two branches -> L2 mean is not a solution) in a setting with
closed-form branches, so the result is exact rather than solver-dependent. The GS version
needs FreeGSNKE + deflated continuation and is the named next step.

Exact Bratu solution
--------------------
  u(x) = -2 ln[ cosh(s*th/2) / cosh(th/4) ],   s = x - 1/2
  where th solves  th = sqrt(2*lam) * cosh(th/4)   (two roots below lam*)
  u''(x)      = -(th^2/2) sech^2(s*th/2)
  lam*exp(u)  =  lam*cosh^2(th/4) * sech^2(s*th/2)
  so u'' + lam*e^u = 0  iff  th^2 = 2*lam*cosh^2(th/4).  Consistent by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
N = 20001                                  # grid for numerical second derivatives
X = np.linspace(0.0, 1.0, N)
H = X[1] - X[0]


# --------------------------------------------------------------------- Bratu, exact branches
def theta_roots(lam: float):
    """Both roots of th = sqrt(2 lam) cosh(th/4); None if lam > lam*."""
    g = lambda th: th - np.sqrt(2.0 * lam) * np.cosh(th / 4.0)
    # g(0) < 0, rises, then falls; bracket around the maximum of g
    grid = np.linspace(1e-9, 60.0, 200_000)
    vals = g(grid)
    sign = np.sign(vals)
    idx = np.where(np.diff(sign) != 0)[0]
    roots = [brentq(g, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-15) for i in idx]
    return sorted(roots)


def bratu_u(x, th):
    s = x - 0.5
    return -2.0 * np.log(np.cosh(s * th / 2.0) / np.cosh(th / 4.0))


def lam_star():
    """lam* = the fold point: where the two roots merge."""
    # at the fold, d/dth [ th / cosh(th/4) ] = 0  =>  cosh(th/4) = (th/4) sinh(th/4)
    f = lambda th: np.cosh(th / 4.0) - (th / 4.0) * np.sinh(th / 4.0)
    th_c = brentq(f, 1e-6, 40.0, xtol=1e-14)
    return th_c ** 2 / (2.0 * np.cosh(th_c / 4.0) ** 2), th_c


# --------------------------------------------------------------------- Bean, exact (unique)
def bean_H(x, Ha, Jc=1.0):
    """Zero-field-cooled slab, field applied at x=0, penetrating inward. Unique by Prigozhin Thm 2."""
    return np.maximum(Ha - Jc * x, 0.0)


# --------------------------------------------------------------------- residual machinery
def d2(f):
    """Second derivative, 4th-order interior, evaluated away from the edges."""
    out = np.full_like(f, np.nan)
    out[2:-2] = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * H * H)
    return out


def rel_residual_bratu(u, lam, interior=200):
    r = d2(u) + lam * np.exp(u)
    src = lam * np.exp(u)
    sl = slice(interior, N - interior)
    return float(np.linalg.norm(r[sl]) / np.linalg.norm(src[sl]))


def rel_residual_bean(Hf, Jc=1.0, tol=1e-6):
    """|dH/dx + Jc| on the penetrated region (where H > 0); the free boundary is excluded."""
    d1 = np.gradient(Hf, H)
    live = Hf > tol
    live[:5] = live[-5:] = False
    # stay clear of the kink
    idx = np.where(live)[0]
    if len(idx) < 20:
        return float("nan")
    sl = slice(idx[0] + 5, idx[-1] - 5)
    return float(np.linalg.norm(d1[sl] + Jc) / (np.abs(Jc) * np.sqrt(max(1, sl.stop - sl.start))))


# --------------------------------------------------------------------- experiment
def main():
    ls, th_c = lam_star()
    print(f"Bratu fold point: lam* = {ls:.9f}   (theta_c = {th_c:.6f})")
    print("reference value from the literature: 3.513830719\n")

    print("=== TEST: 1D Bratu (non-convex, TWO branches per lambda) ===")
    print(f"{'lambda':>8s} {'th_lo':>8s} {'th_hi':>8s} {'||u_lo||':>9s} {'||u_hi||':>9s} "
          f"{'res(lo)':>10s} {'res(hi)':>10s} {'res(MEAN)':>11s} {'L2 sep':>8s}")
    rows = []
    for lam in [0.5, 1.0, 2.0, 3.0, 3.4, 3.5]:
        rts = theta_roots(lam)
        if len(rts) < 2:
            continue
        th_lo, th_hi = rts[0], rts[-1]
        u_lo, u_hi = bratu_u(X, th_lo), bratu_u(X, th_hi)
        u_mean = 0.5 * (u_lo + u_hi)
        r_lo = rel_residual_bratu(u_lo, lam)
        r_hi = rel_residual_bratu(u_hi, lam)
        r_mn = rel_residual_bratu(u_mean, lam)
        sep = float(np.linalg.norm(u_hi - u_lo) / np.sqrt(N))
        print(f"{lam:8.3f} {th_lo:8.4f} {th_hi:8.4f} "
              f"{np.linalg.norm(u_lo)/np.sqrt(N):9.4f} {np.linalg.norm(u_hi)/np.sqrt(N):9.4f} "
              f"{r_lo:10.2e} {r_hi:10.2e} {r_mn:11.4f} {sep:8.4f}")
        rows.append({"problem": "bratu", "lam": lam, "th_lo": th_lo, "th_hi": th_hi,
                     "res_lo": r_lo, "res_hi": r_hi, "res_mean": r_mn, "l2_sep": sep})

    print("\n=== CONTROL: 1D Bean slab (convex, ONE solution per H_a) ===")
    print(f"{'H_a':>8s} {'pen.depth':>10s} {'res(sol)':>10s} {'res(MEAN)':>11s} {'n branches':>11s}")
    for Ha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        Hf = bean_H(X, Ha)
        # the "conditional mean over the solution set" -- the set has ONE element
        H_mean = Hf.copy()
        r_sol = rel_residual_bean(Hf)
        r_mn = rel_residual_bean(H_mean)
        pen = min(Ha, 1.0)
        print(f"{Ha:8.3f} {pen:10.4f} {r_sol:10.2e} {r_mn:11.2e} {1:11d}")
        rows.append({"problem": "bean", "Ha": Ha, "res_sol": r_sol, "res_mean": r_mn,
                     "n_branches": 1})

    # ---------------- the headline comparison
    print("\n" + "=" * 78)
    print("HEADLINE: relative PDE residual of the L2-OPTIMAL (conditional-mean) predictor")
    print("=" * 78)
    br = [r for r in rows if r["problem"] == "bratu"]
    bn = [r for r in rows if r["problem"] == "bean"]
    br_branch = np.mean([0.5 * (r["res_lo"] + r["res_hi"]) for r in br])
    br_mean = np.mean([r["res_mean"] for r in br])
    bn_mean = np.mean([r["res_mean"] for r in bn])
    print(f"  NON-CONVEX (Bratu)  residual of an actual branch : {br_branch:.3e}")
    print(f"  NON-CONVEX (Bratu)  residual of the L2 mean      : {br_mean:.4f}"
          f"   <-- {br_mean/br_branch:.3g}x worse")
    print(f"  CONVEX     (Bean)   residual of the L2 mean      : {bn_mean:.3e}"
          f"   <-- unchanged: the solution set is a singleton")
    print("\n  The L2 objective is blind to this: the mean is by definition the L2-optimal")
    print("  predictor. Only a RESIDUAL term sees that it is not a solution.")

    (HERE / "results.json").write_text(json.dumps(
        {"lam_star": ls, "theta_c": th_c, "rows": rows,
         "summary": {"bratu_branch_residual": br_branch,
                     "bratu_mean_residual": br_mean,
                     "bean_mean_residual": bn_mean}}, indent=2))
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
