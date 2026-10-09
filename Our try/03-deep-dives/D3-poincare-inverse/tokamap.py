#!/usr/bin/env python3
"""Tokamap: an exactly symplectic Poincare map for tokamak field lines.

Balescu, Vlad & Spineanu, Phys. Rev. E 58, 951 (1998). Explicit form:

    P_n     = Psi_n - 1 - (x_L/2pi) sin(2pi T_n)
    Psi_n+1 = 1/2 ( P_n + sqrt(P_n^2 + 4 Psi_n) )
    T_n+1   = T_n + 1/q(Psi_n+1) - (x_L/4pi^2) cos(2pi T_n)/(1+Psi_n+1)^2   (mod 1)

Psi >= 0 is preserved automatically (the magnetic axis Psi = 0 is respected), which is the
advantage over a naive standard map. x_L is the perturbation amplitude: x_L = 0 gives the
integrable case (rigid rotation on each circle).

q-profile: we use a plain monotonic q(Psi) = q0 + (q1-q0)*Psi^2 with q0=1, q1=4, stated
explicitly because it is OUR choice, not Balescu's -- the qualitative phenomena (islands,
overlap, chaos) do not depend on it, but any quantitative number does.

Purpose here is to test conjecture C5: is a Poincare-based INVERSE problem stable enough to be
a 48-hour hackathon task? Forward first, inverse second.
"""
from __future__ import annotations
import numpy as np

Q0, Q1 = 1.0, 4.0


def q_of(psi):
    return Q0 + (Q1 - Q0) * psi ** 2


def step(psi, T, xL):
    P = psi - 1.0 - (xL / (2 * np.pi)) * np.sin(2 * np.pi * T)
    psi_n = 0.5 * (P + np.sqrt(P * P + 4.0 * psi))
    T_n = T + 1.0 / q_of(psi_n) - (xL / (4 * np.pi ** 2)) * np.cos(2 * np.pi * T) / (1.0 + psi_n) ** 2
    return psi_n, np.mod(T_n, 1.0)


def orbit(psi0, T0, xL, n):
    psi = np.empty(n); T = np.empty(n)
    p, t = float(psi0), float(T0)
    for i in range(n):
        psi[i], T[i] = p, t
        p, t = step(p, t, xL)
        if not np.isfinite(p) or p < 0 or p > 20:
            return psi[:i + 1], T[:i + 1]
    return psi, T


def jacobian_det(psi, T, xL, h=1e-7):
    """Numerical det of the one-step Jacobian. Must be 1 for a symplectic (area-preserving) map."""
    p0, t0 = step(psi, T, xL)
    p1, t1 = step(psi + h, T, xL)
    p2, t2 = step(psi, T + h, xL)
    return ((p1 - p0) / h) * ((t2 - t0) / h) - ((p2 - p0) / h) * ((t1 - t0) / h)


def spread(psi0, T0, xL, n=2000, burn=200):
    """Psi-excursion of one orbit: tiny on an invariant circle, large in a chaotic sea."""
    p, t = orbit(psi0, T0, xL, n)
    if len(p) < burn + 50:
        return np.nan
    return float(np.ptp(p[burn:]))


if __name__ == "__main__":
    print("=== symplecticity check: det J must be 1 ===")
    rng = np.random.default_rng(0)
    for xL in [0.0, 0.2, 0.5, 1.0]:
        d = [jacobian_det(rng.uniform(0.05, 2.0), rng.uniform(0, 1), xL) for _ in range(400)]
        d = np.array(d); d = d[np.isfinite(d)]
        print(f"  x_L={xL:4.2f}   det J: mean={d.mean():.9f}  max|det-1|={np.abs(d-1).max():.2e}")

    print("\n=== Psi >= 0 preserved? ===")
    for xL in [0.5, 1.0, 2.0]:
        ok = True
        for _ in range(200):
            p, _ = orbit(rng.uniform(0.02, 3.0), rng.uniform(0, 1), xL, 3000)
            if (p < 0).any():
                ok = False; break
        print(f"  x_L={xL:4.2f}   Psi>=0 held: {ok}")

    print("\n=== transition to chaos: median Psi-spread over a grid of initial conditions ===")
    print(f"{'x_L':>6s} {'median spread':>14s} {'frac spread>0.3':>16s}")
    psis = np.linspace(0.1, 2.0, 24)
    for xL in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
        s = np.array([spread(p0, 0.25, xL) for p0 in psis])
        s = s[np.isfinite(s)]
        print(f"{xL:6.2f} {np.median(s):14.5f} {np.mean(s > 0.3):16.3f}")
