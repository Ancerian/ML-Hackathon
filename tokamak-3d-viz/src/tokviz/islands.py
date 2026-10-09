"""Locating magnetic islands and seeding field lines inside them.

For a single mode ``cos(m theta* - n phi + alpha)`` the island chain on the
phi = 0 plane has its elliptic (O) and hyperbolic (X) points at the 2m angles
where the perturbation is extremal.  Which family is which depends on the signs
of the amplitude and of ``dq/dpsi``, so rather than reasoning about signs we
test numerically: a seed ON the resonant surface at an O-point sits next to an
elliptic fixed point and barely moves in psi_n, while an X-point seed runs
along the separatrix and sweeps roughly the full island width.
"""
from __future__ import annotations

import numpy as np

from .equilibrium import Equilibrium, TWO_PI


def point_at(eq: Equilibrium, tsf, psi_n: float, theta_star: float):
    """(R, Z) on a flux surface at a requested straight-field-line angle."""
    pts = eq.surface(float(psi_n))
    if pts is None:
        return None
    th = np.mod(tsf.value(pts[:, 0], pts[:, 1]), TWO_PI)
    target = np.mod(float(theta_star), TWO_PI)
    d = np.mod(th - target + np.pi, TWO_PI) - np.pi      # signed, in (-pi, pi]
    i = int(np.argmin(np.abs(d)))
    j = (i + 1) % len(th)
    if d[i] * d[j] < 0 and abs(d[i] - d[j]) > 1e-12:
        t = abs(d[i]) / abs(d[i] - d[j])
        return tuple(pts[i] + t * (pts[j] - pts[i]))
    return tuple(pts[i])


def classify_fixed_points(eq: Equilibrium, tsf, fl, mode, n_turns: int = 40):
    """Return ``(o_points, x_points)`` as lists of ``(R, Z)`` on phi = 0.

    Candidate angles are the 2m extrema of the perturbation; each is traced
    briefly and sorted by how far it wanders in psi_n.
    """
    m = int(mode.m)
    s_res = float(mode.psin_res)
    cands = []
    for j in range(2 * m):
        th = (np.pi * j - mode.phase) / m
        p = point_at(eq, tsf, s_res, th)
        if p is not None:
            cands.append(p)
    if not cands:
        return [], []

    r0 = np.array([p[0] for p in cands])
    z0 = np.array([p[1] for p in cands])
    phi = TWO_PI * np.arange(n_turns, dtype=float)
    R, Z, alive = fl.trace(r0, z0, phi, rtol=1e-9, atol=1e-11)
    sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
    exc = sn.max(axis=1) - sn.min(axis=1)
    exc = np.where(alive, exc, np.inf)

    order = np.argsort(exc)
    o_idx = sorted(order[:m])
    x_idx = sorted(order[m:2 * m])
    return [cands[i] for i in o_idx], [cands[i] for i in x_idx]


def island_seeds(eq: Equilibrium, tsf, fl, mode, n_shell: int = 4,
                 frac: float = 0.42):
    """Seeds filling one island: a short radial fan out of each O-point.

    ``frac`` is the fraction of the predicted island half-width to reach, kept
    below 0.5 so the seeds stay inside the separatrix.
    """
    o_pts, _ = classify_fixed_points(eq, tsf, fl, mode)
    if not o_pts:
        return np.zeros(0), np.zeros(0)
    W = float(getattr(mode, "width_psin", 0.0))
    if W <= 0:
        return np.array([p[0] for p in o_pts]), np.array([p[1] for p in o_pts])

    rs, zs = [], []
    for (r, z) in o_pts:
        # step in psi_n along the local grad-psi direction
        gr = eq.dpsi_dR(r, z)
        gz = eq.dpsi_dZ(r, z)
        g2 = gr * gr + gz * gz
        if g2 < 1e-18:
            rs.append(r); zs.append(z); continue
        dpsi_tot = eq.psi_bdy - eq.psi_axis
        for k in range(n_shell):
            f = frac * (k + 1) / n_shell
            d_psi = f * 0.5 * W * dpsi_tot
            rs.append(r + d_psi * gr / g2)
            zs.append(z + d_psi * gz / g2)
    return np.asarray(rs), np.asarray(zs)
