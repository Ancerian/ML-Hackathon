"""Field-line strike pattern on the vessel wall -- the 'footprint' observable.

VC6 asks whether the inverse problem posed on a divertor footprint is harder
than the one posed on radial excursions.  The footprint is the physically
motivated observable named in the parent project's
`02-field-map/F-poincare-and-topology.md` section 6:

> Синтетична обернена задача. Згенерувати слід на диверторі з ВІДОМОГО спектра
> збурень через трасування многовидів, додати шум, СХОВАТИ спектр -- і просити
> команди відновити амплітуди й фази (m, n) з картини смуг.

Implementation note: strike detection is done by dense fixed-step integration
followed by a post-hoc inside/outside test, NOT by `solve_ivp` events.  Events
fire for the whole integration, so they cannot terminate individual seeds in a
batched state vector -- and batching is what makes tracing hundreds of lines
affordable here.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .equilibrium import TWO_PI


class Wall:
    """Closed (R, Z) polygon, with an inside test and an arc-length coordinate."""

    def __init__(self, profile_rz):
        p = np.asarray(profile_rz, dtype=float)
        if np.allclose(p[0], p[-1]):
            p = p[:-1]
        self.poly = p
        self.path = MplPath(np.vstack([p, p[:1]]))
        seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def inside(self, R, Z):
        pts = np.column_stack([np.asarray(R).ravel(), np.asarray(Z).ravel()])
        return self.path.contains_points(pts).reshape(np.shape(R))

    def arclength_of(self, R, Z):
        """Arc-length coordinate of the nearest wall point, normalised to [0, 1)."""
        R = np.atleast_1d(R); Z = np.atleast_1d(Z)
        d = ((self.poly[None, :, 0] - R[:, None]) ** 2
             + (self.poly[None, :, 1] - Z[:, None]) ** 2)
        i = np.argmin(d, axis=1)
        return self.s[i] / self.length


def trace_to_wall(fl, wall, r0, z0, max_turns=120, pts_per_turn=24,
                  rtol=1e-8, atol=1e-10):
    """Follow field lines until they leave the wall.

    Returns a dict with, per seed: whether it struck, the toroidal angle and
    wall arc-length at the strike, and the connection length in toroidal turns.
    """
    r0 = np.atleast_1d(np.asarray(r0, float))
    z0 = np.atleast_1d(np.asarray(z0, float))
    phi = np.linspace(0.0, TWO_PI * max_turns, int(max_turns * pts_per_turn) + 1)
    R, Z, _ = fl.trace(r0, z0, phi, rtol=rtol, atol=atol)

    ins = wall.inside(R, Z)
    n = R.shape[0]
    struck = np.zeros(n, bool)
    phi_hit = np.full(n, np.nan)
    s_hit = np.full(n, np.nan)
    turns = np.full(n, float(max_turns))

    for i in range(n):
        out = np.where(~ins[i])[0]
        if out.size == 0:
            continue
        j = int(out[0])
        if j == 0:
            continue                      # seeded outside; ignore
        # linear interpolation between the last inside and first outside sample
        struck[i] = True
        phi_hit[i] = phi[j]
        s_hit[i] = float(wall.arclength_of(R[i, j], Z[i, j])[0])
        turns[i] = phi[j] / TWO_PI

    return dict(struck=struck, phi=np.mod(phi_hit, TWO_PI), s=s_hit,
                turns=turns, R=R, Z=Z)


def footprint_map(res, n_phi=24, n_s=24):
    """2-D histogram of strike points in (toroidal angle, wall arc-length).

    This is the observable a team would actually be handed: a pattern of
    stripes, with no labels attached.
    """
    ok = res["struck"] & np.isfinite(res["phi"]) & np.isfinite(res["s"])
    H, _, _ = np.histogram2d(res["phi"][ok], res["s"][ok],
                             bins=[n_phi, n_s],
                             range=[[0.0, TWO_PI], [0.0, 1.0]])
    return H


def connection_length_profile(res, psi_n0, levels, band=0.02):
    """Mean connection length per radial level -- the scalar companion to the map."""
    out = np.zeros(len(levels))
    for i, lv in enumerate(levels):
        sel = np.abs(psi_n0 - lv) <= band
        out[i] = float(np.mean(res["turns"][sel])) if sel.any() else 0.0
    return out
