"""Plasma last closed flux surface (decision G2).

Parametric D (Miller form, the one already used by ``tokviz.surfaces.d_shape``)

    R = R0 + a cos(t + asin(delta) sin t),   Z = kappa a sin t,   kappa = b / a

with R0, a, b from Table I.3-1.  Only the triangularity is taken from the
SHAPE of Table IV.2-4 (profiles/plasma_boundary.csv): that table covers the
upper-outboard quadrant only, so the fit fixes the table plasma's inboard
edge at the limiter radius Ri(plasma) = 1.71 m (same machine, same inner
limiter) and fits (a_t, kappa_t, delta) to the 30 points; delta is then
applied to the design D.  The raw table is NOT used as the surface (its top
exceeds b and crosses the vessel wall, see README contradiction 5).  The
result is checked -- and if needed reduced -- so the LCFS stays inside the
vessel inner wall with at least the kernel minimum clearance.
"""
from __future__ import annotations

import numpy as np

from matplotlib.path import Path as _MplPath

from ..geom import Mesh, orient, point_seg_dist
from ...surfaces import revolve
from .base import Part, finish


def miller(R0, a, kappa, delta, t):
    x = np.arcsin(np.clip(delta, -0.999, 0.999))
    return np.stack([R0 + a * np.cos(t + x * np.sin(t)), kappa * a * np.sin(t)], -1)


def _nelder_mead(f, x0, step, iters=600, tol=1e-12):
    n = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_table_delta(table: np.ndarray, r_in: float) -> dict:
    t = np.linspace(-0.1, np.pi / 2 + 0.7, 3000)

    def rms(q):
        a, k, d = q
        if a <= 0.3 or k <= 0.5 or not (-0.95 < d < 0.95):
            return 1e3
        c = miller(r_in + a, a, k, d, t)
        return float(np.sqrt(np.mean(point_seg_dist(table, c, closed=False) ** 2)))

    best, val = _nelder_mead(rms, [1.25, 1.7, 0.4], [0.1, 0.1, 0.1])
    best, val = _nelder_mead(rms, best, [0.02, 0.02, 0.02])
    return {"a_t": float(best[0]), "kappa_t": float(best[1]), "delta": float(best[2]),
            "R0_t": float(r_in + best[0]), "rms_m": val}


def clearance(prof: np.ndarray, wall: np.ndarray) -> tuple:
    inside = bool(_MplPath(wall).contains_points(prof).all())
    d = point_seg_dist(prof, wall)
    return inside, float(d.min()), prof[int(np.argmin(d))]


def plasma_boundary(tr, n_pol: int = 256) -> dict:
    R0, a, b = tr.v("plasma.R0"), tr.v("plasma.a"), tr.v("plasma.b")
    tr.v("plasma.elongation_b_over_a"), tr.v("plasma.R_outer_plasma")
    r_in = tr.v("plasma.R_inner_plasma")
    tr.v("plasma.boundary_table_Zmax")
    table = np.asarray(tr.profile("plasma_boundary"), float)[:, :2]
    wall = np.asarray(tr.profile("vessel_inner"), float)
    c_min = float(tr.k("plasma_min_wall_clearance"))
    kappa = b / a
    fit = fit_table_delta(table, r_in)
    delta = float(np.round(fit["delta"], 4))
    t = np.linspace(0, 2 * np.pi, n_pol, endpoint=False)
    prof = miller(R0, a, kappa, delta, t)
    ok, dmin, where = clearance(prof, wall)
    clamped = False
    if not ok or dmin < c_min:           # reduce |delta| step-wise until it fits
        clamped = True
        for d in np.linspace(delta, 0.0, 200):
            prof = miller(R0, a, kappa, d, t)
            ok, dmin, where = clearance(prof, wall)
            if ok and dmin >= c_min:
                delta = float(d)
                break
    gaps = {"outboard_midplane": float(tr.v("vessel.rs_R_inner_wall_outboard") - (R0 + a)),
            "inboard_midplane": float((R0 - a) - tr.v("vessel.rs_R_inner_wall_inboard"))}
    return {"profile": prof, "R0": R0, "a": a, "b": b, "kappa": kappa, "delta": delta,
            "fit": fit, "clamped": clamped, "min_wall_clearance": dmin,
            "min_clearance_at": [float(where[0]), float(where[1])], "gaps": gaps}


def build_plasma(tr, spec):
    n_pol = int(tr.p("n_pol", 256))
    n_tor = int(tr.p("n_tor", 256))
    pb = plasma_boundary(tr, n_pol)
    v, q = revolve(pb["profile"], n_tor=n_tor)
    m = orient(Mesh(v, q))
    info = {k: pb[k] for k in ("R0", "a", "b", "kappa", "delta", "fit", "clamped",
                               "min_wall_clearance", "min_clearance_at", "gaps")}
    info["form"] = "Miller D: R = R0 + a cos(t + asin(delta) sin t), Z = kappa a sin t"
    info["profile_rz"] = pb["profile"].tolist()
    return finish(tr, [Part("lcfs", m, "plasma", np.eye(4)[None], ["lcfs"],
                            meta={"closed_solid": True, "axisymmetric": True})], info)
