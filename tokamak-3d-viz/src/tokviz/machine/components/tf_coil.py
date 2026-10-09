"""Toroidal-field coils: 32 D-shaped casings (one mesh, 32 instance transforms).

The poloidal section is the ring between ``tf_coil_inner.csv`` (bore) and
``tf_coil_outer.csv`` (casing); each bore point is paired with the point where
its outward normal meets the casing contour.  Toroidally the section is swept
over the conductor-pack width: 0.338 m on the outer part, and on the inner leg
the sides lie on the radial planes phi = +-pi/N (the wedged inner leg: the
coils bear on each other there).  That rule gives 0.26 m (printed "0.26 at
wedge") at R = 0.13/tan(pi/32) = 1.32 m, inside the 1.1165-1.49 m inner leg.
Support pads (|Z| to 2.83 m at R ~ 2.9-3.4 m, digitized) are a separate part.
"""
from __future__ import annotations

import numpy as np

from ..geom import box, ccw, loft, merge, resample_closed
from .base import Layout, Part, finish


def _normals(p: np.ndarray) -> np.ndarray:
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return np.stack([t[:, 1], -t[:, 0]], -1)          # outward for CCW


def _ray_hits(P: np.ndarray, N: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """First intersection of rays P + s N (s > 0) with a closed polyline."""
    A, B = poly, np.roll(poly, -1, 0)
    E = B - A
    out = np.empty_like(P)
    for k, (p, nrm) in enumerate(zip(P, N)):
        den = nrm[0] * E[:, 1] - nrm[1] * E[:, 0]
        w = A - p
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (w[:, 0] * E[:, 1] - w[:, 1] * E[:, 0]) / den
            u = (w[:, 0] * nrm[1] - w[:, 1] * nrm[0]) / den
        ok = (np.abs(den) > 1e-14) & (s > 0) & (u >= 0) & (u <= 1)
        s_min = s[ok].min()
        out[k] = p + s_min * nrm
    return out


def tf_geometry(tr) -> dict:
    """Everything other builders (ring, shell, tests) need about the TF coil."""
    n = int(tr.v("tf_coils.n_coils"))
    bore = ccw(np.asarray(tr.profile("tf_coil_inner"), float))
    outer = ccw(np.asarray(tr.profile("tf_coil_outer"), float))
    w_out = tr.v("tf_coils.outer_leg_toroidal_width")
    gap = float(tr.k("tf_wedge_contact_gap"))

    def half_width(R):
        R = np.asarray(R, float)
        return np.minimum(w_out / 2, R * np.tan(np.pi / n) - gap / 2)

    return {"n": n, "bore": bore, "outer": outer, "w_out": w_out,
            "half_width": half_width}


def casing_top(outer: np.ndarray, R: np.ndarray, sign: int = +1) -> np.ndarray:
    """Max (sign=+1) or min (sign=-1) Z of the casing contour at radius R
    (NaN where the vertical line misses the contour)."""
    P, Q = outer, np.roll(outer, -1, 0)
    out = np.full(len(np.atleast_1d(R)), np.nan)
    for k, r in enumerate(np.atleast_1d(R)):
        a, b = P[:, 0] - r, Q[:, 0] - r
        m = (a * b <= 0) & (a != b)
        if not m.any():
            continue
        f = a[m] / (a[m] - b[m])
        z = P[m, 1] + f * (Q[m, 1] - P[m, 1])
        out[k] = z.max() if sign > 0 else z.min()
    return out


def build_tf(tr, spec):
    L = Layout(tr)
    g = tf_geometry(tr)
    n_s = int(tr.p("n_contour", 320))
    for key in ("bore_R_inner", "bore_R_outer", "overall_width", "overall_height",
                "inner_leg_radial_thickness", "outer_leg_radial_thickness",
                "inner_leg_toroidal_width_wedge", "coil_thickness_table"):
        tr.v(f"tf_coils.{key}")
    bore = resample_closed(g["bore"], n_s)
    # start at the outer midplane
    bore = np.roll(bore, -int(np.argmax(bore[:, 0])), 0)
    outp = _ray_hits(bore, _normals(bore), g["outer"])
    hb, ho = g["half_width"](bore[:, 0]), g["half_width"](outp[:, 0])
    S = np.stack([
        np.c_[outp[:, 0], -ho, outp[:, 1]],
        np.c_[outp[:, 0], +ho, outp[:, 1]],
        np.c_[bore[:, 0], +hb, bore[:, 1]],
        np.c_[bore[:, 0], -hb, bore[:, 1]],
    ], axis=1)                                             # (n_s, 4, 3)
    casing = loft(S, closed_s=True)

    # support pads (digitized): top face at |Z| = 2.83, R range from the note
    z_pad = tr.v("tf_coils.support_pad_Z_outer")
    r0, r1 = tr.k("tf_support_pad_R_range")
    embed = float(tr.k("tf_support_pad_embed"))
    zc = casing_top(g["outer"], np.linspace(r0, r1, 25))
    z_lo = float(np.nanmin(zc)) - embed
    hw = g["w_out"] / 2
    pads = [box((r0, -hw, z_lo), (r1, hw, z_pad)),
            box((r0, -hw, -z_pad), (r1, hw, -z_lo))]
    pad_mesh = merge(pads)

    idx = list(range(L.n_tf))
    T = L.rot_tf(idx)
    labels = [f"tf_{i:02d}" for i in idx]
    parts = [
        Part("casing", casing, "copper_epoxy_steel_case", T, labels,
             meta={"closed_solid": True, "n_contour": n_s}),
        Part("support_pads", pad_mesh, "steel", T, [f"pads_{i:02d}" for i in idx],
             meta={"closed_solid": True}),
    ]
    info = {
        "phi_deg": [float(np.rad2deg(L.tf_phi(i))) for i in idx],
        "spacing_deg": float(np.rad2deg(L.pitch)),
        "bore_R": [float(bore[:, 0].min()), float(bore[:, 0].max())],
        "casing_R": [float(outp[:, 0].min()), float(outp[:, 0].max())],
        "casing_Zmax": float(outp[:, 1].max()),
        "height_over_pads": float(2 * z_pad),
        "wedge_R_for_0.26m": float(0.13 / np.tan(np.pi / g["n"])),
        "support_pad_R": [r0, r1],
    }
    return finish(tr, parts, info)
