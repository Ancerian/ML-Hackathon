"""Rail limiters: outer rail (16 Mo plates 1.00 x 0.60 x 0.01 m, section from
limiter_outer.csv) + upper/lower rails.

The digitized upper/lower rail position (R 2.3, |Z| 1.95 +-0.15 m, from a
figure with inconsistent scales) lies INSIDE the design plasma (b = 2.10 m),
so the upper/lower plates are placed conformal to the LCFS, their face on it,
centred where the ray from (R0, 0) through the digitized point meets the LCFS.
"""
from __future__ import annotations

import numpy as np

from ..geom import arclen, ccw, loft, sample_at, solid_of_revolution
from ...surfaces import offset_closed
from .base import Layout, Part, finish
from .plasma import plasma_boundary


def _outer_plate(tr, width_tor):
    face = np.asarray(tr.profile("limiter_outer", "plate_face"), float)
    back = np.asarray(tr.profile("limiter_outer", "plate_back"), float)
    if np.allclose(face[0], face[-1]):
        face = face[:-1]
    poly = ccw(np.vstack([face, back]))
    S = np.stack([np.c_[poly[:, 0], np.full(len(poly), y), poly[:, 1]]
                  for y in (-width_tor / 2, width_tor / 2)])
    return loft(S, closed_s=False, cap=True)


def _rail_plate(prof, target, R0, arc, thick, width_tor, sign):
    th = np.arctan2(prof[:, 1], prof[:, 0] - R0)
    tt = np.arctan2(target[1], target[0] - R0)
    i0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (th - tt))))))
    s = arclen(prof)
    s_c = s[i0]
    ss = s_c + np.linspace(-arc / 2, arc / 2, 41)
    face = sample_at(prof, ss)
    off = offset_closed(prof, thick)
    # offset points at the same (fractional) vertex index -> along the normals
    fi = np.interp(np.mod(ss, s[-1]), s, np.arange(len(prof) + 1))
    offc = np.vstack([off, off[:1]])
    back = np.stack([np.interp(fi, np.arange(len(offc)), offc[:, j]) for j in (0, 1)], -1)
    poly = np.vstack([face, back[::-1]])
    rc = float(prof[i0, 0])
    dphi = width_tor / rc
    m = solid_of_revolution(poly, n_tor=9, phi0=-dphi / 2, phi1=dphi / 2)
    return m, [float(prof[i0, 0]), float(prof[i0, 1])]


def build_limiters(tr, spec):
    L = Layout(tr)
    n_out = int(tr.v("limiters.outer_limiter_plates"))
    size = tr.v("limiters.plate_size")                  # [poloidal, toroidal, thick]
    tr.v("limiters.plate_material"), tr.v("limiters.n_rail_limiters")
    tr.v("limiters.outer_limiter_R"), tr.v("limiters.plate_orientation")
    k_out = list(tr.k("outer_limiter_sectors_in_octant"))
    k_rail = list(tr.k("rail_limiter_sectors_in_octant"))
    idx_out = sorted(j * L.per_oct + k for j in range(L.n_oct) for k in k_out)
    if len(idx_out) != n_out:
        raise ValueError(f"outer limiter: {len(idx_out)} positions for {n_out} plates")
    parts = [Part("outer_rail_plate", _outer_plate(tr, size[1]), "molybdenum",
                  L.rot_sectors(idx_out), [f"outer_plate_{i:02d}" for i in idx_out],
                  meta={"closed_solid": True, "sector_indices": idx_out})]

    pb = plasma_boundary(tr)
    tgt = tr.v("limiters.upper_lower_rail_limiter_position")
    idx_rail = sorted(j * L.per_oct + k for j in range(L.n_oct) for k in k_rail)
    centres = {}
    for nm, sgn in (("upper_rail_plate", +1), ("lower_rail_plate", -1)):
        m, c = _rail_plate(pb["profile"], (tgt[0], sgn * tgt[1]), pb["R0"],
                           size[0], size[2], size[1], sgn)
        centres[nm] = c
        parts.append(Part(nm, m, "molybdenum", L.rot_sectors(idx_rail),
                          [f"{nm}_{i:02d}" for i in idx_rail],
                          meta={"closed_solid": True, "sector_indices": idx_rail,
                                "contact_point_RZ": c}))
    return finish(tr, parts, {"rail_contact_points": centres,
                              "digitized_rail_position": list(tgt)})
