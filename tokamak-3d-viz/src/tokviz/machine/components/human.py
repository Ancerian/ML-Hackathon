"""Scale figure: a simple 1.80 m person standing on the floor (underside of
the iron feet) outside the machine."""
from __future__ import annotations

import numpy as np

from ..geom import box, cylinder, ellipsoid, merge, rot_z, translate
from .base import Part, finish


def build_human(tr, spec):
    H = float(tr.k("human_height"))
    s = H / 1.80
    legs = [box((-0.07 * s, y - 0.07 * s, 0), (0.07 * s, y + 0.07 * s, 0.84 * s))
            for y in (-0.10 * s, 0.10 * s)]
    torso = box((-0.11 * s, -0.20 * s, 0.84 * s), (0.11 * s, 0.20 * s, 1.46 * s))
    arms = [box((-0.045 * s, y - 0.045 * s, 0.80 * s), (0.045 * s, y + 0.045 * s, 1.44 * s))
            for y in (-0.255 * s, 0.255 * s)]
    neck = cylinder(0.05 * s, 1.46 * s, 1.55 * s, n=12)
    head = ellipsoid(0.095 * s, 0.085 * s, 0.125 * s).transformed(translate(z=1.675 * s))
    m = merge(legs + [torso] + arms + [neck, head])
    r = float(tr.k("human_R"))
    ph = np.deg2rad(tr.k("human_phi_deg"))
    z0 = float(np.min(tr.profile("iron_core_outline", "foot_plate")[:, 1]))
    T = translate(r * np.cos(ph), r * np.sin(ph), z0) @ rot_z(ph + np.pi)
    return finish(tr, [Part("figure_1p80m", m, "human_scale", T[None], ["human"],
                            meta={"height_m": H})], {"floor_Z": z0, "R": r})
