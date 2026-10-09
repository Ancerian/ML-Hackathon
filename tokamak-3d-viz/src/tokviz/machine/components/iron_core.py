"""Transformer iron core (magnetic circuit), from iron_core_outline.csv.

8 outer limbs, 8 upper and 8 lower radial arms, feet: rectangular blocks of
toroidal width from dimensions.yaml, one mesh each + 8 rotation instances.
Central pillar, pillar hub and the two centre pieces are axisymmetric
(checked: centre piece annulus R 0.518-1.732 x 1.405 m of steel = 95 t vs 90 t
printed).  The radial arms are clamped to the 45-degree wedge near the axis so
the 8 arms do not overlap where they meet the pillar.
"""
from __future__ import annotations

import numpy as np

from ..geom import rot_z, solid_of_revolution, wedge_box
from .base import Part, finish


def _rect(poly):
    p = np.asarray(poly, float)
    return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()


def build_iron(tr, spec):
    n_tor = int(tr.p("n_tor", 128))
    n = int(tr.v("iron_core.n_limbs"))
    phi0 = np.deg2rad(tr.k("iron_limb_phi0_deg"))
    T8 = np.stack([rot_z(phi0 + 2 * np.pi * j / n) for j in range(n)])
    lab = lambda nm: [f"{nm}_{j}" for j in range(n)]
    w_arm = tr.v("iron_core.arm_toroidal_width")
    w_limb = tr.v("iron_core.limb_toroidal_width")
    foot_extra = float(tr.k("iron_foot_extra_width"))
    for k in ("limb_R_inner", "limb_R_outer", "upper_arm_Z_top", "upper_arm_Z_bottom",
              "lower_arm_Z_top", "lower_arm_Z_bottom", "central_pillar_R",
              "centre_piece_R_outer", "centre_piece_Z_range", "pillar_hub_Z_top",
              "mass_total"):
        tr.v(f"iron_core.{k}")
    tr.v("overall.overall_diameter"), tr.v("overall.overall_height")
    parts = []
    for nm in ("upper_radial_arm", "lower_radial_arm"):
        r0, r1, z0, z1 = _rect(tr.profile("iron_core_outline", nm))
        m = wedge_box(r0, r1, z0, z1, w_arm / 2, n_wedge=n, n_r=12)
        parts.append(Part(nm, m, "iron_laminated", T8, lab(nm),
                          meta={"closed_solid": True, "R": [r0, r1], "Z": [z0, z1]}))
    for nm, w in (("outer_limb", w_limb), ("foot_plate", w_limb + foot_extra)):
        r0, r1, z0, z1 = _rect(tr.profile("iron_core_outline", nm))
        m = wedge_box(r0, r1, z0, z1, w / 2, n_wedge=None, n_r=1)
        parts.append(Part(nm, m, "iron_laminated" if nm == "outer_limb" else "steel",
                          T8, lab(nm),
                          meta={"closed_solid": True, "R": [r0, r1], "Z": [z0, z1]}))
    for nm in ("central_pillar", "pillar_top_hub", "centre_piece_upper", "centre_piece_lower"):
        poly = np.asarray(tr.profile("iron_core_outline", nm), float)
        m = solid_of_revolution(poly, n_tor=n_tor)
        parts.append(Part(nm, m, "iron_solid", np.eye(4)[None], [nm],
                          meta={"closed_solid": True, "axisymmetric": True}))
    info = {"limb_phi_deg": [float(np.rad2deg(phi0 + 2 * np.pi * j / n)) for j in range(n)]}
    return finish(tr, parts, info)
