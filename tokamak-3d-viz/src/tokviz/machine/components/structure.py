"""Mechanical structure: upper/lower rings, simplified shell, inner cylinder.

* Rings (ring_structure.csv, d7320 x 900): the inner collar (R < R_split) is
  axisymmetric, lifted clear of the TF casing contour and kept under the iron
  centre pieces; the outer part is 32 wedge blocks between the TF coils
  (parallel-sided slots), with the openings for the vertical ports removed in
  the blocks above the vertical-port sectors.
* Shell (simplified): plates filling the gaps between neighbouring TF coils
  along the casing outer contour, outboard of the ring, leaving the
  horizontal-port band free.
* Inner cylinder: the 4 cm stainless cylinder between PF coil 1 and the TF
  inner legs, fitted into the 31.5 mm radial gap left by the database.
"""
from __future__ import annotations

import numpy as np

from ..geom import (ccw, clip_halfplane, merge, resample_closed, slotted_sector,
                    solid_of_revolution)
from ...surfaces import offset_closed
from .base import Layout, Part, finish
from .tf_coil import casing_top, tf_geometry
from .vessel import port_geometry


def _clean(poly, tol=1e-6):
    p = np.asarray(poly, float)
    keep = [0]
    for i in range(1, len(p)):
        if np.hypot(*(p[i] - p[keep[-1]])) > tol:
            keep.append(i)
    p = p[keep]
    if np.hypot(*(p[0] - p[-1])) <= tol:
        p = p[:-1]
    return p


def _collar(poly, sign, outer_tf, cp_R, cp_Z, clear, n=200):
    """Axisymmetric inner part of a ring: push points out of the TF casing
    footprint (to the casing top + clear) and under the centre piece."""
    p = resample_closed(ccw(poly), n)
    zc = casing_top(outer_tf, p[:, 0], sign)
    for i, (r, z) in enumerate(p):
        if np.isfinite(zc[i]) and sign * z < sign * zc[i] + clear and r > outer_tf[:, 0].min():
            p[i, 1] = zc[i] + sign * clear
        if r <= cp_R + clear and sign * p[i, 1] > sign * cp_Z - clear:
            p[i, 1] = cp_Z - sign * clear
    return _clean(p)


def _ring_kind(L, i):
    k = L.sector_kind(i)
    return k if k in ("vport_large", "vport_small") else "plain"


def build_structure(tr, spec):
    L = Layout(tr)
    g = tf_geometry(tr)
    pg = port_geometry(tr)
    n_tor = int(tr.p("n_tor", 192))
    n_blk = int(tr.p("n_tor_block", 8))
    clear = float(tr.k("structure_clearance"))
    r_split = float(tr.k("ring_collar_R_split"))
    open_m = float(tr.k("ring_port_opening_margin"))
    tr.v("structure.ring_outer_diameter"), tr.v("structure.ring_height")
    tr.v("structure.ring_sectors"), tr.v("structure.ring_Z_top")
    cp = tr.profile("iron_core_outline", "centre_piece_upper")
    cp_R, cp_Z = float(np.max(cp[:, 0])), float(np.min(cp[:, 1]))
    tr.v("iron_core.centre_piece_R_outer"), tr.v("iron_core.centre_piece_Z_range")
    half_slot = g["w_out"] / 2 + clear

    parts = []
    for ring, sign in (("upper_ring", +1), ("lower_ring", -1)):
        poly = _clean(tr.profile("ring_structure", ring))
        inner = clip_halfplane(poly, 0, r_split, keep_less=True)
        outer = clip_halfplane(poly, 0, r_split, keep_less=False)
        collar = _collar(inner, sign, g["outer"], cp_R, sign * cp_Z, clear)
        parts.append(Part(f"{ring}_collar", solid_of_revolution(collar, n_tor=n_tor),
                          "magnetic_steel", np.eye(4)[None], [f"{ring}_collar"],
                          meta={"closed_solid": True, "axisymmetric": True}))
        variants = {"plain": [outer]}
        for kind, (ra, rb) in (("vport_large", pg["vl_R"]), ("vport_small", pg["vs_R"])):
            a = clip_halfplane(outer, 0, ra - open_m, keep_less=True)
            b = clip_halfplane(outer, 0, rb + open_m, keep_less=False)
            variants[kind] = [q for q in (a, b) if len(q) >= 3]
        for kind, polys in variants.items():
            idx = [i for i in range(L.n_sec) if _ring_kind(L, i) == kind]
            m = merge([slotted_sector(q, L.pitch, half_slot, n_tor=n_blk) for q in polys])
            parts.append(Part(f"{ring}_block_{kind}", m, "al_alloy_cast",
                              L.rot_sectors(idx), [f"{ring}_block_{i:02d}" for i in idx],
                              meta={"closed_solid": True, "sector_indices": idx}))

    # ---- shell plates between coils (outboard, clear of the ring/port band) --
    t_sh = float(tr.k("shell_plate_thickness"))
    zb = float(tr.k("shell_port_band_half_height"))
    r_min = tr.v("structure.ring_outer_diameter") / 2
    tr.v("structure.shell_description")
    con = resample_closed(g["outer"], 720)
    off = offset_closed(con, t_sh)
    plates = []
    for sgn in (+1, -1):
        sel = (con[:, 0] >= r_min) & (sgn * con[:, 1] >= zb)
        idx = np.flatnonzero(sel)
        if len(idx) < 3:
            continue
        # contiguous run (the selection does not wrap through index 0 here)
        poly = np.vstack([con[idx], off[idx][::-1]])
        plates.append(slotted_sector(poly, L.pitch, half_slot, n_tor=n_blk))
    parts.append(Part("shell_plates", merge(plates), "al_alloy_plate",
                      L.rot_sectors(range(L.n_sec)),
                      [f"shell_{i:02d}" for i in range(L.n_sec)],
                      meta={"closed_solid": True}))

    # ---- inner cylinder ----------------------------------------------------------
    tr.v("structure.inner_cylinder_thickness"), tr.v("structure.inner_cylinder_R")
    tr.v("structure.inner_cylinder_sectors")
    r_pf = tr.v("pf_coils.coil1_outer_diameter") / 2
    r_tf = tr.v("tf_coils.bore_R_inner") - tr.v("tf_coils.inner_leg_radial_thickness")
    c_cyl = float(tr.k("inner_cylinder_radial_clearance"))
    zh = tr.v("tf_coils.straight_inner_leg") / 2
    cyl = solid_of_revolution(np.array([[r_pf + c_cyl, -zh], [r_tf - c_cyl, -zh],
                                        [r_tf - c_cyl, zh], [r_pf + c_cyl, zh]]), n_tor=n_tor)
    parts.append(Part("inner_cylinder", cyl, "stainless_steel", np.eye(4)[None],
                      ["inner_cylinder"], meta={"closed_solid": True, "axisymmetric": True,
                                                "R": [r_pf + c_cyl, r_tf - c_cyl]}))
    info = {"ring_outer_R": float(max(np.max(tr.profile("ring_structure", r)[:, 0])
                                      for r in ("upper_ring", "lower_ring"))),
            "inner_cylinder_thickness_modelled": float(r_tf - r_pf - 2 * c_cyl)}
    return finish(tr, parts, info)
