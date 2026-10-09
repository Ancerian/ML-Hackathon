"""Port stubs: 8 horizontal (0.46 x 0.96 m) + vertical 14 x 83 cm and 7 x 28 cm
ducts, top and bottom.  Modelled as rectangular duct stubs with a flange; the
matching wall openings are cut in the vessel sectors (vessel ``openings``
param).  Duct starts follow the vessel exterior skin, so the stub sits on the
wall around its opening.
"""
from __future__ import annotations

import numpy as np

from ..geom import Mesh, loft, rect_ring
from .base import Layout, Part, finish
from .vessel import port_geometry


def _outboard_R_of_Z(outer):
    p = outer[outer[:, 0] > 3.5]
    o = np.argsort(p[:, 1])
    return lambda z: np.interp(z, p[o, 1], p[o, 0])


def _top_Z_of_R(outer, sign):
    p = outer[(sign * outer[:, 1] > 1.0) & (outer[:, 0] > 2.3)]
    o = np.argsort(p[:, 0])
    return lambda r: np.interp(r, p[o, 0], p[o, 1])


def _frame_loft(rings):
    return loft(np.stack(rings), closed_s=True)


def horizontal_stub(outer, hw, hh, wall, x_end, fl_margin, fl_thick, n_w=4, n_h=8):
    R_of_Z = _outboard_R_of_Z(outer)
    ri = rect_ring(2 * hw, 2 * hh, n_w, n_h)                 # (u=y, v=z)
    ro = rect_ring(2 * (hw + wall), 2 * (hh + wall), n_w, n_h)
    rf = rect_ring(2 * (hw + wall + fl_margin), 2 * (hh + wall + fl_margin), n_w, n_h)

    def ring(uv, x):
        x = np.broadcast_to(x, (len(uv),))
        return np.c_[x, uv[:, 0], uv[:, 1]]

    def on_skin(uv):
        R = R_of_Z(uv[:, 1])
        return np.sqrt(np.maximum(R ** 2 - uv[:, 0] ** 2, 0))

    duct = _frame_loft([ring(ro, on_skin(ro)), ring(ro, x_end), ring(ri, x_end),
                        ring(ri, on_skin(ri))])
    flange = _frame_loft([ring(rf, x_end - fl_thick), ring(rf, x_end), ring(ro, x_end),
                          ring(ro, x_end - fl_thick)])
    return duct, flange


def vertical_stub(outer, rc, length, hw, wall, z_top, fl_margin, fl_thick, sign,
                  n_l=12, n_w=1):
    Z_of_R = _top_Z_of_R(outer, sign)
    ri = rect_ring(length, 2 * hw, n_l, n_w)                 # (u=x-rc, v=y)
    ro = rect_ring(length + 2 * wall, 2 * (hw + wall), n_l, n_w)
    rf = rect_ring(length + 2 * (wall + fl_margin), 2 * (hw + wall + fl_margin), n_l, n_w)

    def ring(uv, z):
        x = rc + uv[:, 0]
        z = np.broadcast_to(z, (len(uv),))
        return np.c_[x, uv[:, 1], z]

    def on_skin(uv):
        return Z_of_R(np.hypot(rc + uv[:, 0], uv[:, 1]))

    zt, zf = sign * z_top, sign * (z_top - fl_thick)
    duct = _frame_loft([ring(ro, on_skin(ro)), ring(ro, zt), ring(ri, zt), ring(ri, on_skin(ri))])
    flange = _frame_loft([ring(rf, zf), ring(rf, zt), ring(ro, zt), ring(ro, zf)])
    return duct, flange


def build_ports(tr, spec):
    L = Layout(tr)
    pg = port_geometry(tr)
    outer = np.asarray(tr.profile("vessel_outer"), float)
    tr.v("ports.horizontal_port_count"), tr.v("ports.vertical_ports_per_octant")
    tr.v("ports.tangential_access")
    parts = []
    # --- horizontal ------------------------------------------------------------
    d, f = horizontal_stub(outer, pg["h_half_w"], pg["h_half_h"],
                           float(tr.k("port_hduct_wall")), float(tr.k("port_hduct_R_end")),
                           float(tr.k("port_flange_margin")), float(tr.k("port_flange_thickness")))
    idx_h = [i for i in range(L.n_sec) if L.sector_kind(i) == "hport"]
    if len(idx_h) != int(tr.v("ports.horizontal_port_count")):
        raise ValueError("horizontal port count does not match the sector layout")
    T = L.rot_sectors(idx_h)
    lab = [f"hport_oct{L.octant_of(i)}" for i in idx_h]
    parts += [Part("hport_duct", d, "inconel600", T, lab, meta={"closed_solid": True}),
              Part("hport_flange", f, "stainless_steel", T, lab, meta={"closed_solid": True})]
    # --- vertical ------------------------------------------------------------------
    wall_v = float(tr.k("port_vduct_wall"))
    flm, flt = float(tr.k("port_vflange_margin")), float(tr.k("port_vflange_thickness"))
    for tag, kind, zkey in (("vl", "vport_large", "port_vlarge_flange_Z"),
                            ("vs", "vport_small", "port_vsmall_flange_Z")):
        r0, r1 = pg[f"{tag}_R"]
        idx = [i for i in range(L.n_sec) if L.sector_kind(i) == kind]
        T = L.rot_sectors(idx)
        for pos, sgn in (("top", +1), ("bottom", -1)):
            d, f = vertical_stub(outer, 0.5 * (r0 + r1), r1 - r0, pg[f"{tag}_half_w"], wall_v,
                                 float(tr.k(zkey)), flm, flt, sgn)
            lab = [f"{kind}_{pos}_oct{L.octant_of(i)}" for i in idx]
            parts += [Part(f"{kind}_{pos}_duct", d, "inconel600", T, lab,
                           meta={"closed_solid": True}),
                      Part(f"{kind}_{pos}_flange", f, "stainless_steel", T, lab,
                           meta={"closed_solid": True})]
    topo = tr.component("port", lambda c: "types" in c.params)   # jet1983 port topology
    info = {"horizontal_sectors": idx_h,
            "port_topology_entry": None if topo is None else topo.name,
            "vertical_large_R": list(pg["vl_R"]), "vertical_small_R": list(pg["vs_R"]),
            "mode": "stubs + wall openings (vessel.params.openings)"}
    return finish(tr, parts, info)
