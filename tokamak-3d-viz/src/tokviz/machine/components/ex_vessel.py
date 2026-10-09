"""Vessel-attached parts added by delta variants (JET as built, 1983).

Types (``components_add`` of machines/jet1983/deltas.yaml):

* ``vessel_restraint_ring`` -- toroidal restraint rings at Z = params.Z: thick
  outer-skin segments on the rigid sectors (between the parallel bellows
  flanges) + electrically insulated bridges across the bellows.
* ``pump_chamber``  -- pumping chamber on the main horizontal port of
  ``params.octant``: vertical cylinder parsed from ``params.envelope``
  (assumed), spool to the port flange, 1.2 m door opposite the port, two
  turbomolecular pumps at its base.
* ``nbi_adaptor``   -- Middle Port Adaptor, box parsed from ``params.envelope``.
* ``rotary_valve``  -- rotary gate valve on the outside of the adaptor of the same
  octant; carries ``params.status`` (not on the machine in 1983).
* ``gas_inlet``     -- gas introduction module (port and size assumed).
* ``port`` with ``per_octant`` -- limiter port stubs (one per octant); a
  ``port`` entry without it is a topology record (realised by ``ports``).

Octants are 1-based here (as in the deltas): octant k spans
phi = 45(k-1) .. 45k deg with the kernel's tf_phi0_deg of the variant, and
its main horizontal port is kernel sector (k-1)*4 + hport_sector_in_octant.
Every size not in the database is a kernel assumption (``tr.k``) so it is
flagged ``assumed`` in the manifest.
"""
from __future__ import annotations

import re

import numpy as np

from ..geom import box, cylinder, merge, rot_z, slotted_sector, solid_of_revolution
from .base import Layout, Part, RecordOnly, finish
from .ports import _frame_loft, _outboard_R_of_Z
from .vessel import port_geometry

# local cylinder (axis along Z) -> axis along +X
_Z_TO_X = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)


def _numbers(text: str) -> list:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", str(text))]


def _xcyl(r, x0, x1, n=40):
    """Solid cylinder along +X from x0 to x1."""
    return cylinder(r, x0, x1, n=n).transformed(_Z_TO_X)


def _port_frame(tr, octant):
    L = Layout(tr)
    i = L.hport_sector(octant)
    phi = float(L.sector_phi(i))
    return L, i, phi


def _status_meta(text):
    if not text:
        return {}
    t = str(text)
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|"
                  r"November|December)\s+(\d{4})", t)
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    out = {"status_note": t,
           "status": "not_installed" if re.search(r"not on the machine|planned", t, re.I)
           else "installed"}
    if m:
        out["installed"] = f"{m.group(2)}-{months.index(m.group(1)) + 1:02d}"
    return out


# ---------------------------------------------------------------------------
# restraint rings
# ---------------------------------------------------------------------------
def build_restraint_ring(tr, spec):
    tr.own()
    L = Layout(tr)
    from .vessel import octant_layout
    lay = octant_layout(tr, L)
    if lay is None:
        raise KeyError("vessel_sector layout (parallel bellows flanges) needed for the rings")
    d = lay["d"]
    zs = [float(z) for z in tr.p("Z")]
    if str(tr.p("side", "outboard")) != "outboard":
        raise ValueError("only outboard restraint rings are modelled")
    t = float(tr.p("outer_skin_thickness_max"))
    hb = float(tr.p("band_height_poloidal"))
    r_ref = float(tr.p("R_outer_skin_at_Z"))
    gap = float(tr.k("restraint_bridge_insulation_gap"))
    n_tor = int(tr.p("n_tor", 12))
    outer = np.asarray(tr.profile("vessel_outer"), float)
    R_of_Z = _outboard_R_of_Z(outer)
    parts, info = [], {"R_outer_skin_at_Z_model": {}}
    for zc in zs:
        zz = np.linspace(zc - hb / 2, zc + hb / 2, 11)
        rs = R_of_Z(zz)
        poly = np.vstack([np.c_[rs, zz], np.c_[rs[::-1] + t, zz[::-1]]])
        info["R_outer_skin_at_Z_model"][f"{zc:+.2f}"] = float(R_of_Z(zc))
        if abs(float(R_of_Z(zc)) - r_ref) > 2e-3:
            raise ValueError(f"{spec.name}: outer skin R at Z={zc} is {float(R_of_Z(zc)):.4f}, "
                             f"entry says {r_ref}")
        skin = slotted_sector(poly, L.pitch, d, n_tor=n_tor)
        a = np.arcsin((d - gap) / poly[:, 0].max())
        bridge = solid_of_revolution(poly, n_tor=5, phi0=-a, phi1=a)
        tag = "upper" if zc > 0 else "lower"
        idx = list(range(L.n_sec))
        parts.append(Part(f"{tag}_ring_skin", skin, "nicrofer7216LC", L.rot_sectors(idx),
                          [f"{tag}_skin_{i:02d}" + ("_end_pair" if i % L.per_oct == 0 else "")
                           for i in idx],
                          meta={"closed_solid": True, "Z": zc,
                                "note": "one segment per rigid pitch; at the octant welds it "
                                        "spans both end pieces (a/b)"}))
        parts.append(Part(f"{tag}_ring_bridge", bridge, "inconel_insulated_bridge",
                          L.rot_tf(idx), [f"{tag}_bridge_{i:02d}" for i in idx],
                          meta={"closed_solid": True, "Z": zc, "insulation_gap_m": gap}))
    info.update({"band_height_m": hb, "radial_thickness_m": t, "bellows_half_length_m": d,
                 "n_rings": len(zs)})
    return finish(tr, parts, info)


# ---------------------------------------------------------------------------
# pumping chambers / NBI adaptors / rotary valves / gas inlets
# ---------------------------------------------------------------------------
def build_pump_chamber(tr, spec):
    tr.own()
    k = int(tr.p("octant"))
    L, i, phi = _port_frame(tr, k)
    env = str(tr.p("envelope"))
    D, H = _numbers(env)[:2]
    door_d = float(tr.p("door_inner_diameter"))
    n_tmp = int(tr.p("turbomolecular_pumps", 2))
    tr.p("material"), tr.p("port"), tr.p("seal_to_torus"), tr.p("bake_C")
    pg = port_geometry(tr)
    x_fl = float(tr.k("port_hduct_R_end"))
    spool = float(tr.k("pump_chamber_spool_length"))
    frame = float(tr.k("pump_chamber_door_frame"))
    boss = float(tr.k("pump_chamber_door_boss"))
    tmp_d, tmp_h = (float(v) for v in tr.k("pump_chamber_turbo_pump_DH"))
    r = D / 2
    xc = x_fl + spool + r
    wall = float(tr.k("port_hduct_wall")) + float(tr.k("port_flange_margin"))
    hw, hh = pg["h_half_w"] + wall, pg["h_half_h"] + wall
    if hw >= r or 2 * hh >= H:
        raise ValueError(f"{spec.name}: chamber envelope smaller than the port flange")
    chamber = cylinder(r, -H / 2, H / 2, n=64).transformed(_move_mat(xc))
    x_meet = xc - np.sqrt(r * r - hw * hw)          # cylinder surface at the spool corners
    spool_m = box((x_fl, -hw, -hh), (x_meet + 1e-3, hw, hh))
    rd = door_d / 2 + frame
    x_door0 = xc + np.sqrt(r * r - rd * rd) - 0.02
    door = _xcyl(rd, x_door0, xc + r + boss)
    pumps = merge([cylinder(tmp_d / 2, -H / 2 - tmp_h, -H / 2, n=24)
                   .transformed(_move_mat(xc, y))
                   for y in np.linspace(-r / 2, r / 2, n_tmp)])
    T = rot_z(phi)[None]
    lab = [f"{spec.name}"]
    meta = {"closed_solid": True, "octant": k, "phi_deg": float(np.rad2deg(phi)),
            "envelope_assumed": True}
    parts = [Part("chamber", chamber, "stainless_steel_304L", T, lab, meta=dict(meta)),
             Part("spool", spool_m, "stainless_steel_304L", T, lab, meta=dict(meta)),
             Part("door", door, "stainless_steel_304L", T, lab,
                  meta={**meta, "door_inner_diameter_m": door_d}),
             Part("turbo_pumps", pumps, "stainless_steel", T, lab,
                  meta={**meta, "count": n_tmp})]
    return finish(tr, parts, {"octant": k, "port_sector": i, "phi_deg": float(np.rad2deg(phi)),
                              "chamber_R_range": [xc - r, xc + r], "chamber_Z_range": [-H / 2, H / 2],
                              "envelope": env, "door_inner_diameter_m": door_d})


def _move_mat(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _adaptor_box(params_env):
    a, b, c = _numbers(params_env)[:3]           # radial x vertical x toroidal
    return a, b, c


def build_nbi_adaptor(tr, spec):
    tr.own()
    k = int(tr.p("octant"))
    L, i, phi = _port_frame(tr, k)
    env = str(tr.p("envelope"))
    a, b, c = _adaptor_box(env)
    tr.p("name"), tr.p("beam_scrapers"), tr.p("seal_to_torus"), tr.p("port")
    x_fl = float(tr.k("port_hduct_R_end"))
    valve = tr.component("rotary_valve", lambda s: int(s.params.get("octant", -1)) == k)
    if valve is not None:
        ow, oh = float(valve.params["opening_horizontal"]), float(valve.params["opening_vertical"])
        if not (c > ow and b > oh):
            raise ValueError(f"{spec.name}: adaptor {c} x {b} m cannot pass the "
                             f"{ow} x {oh} m valve opening")
    m = box((x_fl, -c / 2, -b / 2), (x_fl + a, c / 2, b / 2))
    return finish(tr, [Part("adaptor", m, "stainless_steel", rot_z(phi)[None], [spec.name],
                            meta={"closed_solid": True, "octant": k,
                                  "phi_deg": float(np.rad2deg(phi)), "envelope_assumed": True})],
                  {"octant": k, "port_sector": i, "phi_deg": float(np.rad2deg(phi)),
                   "R_range": [x_fl, x_fl + a], "envelope": env})


def build_rotary_valve(tr, spec):
    tr.own()
    k = int(tr.p("octant"))
    L, i, phi = _port_frame(tr, k)
    ow, oh = float(tr.p("opening_horizontal")), float(tr.p("opening_vertical"))
    tr.p("opening_shape"), tr.p("rotor"), tr.p("bake_C")
    st = _status_meta(tr.p("status"))
    D, H = (float(v) for v in tr.k("rotary_valve_body_DH"))
    if D <= ow or H <= oh:
        raise ValueError(f"{spec.name}: body {D} x {H} m smaller than the opening")
    x_fl = float(tr.k("port_hduct_R_end"))
    ad = tr.component("nbi_adaptor", lambda s: int(s.params.get("octant", -1)) == k)
    a = _adaptor_box(ad.params["envelope"])[0] if ad is not None else 0.0
    x0 = x_fl + a
    body = cylinder(D / 2, -H / 2, H / 2, n=48).transformed(_move_mat(x0 + D / 2))
    return finish(tr, [Part("valve_body", body, "stainless_steel", rot_z(phi)[None], [spec.name],
                            meta={"closed_solid": True, "octant": k,
                                  "phi_deg": float(np.rad2deg(phi)),
                                  "opening_m": [ow, oh], **st})],
                  {"octant": k, "phi_deg": float(np.rad2deg(phi)), "R_range": [x0, x0 + D],
                   **st})


def build_gas_inlet(tr, spec):
    tr.own()
    k = int(tr.p("octant"))
    L, i, phi = _port_frame(tr, k)
    D, Ln = (float(v) for v in tr.k("gas_inlet_module_DL"))
    x_fl = float(tr.k("port_hduct_R_end"))
    m = _xcyl(D / 2, x_fl, x_fl + Ln, n=32)
    return finish(tr, [Part("gas_module", m, "stainless_steel", rot_z(phi)[None], [spec.name],
                            meta={"closed_solid": True, "octant": k,
                                  "phi_deg": float(np.rad2deg(phi)),
                                  "port_assumed": "main horizontal port flange"})],
                  {"octant": k, "port_sector": i, "phi_deg": float(np.rad2deg(phi)),
                   "R_range": [x_fl, x_fl + Ln]})


# ---------------------------------------------------------------------------
# limiter port stubs  (type "port" with per_octant); topology records -> RecordOnly
# ---------------------------------------------------------------------------
def build_port_entry(tr, spec):
    if "per_octant" not in spec.params:
        raise RecordOnly("port topology record: names the port types of an octant; the "
                         "geometry is the base 'ports' component")
    tr.own()
    L = Layout(tr)
    n_per = int(tr.p("per_octant"))
    tr.p("position"), tr.p("shape")
    ks = list(tr.k("limiter_port_sectors_in_octant"))[:n_per]
    D = float(tr.k("limiter_port_diameter"))
    x_end = float(tr.k("limiter_port_R_end"))
    wall = float(tr.k("port_vduct_wall"))
    flm, flt = float(tr.k("port_vflange_margin")), float(tr.k("port_vflange_thickness"))
    outer = np.asarray(tr.profile("vessel_outer"), float)
    R_of_Z = _outboard_R_of_Z(outer)
    th = np.linspace(0, 2 * np.pi, 33)[:-1]

    def circ(r):
        return np.c_[r * np.cos(th), r * np.sin(th)]          # (y, z)

    def ring(uv, x):
        x = np.broadcast_to(x, (len(uv),))
        return np.c_[x, uv[:, 0], uv[:, 1]]

    def on_skin(uv):
        R = R_of_Z(uv[:, 1])
        return np.sqrt(np.maximum(R ** 2 - uv[:, 0] ** 2, 0))

    ri, ro, rf = circ(D / 2), circ(D / 2 + wall), circ(D / 2 + wall + flm)
    duct = _frame_loft([ring(ro, on_skin(ro)), ring(ro, x_end), ring(ri, x_end),
                        ring(ri, on_skin(ri))])
    flange = _frame_loft([ring(rf, x_end - flt), ring(rf, x_end), ring(ro, x_end),
                          ring(ro, x_end - flt)])
    idx = sorted(j * L.per_oct + kk for j in range(L.n_oct) for kk in ks)
    T = L.rot_sectors(idx)
    lab = [f"limiter_port_oct{L.octant_of(i) + 1}" for i in idx]
    parts = [Part("duct", duct, "inconel600", T, lab, meta={"closed_solid": True}),
             Part("flange", flange, "stainless_steel", T, lab, meta={"closed_solid": True})]
    return finish(tr, parts, {"sectors": idx, "phi_deg": [float(np.rad2deg(L.sector_phi(i)))
                                                          for i in idx],
                              "diameter_m": D, "R_end": x_end,
                              "wall_opening": "not cut (stub on the closed outer skin)"})
