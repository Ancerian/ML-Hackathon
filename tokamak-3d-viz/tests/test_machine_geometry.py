#!/usr/bin/env python3
"""Geometry-kernel checks: generated meshes vs the sourced database.

    python tests/test_machine_geometry.py         # standalone
    pytest tests/test_machine_geometry.py         # with pytest

Tolerances: 1 % for text/table/drawing values, 3 % for digitized ones
(the tolerance is chosen from the database entry's own confidence).
Needs numpy, matplotlib (point-in-polygon) and PyYAML; no data download.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from matplotlib.path import Path as MplPath                      # noqa: E402

from tokviz.machine import build_machine, load_machine, load_component, write_bake  # noqa: E402
from tokviz.machine.geom import (apply, chain_loops, point_seg_dist, poly_area,  # noqa: E402
                                 sample_surface, slice_segments)
from tokviz.machine.schema import SchemaError, Tracker  # noqa: E402

JET75 = REPO / "machines" / "jet1975"
_CACHE: dict = {}


def machine():
    if "mb" not in _CACHE:
        db = load_machine(JET75)
        _CACHE["db"], _CACHE["mb"] = db, build_machine(db)
    return _CACHE["db"], _CACHE["mb"]


def tol_of(db, path):
    return 0.03 if db.dim(path).confidence in ("digitized", "assumed") else 0.01


def check(db, path, val, ref=None, index=None, scale=1.0):
    """|val - ref| <= tol * |ref| with ref = scale * database value."""
    if ref is None:
        v = db.value(path)
        ref = (v[index] if index is not None else v) * scale
    t = tol_of(db, path)
    rel = abs(val - ref) / abs(ref)
    assert rel <= t, f"{path}: model {val:.4f} vs {ref:.4f} ({rel*100:.2f} % > {t*100:.0f} %)"
    return rel


def world_points(part, k=0, instances=None):
    X = part.transforms if instances is None else part.transforms[instances]
    if k:
        return sample_surface(part.mesh, list(X), k=k)
    return np.vstack([apply(M, part.mesh.v) for M in X])


def section_y0(mesh, eps=1e-5):
    """Loops of the local-frame section by the half-plane phi = eps (x > 0)."""
    V = mesh.v
    c, s = np.cos(-eps), np.sin(-eps)
    L = np.c_[c * V[:, 0] - s * V[:, 1], s * V[:, 0] + c * V[:, 1], V[:, 2]]
    sg = slice_segments(L, mesh.tris(), L[:, 1])
    sg = sg[(sg[:, :, 0] > 0).all(1)][:, :, [0, 2]]
    return [lp for lp in chain_loops(sg) if len(lp) > 3]


def crossings_at(loop, axis, value):
    """Values of the other coordinate where a polyline crosses coord[axis]=value."""
    other = 1 - axis
    a, b = loop[:-1], loop[1:]
    m = (a[:, axis] - value) * (b[:, axis] - value) < 0
    f = (value - a[m, axis]) / (b[m, axis] - a[m, axis])
    return np.sort(a[m, other] + f * (b[m, other] - a[m, other]))


# ---------------------------------------------------------------------------
# TF solid membership (analytic, from the database profiles)
# ---------------------------------------------------------------------------
class TFSolid:
    def __init__(self, db, mb):
        from tokviz.machine.components.tf_coil import tf_geometry
        tr = Tracker(db, db.component("tf_coils"))
        self.g = tf_geometry(tr)
        tf = mb["tf_coils"]
        self.phis = np.array(tf.info["phi_deg"]) * np.pi / 180
        self.pitch = 2 * np.pi / self.g["n"]
        self.outer = MplPath(self.g["outer"])
        self.bore = MplPath(self.g["bore"])
        pads = tf.part("support_pads").mesh.v
        self.pad_lo = pads.min(0)
        self.pad_hi = pads.max(0)
        self.z_pad = db.value("tf_coils.support_pad_Z_outer")

    def local(self, P):
        phi = np.arctan2(P[:, 1], P[:, 0])
        k = np.round((phi - self.phis[0]) / self.pitch)
        a = self.phis[0] + k * self.pitch
        x = np.cos(a) * P[:, 0] + np.sin(a) * P[:, 1]
        y = -np.sin(a) * P[:, 0] + np.cos(a) * P[:, 1]
        return x, y, P[:, 2]

    def inside(self, P, margin=1e-3):
        x, y, z = self.local(P)
        hw = self.g["half_width"](x) - margin
        cand = np.abs(y) < hw
        idx = np.flatnonzero(cand)
        xz = np.c_[x[idx], z[idx]]
        in_case = self.outer.contains_points(xz) & ~self.bore.contains_points(xz)
        if in_case.any():
            d = np.minimum(point_seg_dist(xz[in_case], self.g["outer"]),
                           point_seg_dist(xz[in_case], self.g["bore"]))
            in_case[np.flatnonzero(in_case)[d < margin]] = False
        hit = np.zeros(len(P), bool)
        hit[idx[in_case]] = True
        az = np.abs(z)
        pad = ((x > self.pad_lo[0] + margin) & (x < self.pad_hi[0] - margin)
               & (np.abs(y) < self.pad_hi[1] - margin)
               & (az > self.pad_hi[2] - (self.z_pad - (-self.pad_lo[2])) + margin)
               & (az < self.z_pad - margin))
        return hit | pad


# ---------------------------------------------------------------------------
# tests: database / loader
# ---------------------------------------------------------------------------
def test_database_loads_and_validates():
    db, mb = machine()
    assert db.lineage == ["jet1975"]
    for sec in ("plasma", "vessel", "tf_coils", "pf_coils", "iron_core", "structure",
                "ports", "limiters", "nbi", "overall", "kernel"):
        assert db.dims.get(sec), sec
    assert all(d.confidence in ("assumed", "digitized", "drawing")
               for d in db.dims["kernel"].values())
    assert not mb.unbuilt


def test_schema_rejects_bad_entries():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "bad").mkdir()
        (tmp / "bad" / "dimensions.yaml").write_text(textwrap.dedent("""
            plasma:
              - {id: R0, value: 2.96, unit: m, source: X, page: 1, confidence: guessed}
        """))
        try:
            load_machine(tmp / "bad")
        except SchemaError as e:
            assert "confidence" in str(e)
        else:
            raise AssertionError("bad confidence accepted")
        (tmp / "bad" / "dimensions.yaml").write_text(textwrap.dedent("""
            plasma:
              - {id: R0, value: 2.96, unit: m, source: X, confidence: table}
        """))
        try:
            load_machine(tmp / "bad")
        except SchemaError as e:
            assert "page" in str(e)
        else:
            raise AssertionError("sourced value without page accepted")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# tests: dimensions of the generated meshes
# ---------------------------------------------------------------------------
def test_vessel_extents_and_wall_thickness():
    db, mb = machine()
    sec = mb["vessel"].part("rigid_sector_plain").mesh
    loops = sorted(section_y0(sec), key=lambda lp: abs(poly_area(lp)))
    assert len(loops) == 2, f"expected inner+outer skin loops, got {len(loops)}"
    inner, outer = loops
    r_mid = np.sort(np.concatenate([crossings_at(inner, 1, 1e-7), crossings_at(outer, 1, 1e-7)]))
    assert len(r_mid) == 4
    check(db, "vessel.rs_R_outer_wall_inboard", r_mid[0])
    check(db, "vessel.rs_R_inner_wall_inboard", r_mid[1])
    check(db, "vessel.rs_R_inner_wall_outboard", r_mid[2])
    check(db, "vessel.rs_R_outer_wall_outboard", r_mid[3])
    check(db, "vessel.rs_half_height_outer", outer[:, 1].max())
    check(db, "vessel.rs_Z_inner_wall_top", inner[:, 1].max())
    th = point_seg_dist(inner, outer)
    check(db, "vessel.wall_thickness_Ri_Ra", th.min())
    check(db, "vessel.wall_thickness_max", th.max())
    # rigid sector + bellows tile 360 deg
    info = mb["vessel"].info
    assert abs((info["sector_angle_deg"] + info["bellows_angle_deg"]) * 32 - 360) < 1e-9


def test_vessel_sectors_octants_and_openings():
    db, mb = machine()
    parts = mb["vessel"].parts
    n = sum(p.n_instances for p in parts)
    assert n == db.value("vessel.n_rigid_sectors") == 32
    octs = [lab.split("_oct")[1] for p in parts for lab in p.labels]
    assert sorted(set(octs)) == [str(i) for i in range(8)]
    assert all(octs.count(str(i)) == 4 for i in range(8))
    assert mb["vessel"].part("rigid_sector_hport").n_instances == \
        db.value("ports.horizontal_port_count") == 8
    # the port opening removes wall: fewer faces than the plain sector
    assert len(mb["vessel"].part("rigid_sector_hport").mesh.q) < \
        len(mb["vessel"].part("rigid_sector_plain").mesh.q)
    for p in mb["bellows"].parts:
        assert p.n_instances == 32 and p.meta["thin_shell"]
        assert abs(p.meta["thickness_m"] - db.value("vessel.bellows_wall_thickness")) < 1e-12


def test_horizontal_port_opening_size():
    db, mb = machine()
    d = mb["ports"].part("hport_duct").mesh.v
    # inner opening = the database 0.46 x 0.96 (duct wall is extra)
    wall = db.value("kernel.port_hduct_wall")
    check(db, "ports.horizontal_port_width", (d[:, 1].max() - d[:, 1].min()) - 2 * wall)
    check(db, "ports.horizontal_port_height", (d[:, 2].max() - d[:, 2].min()) - 2 * wall)


def test_tf_coils_count_spacing_and_extents():
    db, mb = machine()
    tf = mb["tf_coils"]
    cas = tf.part("casing")
    assert cas.n_instances == db.value("tf_coils.n_coils") == 32
    ang = np.sort(np.mod([np.arctan2(M[1, 0], M[0, 0]) for M in cas.transforms], 2 * np.pi))
    np.testing.assert_allclose(np.diff(np.degrees(ang)), 11.25, atol=1e-6)
    loops = sorted(section_y0(cas.mesh), key=lambda lp: abs(poly_area(lp)))
    bore, outer = loops
    check(db, "tf_coils.bore_R_inner", bore[:, 0].min())
    check(db, "tf_coils.bore_R_outer", bore[:, 0].max())
    check(db, "tf_coils.overall_width", outer[:, 0].max() - outer[:, 0].min())
    check(db, "tf_coils.casing_outer_Rout_fit", outer[:, 0].max())
    check(db, "tf_coils.casing_outer_Zmax_fit", outer[:, 1].max())
    check(db, "tf_coils.bore_Zmax_fit", bore[:, 1].max())
    check(db, "tf_coils.inner_leg_radial_thickness", bore[:, 0].min() - outer[:, 0].min())
    pads = tf.part("support_pads").mesh.v
    check(db, "tf_coils.overall_height", pads[:, 2].max() - pads[:, 2].min())
    # toroidal widths: 0.338 on the outer leg; wedge 0.26 on the inner leg
    v = cas.mesh.v
    out_leg = v[v[:, 0] > 4.5]
    check(db, "tf_coils.outer_leg_toroidal_width", 2 * np.abs(out_leg[:, 1]).max())
    leg = v[(np.abs(v[:, 2]) < 1.0) & (v[:, 0] < 1.6)]
    r_a, r_b = leg[:, 0].min(), leg[:, 0].max()
    w_a = 2 * np.abs(leg[np.isclose(leg[:, 0], r_a, atol=1e-6)][:, 1]).max()
    w_b = 2 * np.abs(leg[np.isclose(leg[:, 0], r_b, atol=1e-6)][:, 1]).max()
    r_w = tf.info["wedge_R_for_0.26m"]
    assert r_a < r_w < r_b, "0.26 m wedge width must fall inside the inner leg"
    check(db, "tf_coils.inner_leg_toroidal_width_wedge", np.interp(r_w, [r_a, r_b], [w_a, w_b]))


def test_pf_coil_radii_and_positions():
    db, mb = machine()
    pf = mb["pf_coils"]
    n = sum(p.n_instances for p in pf.parts if p.name.startswith("coil") and "support" not in p.name)
    assert n == db.value("pf_coils.n_pf_coils_total") == 16
    c1 = pf.part("coil1")
    assert c1.n_instances == db.value("pf_coils.coil1_count")
    R1 = np.hypot(c1.mesh.v[:, 0], c1.mesh.v[:, 1])
    check(db, "pf_coils.coil1_outer_diameter", 2 * R1.max())
    check(db, "pf_coils.coil1_winding_radial", R1.max() - R1.min())
    z1 = np.sort(c1.transforms[:, 2, 3])
    check(db, "pf_coils.coil1_pitch_height", float(np.mean(np.diff(z1))))
    assert abs(z1.mean()) < 1e-9
    for k in (2, 3, 4):
        p = pf.part(f"coil{k}")
        assert p.n_instances == db.value(f"pf_coils.coil{k}_count")
        R = np.hypot(p.mesh.v[:, 0], p.mesh.v[:, 1])
        check(db, f"pf_coils.coil{k}_mean_diameter", R.max() + R.min())
        zs = np.sort(p.transforms[:, 2, 3])
        check(db, f"pf_coils.coil{k}_Z", zs[1])
        assert abs(zs[0] + zs[1]) < 1e-9
    c3 = pf.part("coil3").mesh.v
    check(db, "pf_coils.coil3_width", np.ptp(np.hypot(c3[:, 0], c3[:, 1])))
    check(db, "pf_coils.coil3_height", np.ptp(c3[:, 2]))
    c4 = pf.part("coil4").mesh.v
    check(db, "pf_coils.coil4_overall_height", np.ptp(c4[:, 2]))
    check(db, "pf_coils.coil4_R", 0.5 * (np.hypot(c4[:, 0], c4[:, 1]).max()
                                         + np.hypot(c4[:, 0], c4[:, 1]).min()))


def test_iron_core_overall_dimensions():
    db, mb = machine()
    ir = mb["iron_core"]
    limb = ir.part("outer_limb")
    assert limb.n_instances == db.value("iron_core.n_limbs") == 8
    assert ir.part("upper_radial_arm").n_instances == 8
    assert ir.part("lower_radial_arm").n_instances == 8
    check(db, "overall.overall_diameter", 2 * limb.mesh.v[:, 0].max())
    z_top = ir.part("upper_radial_arm").mesh.v[:, 2].max()
    z_bot = ir.part("foot_plate").mesh.v[:, 2].min()
    check(db, "overall.overall_height", z_top - z_bot)
    check(db, "iron_core.limb_toroidal_width", np.ptp(limb.mesh.v[:, 1]))
    # the 8 arms never overlap near the axis (wedge clamp)
    arm = ir.part("upper_radial_arm").mesh.v
    assert np.all(np.abs(arm[:, 1]) <= arm[:, 0] * np.tan(np.pi / 8) + 1e-9)


def test_ring_structure_diameter():
    db, mb = machine()
    st = mb["structure"]
    rmax = max(np.hypot(*world_points(p)[:, :2].T).max() for p in st.parts
               if "ring_block" in p.name)
    check(db, "structure.ring_outer_diameter", 2 * rmax)
    top = max(p.mesh.v[:, 2].max() for p in st.parts if p.name.startswith("upper_ring"))
    check(db, "structure.ring_Z_top", top)
    n_blocks = sum(p.n_instances for p in st.parts if p.name.startswith("upper_ring_block"))
    assert n_blocks == 32


def test_plasma_shape_and_clearance():
    db, mb = machine()
    pl = mb["plasma"]
    v = pl.part("lcfs").mesh.v
    R = np.hypot(v[:, 0], v[:, 1])
    check(db, "plasma.R_inner_plasma", R.min())
    check(db, "plasma.R_outer_plasma", R.max())
    check(db, "plasma.b", v[:, 2].max())
    check(db, "plasma.a", 0.5 * (R.max() - R.min()))
    check(db, "plasma.R0", 0.5 * (R.max() + R.min()))
    i_top = np.argmax(v[:, 2])
    delta_mesh = (0.5 * (R.max() + R.min()) - R[i_top]) / (0.5 * (R.max() - R.min()))
    assert abs(delta_mesh - pl.info["delta"]) < 0.01
    assert 0.2 < pl.info["delta"] < 0.6
    wall = np.asarray(db.profile("vessel_inner").data, float)
    rz = np.c_[R, v[:, 2]]
    assert MplPath(wall).contains_points(rz).all(), "LCFS crosses the vessel inner wall"
    dmin = point_seg_dist(rz[::7], wall).min()
    assert dmin >= db.value("kernel.plasma_min_wall_clearance") - 1e-3, dmin
    gaps = pl.info["gaps"]
    assert 0.15 <= gaps["outboard_midplane"] <= 0.20      # "~16 cm" (Table I.3-1)
    assert 0.03 <= gaps["inboard_midplane"] <= 0.07       # "~5 cm"


def test_limiters_and_counts():
    db, mb = machine()
    lim = mb["limiters"]
    op = lim.part("outer_rail_plate")
    assert op.n_instances == db.value("limiters.outer_limiter_plates") == 16
    size = db.value("limiters.plate_size")
    v = op.mesh.v
    check(db, "limiters.plate_size", np.ptp(v[:, 2]), ref=size[0])
    check(db, "limiters.plate_size", np.ptp(v[:, 1]), ref=size[1])
    check(db, "limiters.outer_limiter_R", v[np.abs(v[:, 2]) < 0.03][:, 0].min())
    wall = np.asarray(db.profile("vessel_inner").data, float)
    for nm in ("upper_rail_plate", "lower_rail_plate"):
        P = world_points(lim.part(nm))
        rz = np.c_[np.hypot(P[:, 0], P[:, 1]), P[:, 2]]
        assert MplPath(wall).contains_points(rz).all(), f"{nm} crosses the vessel wall"
    assert mb["nbi"].part("injector_tank").n_instances == db.value("nbi.injectors_per_port")


# ---------------------------------------------------------------------------
# tests: no interpenetration between key subsystems (sampled)
# ---------------------------------------------------------------------------
def test_no_interpenetration_vessel_tf():
    db, mb = machine()
    tf = TFSolid(db, mb)
    for comp in ("vessel", "bellows"):
        for p in mb[comp].parts:
            P = world_points(p, k=1)
            bad = tf.inside(P)
            assert not bad.any(), f"{comp}/{p.name}: {bad.sum()} samples inside a TF casing"


def test_no_interpenetration_pf_tf():
    db, mb = machine()
    tf = TFSolid(db, mb)
    for p in mb["pf_coils"].parts:
        P = world_points(p, k=2)
        bad = tf.inside(P)
        assert not bad.any(), f"pf_coils/{p.name}: {bad.sum()} samples inside a TF casing"


def test_no_interpenetration_ports_structure_tf():
    db, mb = machine()
    tf = TFSolid(db, mb)
    for comp in ("ports", "structure", "limiters"):
        for p in mb[comp].parts:
            P = world_points(p, k=1)
            bad = tf.inside(P)
            assert not bad.any(), f"{comp}/{p.name}: {bad.sum()} samples inside a TF casing"


def test_no_interpenetration_plasma_vessel():
    db, mb = machine()
    wall_in = np.asarray(db.profile("vessel_inner").data, float)
    P = world_points(mb["plasma"].part("lcfs"), k=1)
    rz = np.c_[np.hypot(P[:, 0], P[:, 1]), P[:, 2]]
    assert MplPath(wall_in).contains_points(rz).all()
    # and every vessel/bellows sample stays outside the LCFS
    lcfs = np.asarray(mb["plasma"].info["profile_rz"])
    for comp in ("vessel", "bellows"):
        for p in mb[comp].parts:
            Q = world_points(p)
            q = np.c_[np.hypot(Q[:, 0], Q[:, 1]), Q[:, 2]]
            assert not MplPath(lcfs).contains_points(q).any(), f"{p.name} inside the plasma"


def test_nbi_clear_of_iron_limbs():
    db, mb = machine()
    ir = mb["iron_core"].part("outer_limb")
    lo, hi = ir.mesh.v.min(0), ir.mesh.v.max(0)
    for p in mb["nbi"].parts:
        P = world_points(p, k=2)
        for M in ir.transforms:
            L = (P - M[:3, 3]) @ M[:3, :3]          # into the limb's local frame
            inside = np.all((L > lo) & (L < hi), axis=1)
            assert not inside.any(), f"nbi/{p.name} hits an iron limb"


# ---------------------------------------------------------------------------
# tests: bake + manifest provenance
# ---------------------------------------------------------------------------
def test_bake_manifest_provenance_every_component():
    db, mb = machine()
    tmp = Path(tempfile.mkdtemp())
    try:
        path = write_bake(mb, tmp / "bake")
        man = json.loads(path.read_text())
        assert man["machine"]["name"] == "jet1975"
        assert set(man["components"]) == set(mb.components)
        for name, c in man["components"].items():
            assert c["provenance"], f"{name}: no provenance"
            for e in c["provenance"]:
                assert e.get("source") or e.get("file"), (name, e)
                assert e.get("confidence"), (name, e)
                if e.get("confidence") not in ("assumed", "unknown") and "file" not in e:
                    assert e.get("page") is not None, (name, e)
            assert name in man["objects"]
            parts = load_component(tmp / "bake" / c["file"])
            assert [p["name"] for p in parts] == list(c["parts"])
            for p in parts:
                meta = c["parts"][p["name"]]
                assert p["v"].dtype == np.float32 and p["q"].dtype == np.int32
                assert p["X"].shape == (meta["n_instances"], 4, 4)
                assert meta["material"]
        assert man["mesh_totals"]["faces"] > 0
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# tests: delta variants (base inheritance, overrides, add/remove)
# ---------------------------------------------------------------------------
DELTA = """
base: jet1975
meta: {machine: "test variant"}
overrides:
  - {path: plasma.b, value: 2.00, unit: m, source: TESTSRC, page: 7, confidence: text, note: "fixture"}
  - {path: tf_coils.n_coils.note, value: "note changed by fixture", source: TESTSRC, page: 7, confidence: text}
  - {path: kernel.vessel_bellows_angle_deg, value: 2.5, unit: deg, source: TESTSRC, page: 8, confidence: assumed}
  - {path: components.vessel.openings, value: false, source: TESTSRC, page: 8, confidence: assumed}
  - {path: plasma.new_quantity, value: 1.23, unit: m, source: TESTSRC, page: 9, confidence: table}
components_remove:
  - {name: nbi, source: TESTSRC, page: 3, confidence: text, note: "not in this variant"}
components_add:
  - name: extra_coils
    type: rect_coil_set
    params: {coils: [{name: extra_upper, R: 6.5, Z: 2.0, dR: 0.2, dZ: 0.3}]}
    source: TESTSRC
    page: 4
    confidence: drawing
  - {name: mystery, type: no_such_builder, params: {x: 1}, source: TESTSRC, page: 5, confidence: text}
"""


def test_delta_variant_inheritance():
    tmp = Path(tempfile.mkdtemp())
    try:
        var = tmp / "variant_a"
        var.mkdir()
        (var / "deltas.yaml").write_text(DELTA)
        # a variant of the variant (chained inheritance)
        var2 = tmp / "variant_b"
        var2.mkdir()
        (var2 / "deltas.yaml").write_text(textwrap.dedent("""
            base: variant_a
            overrides:
              - {path: plasma.R0, value: 3.00, unit: m, source: TEST2, page: 1, confidence: table}
            components_remove: [human]
        """))
        db = load_machine(var2)
        assert db.lineage == ["jet1975", "variant_a", "variant_b"]
        assert db.value("plasma.b") == 2.00 and db.value("plasma.R0") == 3.00
        d = db.dim("plasma.b")
        assert d.source == "TESTSRC" and d.page == 7
        assert d.supersedes["source"] == "JET1975" and d.supersedes["value"] == 2.10
        assert db.dim("tf_coils.n_coils").note == "note changed by fixture"
        assert db.value("tf_coils.n_coils") == 32
        assert db.value("plasma.new_quantity") == 1.23
        names = [c.name for c in db.components]
        assert "nbi" not in names and "human" not in names
        assert "extra_coils" in names and "mystery" in names
        assert any("no builder" in w for w in db.warnings)
        # the base database is untouched
        base = load_machine(JET75)
        assert base.value("plasma.b") == 2.10

        mb = build_machine(db, only=["plasma", "vessel", "extra_coils", "mystery"])
        v = mb["plasma"].part("lcfs").mesh.v
        assert abs(v[:, 2].max() - 2.00) < 1e-3
        prov = {e["path"]: e for e in mb["plasma"].provenance}
        assert prov["plasma.b"]["source"] == "TESTSRC"
        assert prov["plasma.R0"]["source"] == "TEST2"
        assert [p.name for p in mb["vessel"].parts] == ["rigid_sector_plain"]
        assert abs(mb["vessel"].info["bellows_angle_deg"] - 2.5) < 1e-12
        ec = mb["extra_coils"]
        assert ec.spec_provenance["source"] == "TESTSRC"
        Rr = np.hypot(ec.parts[0].mesh.v[:, 0], ec.parts[0].mesh.v[:, 1])
        assert abs(Rr.max() - 6.6) < 1e-9 and abs(Rr.min() - 6.4) < 1e-9
        assert [u["name"] for u in mb.unbuilt] == ["mystery"]
        path = write_bake(mb, tmp / "bake")
        man = json.loads(path.read_text())
        assert man["machine"]["lineage"] == ["jet1975", "variant_a", "variant_b"]
        ops = [a["op"] for a in man["machine"]["overrides_applied"]]
        assert ops.count("override") == 6 and "remove" in ops and "add_component" in ops
        assert man["machine"]["unbuilt_components"][0]["name"] == "mystery"
    finally:
        shutil.rmtree(tmp)


def test_delta_cycle_and_missing_base():
    tmp = Path(tempfile.mkdtemp())
    try:
        for a, b in (("x", "y"), ("y", "x")):
            (tmp / a).mkdir()
            (tmp / a / "deltas.yaml").write_text(f"base: {b}\n")
        for probe, word in ((tmp / "x", "cycle"),):
            try:
                load_machine(probe)
            except SchemaError as e:
                assert word in str(e)
            else:
                raise AssertionError("cycle not detected")
        (tmp / "z").mkdir()
        (tmp / "z" / "deltas.yaml").write_text("base: does_not_exist\n")
        try:
            load_machine(tmp / "z")
        except SchemaError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("missing base not detected")
    finally:
        shutil.rmtree(tmp)


def test_jet1983_variant_if_present():
    d83 = REPO / "machines" / "jet1983"
    if not (d83 / "deltas.yaml").exists():
        print("  (machines/jet1983/deltas.yaml not present - skipped)")
        return
    db = load_machine(d83)
    assert db.lineage == ["jet1975", "jet1983"]
    mb = build_machine(db)
    # overrides flow into the geometry
    assert mb["vessel"].info["rigid_pieces"] == db.value("vessel.n_rigid_sectors")
    assert mb["pf_coils"].part("coil1").n_instances == db.value("pf_coils.coil1_count")
    z1 = np.sort(mb["pf_coils"].part("coil1").transforms[:, 2, 3])
    assert abs(np.mean(np.diff(z1)) - db.value("pf_coils.coil1_pitch_height")) < 1e-9
    # removed / added components
    assert "nbi" not in mb.components
    unbuilt = {u["name"] for u in mb.unbuilt}
    assert "limiters" in unbuilt              # its 1975 inputs were removed
    mods = [n for n, c in mb.components.items() if c.type == "limiter_module"]
    assert len(mods) == 12
    for n in mods:
        prov = mb[n].spec_provenance
        assert prov.get("source") and prov.get("confidence")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
