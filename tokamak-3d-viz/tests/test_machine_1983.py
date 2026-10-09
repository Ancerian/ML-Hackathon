#!/usr/bin/env python3
"""JET as built, 1983 (machines/jet1983 deltas over jet1975): geometry-kernel checks.

    python tests/test_machine_1983.py        # standalone
    pytest tests/test_machine_1983.py

Expected numbers (interpretation of machines/jet1983/deltas.yaml, see
machines/jet1983/ASSUMPTIONS_KERNEL.md):
  * rigid vessel pieces  40 = 8 octants x 5 (end 'a' + end 'b' half sectors at the
    octant welds count as 2 pieces; 3 full sectors per octant); pitch 11.25 deg kept
  * bellows              32 = 8 x 4 (one per TF pitch; 2 concentric plies each,
    i.e. 32 instances of each of the 2 ply meshes)
  * limiter modules      12 (4 graphite at 0/90/180/270 deg, 8 Ni-clad), 0.40 x 0.80 = 0.32 m2
  * PF coil 1            8 coils, pitch 0.533 m (PF total 14)
  * NBI                  none (section removed); rail limiters removed
  * ports (octants 1-based as in the deltas): pumps 1, 5; NBI adaptors + rotary
    valves 4, 8; gas inlets 2, 6; main port of octant k at phi = 45(k-1) + 22.5 deg
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from matplotlib.path import Path as MplPath                      # noqa: E402

from tokviz.machine import build_machine, load_machine, load_component, write_bake  # noqa: E402
from tokviz.machine.geom import point_seg_dist                   # noqa: E402
from test_machine_geometry import TFSolid, world_points          # noqa: E402

JET83 = REPO / "machines" / "jet1983"
_CACHE: dict = {}

NEW_TYPES = ("vessel_restraint_ring", "pump_chamber", "nbi_adaptor", "rotary_valve",
             "gas_inlet", "port", "limiter_module")


def machine():
    if "mb" not in _CACHE:
        db = load_machine(JET83)
        _CACHE["db"], _CACHE["mb"] = db, build_machine(db)
    return _CACHE["db"], _CACHE["mb"]


def manifest():
    if "man" not in _CACHE:
        db, mb = machine()
        tmp = Path(tempfile.mkdtemp())
        path = write_bake(mb, tmp / "bake")
        _CACHE["man"] = json.loads(path.read_text())
        _CACHE["bake"] = tmp / "bake"
        _CACHE["tmp"] = tmp
    return _CACHE["man"]


def new_components(mb):
    return {n: c for n, c in mb.components.items() if c.type in NEW_TYPES}


def spec_params(db, name):
    return db.component(name).params


def phi_deg_of(M):
    return float(np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))) % 360.0


# ---------------------------------------------------------------------------
def test_loads_and_lineage():
    db, mb = machine()
    assert db.lineage == ["jet1975", "jet1983"]
    assert "1983" in db.meta["variant"]
    assert db.dim("kernel.tf_phi0_deg").origin.endswith("jet1983/ASSUMPTIONS_KERNEL.md")
    # nothing silently unbuilt: every entry is built, realised, record-only or removed
    states = {u["name"]: u["state"] for u in mb.unbuilt}
    assert set(states.values()) <= {"realized", "record_only", "removed"}, states
    assert states["limiters"] == "removed"
    for n in ("vessel_octant_layout_1983", "vessel_bellows_1983", "pf_coil1_stack_1983",
              "vessel_bellows_length_estimate", "octant_port_topology_1983"):
        assert states[n] == "realized", n
    for n in ("vessel_bake_operation_1983", "first_wall_1983", "pf_coil_data_1982",
              "iron_core_1982", "mechanical_structure_1982"):
        assert states[n] == "record_only", n


def test_vessel_40_rigid_pieces_32_bellows():
    db, mb = machine()
    v = mb["vessel"]
    counts = {p.name: p.n_instances for p in v.parts}
    assert counts == {"rigid_sector_end_a": 8, "rigid_sector_end_b": 8,
                      "rigid_sector_vport_large": 8, "rigid_sector_hport": 8,
                      "rigid_sector_vport_small": 8}, counts
    assert sum(counts.values()) == 40 == db.value("vessel.n_rigid_sectors")
    assert v.info["rigid_pieces"] == 40 and v.info["rigid_pieces_per_octant"] == 5
    # five pieces per octant (end 'a' closes the previous octant)
    octs = [lab.split("_oct")[1] for p in v.parts for lab in p.labels]
    assert all(octs.count(str(j)) == 5 for j in range(8)), octs
    # bellows: 32 per ply, 2 plies, 2 mm sheet
    b = mb["bellows"]
    assert len(b.parts) == 2
    for p in b.parts:
        assert p.n_instances == 32
        assert abs(p.meta["thickness_m"] - 0.002) < 1e-12
    assert b.info["convolutions"] == 14 and b.info["flanges"] == "parallel"


def test_vessel_widths_match_delta_interpretation():
    db, mb = machine()
    lay = spec_params(db, "vessel_octant_layout_1983")
    w = mb["vessel"].info["widths_deg"]
    for R, tag in ((4.389, "R4389"), (1.664, "R1664")):
        row = w[f"R{R:.3f}"]
        assert abs(row["rigid_full_deg"] - lay[f"rigid_full_width_deg_at_{tag}"]) < 2e-3
        assert abs(row["rigid_end_deg"] - lay[f"rigid_end_width_deg_at_{tag}"]) < 2e-3
        assert abs(row["bellows_deg"] - lay[f"bellows_width_deg_at_{tag}"]) < 2e-3
    d = lay["bellows_toroidal_length"] / 2
    pitch = np.deg2rad(lay["pitch_deg"])
    # every rigid-piece end plate lies on a plane parallel to the bellows plane, at d
    for p in mb["vessel"].parts:
        V = p.mesh.v
        if p.name.startswith("rigid_sector_end"):
            sgn = +1 if p.name.endswith("_b") else -1
            ang = sgn * pitch / 2
            dist = np.abs(-np.sin(ang) * V[:, 0] + np.cos(ang) * V[:, 1])
            weld = np.abs(V[:, 1])                  # the weld plane is phi = 0 (local)
            assert abs(dist.min() - d) < 1e-9 and weld.min() < 1e-9
            assert np.all(sgn * np.arctan2(V[:, 1], V[:, 0]) >= -1e-12)
        else:
            dist = np.minimum(np.abs(-np.sin(pitch / 2) * V[:, 0] + np.cos(pitch / 2) * V[:, 1]),
                              np.abs(np.sin(pitch / 2) * V[:, 0] + np.cos(pitch / 2) * V[:, 1]))
            assert abs(dist.min() - d) < 1e-9, (p.name, dist.min())
    # bellows: constant toroidal length 2d (parallel flanges)
    for p in mb["bellows"].parts:
        y = p.mesh.v[:, 1]
        assert abs(y.max() - d) < 1e-9 and abs(y.min() + d) < 1e-9


def test_octant_frame_and_port_octants():
    db, mb = machine()
    # octant welds at 45 j, main ports at 45 j + 22.5
    assert np.allclose(np.asarray(mb["vessel"].info["octant_boundaries_deg"]) % 360,
                       np.arange(8) * 45.0)
    expect = {"pump_chamber": {1, 5}, "nbi_adaptor": {4, 8}, "rotary_valve": {4, 8},
              "gas_inlet": {2, 6}}
    for typ, octs in expect.items():
        got = {}
        for n, c in mb.components.items():
            if c.type != typ:
                continue
            k = c.info["octant"]
            assert k == spec_params(db, n)["octant"]
            phi = phi_deg_of(c.parts[0].transforms[0])
            assert abs(phi - (45.0 * (k - 1) + 22.5)) < 1e-9, (n, phi)
            got[k] = n
        assert set(got) == octs, (typ, got)
    # the pump/NBI/gas octants carry the kernel's main horizontal port stub
    hp = mb["ports"].part("hport_duct")
    hp_phi = sorted(phi_deg_of(M) for M in hp.transforms)
    assert np.allclose(hp_phi, 45.0 * np.arange(8) + 22.5)
    # one limiter port per octant, next to the main port
    lp = mb["limiter_port"]
    assert lp.parts[0].n_instances == 8
    assert np.allclose(sorted(lp.info["phi_deg"]), (45.0 * np.arange(8) + 33.75))


def test_limiter_modules_12_area_032():
    db, mb = machine()
    mods = {n: c for n, c in mb.components.items() if c.type == "limiter_module"}
    assert len(mods) == 12
    graphite = sorted(round(c.parts[0].meta["phi_deg"]) % 360 for c in mods.values()
                      if spec_params(db, c.name)["material"] == "graphite")
    assert graphite == [0, 90, 180, 270]
    assert sum(1 for c in mods.values() if "Ni" in spec_params(db, c.name)["material"]) == 8
    wall = np.asarray(db.profile("vessel_inner").data, float)
    for n, c in mods.items():
        sp = spec_params(db, n)
        m = c.parts[0].mesh
        R = np.hypot(m.v[:, 0], m.v[:, 1])
        assert abs(R.min() - sp["R_front"]) < 1e-9
        # front face area: quads whose 4 vertices lie on R = R_front (curved face)
        on = np.abs(R - R.min()) < 1e-9
        Q = m.q[on[m.q].all(1)]
        P = m.v[Q]
        area = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1).sum()
        assert abs(area - 0.32) / 0.32 < 0.01, (n, area)
        assert abs(area - sp["area"]) / sp["area"] < 0.01
        # inside the plasma-side wall (no contact with the vessel)
        W = world_points(c.parts[0], k=2)
        rz = np.c_[np.hypot(W[:, 0], W[:, 1]), W[:, 2]]
        assert MplPath(wall).contains_points(rz).all(), f"{n} crosses the vessel wall"
        assert point_seg_dist(rz, wall).min() > 2e-3, n
    fronts = {spec_params(db, n)["material"].split()[0]: spec_params(db, n)["R_front"] for n in mods}
    assert fronts["graphite"] == 4.21 and fronts["Ni-clad"] == 4.31


def test_pf1_eight_coils():
    db, mb = machine()
    pf = mb["pf_coils"]
    c1 = pf.part("coil1")
    assert c1.n_instances == 8 == db.value("pf_coils.coil1_count")
    z = np.sort(c1.transforms[:, 2, 3])
    assert np.allclose(np.diff(z), 0.533)
    assert np.allclose(z, sorted(spec_params(db, "pf_coil1_stack_1983")["Z_centres"]), atol=1e-6)
    n_pf = sum(pf.part(f"coil{k}").n_instances for k in (1, 2, 3, 4))
    assert n_pf == 14 == db.value("pf_coils.n_pf_coils_total")
    assert pf.info["coil1_stack_check"]["max_dZ_m"] < 1e-6


def test_removals():
    db, mb = machine()
    assert "nbi" not in mb.components and "nbi" not in db.dims
    assert not any(c.type == "nbi" for c in db.components)
    for key in ("n_rail_limiters", "outer_limiter_plates", "plate_size", "plate_material",
                "upper_lower_rail_limiter_position", "plate_orientation"):
        assert not db.has(f"limiters.{key}"), key
    assert db.has("limiters.bellows_shield_Re")           # kept (Brochure p.12)
    try:
        db.profile("limiter_outer")
    except KeyError:
        pass
    else:
        raise AssertionError("profile limiter_outer should be removed")
    assert "limiters" not in mb.components


def test_restraint_rings():
    db, mb = machine()
    rr = mb["vessel_restraint_rings"]
    names = {p.name: p for p in rr.parts}
    assert set(names) == {"upper_ring_skin", "upper_ring_bridge", "lower_ring_skin",
                          "lower_ring_bridge"}
    for tag, z0 in (("upper", 1.0), ("lower", -1.0)):
        sk, br = names[f"{tag}_ring_skin"], names[f"{tag}_ring_bridge"]
        assert sk.n_instances == 32 and br.n_instances == 32
        zc = 0.5 * (sk.mesh.v[:, 2].min() + sk.mesh.v[:, 2].max())
        assert abs(zc - z0) < 1e-9
        assert abs(rr.info["R_outer_skin_at_Z_model"][f"{z0:+.2f}"] - 4.293) < 1e-3
        # bridges sit over the bellows (TF azimuths), skins between them, no overlap
        d = spec_params(db, "vessel_octant_layout_1983")["bellows_toroidal_length"] / 2
        assert np.abs(br.mesh.v[:, 1]).max() < d - 1e-3


def test_no_interpenetration_new_parts():
    """Sampled: new parts vs TF casings/pads, the vessel double wall, PF packs,
    iron limbs; bellows/rigid pieces vs TF (their shape changed in 1983)."""
    db, mb = machine()
    tf = TFSolid(db, mb)
    outer = np.asarray(db.profile("vessel_outer").data, float)
    inner = np.asarray(db.profile("vessel_inner").data, float)
    po, pi = MplPath(outer), MplPath(inner)
    packs = [(v["R_in"], v["R_out"], z, pk) for k, v in mb["pf_coils"].info.items()
             if k.startswith("coil") and isinstance(v, dict) and "Z" in v
             for z, pk in [(z, k) for z in v["Z"]]]
    pf_parts = {p.name: p for p in mb["pf_coils"].parts}
    ir = mb["iron_core"].part("outer_limb")
    lo, hi = ir.mesh.v.min(0), ir.mesh.v.max(0)
    min_clear = {}
    for n, c in new_components(mb).items():
        for p in c.parts:
            P = world_points(p, k=2)
            bad = tf.inside(P)
            assert not bad.any(), f"{n}/{p.name}: {bad.sum()} samples inside a TF casing"
            rz = np.c_[np.hypot(P[:, 0], P[:, 1]), P[:, 2]]
            in_wall = po.contains_points(rz) & ~pi.contains_points(rz)
            if in_wall.any():
                dd = np.minimum(point_seg_dist(rz[in_wall], outer), point_seg_dist(rz[in_wall], inner))
                assert dd.max() < 2e-3, f"{n}/{p.name} enters the vessel double wall ({dd.max():.4f} m)"
            for R0, R1, z, pk in packs:
                blk = pf_parts[pk].meta["blocks"]
                for b0, b1 in blk:
                    inside = ((rz[:, 0] > R0 + 1e-3) & (rz[:, 0] < R1 - 1e-3)
                              & (rz[:, 1] > z + b0 + 1e-3) & (rz[:, 1] < z + b1 - 1e-3))
                    assert not inside.any(), f"{n}/{p.name} hits PF {pk}"
            for M in ir.transforms:
                Lc = (P - M[:3, 3]) @ M[:3, :3]
                assert not np.all((Lc > lo) & (Lc < hi), axis=1).any(), f"{n}/{p.name} hits a limb"
            min_clear[f"{n}/{p.name}"] = float(np.min(np.hypot(P[:, 0], P[:, 1])))
    for comp in ("vessel", "bellows"):
        for p in mb[comp].parts:
            bad = tf.inside(world_points(p, k=1))
            assert not bad.any(), f"{comp}/{p.name}: {bad.sum()} samples inside a TF casing"
    # restraint ring vs TF bore: radial clearance at the band edges (analytic)
    bore = tf.g["bore"]
    rr = mb["vessel_restraint_rings"].part("upper_ring_skin").mesh.v
    Rr = np.hypot(rr[:, 0], rr[:, 1])
    b = bore[bore[:, 0] > 3.5]
    o = np.argsort(b[:, 1])
    gap = np.interp(rr[:, 2], b[o, 1], b[o, 0]) - Rr
    assert gap.min() > 0.010, f"restraint ring only {gap.min()*1e3:.1f} mm from the TF bore"


def test_manifest_variant_status_provenance():
    db, mb = machine()
    man = manifest()
    m = man["machine"]
    assert m["name"] == "jet1983" and "1983" in m["variant"]
    comps = man["components"]
    names_db = {c.name for c in db.components}
    listed = set(comps) | {u["name"] for u in m["unbuilt_components"]}
    assert names_db <= listed, names_db - listed
    idx = {e["name"]: e for e in m["component_index"]}
    assert names_db <= set(idx)
    assert idx["nbi"]["state"] == "removed"
    assert idx["profiles/limiter_outer.csv"]["state"] == "removed"
    for name, c in comps.items():
        assert c["variant"] in ("jet1975", "jet1983"), name
        assert c["status"] in ("installed", "not_installed")
        assert c["provenance"], name
        for e in c["provenance"]:
            assert e.get("source") or e.get("file"), (name, e)
            assert e.get("confidence"), (name, e)
            if e.get("confidence") not in ("assumed", "unknown") and "file" not in e:
                assert e.get("page") is not None, (name, e)
        parts = load_component(_CACHE["bake"] / c["file"])
        assert [p["name"] for p in parts] == list(c["parts"])
    for n in new_components(mb):
        assert comps[n]["variant"] == "jet1983", n
        assert any(e["path"] == f"components.{n}" for e in comps[n]["provenance"]) or \
            comps[n]["spec"]["provenance"].get("source"), n
    assert comps["pf_coils"]["variant"] == "jet1975"
    assert "jet1983" in comps["pf_coils"]["modified_by_variants"]
    assert "vessel_octant_layout_1983" in comps["vessel"]["realizes"]
    # rotary valves: not on the machine in 1983 (installed January 1984)
    for n in ("rotary_valve_oct4", "rotary_valve_oct8"):
        c = comps[n]
        assert c["status"] == "not_installed" and c["installed"] == "1984-01"
        assert c["hide_in_strict_variant"] is True and "1984" in c["status_note"]
        assert c["parts"]["valve_body"]["meta"]["status"] == "not_installed"
    shown = [n for n, c in comps.items() if not c["hide_in_strict_variant"]]
    assert len(shown) == len(comps) - 2


def test_base_1975_unchanged():
    base = load_machine(REPO / "machines" / "jet1975")
    mb = build_machine(base, only=["vessel", "bellows", "pf_coils"])
    assert mb["vessel"].info["rigid_pieces"] == 32
    assert abs(mb["vessel"].info["bellows_angle_deg"] - 3.0) < 1e-12
    assert mb["pf_coils"].part("coil1").n_instances == 10
    assert not mb["vessel"].consumed


def teardown_module(module=None):
    if "tmp" in _CACHE:
        shutil.rmtree(_CACHE["tmp"], ignore_errors=True)


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
    teardown_module()
    sys.exit(1 if fails else 0)
