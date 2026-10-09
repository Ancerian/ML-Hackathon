#!/usr/bin/env python3
"""Web GLB export checks (tokviz.machine.export_gltf).

    pytest tests/test_export_gltf.py
    python tests/test_export_gltf.py        # standalone

The export is re-run into a temporary directory (a few seconds), so the test
does not depend on the files in web/; the files in web/ (if present) are
additionally checked against the size budget.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_export_web                                           # noqa: E402
from tokviz.machine import export_gltf as EG                    # noqa: E402

MACHINES = ("jet1975", "jet1983")
REQUIRED_PART_EXTRAS = ("kind", "id", "component", "part", "subsystem", "label_uk",
                        "confidence", "sources", "status", "key_dims", "n_instances")
_cache: dict = {}


def out_dir() -> Path:
    if "dir" not in _cache:
        d = Path(tempfile.mkdtemp(prefix="tokviz_web_"))
        assert run_export_web.main(["--out", str(d)]) == 0
        _cache["dir"] = d
    return _cache["dir"]


def glb(name):
    key = f"glb_{name}"
    if key not in _cache:
        _cache[key] = EG.read_glb(out_dir() / f"{name}.glb")
    return _cache[key]


def nodes_by_name(js):
    return {n["name"]: (i, n) for i, n in enumerate(js["nodes"])}


def world_vertices(js, binb, pred):
    """World-space POSITION data of every mesh node whose part node satisfies pred."""
    W = EG.node_world_matrices(js)
    parent = {c: i for i, n in enumerate(js["nodes"]) for c in n.get("children", [])}
    out = []
    for i, n in enumerate(js["nodes"]):
        if "mesh" not in n:
            continue
        part = js["nodes"][parent[i]]
        if not pred(part.get("extras", {})):
            continue
        for pr in js["meshes"][n["mesh"]]["primitives"]:
            P = EG.accessor_array(js, binb, pr["attributes"]["POSITION"]).astype(float)
            out.append(P @ W[i][:3, :3].T + W[i][:3, 3])
    return np.vstack(out)


# ---------------------------------------------------------------------------
def test_container_and_accessors():
    for name in MACHINES:
        path = out_dir() / f"{name}.glb"
        raw = path.read_bytes()
        assert raw[:4] == b"glTF"
        js, binb = glb(name)                       # validates header / chunk types / alignment
        assert js["asset"]["version"] == "2.0"
        assert "extensionsRequired" not in js
        for ext in js.get("extensionsUsed", []):
            assert not re.search(r"draco|meshopt|texture_basisu|ktx", ext, re.I), ext
        assert js["buffers"][0]["byteLength"] == len(binb)
        for bv in js["bufferViews"]:
            assert bv["byteOffset"] % 4 == 0
            assert bv["byteOffset"] + bv["byteLength"] <= len(binb)
        for m in js["meshes"]:
            for pr in m["primitives"]:
                P = EG.accessor_array(js, binb, pr["attributes"]["POSITION"])
                a = js["accessors"][pr["attributes"]["POSITION"]]
                assert np.allclose(P.min(0), a["min"], atol=1e-6)
                assert np.allclose(P.max(0), a["max"], atol=1e-6)
                assert np.all(np.isfinite(P))
                if pr.get("mode", 4) == 4:
                    N = EG.accessor_array(js, binb, pr["attributes"]["NORMAL"])
                    assert len(N) == len(P)
                    assert np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-3)
                    I = EG.accessor_array(js, binb, pr["indices"])
                    assert len(I) % 3 == 0 and int(I.max()) < len(P)
                    ia = js["accessors"][pr["indices"]]
                    assert ia["min"][0] == int(I.min()) and ia["max"][0] == int(I.max())
                else:
                    assert pr["mode"] == 3                      # LINE_STRIP
                assert 0 <= pr["material"] < len(js["materials"])
        for n in js["nodes"]:
            if "matrix" in n:
                M = np.asarray(n["matrix"]).reshape(4, 4).T
                assert np.allclose(M[3], [0, 0, 0, 1])
                assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-6)


def test_node_tree_names_and_extras():
    for name in MACHINES:
        js, _ = glb(name)
        roots = js["scenes"][0]["nodes"]
        assert len(roots) == 1
        root = js["nodes"][roots[0]]
        assert root["extras"]["kind"] == "machine" and root["extras"]["up_axis"] == "+Y"
        groups = [js["nodes"][c] for c in root["children"]]
        assert [g["name"] for g in groups] == list(EG.GROUPS)
        names = [n["name"] for n in js["nodes"]]
        assert len(names) == len(set(names)), "node names must be unique"
        for n in names:
            assert re.fullmatch(r"[A-Za-z0-9_-]+", n), n     # survives three.js sanitising
        n_parts = 0
        for g in groups:
            assert g["extras"]["kind"] == "group" and g["extras"]["label_uk"]
            for c in g.get("children", []):
                p = js["nodes"][c]
                e = p["extras"]
                n_parts += 1
                for k in REQUIRED_PART_EXTRAS:
                    assert k in e, (p["name"], k)
                assert e["kind"] == "part" and e["subsystem"] == g["name"]
                assert p["name"] == e["id"]
                assert isinstance(e["sources"], list)
                for s in e["sources"]:
                    assert set(s) >= {"doc", "page", "cite"}
                assert isinstance(e["key_dims"], dict)
                kids = [js["nodes"][k] for k in p.get("children", [])]
                assert len(kids) == e["n_instances"] >= 1
                for k in kids:
                    assert "mesh" in k and k["extras"]["kind"] == "instance"
                    assert k["extras"]["part_id"] == p["name"]
        assert n_parts > 40
        # sourced parts really carry sources
        by = nodes_by_name(js)
        tf = by["tf_coils__casing"][1]["extras"]
        assert any(s["doc"] == "EUR 5516e" and s["page"] == 325 for s in tf["sources"])
        assert tf["key_dims"]["overall_height"]["value"] == 5.68
        assert tf["key_dims"]["overall_height"]["cite"] == "EUR 5516e, с. 327"


def test_group_contents_per_variant():
    js75, _ = glb("jet1975")
    js83, _ = glb("jet1983")
    b75, b83 = nodes_by_name(js75), nodes_by_name(js83)
    assert "limiters__upper_rail_plate" in b75 and "nbi__beam_duct" in b75
    assert "limiters__upper_rail_plate" not in b83 and "nbi__beam_duct" not in b83
    assert "limiter_C1__limiter_C1" in b83
    assert b83["limiter_C1__limiter_C1"][1]["extras"]["variant"] == "jet1983"
    rv = b83["rotary_valve_oct4__valve_body"][1]["extras"]
    assert rv["status"] == "not_installed" and rv["subsystem"] == "Extras1983"
    assert len(b83["vessel__rigid_sector_end_a"][1]["children"]) == 8


def test_tf_mesh_reuse():
    for name in MACHINES:
        js, _ = glb(name)
        i, part = nodes_by_name(js)["tf_coils__casing"]
        meshes = {js["nodes"][c]["mesh"] for c in part["children"]}
        assert len(meshes) == 1
        use = Counter(n["mesh"] for n in js["nodes"] if "mesh" in n)
        assert use[meshes.pop()] >= 30
        # the other repeated parts are shared too
        for pid in ("bellows__bellows_outer_ply", "iron_core__outer_limb", "pf_coils__coil1"):
            p = nodes_by_name(js)[pid][1]
            assert len({js["nodes"][c]["mesh"] for c in p["children"]}) == 1


def test_budget():
    for name in MACHINES:
        st = EG.glb_stats(out_dir() / f"{name}.glb")
        assert st["bytes"] <= EG.MAX_FILE_BYTES
        assert st["triangles_rendered"] <= EG.TRI_BUDGET, st["triangles_rendered"]
        assert 1000 <= st["line_points"] <= 5000
        p = st["groups"]["Plasma"]["triangles"]
        assert 7000 <= p <= 20000
    total = sum((out_dir() / f).stat().st_size for f in
                ("jet1975.glb", "jet1983.glb", "jet_meta.json"))
    assert total <= 16 * 1024 * 1024
    for f in ("jet1975.glb", "jet1983.glb"):                 # the published copies
        p = ROOT / "web" / f
        if p.exists():
            assert p.stat().st_size <= EG.MAX_FILE_BYTES


def test_dimensions_iron_and_plasma():
    for name in MACHINES:
        js, binb = glb(name)
        # laminated iron (limbs + radial arms); the steel foot plates stick out further
        iron = world_vertices(js, binb, lambda e: e.get("part") in (
            "outer_limb", "upper_radial_arm", "lower_radial_arm"))
        r = np.hypot(iron[:, 0], iron[:, 2])
        assert abs(2 * r.max() - 14.8) < 0.15, 2 * r.max()
        arms = world_vertices(js, binb, lambda e: e.get("part") in ("upper_radial_arm",
                                                                     "lower_radial_arm"))
        h = arms[:, 1].max() - arms[:, 1].min()
        assert abs(h - 11.5) < 0.15, h                       # Y-up: height along y
        lcfs = world_vertices(js, binb, lambda e: e.get("id") == "plasma__lcfs")
        rr = np.hypot(lcfs[:, 0], lcfs[:, 2])
        assert abs(0.5 * (rr.min() + rr.max()) - 2.96) < 0.01
        assert abs(0.5 * (rr.max() - rr.min()) - 1.25) < 0.01
        assert abs(lcfs[:, 1].max() - 2.10) < 0.02 and abs(lcfs[:, 1].min() + 2.10) < 0.02
        vessel = world_vertices(js, binb, lambda e: e.get("subsystem") == "Vessel")
        assert abs(vessel[:, 1].max() - 2.315) < 0.01        # outer skin top, kernel Z
        st = EG.glb_stats(out_dir() / f"{name}.glb")
        lo, hi = np.asarray(st["bbox"])
        assert np.all(np.abs(lo) < 9.5) and np.all(np.abs(hi) < 9.5)


def test_materials():
    for name in MACHINES:
        js, _ = glb(name)
        by = nodes_by_name(js)
        inst = js["nodes"][by["plasma__lcfs"][1]["children"][0]]
        mat = js["materials"][js["meshes"][inst["mesh"]]["primitives"][0]["material"]]
        assert mat["alphaMode"] == "BLEND" and mat["pbrMetallicRoughness"]["baseColorFactor"][3] < 1
        assert max(mat["emissiveFactor"]) > 0.5 and mat["doubleSided"]
        for m in js["materials"]:
            f = m["pbrMetallicRoughness"]
            assert len(f["baseColorFactor"]) == 4
            assert 0 <= f["metallicFactor"] <= 1 and 0 <= f["roughnessFactor"] <= 1


def test_meta():
    m = json.loads((out_dir() / "jet_meta.json").read_text())
    for name in MACHINES:
        subs = m["machines"][name]["subsystems"]
        assert list(subs) == list(EG.GROUPS)
        for gid, s in subs.items():
            assert s["label_uk"] and re.fullmatch(r"#[0-9a-f]{6}", s["colour"])
        assert subs["TF"]["key_dims"]["n_coils"]["value"] == 32
    g = m["global"]
    assert g["R0"]["value"] == 2.96 and g["B0"]["value"] == 2.77
    assert g["I_p"]["value"] == 3.8e6
    assert abs(g["ripple"]["model_pct"] - 3.67) < 0.01 and g["ripple"]["source_pct"] == 3.6
    assert "3,67 %" in g["ripple"]["text_uk"] and "3,6 %" in g["ripple"]["text_uk"]
    assert m["global"]["masses"]["jet1983"]["iron_core"]["value"] == 2.8e6
    assert len(m["changes_1983"]) >= 6


def test_trimesh_roundtrip():
    try:
        import trimesh
    except ImportError:                                     # optional cross-check
        return
    for name in MACHINES:
        sc = trimesh.load(out_dir() / f"{name}.glb", force="scene")
        tri = sum(len(sc.geometry[g].faces) for _, g in
                  (sc.graph[n] for n in sc.graph.nodes_geometry)
                  if hasattr(sc.geometry[g], "faces"))
        assert tri == EG.glb_stats(out_dir() / f"{name}.glb")["triangles_rendered"]


if __name__ == "__main__":
    fails = 0
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {k}")
            except Exception as exc:                        # noqa: BLE001
                fails += 1
                print(f"FAIL {k}: {exc!r}")
    raise SystemExit(1 if fails else 0)
