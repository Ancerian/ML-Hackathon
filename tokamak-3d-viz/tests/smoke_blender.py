"""Blender-side smoke test: build every material and node group, check the API.

    blender --background --python tests/smoke_blender.py -- --bake data/bake01

Exits non-zero on failure, so it is usable as a CI gate.
"""
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "blender"))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
bake = argv[argv.index("--bake") + 1] if "--bake" in argv else "data/bake01"

import numpy as np
from tokblend import materials as M
from tokblend import cutaway as CU
from tokblend.bl import (eevee_engine_id, enable_cycles, set_modifier_input,
                         get_modifier_input, instance_points_group)
from tokblend import compositor as CO

fails = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        fails.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


print(f"Blender {bpy.app.version_string}, numpy {np.__version__}")
check("eevee engine id", lambda: eevee_engine_id())
check("cycles enable", lambda: enable_cycles())

gz = np.load(os.path.join(bake, "psi_n_grid.npz"))
R_rng = (float(gz["R_scaled"][0]), float(gz["R_scaled"][-1]))
Z_rng = (float(gz["Z_scaled"][0]), float(gz["Z_scaled"][-1]))

check("plasma volume material",
      lambda: M.plasma_volume("V", gz["psi_n"], R_rng, Z_rng))
for preset in ("tungsten", "beryllium", "steel", "copper"):
    check(f"metal:{preset}", lambda p=preset: M.metal_material(f"M_{p}", p))
check("flux surface material", lambda: M.flux_surface_material("F"))
check("emissive material", lambda: M.emissive_material("E"))
check("matte material", lambda: M.matte_material("Mt"))
check("wedge cut group", lambda: CU.wedge_cut_group("W"))
check("point instancer group", lambda: instance_points_group("P"))
check("compositor", lambda: CO.build())


def modifier_roundtrip():
    g = CU.wedge_cut_group("W")
    bpy.ops.mesh.primitive_cube_add()
    ob = bpy.context.active_object
    mods = CU.apply_wedge_cut([ob], cut=0.4)
    got = get_modifier_input(mods[0], g["_sockets"]["Cut"])
    assert abs(float(got) - 0.4) < 1e-6, f"round-trip gave {got!r}"


check("GN modifier input round-trip", modifier_roundtrip)


def no_undefined_nodes():
    bad = []
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        for n in mat.node_tree.nodes:
            if n.bl_idname == "NodeUndefined":
                bad.append(f"{mat.name}/{n.name}")
    for g in bpy.data.node_groups:
        for n in g.nodes:
            if n.bl_idname == "NodeUndefined":
                bad.append(f"{g.name}/{n.name}")
    assert not bad, f"undefined nodes: {bad}"


check("no undefined nodes", no_undefined_nodes)


def links_resolve():
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        for lk in mat.node_tree.links:
            assert lk.is_valid, f"{mat.name}: invalid link"


check("all links valid", links_resolve)

print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'ALL PASS'}")
sys.exit(1 if fails else 0)
