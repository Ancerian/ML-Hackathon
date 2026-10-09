"""Blender-side smoke test for the JET engineering scene (phase F3).

    blender --background --factory-startup --python tests/smoke_blender_jet.py -- \
        --bake data/bake_jet1975 [--variant-bake data/bake_jet1983]

Builds the scene in memory (no render, nothing written) and checks:
scene builds; one collection per subsystem; the 32 TF coils are linked
duplicates of ONE mesh; every machine object has a mapped material; bellows
carry Solidify with the recorded thickness; cameras C1..C6 exist (C5/C6
orthographic); the cut rig is present, wired into every machine material and
round-trips; the exploded variant shares meshes; the variant swap and the
strict-year filter work.  Exits non-zero on failure (CI gate).
"""
import json
import math
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "blender"))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
bake = argv[argv.index("--bake") + 1] if "--bake" in argv else "data/bake_jet1975"
variant = argv[argv.index("--variant-bake") + 1] if "--variant-bake" in argv else None

import build_jet_scene as B                      # noqa: E402
from tokblend import jet_cut as JC               # noqa: E402

fails = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        fails.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


cli = ["--bake", bake] + (["--variant-bake", variant, "--strict-year", "1983"]
                          if variant else [])
args = B.parse_args(cli)
S = {}


def build():
    S["js"] = B.JetScene(args, log=lambda *a: None).build()


check("scene builds", build)
if "js" not in S:
    print("FAILED: scene did not build")
    sys.exit(1)
js = S["js"]
base = js.tables[0]
man = base.manifest


def subsystems():
    need = {"TF", "Vessel", "Bellows", "Ports", "PF", "IronCore", "Structure",
            "Limiters", "NBI", "Plasma", "FieldLines", "Human"}
    have = {c.get("tok_subsystem") for c in bpy.data.collections
            if c.get("tok_machine") == base.name}
    miss = need - have
    assert not miss, f"missing subsystem collections: {sorted(miss)}"
    for s in need:
        assert s in bpy.data.collections, f"collection {s!r} not named plainly"


check("one collection per subsystem", subsystems)


def tf_linked():
    tf = [o for o in base.colls["TF"].objects if o.get("tok_part") == "casing"]
    n_expect = man["components"]["tf_coils"]["parts"]["casing"]["n_instances"]
    assert len(tf) == n_expect == 32, f"{len(tf)} TF casings (manifest {n_expect})"
    meshes = {o.data.name for o in tf}
    assert len(meshes) == 1, f"TF casings use {len(meshes)} meshes, expected 1"
    assert tf[0].data.users >= 32, f"mesh users {tf[0].data.users}"
    phis = sorted(round(math.degrees(math.atan2(o.matrix_world[1][0],
                                                o.matrix_world[0][0])) % 360, 2) for o in tf)
    steps = {round(b - a, 2) for a, b in zip(phis, phis[1:])}
    assert steps == {11.25}, f"TF spacing {steps}"


check("32 TF coils = linked duplicates of one mesh, 11.25 deg apart", tf_linked)


def instance_counts():
    for comp, c in man["components"].items():
        want = c["n_instances_total"]
        got = sum(1 for o in base.objects if o["tok_component"] == comp)
        assert got == want, f"{comp}: {got} objects vs manifest {want}"


check("object count == manifest instances, every component", instance_counts)


def materials():
    bad, unm = [], []
    for o in [o for tb in js.tables for o in tb.objects]:
        if not o.data.materials or o.data.materials[0] is None:
            bad.append(o.name)
        elif o.data.materials[0].name.startswith("JET_Unmapped"):
            unm.append((o["tok_material_tag"], o.name))
    assert not bad, f"{len(bad)} objects without material, e.g. {bad[:3]}"
    assert not unm, f"unmapped material tags: {sorted(set(t for t, _ in unm))}"
    tags = {o["tok_material_tag"]: o.data.materials[0].name for o in base.objects}
    assert "Copper" in tags["copper_epoxy"], tags["copper_epoxy"]
    assert "Inconel" in tags["inconel600"], tags["inconel600"]
    assert "Iron" in tags["iron_laminated"], tags["iron_laminated"]


check("materials assigned and mapped", materials)


def provenance_props():
    o = next(o for o in base.objects if o["tok_component"] == "tf_coils")
    for k in ("tok_component", "tok_sources", "tok_pages", "tok_confidence",
              "tok_status_tag", "tok_subsystem"):
        assert k in o.keys(), f"missing {k}"
    assert 325 in list(o["tok_pages"]), list(o["tok_pages"])
    prov = json.loads(o.data["tok_provenance_json"])
    assert any(e.get("page") == 83 for e in prov)


check("provenance custom properties", provenance_props)


def bellows():
    b = base.colls["Bellows"].objects
    assert len(b) == 64, len(b)
    for o in b:
        m = o.modifiers.get("Solidify")
        assert m is not None, o.name
        assert abs(m.thickness - o["tok_thin_shell_m"]) < 1e-9
    assert abs(b[0].modifiers["Solidify"].thickness - 0.0017) < 1e-6


check("bellows thin shells -> Solidify(recorded thickness)", bellows)


def cameras():
    for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
        cam = js.cams[base.label][k]
        assert cam.type == "CAMERA", k
    assert js.cams[base.label]["C5"].data.type == "ORTHO"
    assert js.cams[base.label]["C6"].data.type == "ORTHO"
    assert js.cams[base.label]["C3"].data.lens <= 16, "C3 should be wide"
    # C3 eye must be inside the vessel (between the inner and outer wall)
    e = js.cams[base.label]["C3"].location
    r = math.hypot(e.x, e.y)
    assert 1.7 < r < 4.3 and abs(e.z) < 1.5, (r, e.z)


check("cameras C1..C6 exist (C5/C6 ortho, C3 in-vessel wide)", cameras)


def cut_rig():
    g = bpy.data.node_groups.get(JC.GROUP)
    assert g is not None, "no TokJetCut group"
    mach_mats = {o.data.materials[0] for o in base.objects
                 if o["tok_subsystem"] != "Plasma" and o.data.materials}
    unwrapped = [m.name for m in mach_mats if not m.get("tok_cut_wrapped")]
    assert not unwrapped, f"materials without cut rig: {unwrapped}"
    JC.set_cut(wedge=(0.3, 1.1), plane=None)
    c = JC.get_cut()
    assert c["WedgeOn"] == 1.0 and abs(c["WedgeStart"] - 0.3) < 1e-6 and c["PlaneOn"] == 0
    JC.set_cut(None, ((0.0, 0.0, 1.0), 0.25))
    c = JC.get_cut()
    assert c["WedgeOn"] == 0.0 and c["PlaneOn"] == 1.0 and abs(c["PlaneD"] - 0.25) < 1e-6
    nocut = [o for o in base.objects if o.get("tok_nocut", 0) > 0.5]
    assert any("central_pillar" in o["tok_part"] for o in nocut)
    assert any(o["tok_part"].startswith("coil1") for o in nocut)
    assert not any(o["tok_subsystem"] == "TF" for o in nocut)
    # every shot applies without error
    for k in B.SHOTS:
        js.apply_shot(k)


check("cut rig present, wired, round-trips; all shots apply", cut_rig)


def f5_rig():
    """F5: cut parameters are scene uniforms (animatable without shader
    recompiles); back faces without a cut shade as dark depth; technical
    style switch; TF = steel case outside, copper on cut faces."""
    sc = bpy.context.scene
    JC.set_cut(wedge=(0.1, 0.2), plane=None)
    assert abs(sc["tok_WedgeStart"] - 0.1) < 1e-9 and sc["tok_WedgeOn"] == 1.0
    g = bpy.data.node_groups[JC.GROUP]
    assert not [n for n in g.nodes if n.bl_idname == "ShaderNodeValue"], \
        "cut parameters must not be Value nodes (EEVEE recompiles per change)"
    vl = [n for n in g.nodes if n.bl_idname == "ShaderNodeAttribute"
          and n.attribute_type == "VIEW_LAYER"]
    assert len(vl) >= 7, len(vl)
    assert {"Keep", "CutOn", "Drawing"} <= {s.name for s in g.interface.items_tree
                                           if getattr(s, "in_out", "") == "OUTPUT"}
    JC.set_drawing(True)
    assert sc["tok_Drawing"] == 1.0
    JC.set_drawing(False)
    mats = {o.data.materials[0] for o in base.objects
            if o["tok_subsystem"] not in ("Human", "Plasma")}
    for m in mats:
        nt = m.node_tree
        assert "DepthFill" in nt.nodes and "DrawingFlat" in nt.nodes, m.name
    tf = next(o for o in base.objects if o["tok_part"] == "casing").data.materials[0]
    assert "SteelCase" in tf.name, tf.name
    sec = tf.node_tree.nodes["SectionFill"].inputs["Color"].default_value
    assert sec[0] > 0.6 and sec[2] < 0.2, f"TF cut face not copper: {tuple(sec)}"
    # the in-vessel camera sits near the OUTBOARD midplane
    e = js.cams[base.label]["C3"].location
    assert math.hypot(e.x, e.y) > 3.5, "C3 eye should be near the outboard wall"
    # C6 places the scale figure next to the iron (R ~ 8 m)
    assert 7.4 < math.hypot(*js.c6_human) < 8.5, js.c6_human
    for k in B.SHOTS:
        js.apply_shot(k)
    js.apply_shot("C1")
    h = base.objects_in("Human")[0]
    assert abs(math.hypot(h.location.x, h.location.y) - base.manifest["components"]
               ["human"]["info"]["R"]) < 0.05, "figure not restored after C6"


check("F5 rig: uniform cut params, depth fill, drawing style, TF case, C3/C6", f5_rig)


def exploded():
    ex = js.exploded[base.label]
    objs = list(ex.objects)
    assert len(objs) > 20, len(objs)
    for d in objs:
        src = bpy.data.objects[d["tok_exploded_from"]]
        assert d.data is src.data, "exploded duplicate does not share the mesh"
        assert d["tok_explode"] == 1.0
    # base objects untouched
    assert all(o.get("tok_explode", 0) == 0 for o in base.objects)


check("exploded octant = non-destructive linked duplicates", exploded)


def plasma():
    pl = js.plasmas[base.label]
    vol = pl.volume_objs
    assert vol and vol[0].data.materials, "no plasma volume"
    out = next(n for n in vol[0].data.materials[0].node_tree.nodes
               if n.bl_idname == "ShaderNodeOutputMaterial")
    assert out.inputs["Volume"].links, "plasma material has no volume output"
    if len(js.tables) > 1:
        pv = js.plasmas[js.tables[1].label]
        assert pv.source.startswith("physics"), f"variant plasma: {pv.source}"
    print(f"        psi_N source: {pl.source}; flux surfaces {len(pl.surfaces)}, "
          f"field-line objects {len(pl.lines)}")


check("plasma volume with psi_N gradient", plasma)


def variant_swap():
    if len(js.tables) < 2:
        print("        (no --variant-bake: swap tested on one machine only)")
        js.set_active("variant")
        assert js.active is base
        return
    v = js.tables[1]
    js.set_active("variant")
    assert js.active is v and base.root.hide_render and not v.root.hide_render
    hidden = [o for o in v.objects if o.get("tok_strict_hidden")]
    tags = {o["tok_status_tag"] for o in hidden}
    assert hidden and all(o.hide_render for o in hidden), "strict-1983 hid nothing"
    assert any("rotary_valve" in o["tok_component"] for o in hidden), tags
    js.set_active("base")
    assert js.active is base and v.root.hide_render


check("variant swap + strict-1983 status filter", variant_swap)


def no_undefined_nodes():
    bad = []
    for mat in bpy.data.materials:
        if mat.node_tree:
            bad += [f"{mat.name}/{n.name}" for n in mat.node_tree.nodes
                    if n.bl_idname == "NodeUndefined"]
            assert all(lk.is_valid for lk in mat.node_tree.links), mat.name
    for g in bpy.data.node_groups:
        bad += [f"{g.name}/{n.name}" for n in g.nodes if n.bl_idname == "NodeUndefined"]
    assert not bad, bad


check("no undefined nodes, all links valid", no_undefined_nodes)

print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'ALL PASS'}")
sys.exit(1 if fails else 0)
