"""Section / cutaway rig for the engineering (JET) scene -- shader based.

Why a shader rig here and not the Geometry Nodes wedge of :mod:`cutaway`
-----------------------------------------------------------------------
The machine bake is built from LINKED DUPLICATES: one mesh in a LOCAL frame,
many objects carrying world transforms.  A GN wedge evaluates in object space,
so it would cut each TF coil / sector at its own local angle.  A world-space
test in the shader is exact per pixel, needs no per-object parameters, costs
nothing to re-evaluate between shots, and works identically in EEVEE and
Cycles.  The wedge maths (``phi = atan2(y, x)``, ``wrap(phi - Start, 2 pi) <
Width``) is the same as :func:`tokblend.cutaway.wedge_cut_group`, so the two
rigs agree on angles; ``Start``/``Width`` are in radians as there.

One shader node group ``TokJetCut`` (output ``Keep``: 1 keep, 0 clip) is
inserted in every machine material.  Its parameters are SCENE custom
properties ``tok_<name>`` read by View-Layer Attribute nodes (GPU uniforms:
animatable, no shader recompilation), so a single assignment re-cuts the
whole scene:

    WedgeOn, WedgeStart, WedgeWidth      toroidal wedge (octant cutaway)
    PlaneOn, PlaneN (vector), PlaneD     half-space  n.P > d  is removed

Per-object behaviour comes from custom properties read by Attribute(OBJECT):

    tok_nocut    1 -> the wedge never removes it (central column)
    tok_explode  1 -> INVERTED wedge: keep only what is inside it (the
                      exploded octant duplicates)
    tok_off      world offset of an exploded duplicate; subtracted before the
                 test so a displaced piece is still trimmed at its home angle

Section fill
------------
Surfaces seen from BEHIND (``Geometry.Backfacing``) are shaded with a flat
section colour -- but only while a cut is active (output ``CutOn``); without a
cut a back face can only be the inside of an opening, shaded as dark depth.  Every closed solid in the bake has outward normals, so after a
cut the inside of the far wall reads as a filled section face (a "cap") without
any Boolean.  This is the classic technical-illustration trick.
"""
from __future__ import annotations

import math

import bpy

from .bl import sin, sout, set_transparency

GROUP = "TokJetCut"
PARAMS = ("WedgeOn", "WedgeStart", "WedgeWidth", "PlaneOn", "PlaneD", "Drawing")

#: colour of a back face seen WITHOUT a cut (open port bores, gaps): a dark
#: cavity, never the section code (phase F5 fix -- port holes read as filled)
DEPTH_COLOR = (0.012, 0.012, 0.014)


def cut_group():
    g = bpy.data.node_groups.get(GROUP)
    if g is not None:
        return g
    g = bpy.data.node_groups.new(GROUP, "ShaderNodeTree")
    g.interface.new_socket("Keep", in_out="OUTPUT", socket_type="NodeSocketFloat")
    # 1 when a back face can be a cut face (a cut is on, or exploded piece)
    g.interface.new_socket("CutOn", in_out="OUTPUT", socket_type="NodeSocketFloat")
    # 1 = technical-illustration style (flat colour-coded faces)
    g.interface.new_socket("Drawing", in_out="OUTPUT", socket_type="NodeSocketFloat")
    n, L = g.nodes, g.links

    def val(name, v, x, y):
        # parameter = SCENE custom property "tok_<name>" read through a
        # View-Layer attribute: a GPU uniform, so changing / animating it
        # never forces EEVEE to recompile the machine shaders (a Value node
        # is a compile-time constant: 15-130 s per animation frame).
        at = n.new("ShaderNodeAttribute")
        at.attribute_type = "VIEW_LAYER"
        at.attribute_name = PROP + name
        at.location = (x - 180, y)
        nd = n.new("ShaderNodeMath")
        nd.operation = "ADD"
        nd.inputs[1].default_value = 0.0
        nd.name = nd.label = name
        nd.location = (x, y)
        L.new(sout(at, "Factor"), nd.inputs[0])
        return nd

    def math_(op, x, y, a=None, b=None, label=None, clamp=False):
        nd = n.new("ShaderNodeMath")
        nd.operation = op
        nd.location = (x, y)
        nd.use_clamp = clamp
        if label:
            nd.label = label
        for i, s in enumerate((a, b)):
            if s is None:
                continue
            if isinstance(s, (int, float)):
                nd.inputs[i].default_value = float(s)
            else:
                L.new(s, nd.inputs[i])
        return nd

    def attr(name, x, y):
        nd = n.new("ShaderNodeAttribute")
        nd.attribute_type = "OBJECT"
        nd.attribute_name = name
        nd.location = (x, y)
        nd.label = name
        return nd

    geo = n.new("ShaderNodeNewGeometry"); geo.location = (-1600, 0)
    a_off = attr("tok_off", -1600, -300)
    a_noc = attr("tok_nocut", -1600, -500)
    a_exp = attr("tok_explode", -1600, -700)

    p = n.new("ShaderNodeVectorMath"); p.operation = "SUBTRACT"; p.location = (-1350, 0)
    p.label = "P - tok_off"
    L.new(sout(geo, "Position"), p.inputs[0])
    L.new(sout(a_off, "Vector"), p.inputs[1])
    sep = n.new("ShaderNodeSeparateXYZ"); sep.location = (-1150, 0)
    L.new(p.outputs[0], sin(sep, "Vector"))

    w_on = val("WedgeOn", 0.0, -1150, 300)
    w_st = val("WedgeStart", 0.0, -1150, 200)
    w_wd = val("WedgeWidth", math.pi / 2, -1150, 100)
    p_on = val("PlaneOn", 0.0, -1150, -400)
    p_d = val("PlaneD", 0.0, -1150, -500)
    p_n = n.new("ShaderNodeAttribute"); p_n.name = p_n.label = "PlaneN"
    p_n.attribute_type = "VIEW_LAYER"
    p_n.attribute_name = PROP + "PlaneN"
    p_n.location = (-1150, -650)

    # --- wedge ----------------------------------------------------------------
    phi = math_("ARCTAN2", -950, 0, sout(sep, "Y"), sout(sep, "X"), "phi")
    rel = math_("SUBTRACT", -780, 0, phi.outputs[0], w_st.outputs[0])
    wrap = n.new("ShaderNodeMath"); wrap.operation = "WRAP"; wrap.location = (-610, 0)
    L.new(rel.outputs[0], wrap.inputs[0])
    wrap.inputs[1].default_value = 2.0 * math.pi
    wrap.inputs[2].default_value = 0.0
    inside = math_("LESS_THAN", -440, 0, wrap.outputs[0], w_wd.outputs[0], "inside wedge")

    not_noc = math_("SUBTRACT", -610, -300, 1.0, sout(a_noc, "Factor"), "1-nocut")
    not_exp = math_("SUBTRACT", -610, -450, 1.0, sout(a_exp, "Factor"), "1-explode")
    c1 = math_("MULTIPLY", -270, 0, inside.outputs[0], w_on.outputs[0])
    c2 = math_("MULTIPLY", -110, 0, c1.outputs[0], not_noc.outputs[0])
    clip_main = math_("MULTIPLY", 50, 0, c2.outputs[0], not_exp.outputs[0], "clip main")
    outside = math_("SUBTRACT", -270, -200, 1.0, inside.outputs[0], "outside wedge")
    clip_exp = math_("MULTIPLY", 50, -200, outside.outputs[0], sout(a_exp, "Factor"),
                     "clip exploded")

    # --- plane ----------------------------------------------------------------
    dot = n.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    dot.location = (-950, -550)
    L.new(p.outputs[0], dot.inputs[0])
    L.new(sout(p_n, "Vector"), dot.inputs[1])
    side = math_("GREATER_THAN", -610, -600, sout(dot, "Value"), p_d.outputs[0], "n.P > d")
    clip_pl = math_("MULTIPLY", 50, -500, side.outputs[0], p_on.outputs[0], "clip plane")

    s1 = math_("ADD", 220, -100, clip_main.outputs[0], clip_exp.outputs[0])
    s2 = math_("ADD", 380, -200, s1.outputs[0], clip_pl.outputs[0], clamp=True)
    keep = math_("SUBTRACT", 540, -200, 1.0, s2.outputs[0], "Keep")
    go = n.new("NodeGroupOutput"); go.location = (720, -200)
    L.new(keep.outputs[0], go.inputs[0])
    on1 = math_("MAXIMUM", 220, -800, w_on.outputs[0], p_on.outputs[0])
    on2 = math_("MAXIMUM", 380, -800, on1.outputs[0], sout(a_exp, "Factor"), "CutOn")
    L.new(on2.outputs[0], go.inputs[1])
    drw = val("Drawing", 0.0, 380, -1000)
    L.new(drw.outputs[0], go.inputs[2])
    _defaults()
    return g


PROP = "tok_"
_DEFAULTS = {"WedgeOn": 0.0, "WedgeStart": 0.0, "WedgeWidth": math.pi / 2,
             "PlaneOn": 0.0, "PlaneD": 0.0, "Drawing": 0.0}


def _scene():
    return bpy.context.scene


def _defaults():
    sc = _scene()
    for k, v in _DEFAULTS.items():
        if PROP + k not in sc.keys():
            sc[PROP + k] = float(v)
    if PROP + "PlaneN" not in sc.keys():
        sc[PROP + "PlaneN"] = (0.0, 1.0, 0.0)


def _set(k, v):
    _scene()[PROP + k] = v


def set_cut(wedge=None, plane=None):
    """Configure the rig (scene custom properties ``tok_*``).

    ``wedge`` = (start_rad, width_rad) or None (off);
    ``plane`` = ((nx, ny, nz), d) or None (off).  Removes ``n.P > d``.
    """
    cut_group()
    if wedge is None:
        _set("WedgeOn", 0.0)
    else:
        _set("WedgeOn", 1.0)
        _set("WedgeStart", float(wedge[0]))
        _set("WedgeWidth", float(wedge[1]))
    if plane is None:
        _set("PlaneOn", 0.0)
    else:
        (nx, ny, nz), d = plane
        _set("PlaneOn", 1.0)
        _set("PlaneN", (float(nx), float(ny), float(nz)))
        _set("PlaneD", float(d))
    _scene().update_tag()


def set_drawing(on):
    """Switch every wrapped material to the technical-illustration style."""
    cut_group()
    _set("Drawing", 1.0 if on else 0.0)


def get_cut():
    cut_group()
    sc = _scene()
    return {k: float(sc[PROP + k]) for k in PARAMS}


def plane_normal():
    cut_group()
    return tuple(_scene()[PROP + "PlaneN"])


def wrap_material(mat, section_color=(0.55, 0.12, 0.10), section_strength=0.9,
                  section=True):
    """Insert section fill + cut clipping in front of the material output.

    ``section=False`` for translucent shells (flux surfaces): clip only."""
    if mat.get("tok_cut_wrapped"):
        return mat
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    if not out.inputs["Surface"].links:
        return mat
    src = out.inputs["Surface"].links[0].from_socket
    x, y = out.location
    L = nt.links

    grp = nt.nodes.new("ShaderNodeGroup"); grp.location = (x - 350, y + 450)
    grp.node_tree = cut_group()
    grp.label = grp.name = "TokJetCut"

    if section:
        geo = nt.nodes.new("ShaderNodeNewGeometry"); geo.location = (x - 900, y + 400)
        geo.label = "CutBackfacing"
        sec = nt.nodes.new("ShaderNodeEmission"); sec.location = (x - 900, y + 150)
        sec.label = sec.name = "SectionFill"
        sec.inputs["Color"].default_value = (*section_color, 1.0)
        sec.inputs["Strength"].default_value = float(section_strength)
        # back face without any cut = the dark depth of an opening
        dep = nt.nodes.new("ShaderNodeEmission"); dep.location = (x - 900, y - 50)
        dep.label = dep.name = "DepthFill"
        dep.inputs["Color"].default_value = (*DEPTH_COLOR, 1.0)
        dep.inputs["Strength"].default_value = 1.0
        mixd = nt.nodes.new("ShaderNodeMixShader"); mixd.location = (x - 700, y + 100)
        mixd.label = "CutOn ? section : depth"
        L.new(grp.outputs["CutOn"], mixd.inputs[0])
        L.new(sout(dep, "Emission"), mixd.inputs[1])
        L.new(sout(sec, "Emission"), mixd.inputs[2])
        # technical-illustration front face: flat, lightened section colour
        # with a faint facing term so shapes stay legible
        tint = tuple(0.42 * c + 0.58 for c in section_color)
        lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.location = (x - 1100, y + 650)
        lw.inputs["Blend"].default_value = 0.5
        fr = nt.nodes.new("ShaderNodeMapRange"); fr.location = (x - 900, y + 650)
        fr.inputs["From Min"].default_value = 0.0
        fr.inputs["From Max"].default_value = 1.0
        fr.inputs["To Min"].default_value = 1.0
        fr.inputs["To Max"].default_value = 0.78
        L.new(sout(lw, "Facing"), fr.inputs["Value"])
        flat = nt.nodes.new("ShaderNodeEmission"); flat.location = (x - 700, y + 650)
        flat.label = flat.name = "DrawingFlat"
        flat.inputs["Color"].default_value = (*tint, 1.0)
        L.new(fr.outputs["Result"], flat.inputs["Strength"])
        mixf = nt.nodes.new("ShaderNodeMixShader"); mixf.location = (x - 500, y + 500)
        mixf.label = "Drawing ? flat : shaded"
        L.new(grp.outputs["Drawing"], mixf.inputs[0])
        L.new(src, mixf.inputs[1])
        L.new(sout(flat, "Emission"), mixf.inputs[2])
        mixb = nt.nodes.new("ShaderNodeMixShader"); mixb.location = (x - 350, y + 200)
        mixb.label = "Backface=section"
        L.new(sout(geo, "Backfacing"), mixb.inputs[0])
        L.new(mixf.outputs[0], mixb.inputs[1])
        L.new(mixd.outputs[0], mixb.inputs[2])
        body = mixb.outputs[0]
    else:
        body = src

    tr = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.location = (x - 350, y - 50)
    mixc = nt.nodes.new("ShaderNodeMixShader"); mixc.location = (x - 150, y + 200)
    mixc.label = "Clip"
    L.new(grp.outputs["Keep"], mixc.inputs[0])
    L.new(sout(tr, "BSDF"), mixc.inputs[1])
    L.new(body, mixc.inputs[2])
    L.new(mixc.outputs[0], out.inputs["Surface"])

    if section:
        set_transparency(mat, "DITHERED")      # binary alpha -> exact, no noise
    mat.use_backface_culling = False
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    mat["tok_cut_wrapped"] = 1
    return mat
