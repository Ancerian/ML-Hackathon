"""Animated cutaway rig.

Two mechanisms, both driven by a single 0 -> 1 parameter so the still set and
any later animation come from the same rig:

1. :func:`wedge_cut_group` -- Geometry Nodes, deletes faces inside a toroidal
   wedge.  Non-destructive, fast, exact, and it leaves the interior open, which
   is what a cutaway wants.  No caps.
2. :func:`boolean_cut` -- a Boolean modifier against a wedge solid.  Slower but
   produces capped, watertight sections.  Blender 5.x adds the 'MANIFOLD'
   solver, which is markedly more robust than 'EXACT' on generated geometry.

A third, shader-only variant lives in :func:`clip_plane_nodes` for real-time
use during a lecture, where modifier re-evaluation would stall the viewport.
"""
from __future__ import annotations

import math

import bpy

from .bl import sin, sout, IS_5X, set_modifier_input


def wedge_cut_group(name="TokWedgeCut"):
    """Geometry Nodes group: delete faces whose toroidal angle is in a wedge.

    Inputs: Geometry, Cut (0..1), Start (rad), Width (rad).
    ``Cut`` scales the wedge from nothing to its full ``Width``, so animating
    0 -> 1 opens the machine progressively.
    """
    g = bpy.data.node_groups.get(name)
    if g is not None:
        return g
    g = bpy.data.node_groups.new(name, "GeometryNodeTree")

    iface = g.interface
    s_geo_in = iface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_cut = iface.new_socket("Cut", in_out="INPUT", socket_type="NodeSocketFloat")
    s_start = iface.new_socket("Start", in_out="INPUT", socket_type="NodeSocketFloat")
    s_width = iface.new_socket("Width", in_out="INPUT", socket_type="NodeSocketFloat")
    iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    s_cut.default_value = 0.0
    s_cut.min_value = 0.0
    s_cut.max_value = 1.0
    s_start.default_value = 0.0
    s_width.default_value = math.pi / 2.0

    n = g.nodes
    gi = n.new("NodeGroupInput"); gi.location = (-900, 0)
    go = n.new("NodeGroupOutput"); go.location = (600, 0)

    pos = n.new("GeometryNodeInputPosition"); pos.location = (-900, -300)
    sep = n.new("ShaderNodeSeparateXYZ"); sep.location = (-720, -300)
    g.links.new(sout(pos, "Position"), sin(sep, "Vector"))

    ang = n.new("ShaderNodeMath"); ang.operation = "ARCTAN2"; ang.location = (-540, -300)
    ang.label = "phi = atan2(y, x)"
    g.links.new(sout(sep, "Y"), ang.inputs[0])
    g.links.new(sout(sep, "X"), ang.inputs[1])

    # rotate so the wedge starts at 'Start', then wrap into [0, 2pi)
    sub = n.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.location = (-360, -300)
    g.links.new(sout(ang, "Value"), sub.inputs[0])
    g.links.new(sout(gi, "Start"), sub.inputs[1])

    wrap = n.new("ShaderNodeMath"); wrap.operation = "WRAP"; wrap.location = (-180, -300)
    wrap.label = "wrap to [0, 2pi)"
    g.links.new(sout(sub, "Value"), wrap.inputs[0])
    wrap.inputs[1].default_value = 2.0 * math.pi
    wrap.inputs[2].default_value = 0.0

    # effective wedge = Cut * Width
    eff = n.new("ShaderNodeMath"); eff.operation = "MULTIPLY"; eff.location = (-360, -520)
    g.links.new(sout(gi, "Cut"), eff.inputs[0])
    g.links.new(sout(gi, "Width"), eff.inputs[1])

    less = n.new("ShaderNodeMath"); less.operation = "LESS_THAN"; less.location = (0, -400)
    less.label = "inside wedge?"
    g.links.new(sout(wrap, "Value"), less.inputs[0])
    g.links.new(sout(eff, "Value"), less.inputs[1])

    dele = n.new("GeometryNodeDeleteGeometry"); dele.location = (300, 0)
    dele.domain = "FACE"
    dele.mode = "ALL"
    g.links.new(sout(gi, "Geometry"), sin(dele, "Geometry"))
    g.links.new(sout(less, "Value"), sin(dele, "Selection"))
    g.links.new(sout(dele, "Geometry"), go.inputs[0])

    g["_sockets"] = {"Cut": s_cut.identifier, "Start": s_start.identifier,
                     "Width": s_width.identifier}
    return g


def apply_wedge_cut(objs, cut=0.0, start=0.0, width=math.pi / 2.0,
                    name="TokWedgeCut"):
    """Add the cutaway modifier to each object and set its parameters."""
    g = wedge_cut_group(name)
    ids = g["_sockets"]
    mods = []
    for ob in objs:
        m = ob.modifiers.get("Cutaway")
        if m is None:
            m = ob.modifiers.new("Cutaway", "NODES")
        m.node_group = g
        set_modifier_input(m, ids["Cut"], float(cut))
        set_modifier_input(m, ids["Start"], float(start))
        set_modifier_input(m, ids["Width"], float(width))
        mods.append(m)
    return mods


def set_cut(objs, cut):
    for ob in objs:
        m = ob.modifiers.get("Cutaway")
        if m is None or m.node_group is None:
            continue
        set_modifier_input(m, m.node_group["_sockets"]["Cut"], float(cut))
        ob.update_tag()


def wedge_solid(name, r_max, z_max, start=0.0, width=math.pi / 2.0, coll=None):
    """A solid wedge for the Boolean-based, capped cutaway."""
    import numpy as np
    from .bl import mesh_object
    a0, a1 = start, start + width
    R = r_max * 1.6
    Z = z_max * 1.6
    ring = [(0.0, 0.0), (R * math.cos(a0), R * math.sin(a0)),
            (R * math.cos(0.5 * (a0 + a1)), R * math.sin(0.5 * (a0 + a1))),
            (R * math.cos(a1), R * math.sin(a1))]
    verts = [(x, y, -Z) for x, y in ring] + [(x, y, Z) for x, y in ring]
    n = len(ring)
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    return mesh_object(name, np.array(verts), faces, coll, smooth=False)


def boolean_cut(ob, cutter, solver=None):
    """Capped section via a Boolean difference.

    Blender 5.x adds the 'MANIFOLD' solver, which handles generated,
    self-touching geometry far better than 'EXACT'.
    """
    m = ob.modifiers.new("CutawayBool", "BOOLEAN")
    m.operation = "DIFFERENCE"
    m.object = cutter
    if solver is None:
        solver = "MANIFOLD" if IS_5X else "EXACT"
    try:
        m.solver = solver
    except TypeError:
        m.solver = "EXACT"
    return m


def clip_plane_nodes(mat, normal=(0.0, 1.0, 0.0), offset=0.0):
    """Shader-only clipping, for real-time viewport use.

    Adds ``Geometry.Position -> Vector Math (Dot Product) -> Math (Less Than)``
    driving a Transparent BSDF mix.  Needs a transparent render method, so the
    material is switched to BLENDED.
    """
    from .bl import NodeBuilder, set_transparency
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    src = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].links else None
    if src is None:
        return mat
    nb = NodeBuilder(nt, x0=out.location[0] - 700, dy=-240)
    geo = nb.new("ShaderNodeNewGeometry", col=0, row=3, label="ClipGeom")
    dot = nb.new("ShaderNodeVectorMath", col=1, row=3, operation="DOT_PRODUCT",
                 label="ClipDot")
    nt.links.new(sout(geo, "Position"), dot.inputs[0])
    dot.inputs[1].default_value = tuple(normal)
    lt = nb.new("ShaderNodeMath", col=2, row=3, operation="LESS_THAN", label="ClipTest")
    nt.links.new(sout(dot, "Value"), lt.inputs[0])
    lt.inputs[1].default_value = float(offset)
    tr = nb.new("ShaderNodeBsdfTransparent", col=2, row=4, label="ClipTransparent")
    mix = nb.new("ShaderNodeMixShader", col=3, row=3, label="ClipMix")
    nt.links.new(sout(lt, "Value"), mix.inputs[0])
    nt.links.new(src, mix.inputs[1])
    nt.links.new(sout(tr, "BSDF"), mix.inputs[2])
    nt.links.new(sout(mix, "Shader"), out.inputs["Surface"])
    set_transparency(mat, "BLENDED")
    return mat
