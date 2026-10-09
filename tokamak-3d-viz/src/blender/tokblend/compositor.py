"""Compositor: glare on the plasma, plus the Cryptomatte passes the
annotation layer masks against.

Blender 5.x reworked this area and the old recipes do not run:

============================  =========================  =====================
thing                         4.2 LTS                     5.x
============================  =========================  =====================
tree location                 ``Scene.node_tree``         ``Scene.compositing_node_group``
final output node             ``CompositorNodeComposite`` ``NodeGroupOutput``
mix / math / ramp             ``CompositorNodeMixRGB``    ``ShaderNodeMix``
                              ``CompositorNodeMath``      ``ShaderNodeMath``
Glare settings                node properties             input SOCKETS
                              (``glare_type`` etc.)       ('Type', 'Size', ...)
============================  =========================  =====================

EEVEE also lost its built-in Bloom at 4.2, so glare is no longer optional
polish -- it is the only way to get plasma bloom in the real-time path.
"""
from __future__ import annotations

import bpy

from .bl import sin, sout, IS_5X


#: Glare 'Type' is a MENU socket in 5.x and takes these exact display strings
#: -- not the uppercase enum identifiers the 4.2 node property used.
GLARE_TYPES = ("Bloom", "Ghosts", "Streaks", "Fog Glow", "Simple Star",
               "Sun Beams", "Kernel")
#: 4.2 node-property spelling, for the legacy branch
GLARE_TYPES_LEGACY = {"Bloom": "FOG_GLOW", "Ghosts": "GHOSTS",
                      "Streaks": "STREAKS", "Fog Glow": "FOG_GLOW",
                      "Simple Star": "SIMPLE_STAR"}


def build(glare_type="Bloom", size=0.35, strength=0.18, threshold=1.6,
          quality="High"):
    """Wire Render Layers -> Glare -> output, on whichever API this build has."""
    sc = bpy.context.scene
    sc.render.use_compositing = True

    if IS_5X and hasattr(sc, "compositing_node_group"):
        tree = bpy.data.node_groups.get("TokComp")
        if tree is None:
            tree = bpy.data.node_groups.new("TokComp", "CompositorNodeTree")
            tree.interface.new_socket("Image", in_out="OUTPUT",
                                      socket_type="NodeSocketColor")
        sc.compositing_node_group = tree
        out = next((n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"), None)
        if out is None:
            out = tree.nodes.new("NodeGroupOutput")
        out.location = (600, 0)
    else:                                        # 4.x
        sc.use_nodes = True
        tree = sc.node_tree
        out = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"), None)
        if out is None:
            out = tree.nodes.new("CompositorNodeComposite")
        out.location = (600, 0)

    rl = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"), None)
    if rl is None:
        rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)

    gl = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeGlare"), None)
    if gl is None:
        gl = tree.nodes.new("CompositorNodeGlare")
    gl.location = (100, 0)

    # 5.x exposes Glare through sockets; 4.2 through node properties.
    if any(s.name == "Type" for s in gl.inputs):          # 5.x: sockets
        _set_menu_socket(gl, "Type", glare_type)
        _set_menu_socket(gl, "Quality", quality)
        # NOTE: 'Size' is a 0..1 FLOAT here.  In 4.2 it was an integer number
        # of blur steps (1..9); passing 8 to the 5.x socket gives an enormous,
        # obviously wrong glare.
        for nm, val in (("Size", float(size)),
                        ("Strength", float(strength)),
                        ("Threshold", float(threshold))):
            for sk in gl.inputs:
                if sk.name == nm:
                    sk.default_value = val
                    break
    else:                                                 # 4.2: properties
        gl.glare_type = GLARE_TYPES_LEGACY.get(glare_type, "FOG_GLOW")
        gl.quality = quality.upper()
        gl.threshold = float(threshold)
        gl.mix = float(strength) * 2.0 - 1.0
        gl.size = max(1, min(9, int(round(size * 9))))

    tree.links.new(sout(rl, "Image"), sin(gl, "Image"))
    tree.links.new(sout(gl, "Image"), out.inputs[0])
    return tree


def _set_menu_socket(node, name, value):
    """Set a NodeSocketMenu by its display string.

    These sockets report an EMPTY ``enum_items`` (the menu is built
    dynamically), so the accepted values cannot be discovered by
    introspection -- they must be the exact display strings.
    """
    for s in node.inputs:
        if s.name != name:
            continue
        try:
            s.default_value = value
        except (TypeError, ValueError) as e:
            print(f"[compositor] {name}={value!r} rejected: {e}")
        return
