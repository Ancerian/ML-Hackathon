"""Version-tolerant helpers over the Blender Python API.

Verified by introspection against Blender 5.2.1 LTS; every place where 4.2 LTS
differs is handled here rather than scattered through the build scripts.

Differences handled
-------------------
* render engine id: ``BLENDER_EEVEE`` (5.x) vs ``BLENDER_EEVEE_NEXT`` (4.2)
* material transparency: ``surface_render_method`` (4.2+) vs ``blend_method``
* compositor: ``Scene.compositing_node_group`` + ``NodeGroupOutput`` (5.x) vs
  ``Scene.node_tree`` + ``CompositorNodeComposite`` (4.x)
* nodes with repeated socket names (Mix, Map Range) -- resolved by filtering on
  ``socket.enabled`` instead of guessing indices
"""
from __future__ import annotations

import bpy

IS_5X = bpy.app.version[0] >= 5


# ---------------------------------------------------------------------------
# sockets
# ---------------------------------------------------------------------------
#: Socket names that were renamed between versions.  Each key maps to the
#: alternatives to try, so one node tree builds on 4.2 LTS and 5.x alike.
#:  * 'Fac' -> 'Factor'            : renamed in 5.0 (ColorRamp, Mix Shader,
#:                                   Noise/Voronoi/Attribute outputs)
#:  * Principled BSDF v2 renames   : landed in 4.0
_ALIASES = {
    "Fac": ("Factor",),
    "Factor": ("Fac",),
    "Specular": ("Specular IOR Level",),
    "Specular IOR Level": ("Specular",),
    "Subsurface": ("Subsurface Weight",),
    "Subsurface Weight": ("Subsurface",),
    "Transmission": ("Transmission Weight",),
    "Transmission Weight": ("Transmission",),
    "Clearcoat": ("Coat Weight",),
    "Coat Weight": ("Clearcoat",),
    "Sheen": ("Sheen Weight",),
    "Sheen Weight": ("Sheen",),
    "Emission": ("Emission Color",),
    "Emission Color": ("Emission",),
}


def _pick(sockets, name, idx, what, node):
    for cand in (name,) + _ALIASES.get(name, ()):
        hits = [s for s in sockets if s.name == cand and s.enabled]
        if not hits:
            hits = [s for s in sockets if s.name == cand]
        if hits:
            return hits[min(idx, len(hits) - 1)]
    raise KeyError(
        f"{node.bl_idname} has no {what} {name!r}; available: "
        f"{[s.name for s in sockets]}"
    )


def sin(node, name, idx=0):
    """The ``idx``-th ENABLED input socket called ``name``.

    ``ShaderNodeMix`` and ``ShaderNodeMapRange`` expose one socket per data
    type under the same name; only the ones matching the node's current
    ``data_type`` are enabled, so filtering on ``enabled`` is the only safe way
    to address them by name.
    """
    return _pick(node.inputs, name, idx, "input", node)


def sout(node, name, idx=0):
    return _pick(node.outputs, name, idx, "output", node)


def set_in(node, name, value, idx=0):
    sin(node, name, idx).default_value = value


class NodeBuilder:
    """Thin wrapper that lays nodes out on a grid and links by socket name."""

    def __init__(self, tree, x0=-1400, dy=-260):
        self.tree = tree
        self.x0 = x0
        self.dy = dy
        self._col = {}

    def new(self, idname, col=0, row=0, label=None, **props):
        n = self.tree.nodes.new(idname)
        n.location = (self.x0 + col * 220, row * self.dy)
        if label:
            n.label = label
            n.name = label
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, a_out, b, b_in, a_idx=0, b_idx=0):
        self.tree.links.new(sout(a, a_out, a_idx), sin(b, b_in, b_idx))


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------
def eevee_engine_id() -> str:
    """The EEVEE identifier this build accepts."""
    for cand in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            bpy.context.scene.render.engine = cand
            return cand
        except Exception:
            continue
    raise RuntimeError("no EEVEE engine available")


def enable_cycles() -> bool:
    """Enable the Cycles add-on and select the Metal GPU when present."""
    import addon_utils
    try:
        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception:
        return False
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return False
    for dt in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = dt
            break
        except Exception:
            continue
    try:
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        for d in prefs.devices:
            d.use = (d.type != "CPU") if gpus else True
    except Exception:
        pass
    return True


def set_transparency(mat, mode="DITHERED"):
    """Blended/dithered transparency across 4.2+ and legacy builds."""
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = mode           # 'DITHERED' | 'BLENDED'
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "HASHED" if mode == "DITHERED" else "BLEND"


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------
def collection(name, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def mesh_object(name, verts, faces, coll=None, smooth=True):
    """Build a mesh object from vertex/face arrays."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    me.validate(verbose=False)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def store_attribute(mesh, name, values, domain="POINT", dtype="FLOAT"):
    """Write a named attribute (e.g. psi_n per vertex) for shaders to read."""
    attr = mesh.attributes.get(name)
    if attr is None:
        attr = mesh.attributes.new(name=name, type=dtype, domain=domain)
    attr.data.foreach_set("value", list(values))
    mesh.update()
    return attr


def curve_object(name, polylines, coll=None, bevel=0.0, resolution=2):
    """A single curve object holding many 3-D polylines."""
    cu = bpy.data.curves.new(name, type="CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = resolution
    cu.bevel_depth = bevel
    cu.bevel_resolution = 2
    for pl in polylines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pl) - 1)
        flat = []
        for p in pl:
            flat += [float(p[0]), float(p[1]), float(p[2]), 1.0]
        sp.points.foreach_set("co", flat)
    ob = bpy.data.objects.new(name, cu)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def points_object(name, points, coll=None, attributes=None):
    """Vertex-only mesh holding a point set, with optional named attributes.

    A real ``PointCloud`` datablock exists (``bpy.data.pointclouds``; populate
    it with ``PointCloud.resize(n)``, NOT ``points.add()``), but a vertex mesh
    plus a Geometry Nodes instancer behaves identically on 4.2 and 5.x and is
    far easier to shade and size.
    """
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in points], [], [])
    me.update()
    for attr_name, values in (attributes or {}).items():
        store_attribute(me, attr_name, values)
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def set_modifier_input(mod, identifier, value):
    """Set a Geometry Nodes modifier input, on 4.x or 5.x.

    The API changed outright in Blender 5.0:

        4.2 LTS :  ``mod["Socket_2"] = value``            (an ID property)
        5.x     :  ``getattr(mod.properties.inputs, "Socket_2").value = value``

    In 5.x ``NodesModifier`` no longer supports ID-property assignment at all,
    so the old idiom raises ``TypeError`` rather than silently doing nothing.
    """
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        sock = getattr(props.inputs, identifier, None)
        if sock is not None and hasattr(sock, "value"):
            sock.value = value
            return True
    try:
        mod[identifier] = value
        return True
    except TypeError:
        return False


def get_modifier_input(mod, identifier, default=None):
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        sock = getattr(props.inputs, identifier, None)
        if sock is not None and hasattr(sock, "value"):
            return sock.value
    try:
        return mod[identifier]
    except (TypeError, KeyError):
        return default


def instance_points_group(name="TokPointInstancer", radius=0.02, subdiv=1,
                         material=None):
    """GN group that instances a small sphere on every point.

    A Set Material node is included deliberately: geometry GENERATED inside a
    Geometry Nodes tree does not inherit the object's material slots, so
    instanced primitives render with Blender's default white no matter what is
    appended to the object.  (Measured the hard way: three differently
    coloured puncture classes all came out identical white.)
    """
    g = bpy.data.node_groups.get(name)
    if g is not None:
        if material is not None:
            for n in g.nodes:
                if n.bl_idname == "GeometryNodeSetMaterial":
                    n.inputs["Material"].default_value = material
        return g
    g = bpy.data.node_groups.new(name, "GeometryNodeTree")
    g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_r = g.interface.new_socket("Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    s_r.default_value = float(radius)
    s_r.min_value = 0.0
    g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    gi = g.nodes.new("NodeGroupInput"); gi.location = (-600, 0)
    go = g.nodes.new("NodeGroupOutput"); go.location = (500, 0)
    ico = g.nodes.new("GeometryNodeMeshIcoSphere"); ico.location = (-350, -220)
    ico.inputs["Subdivisions"].default_value = int(subdiv)
    g.links.new(gi.outputs["Radius"], ico.inputs["Radius"])
    iop = g.nodes.new("GeometryNodeInstanceOnPoints"); iop.location = (100, 0)
    g.links.new(gi.outputs["Geometry"], iop.inputs["Points"])
    g.links.new(ico.outputs["Mesh"], iop.inputs["Instance"])

    setm = g.nodes.new("GeometryNodeSetMaterial"); setm.location = (300, 0)
    if material is not None:
        setm.inputs["Material"].default_value = material
    g.links.new(iop.outputs["Instances"], setm.inputs["Geometry"])
    g.links.new(setm.outputs["Geometry"], go.inputs[0])

    g["_sockets"] = {"Radius": s_r.identifier}
    return g


def instance_points(ob, radius=0.02, name="TokPointInstancer", material=None):
    """Attach the sphere instancer.  Each caller may pass its own group name so
    classes with different colours do not share one group."""
    g = instance_points_group(name, radius=radius, material=material)
    m = ob.modifiers.get("Instancer") or ob.modifiers.new("Instancer", "NODES")
    m.node_group = g
    set_modifier_input(m, g["_sockets"]["Radius"], float(radius))
    return m


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
