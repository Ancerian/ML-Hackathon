"""Second introspection pass, with Cycles enabled and dynamic enums probed.

Many Blender enums are built dynamically and report an EMPTY ``enum_items``
even though they accept values (view_transform, cycles.denoiser, Glare's menu
sockets...).  Those can only be discovered by assignment, which is what this
script does.

    blender --background --python tools/introspect_cycles.py
"""
import addon_utils
import bpy

OUT = []
p = lambda *a: OUT.append(" ".join(str(x) for x in a))

p("=== CYCLES ===")
addon_utils.enable("cycles", default_set=True, persistent=True)
sc = bpy.context.scene
for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
    try:
        sc.render.engine = eng
        p(f"  engine {eng}: ACCEPTED")
    except Exception:
        p(f"  engine {eng}: REJECTED")

prefs = bpy.context.preferences.addons["cycles"].preferences
for dt in ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI", "NONE"):
    try:
        prefs.compute_device_type = dt
        p(f"  compute_device_type {dt}: ACCEPTED")
    except Exception:
        p(f"  compute_device_type {dt}: REJECTED")
try:
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        p(f"  device {d.name!r} type={d.type}")
except Exception as e:
    p("  devices err:", e)

sc.render.engine = "CYCLES"
for name, cands in (("denoiser", ("OPENIMAGEDENOISE", "OPTIX")),
                    ("sampling_pattern", ("AUTOMATIC", "SOBOL_BURLEY",
                                          "TABULATED_SOBOL", "BLUE_NOISE"))):
    for c in cands:
        try:
            setattr(sc.cycles, name, c)
            p(f"  cycles.{name} {c}: ACCEPTED")
        except Exception:
            p(f"  cycles.{name} {c}: REJECTED")
p(f"  cycles.volume_bounces default = {sc.cycles.volume_bounces} (raise for plasma)")

p("\n=== COLOR MANAGEMENT (dynamic enum) ===")
for vt in ("Standard", "Filmic", "AgX", "Khronos PBR Neutral", "Raw", "False Color"):
    try:
        sc.view_settings.view_transform = vt
        p(f"  view_transform {vt!r}: ACCEPTED")
    except Exception:
        p(f"  view_transform {vt!r}: REJECTED")

p("\n=== GLARE MENU SOCKETS (enum_items is empty; probe by assignment) ===")
t = bpy.data.node_groups.new("C", "CompositorNodeTree")
g = t.nodes.new("CompositorNodeGlare")
p("  inputs:", [s.name for s in g.inputs])
p("  outputs:", [s.name for s in g.outputs])
ty = next(s for s in g.inputs if s.name == "Type")
for c in ("Bloom", "Ghosts", "Streaks", "Fog Glow", "Simple Star",
          "Sun Beams", "Kernel", "BLOOM"):
    try:
        ty.default_value = c
        p(f"  Type {c!r}: ACCEPTED")
    except Exception:
        p(f"  Type {c!r}: REJECTED")

p("\n=== GEOMETRY NODES MODIFIER INPUTS (5.x moved these) ===")
gn = bpy.data.node_groups.new("G", "GeometryNodeTree")
gn.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
sr = gn.interface.new_socket("Radius", in_out="INPUT", socket_type="NodeSocketFloat")
gn.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
bpy.ops.mesh.primitive_cube_add()
m = bpy.context.active_object.modifiers.new("M", "NODES")
m.node_group = gn
p(f"  socket identifier: {sr.identifier!r}")
try:
    m[sr.identifier] = 0.5
    p("  4.x style  mod[id] = v      : ACCEPTED")
except Exception as e:
    p(f"  4.x style  mod[id] = v      : {type(e).__name__}")
try:
    getattr(m.properties.inputs, sr.identifier).value = 0.5
    p("  5.x style  properties.inputs: ACCEPTED")
except Exception as e:
    p(f"  5.x style  properties.inputs: {type(e).__name__}")

print("\n".join(OUT))
