"""Switch the open scene between the six documented shots.

Run from Blender's Scripting workspace: open this file in the Text Editor and
press Run Script.  Edit SHOT below, or call set_shot("04_poincare") from the
Python console.

Each shot sets three things: how far the cutaway wedge is open, which
collections are visible, and the active camera.  Without this the file opens
with the machine sealed and every collection visible at once, which shows
nothing.
"""
import bpy

SHOT = "02_cutaway"          # <- change me and press Run Script

SHOTS = {
    "01_exterior": dict(cut=0.00, show=("Machine",)),
    "02_cutaway":  dict(cut=1.00, show=("Machine", "PlasmaVolume", "FieldLines")),
    "03_plasma":   dict(cut=0.00, show=("PlasmaVolume", "FieldLines")),
    "04_poincare": dict(cut=0.00, show=("Diagnostics",)),
    "05_qprofile": dict(cut=0.55, show=("FluxSurfaces", "FieldLines")),
    "06_divertor": dict(cut=1.00, show=("Machine", "PlasmaVolume")),
}


def set_modifier_input(mod, identifier, value):
    """Set a Geometry Nodes input on 4.x or 5.x (the API changed at 5.0)."""
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


def set_cut(value):
    n = 0
    for ob in bpy.data.objects:
        m = ob.modifiers.get("Cutaway") if ob.type == "MESH" else None
        if m is None or m.node_group is None:
            continue
        set_modifier_input(m, m.node_group["_sockets"]["Cut"], float(value))
        ob.update_tag()
        n += 1
    return n


def set_visible(names):
    names = set(names) | {"Rig"}
    for c in bpy.data.collections:
        vis = c.name in names
        c.hide_viewport = not vis
        c.hide_render = not vis
    # the viewport also honours the per-view-layer exclude flag
    vl = bpy.context.view_layer
    for lc in vl.layer_collection.children:
        lc.exclude = lc.name not in names


def set_shot(name=SHOT):
    if name not in SHOTS:
        raise KeyError(f"unknown shot {name!r}; try one of {list(SHOTS)}")
    s = SHOTS[name]
    n = set_cut(s["cut"])
    set_visible(s["show"])
    cam = bpy.data.objects.get(f"CAM_{name}")
    if cam is not None:
        bpy.context.scene.camera = cam
    print(f"[set_shot] {name}: cut={s['cut']} on {n} objects, "
          f"showing {', '.join(s['show'])}, camera={cam.name if cam else 'unchanged'}")
    return name


if __name__ == "__main__":
    set_shot(SHOT)
