"""Cameras and lighting.

The plasma is the scene's dominant light source.  In Cycles that works for
free.  In EEVEE it does NOT: volume emission never illuminates surrounding
surfaces, so :func:`plasma_proxy_lights` adds a ring of point lights on the
magnetic axis, tinted and scaled to approximate what the volume would have
emitted.  Those lights are excluded from Cycles renders to avoid double
counting.
"""
from __future__ import annotations

import math

import bpy


def frame_distance(radius, lens, sensor=36.0, margin=1.30):
    """Camera distance that fits a sphere of ``radius`` in frame.

    ``tan(hfov/2) = sensor / (2 lens)``, so ``d = radius / tan(hfov/2)``.
    """
    return float(margin * radius * (2.0 * lens) / sensor)


def add_camera(name, location, look_at=(0.0, 0.0, 0.0), lens=50.0,
               coll=None, clip=(0.1, 2000.0)):
    """Camera with its OWN track target.

    Each camera gets a private empty.  Sharing one target across the set looks
    tidy but is a trap: moving it for the last camera silently re-aims every
    other camera at the same point.
    """
    cam_d = bpy.data.cameras.new(name)
    cam_d.lens = lens
    cam_d.clip_start, cam_d.clip_end = clip
    ob = bpy.data.objects.new(name, cam_d)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = location
    tgt = bpy.data.objects.new(f"{name}_Target", None)
    (coll or bpy.context.scene.collection).objects.link(tgt)
    tgt.location = look_at
    tgt.empty_display_size = 0.5
    con = ob.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return ob


def _orbit(radius, lens, az_deg, el_deg, margin=1.30, centre=(0.0, 0.0, 0.0)):
    """Position on a sphere that frames ``radius``, at azimuth/elevation."""
    d = frame_distance(radius, lens, margin=margin)
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (centre[0] + d * math.cos(el) * math.cos(az),
            centre[1] + d * math.cos(el) * math.sin(az),
            centre[2] + d * math.sin(el))


def standard_cameras(scene_radius, R0, a, plasma_radius=None, coll=None):
    """The six framings of the still set.

    Distances are derived from the ACTUAL scene bounding radius, not from R0 --
    the cryostat is roughly twice R0 across, so framing on R0 puts the camera
    inside the machine.
    """
    pr = float(plasma_radius if plasma_radius is not None else R0 * 1.45)
    cams = {}
    cams["01_exterior"] = add_camera(
        "CAM_01_exterior", _orbit(scene_radius, 42.0, -140.0, 18.0),
        (0, 0, 0), lens=42.0, coll=coll)
    cams["02_cutaway"] = add_camera(
        "CAM_02_cutaway", _orbit(scene_radius, 50.0, -128.0, 26.0, margin=1.15),
        (0, 0, 0), lens=50.0, coll=coll)
    cams["03_plasma"] = add_camera(
        "CAM_03_plasma", _orbit(pr, 65.0, -125.0, 22.0, margin=1.25),
        (0, 0, 0), lens=65.0, coll=coll)
    # Straight down -Y onto the phi = 0 plane, where the punctures live.
    # The section is a TALL poloidal cross-section: it spans about 2a in R but
    # 2 kappa a in Z, so frame on the half-DIAGONAL, not on the minor radius,
    # or the outer island chains fall outside a 16:9 frame.
    sec_radius = a * 1.85
    cams["04_poincare"] = add_camera(
        "CAM_04_poincare",
        (R0, -frame_distance(sec_radius, 80.0, margin=1.22), 0.0),
        (R0, 0.0, 0.0), lens=80.0, coll=coll)
    cams["05_qprofile"] = add_camera(
        "CAM_05_qprofile", _orbit(pr, 55.0, -135.0, 42.0, margin=1.30),
        (0, 0, 0), lens=55.0, coll=coll)
    cams["06_divertor"] = add_camera(
        "CAM_06_divertor",
        _orbit(a * 1.9, 85.0, -118.0, -26.0, margin=1.25,
               centre=(R0 * 0.55, 0.0, -a * 1.05)),
        (R0 * 0.85, 0.0, -a * 1.15), lens=85.0, coll=coll)
    return cams


def plasma_proxy_lights(R0, z0=0.0, n=8, energy=9000.0,
                        color=(1.0, 0.72, 0.42), coll=None, radius=0.6):
    """Ring of point lights on the magnetic axis, for EEVEE only.

    EEVEE cannot light surfaces from an emissive volume, so without these the
    vessel interior goes black in the real-time path.  They are excluded from
    Cycles via :func:`set_engine_visibility`.
    """
    out = []
    for i in range(n):
        ph = 2 * math.pi * i / n
        lt = bpy.data.lights.new(f"PlasmaProxy_{i:02d}", type="POINT")
        lt.energy = energy / n
        lt.color = color
        lt.shadow_soft_size = radius
        ob = bpy.data.objects.new(f"PlasmaProxy_{i:02d}", lt)
        (coll or bpy.context.scene.collection).objects.link(ob)
        ob.location = (R0 * math.cos(ph), R0 * math.sin(ph), z0)
        out.append(ob)
    return out


def set_engine_visibility(objs, in_cycles=True, in_eevee=True):
    """Hide proxy lights from whichever engine does not need them.

    ``hide_render`` is global, so engine-specific exclusion is done by toggling
    it right before each render instead; this helper records the intent on the
    object for :mod:`tokblend.render` to act on.
    """
    for ob in objs:
        ob["use_in_cycles"] = bool(in_cycles)
        ob["use_in_eevee"] = bool(in_eevee)


def key_rig(scene_radius, coll=None, irradiance=1.6):
    """Three-point exterior rig, placed OUTSIDE the machine.

    Lamp power is set from the distance, not from R0.  A point/area lamp of
    ``P`` watts at distance ``d`` gives roughly ``P / (4 pi d^2)`` W/m^2, so
    holding the irradiance fixed means ``P = irradiance * 4 pi d^2``.  Sizing
    lamps off R0 instead puts them INSIDE a cryostat that is about 3x R0 across
    and blows the image out completely.
    """
    import math as _m
    out = []
    d = float(scene_radius) * 1.9
    p_of = lambda f: irradiance * 4.0 * _m.pi * (d ** 2) * f
    specs = [("Key", (d * 0.62, -d * 0.68, d * 0.38), p_of(1.00), (1.0, 0.96, 0.90), scene_radius * 0.55),
             ("Fill", (-d * 0.70, -d * 0.45, d * 0.12), p_of(0.32), (0.72, 0.82, 1.00), scene_radius * 0.80),
             ("Rim", (-d * 0.30, d * 0.80, d * 0.45), p_of(0.55), (0.85, 0.90, 1.00), scene_radius * 0.45)]
    for nm, loc, en, col, size in specs:
        lt = bpy.data.lights.new(f"L_{nm}", type="AREA")
        lt.energy = en
        lt.color = col
        lt.size = size
        lt.shape = "DISK"
        ob = bpy.data.objects.new(f"L_{nm}", lt)
        (coll or bpy.context.scene.collection).objects.link(ob)
        ob.location = loc
        tgt = bpy.data.objects.get("LightTarget")
        if tgt is None:
            tgt = bpy.data.objects.new("LightTarget", None)
            (coll or bpy.context.scene.collection).objects.link(tgt)
            tgt.location = (0.0, 0.0, 0.0)
        con = ob.constraints.new("TRACK_TO")
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        out.append(ob)
    return out


def dark_world(strength=0.02, color=(0.02, 0.03, 0.05)):
    w = bpy.data.worlds.get("TokWorld") or bpy.data.worlds.new("TokWorld")
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (*color, 1.0)
        bg.inputs["Strength"].default_value = float(strength)
    bpy.context.scene.world = w
    return w
