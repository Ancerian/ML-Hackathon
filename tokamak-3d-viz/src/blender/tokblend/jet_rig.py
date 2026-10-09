"""Cameras, lights, exploded-octant variant and captions for the JET scene.

Framing is computed, not guessed: every camera distance / ortho scale is
solved so that the world bounding boxes of what the shot shows fit the 16:9
frame with a margin (the DIII-D lesson in docs/BLUEPRINT.md §9.4: frame on the
REAL bounding radius, never on R0).  Scale is 1:1 (metres).

Cameras
-------
C1  exterior three-quarter, 1.8 m human figure for scale
C2  1.5-octant cutaway, all layers plasma -> vessel -> bellows -> TF ->
    structure -> PF -> iron visible in the opening
C3  inside the vessel looking toroidally (wide lens): limiters, ports
C4  exploded octant (linked duplicates offset along their own axes; the
    base objects are untouched -- a scene VARIANT, not an edit)
C5  orthographic elevation section (R-Z plane through an iron limb)
C6  orthographic plan section (midplane)
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector, Matrix

from .bl import collection
from . import jet_cut as JC


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------
def bbox_points(objs):
    pts = []
    for ob in objs:
        if ob.type != "MESH":
            continue
        mw = ob.matrix_world
        pts += [mw @ Vector(c) for c in ob.bound_box]
    return np.array([[p.x, p.y, p.z] for p in pts]) if pts else np.zeros((1, 3))


def _basis(direction, up=(0, 0, 1)):
    f = np.asarray(direction, float)
    f = f / np.linalg.norm(f)                    # camera looks along f
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u


def fit_perspective(points, target, view_dir, lens, aspect=16 / 9, sensor=36.0,
                    margin=1.08):
    """Distance along -view_dir from target so all points fit the frame."""
    f, r, u = _basis(view_dir)
    tanh = (sensor / 2) / lens / margin
    tanv = tanh / aspect
    d = 0.0
    P = np.asarray(points) - np.asarray(target)
    for p in P:
        pf, pr, pu = p @ f, p @ r, p @ u
        # camera at -d f; depth of p = d + pf
        d = max(d, abs(pr) / tanh - pf, abs(pu) / tanv - pf)
    return d


def add_camera(name, target, view_dir, lens=35.0, dist=None, coll=None,
               ortho_scale=None, clip=(0.05, 500.0), up=(0, 0, 1)):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start, cd.clip_end = clip
    ob = bpy.data.objects.new(name, cd)
    (coll or bpy.context.scene.collection).objects.link(ob)
    f, r, u = _basis(view_dir, up)
    loc = np.asarray(target) - f * (dist if dist is not None else 30.0)
    ob.location = Vector(loc.tolist())
    # camera looks along its local -Z with local +Y up
    rot = Matrix(((r[0], u[0], -f[0]), (r[1], u[1], -f[1]), (r[2], u[2], -f[2])))
    ob.rotation_euler = rot.to_euler()
    if ortho_scale is not None:
        cd.type = "ORTHO"
        cd.ortho_scale = float(ortho_scale)
    ob["tok_target"] = list(map(float, target))
    return ob


def _dir(az_deg, el_deg):
    """Unit vector FROM the camera TOWARD the target, for a camera placed at
    azimuth az (deg) and elevation el above the horizon."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))


def fit_ortho(points, view_dir, aspect=16 / 9, margin=1.06, up=(0, 0, 1)):
    f, r, u = _basis(view_dir, up)
    P = np.asarray(points)
    cx = P @ r
    cy = P @ u
    w = (cx.max() - cx.min())
    h = (cy.max() - cy.min())
    centre = (0.5 * (cx.max() + cx.min())) * r + (0.5 * (cy.max() + cy.min())) * u
    return max(w, h * aspect) * margin, centre


# ---------------------------------------------------------------------------
# lights / world
# ---------------------------------------------------------------------------
def world(color=(0.030, 0.034, 0.040), strength=1.0, name="JET_World"):
    w = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = float(strength)
    bpy.context.scene.world = w
    return w


def sun(name, az_deg, el_deg, strength, color=(1, 1, 1), angle_deg=2.0, coll=None):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = float(strength)
    ld.color = color
    ld.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    d = _dir(az_deg, el_deg)                      # light travels along d
    f, r, u = _basis(d)
    ob.rotation_euler = Matrix(((r[0], u[0], -f[0]), (r[1], u[1], -f[1]),
                                (r[2], u[2], -f[2]))).to_euler()
    return ob


def point(name, loc, power, color=(1, 1, 1), radius=0.25, coll=None):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = float(power)
    ld.color = color
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    return ob


def interior_lights(R0, coll, n=16, power=60.0, skip_near=None, skip_r=1.2,
                    rows=(0.9, -0.9)):
    """Work lights around the vessel (on the plasma axis, plasma hidden).

    Every shadowed EEVEE point light costs shadow-map pages; 16 of them made
    the in-vessel frame ~10x slower than the exterior ones, so keep n small."""
    out = []
    for i in range(n):
        ph = 2 * math.pi * (i + 0.5) / n
        for z in rows:
            p = Vector((R0 * math.cos(ph), R0 * math.sin(ph), z))
            if skip_near is not None and (p - Vector(skip_near)).length < skip_r:
                continue
            lt = point(f"Work_{i:02d}_{'u' if z > 0 else 'd'}", p, power,
                       (1.0, 0.93, 0.84), 0.35, coll)
            if hasattr(lt.data, "shadow_maximum_resolution"):
                # coarser shadow texels: halves the in-vessel render time
                lt.data.shadow_maximum_resolution = 0.03
            out.append(lt)
    return out


# ---------------------------------------------------------------------------
# exploded octant (non-destructive variant)
# ---------------------------------------------------------------------------
#: subsystem -> (radial offset of DISCRETE parts [m], vertical offset [m],
#: vertical also applied to discrete parts?).  "Own axes": axisymmetric rings
#: (PF coils, ring collars) move along Z only; discrete parts (TF coils,
#: sectors, limbs, blocks) move radially along the octant bisector.
EXPLODE = {
    # F5: vessel pulled further out of the TF bore (it was hidden behind the
    # displaced TF coils), TF / structure / iron spaced ~2.3 m apart so each
    # layer reads on its own; ring blocks lifted, shell plates stay at Z=0
    "Vessel": (2.6, 0.0, False), "Bellows": (2.6, 0.0, False),
    "Ports": (2.6, 0.0, False), "Limiters": (2.6, 0.0, False),
    "Extras": (2.6, 0.0, False), "NBI": (2.6, 0.0, False),
    "TF": (5.9, 0.0, False),
    "Structure": (8.5, 1.2, True),
    "PF": (0.0, 1.8, False),
    "IronCore": (11.0, 0.0, False),
}


def _phi_in(phi, start, width):
    return np.mod(phi - start, 2 * np.pi) < width


def build_exploded(table, start, width, coll_name, log=print, max_samples=600):
    """Linked duplicates of everything (outboard of the central column) that
    reaches into the wedge, displaced along their own axes per subsystem.  The
    shader rig keeps only the part INSIDE the wedge (``tok_explode``), tested
    at the duplicate's home position (``tok_off``)."""
    coll = collection(coll_name, parent=table.root)
    coll["tok_role"] = "exploded"
    mid = start + 0.5 * width
    u = np.array([math.cos(mid), math.sin(mid), 0.0])
    n = 0
    for ob in table.objects:
        sub = ob.get("tok_subsystem")
        if sub not in EXPLODE or ob.get("tok_nocut", 0) > 0.5:
            continue
        if ob.get("tok_strict_hidden", 0):
            continue
        me = ob.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)[:: max(1, nv // max_samples)]
        M = np.array(ob.matrix_world)
        w = co @ M[:3, :3].T + M[:3, 3]
        ph = np.arctan2(w[:, 1], w[:, 0])
        if not _phi_in(ph, start, width).any():
            continue
        dr, dz, dz_discrete = EXPLODE[sub]
        lo, hi = w.min(0), w.max(0)
        ring = (math.hypot(*(0.5 * (lo[:2] + hi[:2]))) < 0.3 and (hi[0] - lo[0]) > 1.5)
        zc = float(0.5 * (lo[2] + hi[2]))
        zs = math.copysign(1.0, zc) if abs(zc) > 0.4 else 0.0
        if ring:
            off = np.array([0.0, 0.0, dz * zs])
        else:
            off = u * dr + np.array([0.0, 0.0, dz * zs if dz_discrete else 0.0])
        dup = ob.copy()                         # linked duplicate: shares mesh
        dup.name = ("EX_" + ob.name)[:63]
        coll.objects.link(dup)
        dup.matrix_world = Matrix.Translation(Vector(off.tolist())) @ ob.matrix_world
        dup["tok_explode"] = 1.0
        dup["tok_off"] = tuple(float(x) for x in off)
        dup["tok_exploded_from"] = ob.name
        n += 1
    log(f"[explode] {n} linked duplicates in {coll.name} "
        f"(wedge {math.degrees(start):.2f}..{math.degrees(start + width):.2f} deg)")
    return coll


# ---------------------------------------------------------------------------
# captions (burnt in with the render stamp; no image libraries needed)
# ---------------------------------------------------------------------------
SOURCE_LABEL = {"JET1975": "EUR 5516e (1975)",
                "JET_JU_1983_Progress_Report.pdf": "EUR 9472 EN (1984)",
                "JET_JU_Brochure.pdf": "JET JU brochure (1982)",
                "JET_JU_1979_Annual_Report.pdf": "JET JU Annual Report 1979",
                "JET_Wesson_1999_Science_of_JET.pdf": "Wesson, The Science of JET (1999)"}


def set_caption(text, size=18):
    r = bpy.context.scene.render
    r.use_stamp = True
    for p in ("use_stamp_date", "use_stamp_time", "use_stamp_render_time",
              "use_stamp_frame", "use_stamp_frame_range", "use_stamp_memory",
              "use_stamp_hostname", "use_stamp_camera", "use_stamp_lens",
              "use_stamp_scene", "use_stamp_marker", "use_stamp_filename",
              "use_stamp_sequencer_strip", "use_stamp_labels"):
        if hasattr(r, p):
            setattr(r, p, False)
    r.use_stamp_note = True
    r.stamp_note_text = text
    r.stamp_font_size = int(size)
    r.stamp_foreground = (0.92, 0.92, 0.92, 1.0)
    r.stamp_background = (0.0, 0.0, 0.0, 0.55)
