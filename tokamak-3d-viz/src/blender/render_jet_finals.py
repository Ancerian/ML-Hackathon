#!/usr/bin/env python3
"""Phase F5 deliverables for the JET reconstruction (runs inside Blender).

    blender -b --factory-startup --python src/blender/render_jet_finals.py -- \
        --mode stills   --bake data/bake_jet1975 --variant-bake data/bake_jet1983 \
        [--active variant --strict-year 1983] --cams C1,C2,C3,C4 --tag jet1975
    ... --mode sections --tag jet1975          (C5 + C6, technical style, 3000 px)
    ... --mode anim     [--frames 1-960]       (EEVEE 1920x1080 @ 24 fps)
    ... --mode encode                          (frames -> MP4 through Blender's FFmpeg)

Outputs
-------
stills    render/jet/final/<tag>_<C>.png   + <tag>_<C>.json (label anchors in
          pixels, title, attribution) for ``src/overlay_jet.py``
sections  render/jet/sections/<tag>_<C>_raw.png + .json (orthographic camera
          mapping world -> pixel) for ``src/overlay_jet.py --sections``
anim      render/jet/anim/frames/####.png ; ``--mode encode`` -> jet_anim.mp4

Captions are NOT burnt in (no render stamp): short title + leader-line labels
+ attribution are a matplotlib overlay (G3).  Anchors are found by ray casting
from the camera through the scene, honouring the shader cut (a hit inside the
removed region is stepped through, like the renderer does), so every leader
line ends on a pixel where that component is actually visible.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_jet_scene as B                                     # noqa: E402
from tokblend import jet_cut as JC                              # noqa: E402
from tokblend import jet_rig as RG                              # noqa: E402
from tokblend import render as RD                               # noqa: E402

DEG = math.pi / 180.0

ATTRIB_1975 = "Власна 3D-реконструкція; побудовано за даними EUR 5516e (1975)"
ATTRIB_1983 = ("Власна 3D-реконструкція; побудовано за даними EUR 5516e (1975), "
               "JET JU 1983")

TITLES = {
    "C1": "загальний вигляд",
    "C2": "розріз 1,5 октанта",
    "C3": "усередині вакуумної камери",
    "C4": "рознесений октант (схематично)",
    "C5": "вертикальний переріз R–Z через ярмо",
    "C6": "план, переріз у площині Z = 0",
}

#: label key -> (text, object filter).  Filter keys: subs (subsystems),
#: part (regex on tok_part), comp (regex on tok_component).
LABELS = {
    "plasma": ("Плазма: магнітні поверхні ψN, силові лінії", dict(subs=("Plasma",))),
    "vessel": ("Вакуумна камера (жорсткі сектори)", dict(subs=("Vessel",))),
    "bellows": ("Сильфони", dict(subs=("Bellows",))),
    "tf": ("Котушки тороїдального поля (TF)", dict(subs=("TF",), part="casing")),
    "pf1": ("PF 1 (центральний соленоїд)", dict(subs=("PF",), part="^coil1$")),
    "pf2": ("PF 2", dict(subs=("PF",), part="^coil2$")),
    "pf3": ("PF 3", dict(subs=("PF",), part="^coil3$")),
    "pf4": ("PF 4", dict(subs=("PF",), part="^coil4$")),
    "iron": ("Залізне ярмо (магнітопровід)", dict(subs=("IronCore",))),
    "ring": ("Механічна структура: кільце", dict(subs=("Structure",), part="ring")),
    "shell": ("Механічна структура: оболонка", dict(subs=("Structure",), part="shell")),
    "limiters": ("Лімітери", dict(subs=("Limiters",))),
    "ports": ("Порти", dict(subs=("Ports",))),
    "nbi": ("Інжектори нейтральних пучків (NBI)", dict(subs=("NBI",), comp="^nbi$")),
    "nbi83": ("Адаптер порту NBI", dict(subs=("NBI",), comp="nbi_adaptor")),
    "pump": ("Камера відкачки", dict(subs=("Extras",), comp="pump_chamber")),
    "human": ("Фігура 1,8 м", dict(subs=("Human",))),
}

SHOT_LABELS = {
    "C1": ["tf", "vessel", "pf3", "pf4", "iron", "ring", "shell", "ports", "nbi", "nbi83",
           "pump", "human"],
    "C2": ["plasma", "vessel", "bellows", "tf", "pf1", "pf2", "pf3", "pf4", "iron", "ring",
           "shell", "limiters", "ports", "pump", "human"],
    "C3": ["vessel", "bellows", "limiters", "ports"],
    "C4": ["plasma", "vessel", "bellows", "tf", "pf1", "pf2", "pf3", "pf4", "iron", "ring",
           "shell", "ports", "human"],
}
#: C3 wording: what the viewer sees from inside
C3_TEXT = {"vessel": "Стінка жорсткого сектора (Inconel 600)",
           "bellows": "Сильфонна секція (Inconel 625)",
           "limiters": "Рейковий лімітер (Mo-пластини)",
           "ports": "Отвори портів"}


def argv_after():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stills", "sections", "anim", "encode", "probe"],
                    required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--res", default="3840x2160")
    ap.add_argument("--frames", default="1-960")
    ap.add_argument("--anim-samples", type=int, default=48)
    ap.add_argument("--anim-res", default="1920x1080")
    ap.add_argument("--out-root", default="render/jet")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--save-blend", default=None)
    a, rest = ap.parse_known_args(argv)
    return a, rest


# ---------------------------------------------------------------------------
# cut test in Python (same maths as the TokJetCut shader group)
# ---------------------------------------------------------------------------
def cut_state():
    st = JC.get_cut()
    st["PlaneN"] = np.array(JC.plane_normal())
    return st


def kept(ob, p, st):
    off = np.array(ob.get("tok_off", (0.0, 0.0, 0.0)), float)
    q = np.asarray(p, float) - off
    noc = float(ob.get("tok_nocut", 0.0))
    exp = float(ob.get("tok_explode", 0.0))
    phi = math.atan2(q[1], q[0])
    rel = (phi - st["WedgeStart"]) % (2 * math.pi)
    inside = 1.0 if rel < st["WedgeWidth"] else 0.0
    clip = inside * st["WedgeOn"] * (1 - noc) * (1 - exp) + (1 - inside) * exp
    if st["PlaneOn"] > 0.5 and float(q @ st["PlaneN"]) > st["PlaneD"]:
        clip += 1
    return clip < 0.5


def _transparent(ob):
    return (ob.get("tok_subsystem") in ("Plasma", "FieldLines") or ob.name == "Floor"
            or ob.type != "MESH")


def visible_from(cam, p, targets, st, depsgraph, max_steps=60):
    sc = bpy.context.scene
    o = Vector(cam.matrix_world.translation)
    pv = Vector(p)
    d = pv - o
    dist = d.length
    d.normalize()
    travelled = 0.0
    for _ in range(max_steps):
        hit, loc, nrm, idx, ob, mw = sc.ray_cast(depsgraph, o, d,
                                                 distance=dist - travelled + 0.05)
        if not hit:
            return False                    # the sample face itself was not hit
        ob = ob.original if hasattr(ob, "original") else ob
        step = (loc - o).length
        if ob.name in targets:
            return (loc - pv).length < 0.35 and kept(ob, loc, st)
        if _transparent(ob) or not kept(ob, loc, st):
            o = loc + d * 2e-3
            travelled += step + 2e-3
            continue
        return False
    return False


def face_samples(objs, n=500, seed=1):
    rng = random.Random(seed)
    pool = []
    for ob in objs:
        if ob.type != "MESH" or not ob.visible_get():
            continue
        me = ob.data
        m = len(me.polygons)
        if m == 0:
            continue
        cen = np.empty(m * 3, np.float32)
        me.polygons.foreach_get("center", cen)
        area = np.empty(m, np.float32)
        me.polygons.foreach_get("area", area)
        cen = cen.reshape(-1, 3)
        # skip slivers; weight by area so big readable faces win
        ok = np.nonzero(area > np.percentile(area, 30))[0]
        M = np.array(ob.matrix_world)
        for i in ok[:: max(1, len(ok) // 60)]:
            w = M[:3, :3] @ cen[i] + M[:3, 3]
            pool.append((ob, w, float(area[i])))
    rng.shuffle(pool)
    return pool[:n]


def select_objs(js, spec):
    import re
    tb = js.active
    objs = []
    pool = list(tb.objects) + list(js.exploded[tb.label].objects)
    pl = js.plasmas[tb.label]
    if "Plasma" in spec.get("subs", ()):
        return [o for o in pl.surfaces + pl.volume_objs if o.visible_get()]
    for o in pool:
        if o.get("tok_subsystem") not in spec.get("subs", ()):
            continue
        if "part" in spec and not re.search(spec["part"], o.get("tok_part", "")):
            continue
        if "comp" in spec and not re.search(spec["comp"], o.get("tok_component", "")):
            continue
        if not o.visible_get():
            continue
        objs.append(o)
    return objs


def find_anchors(js, key, cam, W, H, keys=None, texts=None):
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    st = cut_state()
    out = []
    for lk in keys or SHOT_LABELS.get(key, []):
        text, spec = LABELS[lk]
        if texts and lk in texts:
            text = texts[lk]
        objs = select_objs(js, spec)
        if not objs:
            continue
        # exploded view: label the displaced pieces when they are visible
        ex = [o for o in objs if o.get("tok_explode", 0) > 0.5]
        groups = [ex, objs] if (key == "C4" and ex) else [objs]
        best = None
        for grp in groups:
            names = {o.name for o in grp}
            vis = []
            for ob, w, area in face_samples(grp):
                ndc = world_to_camera_view(sc, cam, Vector(w))
                if not (0.04 < ndc.x < 0.96 and 0.05 < ndc.y < 0.93 and ndc.z > 0):
                    continue
                if not kept(ob, w, st):
                    continue
                if lk == "plasma":
                    ok = _plasma_visible(cam, w, st, dg)
                else:
                    ok = visible_from(cam, w, names, st, dg)
                if ok:
                    vis.append((ndc.x * W, (1 - ndc.y) * H, ndc.z))
            if len(vis) >= 2:
                best = vis
                break
        if not best and lk == "ports":
            best = _port_openings(js, cam, W, H, st, dg)
        if not best:
            js.log(f"[anchor] {key}/{lk}: not visible")
            continue
        # prefer the nearer part of what is visible (large on screen), then
        # the sample closest to the median of that subset
        dep = np.array([z for _, _, z in best])
        near = [b for b in best if b[2] <= np.percentile(dep, 45)] or best
        P = np.array([(x, y) for x, y, _ in near])
        med = np.median(P, axis=0)
        i = int(np.argmin(np.hypot(*(P - med).T)))
        out.append({"key": lk, "text": text, "x": float(P[i, 0]), "y": float(P[i, 1]),
                    "n_visible": len(best)})
        js.log(f"[anchor] {key}/{lk}: ({P[i, 0]:.0f}, {P[i, 1]:.0f}) from {len(best)} samples")
    return out


def _port_openings(js, cam, W, H, st, dg):
    """Port OPENINGS seen from inside: the centre of each duct's inner end
    (an empty hole, so a face sample cannot be hit); visible when the ray
    to it is not blocked before it."""
    sc = bpy.context.scene
    out = []
    o = Vector(cam.matrix_world.translation)
    for ob in js.active.objects_in("Ports"):
        if not ob.visible_get() or "duct" not in ob.get("tok_part", ""):
            continue
        bb = RG.bbox_points([ob])
        R = np.hypot(bb[:, 0], bb[:, 1])
        c = bb.mean(0)
        phi = math.atan2(c[1], c[0])
        if (bb[:, 2].max() - bb[:, 2].min()) > (R.max() - R.min()):
            continue                                # vertical port: skip here
        p = Vector((R.min() * math.cos(phi), R.min() * math.sin(phi), c[2]))
        ndc = world_to_camera_view(sc, cam, p)
        if not (0.04 < ndc.x < 0.96 and 0.05 < ndc.y < 0.93 and ndc.z > 0):
            continue
        d = p - o
        dist = d.length
        d.normalize()
        hit, loc, *_ = sc.ray_cast(dg, o, d, distance=dist - 0.15)
        if not hit:
            out.append((ndc.x * W, (1 - ndc.y) * H, ndc.z))
    return out


def _plasma_visible(cam, w, st, dg):
    sc = bpy.context.scene
    o = Vector(cam.matrix_world.translation)
    d = Vector(w) - o
    dist = d.length
    d.normalize()
    trav = 0.0
    for _ in range(60):
        hit, loc, n, i, ob, mw = sc.ray_cast(dg, o, d, distance=dist - trav - 0.01)
        if not hit:
            return True
        ob = ob.original if hasattr(ob, "original") else ob
        if _transparent(ob) or not kept(ob, loc, st):
            step = (loc - o).length
            o = loc + d * 2e-3
            trav += step + 2e-3
            continue
        return False
    return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def variant_name(js):
    tb = js.active
    meta = tb.manifest.get("machine", {}).get("meta", {})
    return meta.get("variant") or meta.get("machine") or tb.name


def is_1983(js):
    return "1983" in js.active.label


def cycles(samples, res):
    sc = bpy.context.scene
    RD.setup_cycles(samples=samples, threshold=0.01, volume_bounces=2)
    cy = sc.cycles
    cy.transparent_max_bounces = 128      # clip-by-transparency through many layers
    cy.max_bounces = 8
    cy.glossy_bounces = 4
    cy.diffuse_bounces = 3
    cy.transmission_bounces = 4
    cy.use_adaptive_sampling = True
    cy.adaptive_min_samples = 32
    cy.time_limit = 0
    try:
        cy.use_light_tree = True
    except Exception:
        pass
    w, h = (int(x) for x in res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.color_depth = "8"
    sc.render.use_stamp = False


def eevee(samples, res):
    sc = bpy.context.scene
    RD.setup_eevee(samples=samples)
    try:
        sc.eevee.shadow_pool_size = "1024"
    except Exception:
        pass
    w, h = (int(x) for x in res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.color_depth = "8"
    sc.render.use_stamp = False


def shot_meta(js, key, cam, W, H, seconds, engine, samples, extra=None):
    y83 = is_1983(js)
    strict = js.args.strict_year
    title = f"JET {'1983' if y83 else '1975'} — {TITLES[key]}"
    sub = variant_name(js)
    if strict:
        sub += f"; суворо станом на {strict} р."
    pl = js.plasmas[js.active.label]
    d = {"key": key, "machine": js.active.name, "variant": sub, "title": title,
         "attribution": ATTRIB_1983 if y83 else ATTRIB_1975,
         "plasma_note": pl.caption_note(), "psi_source": pl.source,
         "W": W, "H": H, "engine": engine, "samples": samples,
         "render_seconds": round(seconds, 1),
         "camera": {"name": cam.name, "type": cam.data.type, "lens": cam.data.lens,
                    "ortho_scale": cam.data.ortho_scale,
                    "matrix_world": [list(r) for r in cam.matrix_world]}}
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def do_stills(js, a, cams):
    out_dir = os.path.join(a.out_root, "final")
    os.makedirs(out_dir, exist_ok=True)
    cycles(a.samples, a.res)
    W, H = (int(x) for x in a.res.lower().split("x"))
    times = {}
    for key in cams:
        js.apply_shot(key)
        cam = js.cams[js.active.label][key]
        base = os.path.join(out_dir, f"{a.tag}_{key}")
        texts = C3_TEXT if key == "C3" else None
        anchors = find_anchors(js, key, cam, W, H, texts=texts)
        t0 = time.time()
        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = base + ".png"
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        times[key] = dt
        print(f"[final] {a.tag}_{key}: {dt:.1f}s", flush=True)
        meta = shot_meta(js, key, cam, W, H, dt, "CYCLES", a.samples,
                         {"labels": anchors, "image": os.path.basename(base) + ".png"})
        json.dump(meta, open(base + ".json", "w"), indent=1, ensure_ascii=False)
    tf = os.path.join(out_dir, "render_times.json")
    prev = json.load(open(tf)) if os.path.exists(tf) else {}
    prev.update({f"{a.tag}_{k}": {"seconds": round(v, 1), "engine": "CYCLES Metal GPU",
                                  "res": a.res, "samples": a.samples,
                                  "adaptive_threshold": 0.01, "denoiser": "OIDN"}
                 for k, v in times.items()})
    json.dump(prev, open(tf, "w"), indent=1)
    return times


def do_sections(js, a):
    out_dir = os.path.join(a.out_root, "sections")
    os.makedirs(out_dir, exist_ok=True)
    W, H = (int(x) for x in js.args.section_res.lower().split("x"))
    eevee(64, js.args.section_res)
    times = {}
    for key in ("C5", "C6"):
        js.apply_shot(key)
        cam = js.cams[js.active.label][key]
        base = os.path.join(out_dir, f"{a.tag}_{key}_raw")
        t0 = time.time()
        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = base + ".png"
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        times[key] = dt
        f, r, u = RG._basis(-Vector(cam.matrix_world.col[2][:3]),
                            tuple(cam.matrix_world.col[1][:3]))
        c = np.array(cam.matrix_world.translation)
        fr = js.frames[js.active.label]
        tb = js.active
        comps = tb.manifest.get("components", {})
        extra = {
            "image": os.path.basename(base) + ".png",
            "ortho": {"right": list(map(float, r)), "up": list(map(float, u)),
                      "centre": list(map(float, c)), "scale": float(cam.data.ortho_scale)},
            "frame": {"section_phi_deg": math.degrees(fr["section_phi"]),
                      "rot_deg": math.degrees(fr["rot"]), "b0_deg": math.degrees(fr["b0"])},
            "human_xy": list(getattr(js, "c6_human", (0, 0))),
            "floor_Z": float(comps.get("human", {}).get("info", {}).get("floor_Z", -5.72)),
            "nbi_present": bool(tb.objects_in("NBI")),
        }
        meta = shot_meta(js, key, cam, W, H, dt, "EEVEE", 64, extra)
        json.dump(meta, open(base + ".json", "w"), indent=1, ensure_ascii=False)
        print(f"[section] {a.tag}_{key}: {dt:.1f}s", flush=True)
    return times


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


FPS = 24
T_ORBIT, T_OPEN, T_IN, T_BACK = 10.0, 8.0, 12.0, 10.0


def anim_setup(js, a):
    """Keyframe the camera, the cut wedge, the flux-surface alpha and the
    field-line build for the four phases (orbit / open / inward / section)."""
    sc = bpy.context.scene
    tb = js.active
    fr = js.frames[tb.label]
    js.apply_shot("C2")                           # subsystems + plasma surfaces
    for ob in js.plasmas[tb.label].volume_objs:
        ob.hide_render = ob.hide_viewport = True
    pl = js.plasmas[tb.label]
    for ob in pl.surfaces:
        ob.hide_render = ob.hide_viewport = False
    n_total = int(round((T_ORBIT + T_OPEN + T_IN + T_BACK) * FPS))
    sc.frame_start, sc.frame_end = 1, n_total
    sc.render.fps = FPS

    cam_d = bpy.data.cameras.new("CAM_anim")
    cam = bpy.data.objects.new("CAM_anim", cam_d)
    bpy.data.collections["Rig"].objects.link(cam)
    cam_d.sensor_width = 36.0
    cam_d.clip_start, cam_d.clip_end = 0.05, 400.0
    tgt = bpy.data.objects.new("CAM_anim_target", None)
    bpy.data.collections["Rig"].objects.link(tgt)
    tt = cam.constraints.new("TRACK_TO")
    tt.target = tgt
    tt.track_axis, tt.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    sc.camera = cam

    c1 = js.cams[tb.label]["C1"]
    c2 = js.cams[tb.label]["C2"]
    r1 = Vector(c1.location).length
    r2 = Vector(c2.location).length
    w2s, w2w = fr["wedge_c2"]
    bis0 = w2s + 0.5 * w2w                        # C2 bisector azimuth
    phs = fr["section_phi"]
    # final wedge = half-space n.P > 0 of the C5 section: start phs, width pi;
    # choose the start branch closest to the C2 wedge
    s_end = phs
    while s_end - w2s > math.pi:
        s_end -= 2 * math.pi
    while s_end - w2s < -math.pi:
        s_end += 2 * math.pi
    bis_end = s_end + 0.5 * math.pi
    # camera distance for the final "elevation" (perspective, 38 mm): fit C5
    c5 = js.cams[tb.label]["C5"]
    d_end = c5.data.ortho_scale * 1.18 / (36.0 / 38.0)
    z_end = float(np.array(c5.matrix_world.translation)[2])

    JC.set_cut(None, None)
    s_on, s_st, s_wd = "tok_WedgeOn", "tok_WedgeStart", "tok_WedgeWidth"

    flux_mat = next(m for m in bpy.data.materials if m.name.startswith("JET_FluxSurface_"))
    alpha_node = next(n for n in flux_mat.node_tree.nodes if n.label == "Alpha")
    _uniform_into(flux_mat, alpha_node.inputs[1], "tok_flux_alpha")
    alpha_sock = "tok_flux_alpha"
    alpha_full = 0.55
    line_mat = next(m for m in bpy.data.materials if m.name.startswith("JET_FieldLine_"))
    _uniform_into(line_mat, _fade_node(line_mat), "tok_line_fade")
    lfade = "tok_line_fade"
    lines = pl.lines[0] if pl.lines else None
    if lines is not None:
        lines.data.bevel_factor_mapping_end = "SPLINE"

    f1 = T_ORBIT * FPS
    f2 = f1 + T_OPEN * FPS
    f3 = f2 + T_IN * FPS
    az_c1 = math.atan2(c1.location.y, c1.location.x)
    az_orbit0 = bis0 - 150 * DEG
    el1 = math.asin(max(-1, min(1, c1.location.z / r1)))
    el2 = math.asin(max(-1, min(1, c2.location.z / r2)))
    tgt_c1 = Vector(c1.get("tok_target", (0, 0, 0)))
    R0 = 2.96
    for f in range(1, n_total + 1):
        t = (f - 1)
        if t <= f1:                                # 1 orbit
            u = smooth(t / f1) * 0.85 + 0.15 * (t / f1)
            az = lerp(az_orbit0, bis0, u)
            el = lerp(el1, el2, u)
            rr = lerp(r1 * 1.02, r2, u)
            look = tgt_c1.lerp(Vector((0, 0, 0)), u)
            lens = lerp(35.0, 40.0, u)
            won, wst, wwd = 0.0, bis0, 0.0
            alpha, build, lf = 0.0, 0.0, 0.0
        elif t <= f2:                              # 2 open the octant cutaway
            u = smooth((t - f1) / (f2 - f1))
            az, el = bis0, el2
            rr = lerp(r2, r2 * 0.93, u)
            look = Vector((0, 0, 0))
            lens = 40.0
            ww = w2w * u
            won, wst, wwd = 1.0, bis0 - 0.5 * ww, max(ww, 1e-4)
            alpha, build, lf = 0.0, 0.0, 0.0
        elif t <= f3:                              # 3 inward to the plasma
            u = smooth((t - f2) / (f3 - f2))
            p0 = Vector((r2 * 0.93 * math.cos(el2) * math.cos(bis0),
                         r2 * 0.93 * math.cos(el2) * math.sin(bis0),
                         r2 * 0.93 * math.sin(el2)))
            eye_in = Vector((7.4 * math.cos(bis0 - 6 * DEG), 7.4 * math.sin(bis0 - 6 * DEG), 2.1))
            pos = p0.lerp(eye_in, u)
            look = Vector((0, 0, 0)).lerp(Vector((1.4 * math.cos(bis0), 1.4 * math.sin(bis0),
                                                  -0.2)), u)
            lens = lerp(40.0, 24.0, u)
            won, wst, wwd = 1.0, w2s, w2w
            alpha = alpha_full * smooth((t - f2) / (0.45 * (f3 - f2)))
            build = smooth((t - f2 - 0.3 * (f3 - f2)) / (0.65 * (f3 - f2)))
            lf = 1.0
            _key_cam(cam, tgt, cam_d, pos, look, lens, f)
            _key_cut(s_on, s_st, s_wd, won, wst, wwd, f)
            _key_plasma(alpha_sock, alpha, lines, build, lfade, lf, f)
            continue
        else:                                      # 4 pull back to the elevation
            u = smooth((t - f3) / (n_total - f3))
            eye_in = Vector((7.4 * math.cos(bis0 - 6 * DEG), 7.4 * math.sin(bis0 - 6 * DEG), 2.1))
            az0 = math.atan2(eye_in.y, eye_in.x)
            az = lerp(az0, bis_end, u)
            rr0 = eye_in.length
            el0 = math.asin(eye_in.z / rr0)
            rr = lerp(rr0, d_end, u)
            el = lerp(el0, math.asin(max(-1, min(1, z_end / d_end))) if d_end else 0.0, u)
            look = Vector((1.4 * math.cos(bis0), 1.4 * math.sin(bis0), -0.2)).lerp(
                Vector((0, 0, z_end)), u)
            lens = lerp(24.0, 38.0, u)
            wst = lerp(w2s, s_end, u)
            wwd = lerp(w2w, math.pi, u)
            won = 1.0
            alpha = alpha_full
            build = 1.0
            lf = 1.0 - smooth((t - f3) / (0.4 * (n_total - f3)))
        pos = Vector((rr * math.cos(el) * math.cos(az), rr * math.cos(el) * math.sin(az),
                      rr * math.sin(el)))
        _key_cam(cam, tgt, cam_d, pos, look, lens, f)
        _key_cut(s_on, s_st, s_wd, won, wst, wwd, f)
        _key_plasma(alpha_sock, alpha, lines, build, lfade, lf, f)
    # linear interpolation everywhere: the smoothing is already in the keys
    for idb in (cam, tgt, cam_d, sc, lines.data if lines is not None else None,
                lines):
        if idb is None or idb.animation_data is None or idb.animation_data.action is None:
            continue
        for fc in _fcurves(idb.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    return cam, n_total


def _fcurves(action):
    if hasattr(action, "fcurves"):
        try:
            return list(action.fcurves)
        except Exception:
            pass
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def _uniform_into(mat, socket, prop):
    """Feed ``socket`` from the scene property ``prop`` (View-Layer attribute:
    a uniform, animatable without shader recompilation)."""
    nt = mat.node_tree
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "VIEW_LAYER"
    at.attribute_name = prop
    nt.links.new(at.outputs["Factor"], socket)
    bpy.context.scene[prop] = 0.0


def _fade_node(mat):
    """Insert Mix(Transparent, surface) with a keyable factor (1 = visible)."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    src = out.inputs["Surface"].links[0].from_socket
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mx = nt.nodes.new("ShaderNodeMixShader")
    mx.label = mx.name = "Fade"
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(src, mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    from tokblend.bl import set_transparency
    set_transparency(mat, "BLENDED")
    return mx.inputs[0]


def _key_cam(cam, tgt, cam_d, pos, look, lens, f):
    cam.location = pos
    cam.keyframe_insert("location", frame=f)
    tgt.location = look
    tgt.keyframe_insert("location", frame=f)
    cam_d.lens = lens
    cam_d.keyframe_insert("lens", frame=f)


def _key_prop(prop, v, f):
    sc = bpy.context.scene
    sc[prop] = float(v)
    sc.keyframe_insert(f'["{prop}"]', frame=f)


def _key_cut(s_on, s_st, s_wd, on, st, wd, f):
    _key_prop(s_on, on, f)
    _key_prop(s_st, st, f)
    _key_prop(s_wd, wd, f)


def _key_plasma(alpha_sock, alpha, lines, build, lfade, lf, f):
    _key_prop(alpha_sock, alpha, f)
    if lines is not None:
        lines.data.bevel_factor_end = max(build, 1e-3)
        lines.data.keyframe_insert("bevel_factor_end", frame=f)
        lines.hide_render = build < 1e-3
        lines.keyframe_insert("hide_render", frame=f)
    _key_prop(lfade, lf, f)


def do_anim(js, a):
    sc = bpy.context.scene
    cam, n_total = anim_setup(js, a)
    eevee(a.anim_samples, a.anim_res)
    ee = sc.eevee
    try:
        ee.use_raytracing = False             # 3x faster; AO-like look is enough here
    except Exception:
        pass
    # Measured at frame 700 (960x540, 16 spp): 99 s with transparent shadows
    # through the shader-clipped machine, 0.8 s without -- visually the same
    # (the world light dominates in the opening).  Key sun only casts shadows.
    for m in bpy.data.materials:
        if hasattr(m, "use_transparent_shadow"):
            m.use_transparent_shadow = False
    for s_ in js.suns[1:]:
        s_.data.use_shadow = False
    try:
        ee.shadow_resolution_scale = 0.5
    except Exception:
        pass
    fdir = os.path.join(a.out_root, "anim", "frames")
    os.makedirs(fdir, exist_ok=True)
    if "," in a.frames:                       # spot-check individual frames
        tdir = os.path.join(a.out_root, "anim", "_test")
        os.makedirs(tdir, exist_ok=True)
        for f in (int(x) for x in a.frames.split(",")):
            sc.frame_set(f)
            sc.render.filepath = os.path.join(os.path.abspath(tdir), f"f{f:04d}.png")
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"[anim-test] frame {f}: {time.time() - t0:.1f}s", flush=True)
        return
    lo, hi = (int(x) for x in a.frames.split("-"))
    hi = min(hi, n_total)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = os.path.join(os.path.abspath(fdir), "")
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
    t0 = time.time()
    sc.frame_start, sc.frame_end = lo, hi
    bpy.ops.render.render(animation=True)
    dt = time.time() - t0
    info = {"frames": [lo, hi], "n_total": n_total, "fps": FPS, "seconds": round(dt, 1),
            "per_frame_s": round(dt / max(1, hi - lo + 1), 2), "engine": "EEVEE",
            "samples": a.anim_samples, "res": a.anim_res,
            "phases_s": {"orbit": T_ORBIT, "open": T_OPEN, "inward": T_IN, "section": T_BACK}}
    tf = os.path.join(a.out_root, "anim", "render_info.json")
    prev = json.load(open(tf)) if os.path.exists(tf) else {}
    prev[f"{lo}-{hi}"] = info
    json.dump(prev, open(tf, "w"), indent=1)
    print(f"[anim] frames {lo}-{hi}: {dt:.1f}s ({info['per_frame_s']} s/frame)", flush=True)


def do_encode(a):
    """Frames (the annotated copies when present) -> H.264 MP4 via the VSE."""
    fdir = os.path.abspath(os.path.join(a.out_root, "anim", "frames_annotated"))
    if not os.path.isdir(fdir) or not [f for f in os.listdir(fdir) if f.endswith(".png")]:
        fdir = os.path.abspath(os.path.join(a.out_root, "anim", "frames"))
    files = sorted(f for f in os.listdir(fdir) if f.endswith(".png"))
    sc = bpy.context.scene
    img = bpy.data.images.load(os.path.join(fdir, files[0]))
    w, h = img.size
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.sequence_editor_create()
    seq = sc.sequence_editor
    coll = getattr(seq, "strips", None)
    if coll is None:
        coll = seq.sequences
    st = coll.new_image("frames", os.path.join(fdir, files[0]), channel=1, frame_start=1)
    for fn in files[1:]:
        st.elements.append(fn)
    sc.frame_start, sc.frame_end = 1, len(files)
    sc.render.fps = FPS
    ims = sc.render.image_settings
    if hasattr(ims, "media_type"):             # Blender 5.x: video is a media type
        ims.media_type = "VIDEO"
    ims.file_format = "FFMPEG"
    ff = sc.render.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    ff.constant_rate_factor = "HIGH"
    ff.ffmpeg_preset = "GOOD"
    ff.gopsize = FPS
    ff.audio_codec = "NONE"
    try:
        sc.view_settings.view_transform = "Standard"
    except Exception:
        pass
    out = os.path.abspath(os.path.join(a.out_root, "anim", "jet_anim.mp4"))
    sc.render.filepath = out
    sc.render.use_file_extension = False
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    print(f"[encode] {len(files)} frames -> {out} in {time.time() - t0:.1f}s", flush=True)


def main():
    a, rest = parse(argv_after())
    if a.mode == "encode":
        do_encode(a)
        return 0
    extra = ["--no-caption"]
    if a.mode == "sections":
        extra += ["--technical", "--section-res", "3000x2400"]
    args = B.parse_args(rest + extra)
    js = B.JetScene(args).build()
    js.set_active(args.active)
    if a.tag is None:
        a.tag = ("jet1983" if is_1983(js) else "jet1975") + (
            f"_strict{args.strict_year}" if args.strict_year and is_1983(js) else "")
    if a.mode == "stills":
        cams = [c.strip() for c in args.cams.split(",") if c.strip()]
        t = do_stills(js, a, cams)
        print(f"[final] times {t}", flush=True)
    elif a.mode == "sections":
        do_sections(js, a)
    elif a.mode == "anim":
        do_anim(js, a)
    elif a.mode == "probe":
        # quick look-dev: EEVEE, small, all requested cams, with anchors
        eevee(32, a.res)
        W, H = (int(x) for x in a.res.lower().split("x"))
        cams = [c.strip() for c in args.cams.split(",") if c.strip()]
        eng = (a.engine or "EEVEE").upper()
        if eng == "CYCLES":
            cycles(a.samples, a.res)
        out_dir = os.path.join(a.out_root, "_probe")
        os.makedirs(out_dir, exist_ok=True)
        for key in cams:
            js.apply_shot(key)
            cam = js.cams[js.active.label][key]
            base = os.path.join(out_dir, f"{a.tag}_{key}")
            anchors = find_anchors(js, key, cam, W, H,
                                   texts=C3_TEXT if key == "C3" else None) \
                if key in SHOT_LABELS else []
            t0 = time.time()
            bpy.context.scene.camera = cam
            bpy.context.scene.render.filepath = base + ".png"
            bpy.ops.render.render(write_still=True)
            dt = time.time() - t0
            meta = shot_meta(js, key, cam, W, H, dt, eng, a.samples,
                             {"labels": anchors, "image": os.path.basename(base) + ".png"})
            json.dump(meta, open(base + ".json", "w"), indent=1, ensure_ascii=False)
            print(f"[probe] {key} {dt:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
