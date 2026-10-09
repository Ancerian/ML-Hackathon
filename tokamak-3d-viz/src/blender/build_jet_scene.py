#!/usr/bin/env python3
"""Build the JET engineering scene (phase F3) from a machine bake.

    blender --background --python src/blender/build_jet_scene.py -- \
        --bake data/bake_jet1975 [--variant-bake data/bake_jet1983] \
        [--active variant] [--strict-year 1983] \
        --out jet.blend --render previews --engine EEVEE --res 1920x1080

Writes ``render/jet/preview_C1.png`` .. ``preview_C6.png`` and
``render/jet/render_times.json``.

Scene layout
------------
``JET_<label>``           root collection per loaded machine (``1975``, ``1983``)
  ``TF``, ``Vessel``, ...   one collection per subsystem (variant: ``TF_1983``)
  ``Exploded_<label>``      C4 variant: linked duplicates, offset (non-destructive)
  ``PlasmaSections_<label>``  psi_N-coloured section cards for C5 / C6
``Environment``           floor, sun lights, world
``InteriorLights``        work lights for C3
``Rig``                   cameras C1..C6

``--active variant`` swaps which machine renders (collections are swapped, not
rebuilt).  ``--strict-year 1983`` hides objects whose status tag says they
arrived later (e.g. the 1983 rotary valves, "installed January 1984").

The DIII-D pipeline (build_scene.py / tokblend.machine) is not used or changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokblend import machine_table as MT
from tokblend import jet_materials as JM
from tokblend import jet_cut as JC
from tokblend import jet_plasma as JP
from tokblend import jet_rig as RG
from tokblend import materials as M
from tokblend import render as RD
from tokblend import compositor as CO
from tokblend.bl import collection, clear_scene

DEG = math.pi / 180.0
PLAN_Z = 0.0


def machine_frame(tb):
    """Angles RELATIVE TO THE MACHINE, read from its own bake.

    The 1983 bake is rotated by -5.625 deg against 1975 (tf_phi0) and puts
    the iron limbs on the octant welds, so nothing here is an absolute
    azimuth: octant boundary b0 from ``vessel.info.octant_boundaries_deg``,
    limb azimuth from ``iron_core.info.limb_phi_deg``.
    """
    comps = tb.manifest.get("components", {})
    b0 = float(((comps.get("vessel", {}).get("info", {}) or {})
                .get("octant_boundaries_deg") or [5.625])[0])
    limb = float(((comps.get("iron_core", {}).get("info", {}) or {})
                  .get("limb_phi_deg") or [b0 + 11.25])[0])
    return {
        "b0": b0 * DEG,
        "rot": (b0 - 5.625) * DEG,           # rotation relative to the 1975 frame
        # wedges (start, width) on REAL octant boundaries (welds, p.299)
        "wedge_c2": (b0 * DEG - math.pi, 67.5 * DEG),          # 1.5 octants
        "wedge_c4": (b0 * DEG - 0.75 * math.pi, 45.0 * DEG),   # 1 octant
        # R-Z section through an iron limb.  Nudged 0.25 deg off the limb
        # axis: in 1983 the limbs sit ON the octant welds, where the two
        # half-sector end caps are coplanar -- a section exactly there cuts
        # through coincident faces (black z-fighting band in the first test).
        "section_phi": (limb + 0.25) * DEG,
    }

ALL_SUBS = MT.SUBSYSTEMS
NO_PLASMA = tuple(s for s in ALL_SUBS if s not in ("Plasma", "FieldLines"))

SHOTS = {
    "C1": dict(subs=NO_PLASMA, cut=None, floor=True, look="exterior",
               title="загальний вигляд; фігура людини 1,8 м для масштабу"),
    "C2": dict(subs=ALL_SUBS, cut=("wedge", "wedge_c2"), floor=True, look="exterior",
               plasma="surfaces",
               title="розріз 1,5 октанта: плазма - камера - сильфони - TF - "
                     "структура - PF - залізо"),
    "C3": dict(subs=NO_PLASMA, cut=None, floor=False, look="interior",
               title="усередині вакуумної камери (плазму приховано): лімітери, порти"),
    "C4": dict(subs=ALL_SUBS, cut=("wedge", "wedge_c4"), floor=True, look="exterior",
               plasma="surfaces",
               exploded=True,
               title="рознесений октант (зміщення вздовж власних осей, схематично)"),
    "C5": dict(subs=NO_PLASMA, cut=("plane_v", "section_phi"), floor=False, look="drawing",
               cards="rz", title="вертикальний переріз R-Z через ярмо (ортографія)"),
    "C6": dict(subs=NO_PLASMA, cut=("plane_h", "plan_z"), floor=False, look="drawing",
               cards="plan", title="план, переріз у площині Z = 0 (ортографія)"),
}


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _label_for(bake_dir, man, taken):
    name = man.get("machine", {}).get("name", os.path.basename(bake_dir.rstrip("/")))
    digits = "".join(ch for ch in name if ch.isdigit()) or name
    lab = digits
    while lab in taken:
        lab += "b"
    return lab


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
class JetScene:
    def __init__(self, args, log=print):
        self.args = args
        self.log = log
        self.tables = []
        self.plasmas = {}
        self.cards = {}         # label -> {"rz": [...], "plan": ob}
        self.exploded = {}      # label -> collection
        self.cams = {}          # label -> {"C1": cam, ...}
        self.frames = {}        # label -> machine_frame()
        self.active = None

    def load(self, bake_dir, primary):
        man = json.load(open(os.path.join(bake_dir, "manifest.json")))
        lab = _label_for(bake_dir, man, {t.label for t in self.tables})
        tb = MT.MachineTable(bake_dir, lab)
        self.primary_flags = getattr(self, "primary_flags", {})
        self.primary_flags[lab] = primary
        tb.build(lambda tag, c, p: JM.material_for_tag(tag, c, p, plasma_mat=None),
                 primary=primary, log=self.log)
        tb.ensure_colls(("Plasma", "FieldLines"), primary=primary)
        # human figure and plasma never take part in the wedge cut
        for ob in tb.objects_in("Human", "Plasma"):
            ob["tok_nocut"] = 1.0
        # plasma: volume (psi_N from physics bake, else schematic proxy) or shell
        # a variant without its own physics bake borrows the primary one
        # (same plasma design; documented in the caption note)
        fb = None if primary else self.args.bake
        pl = JP.JetPlasma(tb, log=self.log, physics_fallback=fb)
        if self.args.plasma == "shell":
            pmat = JM.plasma_shell()
        else:
            pmat = pl.volume_material()
        for ob in tb.objects_in("Plasma"):
            if not ob.data.materials:
                ob.data.materials.append(pmat)
        # with a physics bake, the volume is bounded by the PHYSICS LCFS (the
        # machine bake's Miller-D shell stays in the file, hidden)
        pv = pl.physics_volume_object(tb.colls["Plasma"], pmat)
        if pv is not None:
            for ob in tb.objects_in("Plasma"):
                ob.hide_render = ob.hide_viewport = True
                ob["tok_note"] = "superseded by PlasmaVolume_physics"
            pl.volume_objs = [pv]
        else:
            pl.volume_objs = list(tb.objects_in("Plasma"))
        flux_mat = M.flux_surface_material(f"JET_FluxSurface_{lab}", alpha=0.55, emission=1.3)
        JC.wrap_material(flux_mat, section=False)        # nested shells are cut
        line_mat = M.emissive_material(f"JET_FieldLine_{lab}", (0.30, 0.72, 1.0, 1.0), 1.4)
        pl.physics_extras(tb.colls["Plasma"], tb.colls["FieldLines"], flux_mat, line_mat,
                          bevel=0.013)
        self.plasmas[lab] = pl
        fr = machine_frame(tb)
        fr["plan_z"] = PLAN_Z
        self.frames[lab] = fr
        # section cards
        sc = collection(f"PlasmaSections_{lab}", parent=tb.root)
        cmat = JM.plasma_section_material()
        self.cards[lab] = {"rz": pl.elevation_cards(sc, cmat, fr["section_phi"]),
                           "plan": pl.plan_card(sc, cmat, z=PLAN_Z - 0.004),
                           "coll": sc}
        self.tables.append(tb)
        return tb

    def build(self):
        a = self.args
        clear_scene()
        sc = bpy.context.scene
        sc.unit_settings.system = "METRIC"
        sc.unit_settings.scale_length = 1.0
        JC.cut_group()

        base = self.load(a.bake, primary=True)
        if a.variant_bake:
            self.load(a.variant_bake, primary=False)
        if a.strict_year:
            MT.apply_strict_year(self.tables, a.strict_year, log=self.log)

        for tb in self.tables:
            w4 = self.frames[tb.label]["wedge_c4"]
            self.exploded[tb.label] = RG.build_exploded(
                tb, w4[0], w4[1], f"Exploded_{tb.label}", log=self.log)

        # environment ---------------------------------------------------------
        env = collection("Environment")
        floor_z = float(base.manifest["components"].get("human", {})
                        .get("info", {}).get("floor_Z", -5.7229)) - 0.002
        bpy.ops.mesh.primitive_plane_add(size=90.0, location=(0, 0, floor_z))
        fl = bpy.context.active_object
        fl.name = "Floor"
        for c in list(fl.users_collection):
            c.objects.unlink(fl)
        env.objects.link(fl)
        fl.data.materials.append(JM.floor_material())
        self.floor = fl
        self.suns = [RG.sun("Sun_Key", -95.0, 48.0, 3.2, (1.0, 0.97, 0.92), 3.0, env),
                     RG.sun("Sun_Fill", -200.0, 18.0, 0.9, (0.80, 0.87, 1.0), 12.0, env),
                     RG.sun("Sun_Rim", 60.0, 30.0, 1.4, (0.9, 0.93, 1.0), 4.0, env)]
        il = collection("InteriorLights")
        self.work = RG.interior_lights(base.plasma_info.get("R0", 2.96), il, n=10,
                                       power=a.work_power,
                                       skip_near=self._c3_eye(base)[0], rows=(0.0,))
        self.il = il

        # cameras: one set per loaded machine, in its own frame ---------------------
        rig = collection("Rig")
        for tb in self.tables:
            self.cams[tb.label] = self._cameras(tb, rig,
                                                "" if self.primary_flags[tb.label]
                                                else f"_{tb.label}")

        CO.build(glare_type="Bloom", size=0.35, strength=0.14, threshold=2.0)
        w, h = (int(x) for x in a.res.lower().split("x"))
        RD.setup_common(res=(w, h), depth="8")
        self.set_active(a.active)
        self.apply_shot("C2")
        return self

    # -- cameras -------------------------------------------------------------------
    def _machine_objs(self, tb, subs=None, include_human=True):
        subs = subs or [s for s in ALL_SUBS if s not in ("Plasma", "FieldLines")]
        objs = tb.objects_in(*subs)
        if not include_human:
            objs = [o for o in objs if o.get("tok_subsystem") != "Human"]
        return objs

    def _c3_eye(self, tb):
        """In-vessel eye near the OUTBOARD midplane looking toroidally (F5).

        Eye at R = 3.93 m (the rigid-sector plasma-side wall is at 4.389 m,
        the outer rail limiter face at 4.21 m), 0.30 m above the midplane,
        25 deg before a horizontal port (ports at 28.125 + 45 k deg in the
        machine frame; the octant opposite the NBI one).  The target lies
        42 deg further on at R = 2.85 m, so the torus curves away to the
        left: inboard wall left, limiter rail + port openings right, the
        D-shaped section (sector joints, bellows) receding in the middle."""
        rot = machine_frame(tb)["rot"]
        ph = (208.125 - 25.0) * DEG + rot
        eye = (3.93 * math.cos(ph), 3.93 * math.sin(ph), 0.30)
        ph2 = ph + 42.0 * DEG
        tgt = (2.85 * math.cos(ph2), 2.85 * math.sin(ph2), -0.10)
        return eye, tgt

    def _cameras(self, tb, rig, suffix=""):
        fr = self.frames[tb.label]
        pts_all = RG.bbox_points(self._machine_objs(tb))
        pts_mach = RG.bbox_points(self._machine_objs(tb, include_human=False))
        lo, hi = pts_mach.min(0), pts_mach.max(0)
        centre = 0.5 * (lo + hi)
        # headroom for the caption strip burnt in at the top of the frame
        # (the caption is a matplotlib overlay now, --no-caption: less headroom)
        hf = 0.07 if self.args.no_caption else 0.16
        head = np.array([[0.0, 0.0, hi[2] + hf * (hi[2] - lo[2])]])
        rot = math.degrees(fr["rot"])
        cams = {}
        # C1: three-quarter exterior, human figure in front-left
        d = RG._dir(-138.0 + rot, 14.0)
        tgt = (centre[0], centre[1], centre[2] - 0.2)
        pts = np.vstack([pts_all, head])
        cams["C1"] = RG.add_camera("CAM_C1_exterior" + suffix, tgt, d, lens=35.0, coll=rig,
                                   dist=RG.fit_perspective(pts, tgt, d, 35.0, margin=1.05))
        # C2: cutaway, camera on the wedge bisector
        w2 = fr["wedge_c2"]
        d = RG._dir(math.degrees(w2[0] + 0.5 * w2[1]), 26.0)
        tgt = (0.0, 0.0, 0.0)
        pts = np.vstack([pts_mach, head])
        cams["C2"] = RG.add_camera("CAM_C2_cutaway" + suffix, tgt, d, lens=40.0, coll=rig,
                                   dist=RG.fit_perspective(pts, tgt, d, 40.0, margin=1.03))
        # C3: inside the vessel, toroidal view, wide lens
        eye, look = self._c3_eye(tb)
        d = np.asarray(look) - np.asarray(eye)
        cams["C3"] = RG.add_camera("CAM_C3_invessel" + suffix, look, d, lens=12.0, coll=rig,
                                   dist=float(np.linalg.norm(d)), clip=(0.02, 100.0))
        # C4: exploded octant (fit includes the displaced duplicates)
        ex = self.exploded[tb.label]
        pts_ex = np.vstack([pts_mach, RG.bbox_points(list(ex.objects)), head])
        w4 = fr["wedge_c4"]
        d = RG._dir(math.degrees(w4[0] + 0.5 * w4[1]) - 64.0, 22.0)
        c4 = 0.5 * (pts_ex.min(0) + pts_ex.max(0))
        tgt = (c4[0], c4[1], 0.0)
        cams["C4"] = RG.add_camera("CAM_C4_exploded" + suffix, tgt, d, lens=35.0, coll=rig,
                                   dist=RG.fit_perspective(pts_ex, tgt, d, 35.0, margin=1.03))
        # C5: orthographic elevation; the camera sits in the REMOVED half-space
        # (n.P > 0) looking back along -n at the cut face
        ph = fr["section_phi"]
        n = np.array([-math.sin(ph), math.cos(ph), 0.0])
        d = -n
        sa = self.section_aspect()
        scale, c = RG.fit_ortho(pts_all, d, aspect=sa, margin=1.03)
        cams["C5"] = RG.add_camera("CAM_C5_elevation" + suffix, tuple(c), d, lens=50.0,
                                   coll=rig, dist=60.0, ortho_scale=scale, clip=(0.1, 200.0))
        # C6: orthographic plan from above.  F5: the NBI (the one long
        # appendage) points to image LEFT so the ø14.8 m iron fills the frame
        # height; the 1.8 m figure is moved next to the iron, image right.
        nbi = tb.objects_in("NBI") or tb.objects_in("Extras")
        if nbi:
            pn = RG.bbox_points(nbi).mean(0)
            nphi = math.atan2(pn[1], pn[0])
        else:
            nphi = math.pi
        right = np.array([-math.cos(nphi), -math.sin(nphi), 0.0])
        up = np.array([-right[1], right[0], 0.0])
        d = (0.0, 0.0, -1.0)
        self.c6_human = self._c6_human_spot(tb, nphi)
        pts = np.vstack([pts_mach, np.array([[*self.c6_human[:2], 0.0]])])
        scale, c = RG.fit_ortho(pts, d, aspect=sa, margin=1.03, up=tuple(up))
        c = (c[0], c[1], 0.0)
        cams["C6"] = RG.add_camera("CAM_C6_plan" + suffix, c, d, lens=50.0, coll=rig,
                                   up=tuple(up), dist=60.0, ortho_scale=scale,
                                   clip=(0.1, 200.0))
        for k, cam in cams.items():
            cam["tok_machine"] = tb.name
            self.log(f"[camera] {k} {cam.name} at {tuple(round(x, 2) for x in cam.location)} "
                     f"{cam.data.type} lens={cam.data.lens:.0f} "
                     + (f"ortho={cam.data.ortho_scale:.2f}" if cam.data.type == 'ORTHO' else ""))
        return cams

    def section_aspect(self):
        w, h = (int(x) for x in self.args.section_res.lower().split("x"))
        return w / h

    def _c6_human_spot(self, tb, nbi_phi):
        """Plan position for the scale figure in C6: just outside the iron
        (R 7.38 m), half-way between two limbs, roughly opposite the NBI."""
        limbs = ((tb.manifest.get("components", {}).get("iron_core", {})
                  .get("info", {}) or {}).get("limb_phi_deg") or [16.875])
        cands = [math.radians(l + 22.5) for l in limbs]
        want = nbi_phi + math.pi
        best = min(cands, key=lambda a: abs(math.atan2(math.sin(a - want),
                                                       math.cos(a - want))))
        r = 7.95
        return (r * math.cos(best), r * math.sin(best))

    def _place_human(self, key):
        tb = self.active
        for ob in tb.objects_in("Human"):
            if "tok_home" not in ob.keys():
                ob["tok_home"] = [ob.matrix_world[i][3] for i in range(3)]
            home = list(ob["tok_home"])
            if key == "C6" and getattr(self, "c6_human", None) is not None:
                ob.location = (self.c6_human[0], self.c6_human[1], home[2])
            else:
                ob.location = home

    # -- state -----------------------------------------------------------------------
    def set_active(self, which):
        idx = 1 if (which == "variant" and len(self.tables) > 1) else 0
        self.active = self.tables[idx]
        for i, tb in enumerate(self.tables):
            tb.root.hide_render = tb.root.hide_viewport = (i != idx)
        self.log(f"[variant] active machine: {self.active.name} ({self.active.label})")

    def apply_shot(self, key):
        shot = SHOTS[key]
        tb = self.active
        for sub, c in tb.colls.items():
            vis = sub in shot["subs"]
            c.hide_render = c.hide_viewport = not vis
        for lab, ex in self.exploded.items():
            vis = bool(shot.get("exploded")) and lab == tb.label
            ex.hide_render = ex.hide_viewport = not vis
        for lab, cd in self.cards.items():
            cd["coll"].hide_render = cd["coll"].hide_viewport = (lab != tb.label or
                                                                   not shot.get("cards"))
            for ob in cd["rz"]:
                ob.hide_render = ob.hide_viewport = shot.get("cards") != "rz"
            cd["plan"].hide_render = cd["plan"].hide_viewport = shot.get("cards") != "plan"
        self.floor.hide_render = self.floor.hide_viewport = not shot["floor"]
        # plasma representation: 'volume' (glow, uncut) | 'surfaces' (nested
        # psi_N shells, cut with the machine) | 'both'; falls back to the
        # volume when the physics bake has no flux surfaces
        pl = self.plasmas[tb.label]
        mode = self.args.plasma_mode or shot.get("plasma", "volume")
        if mode != "volume" and not pl.surfaces:
            mode = "volume"
        for ob in pl.volume_objs:
            ob.hide_render = ob.hide_viewport = mode == "surfaces"
        for ob in pl.surfaces:
            ob.hide_render = ob.hide_viewport = mode == "volume"
        # cut rig
        cut = shot["cut"]
        fr = self.frames[tb.label]
        if cut is None:
            JC.set_cut(None, None)
        elif cut[0] == "wedge":
            JC.set_cut(wedge=fr[cut[1]], plane=None)
        elif cut[0] == "plane_v":
            phi = fr[cut[1]]
            JC.set_cut(None, ((-math.sin(phi), math.cos(phi), 0.0), 0.0))
        elif cut[0] == "plane_h":
            JC.set_cut(None, ((0.0, 0.0, 1.0), float(fr[cut[1]])))
        # look
        look = shot["look"]
        interior = look == "interior"
        self.il.hide_render = self.il.hide_viewport = not interior
        for s in self.suns:
            s.hide_render = s.hide_viewport = interior
        vs = bpy.context.scene.view_settings
        JC.set_drawing(look == "drawing" and self.args.technical)
        if look == "drawing" and self.args.technical:
            # technical illustration: flat colour-coded faces on white
            RG.world(color=(1.0, 1.0, 1.0), strength=1.0)
            vs.view_transform, vs.look = "Standard", "None"
        elif look == "drawing":
            RG.world(color=(0.80, 0.81, 0.82), strength=1.0)
            self.suns[0].data.energy = 2.2
            # crisper, drawing-like tone curve for the orthographic sheets
            vs.view_transform, vs.look = "Standard", "None"
        elif interior:
            RG.world(color=(0.02, 0.022, 0.025), strength=0.3)
            vs.view_transform, vs.look = "AgX", "AgX - Base Contrast"
        else:
            RG.world(color=(0.030, 0.034, 0.040), strength=4.0)
            self.suns[0].data.energy = 3.2
            vs.view_transform, vs.look = "AgX", "AgX - Base Contrast"
        self._place_human(key)
        bpy.context.scene.camera = self.cams[tb.label][key]
        if not self.args.no_caption:
            RG.set_caption(self.caption(key), size=max(12, int(self.res_h() / 60)))
        return shot

    def res_h(self):
        return int(self.args.res.lower().split("x")[1])

    def caption(self, key):
        tb = self.active
        shot = SHOTS[key]
        subs = [s for s in shot["subs"] if s not in ("Human",)]
        by_src = list(tb.pages_by_source(subs, limit=7).items())
        meta = tb.manifest.get("machine", {}).get("meta", {})
        var = meta.get("variant") or meta.get("machine") or tb.name
        pl_long = pl_short = ""
        if "Plasma" in shot["subs"] or shot.get("cards"):
            pl_long = "  |  " + self.plasmas[tb.label].caption_note()
            pl_short = "  |  " + self.plasmas[tb.label].caption_note().split(" (")[0]
        strict = f"  |  суворо на {self.args.strict_year} р." if self.args.strict_year else ""

        def build(n_src, n_pg, pl):
            refs = "; ".join(f"{RG.SOURCE_LABEL.get(s, s)}, с. {', '.join(map(str, p[:n_pg]))}"
                             for s, p in by_src[:n_src])
            return (f"{key} - {var}: {shot['title']}.  Власна 3D-реконструкція (не копія "
                    f"креслень); побудовано за даними {refs} (сторінки PDF).{pl}{strict}")

        # stamp_note_text accepts 767 bytes, but the burnt-in stamp is cut at
        # ~512 UTF-8 bytes (measured: Cyrillic = 2 bytes).  Shorten
        # progressively instead of letting Blender truncate mid-word.
        budget = 500
        for n_src, n_pg, pl in ((3, 7, pl_long), (3, 5, pl_long), (3, 5, pl_short),
                                (2, 5, pl_short), (2, 4, pl_short), (1, 5, pl_short),
                                (1, 4, "")):
            text = build(n_src, n_pg, pl)
            if len(text.encode("utf-8")) <= budget:
                return text
        return text.encode("utf-8")[:budget].decode("utf-8", "ignore")


# ---------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--variant-bake", default=None)
    ap.add_argument("--active", choices=["base", "variant"], default="base")
    ap.add_argument("--strict-year", type=int, default=None)
    ap.add_argument("--strict-1983", action="store_true",
                    help="shorthand for --strict-year 1983")
    ap.add_argument("--out", default="jet.blend")
    ap.add_argument("--render", choices=["none", "previews"], default="none")
    ap.add_argument("--cams", default="C1,C2,C3,C4,C5,C6")
    ap.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--render-dir", default="render/jet")
    ap.add_argument("--plasma", choices=["volume", "shell"], default="volume")
    ap.add_argument("--plasma-mode", choices=["volume", "surfaces", "both"], default=None,
                    help="override the per-shot plasma representation")
    ap.add_argument("--work-power", type=float, default=40.0)
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--technical", action="store_true",
                    help="C5/C6 in the technical-illustration style (flat colour-coded)")
    ap.add_argument("--section-res", default="1920x1080",
                    help="frame used to fit the C5/C6 orthographic cameras")
    args = ap.parse_args(argv)
    if args.strict_1983 and not args.strict_year:
        args.strict_year = 1983
    return args


def main():
    args = parse_args(argv_after_dashdash())

    t0 = time.time()
    js = JetScene(args).build()
    print(f"[build] scene built in {time.time() - t0:.1f}s; objects: "
          f"{len(bpy.data.objects)}, meshes: {len(bpy.data.meshes)}", flush=True)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    print(f"[build] saved {args.out}", flush=True)

    if args.render == "previews":
        os.makedirs(args.render_dir, exist_ok=True)
        eng = "CYCLES" if args.engine == "CYCLES" else "BLENDER_EEVEE"
        if eng == "CYCLES":
            RD.setup_cycles(samples=args.samples)
        else:
            RD.setup_eevee(samples=args.samples)
            bpy.context.scene.eevee.volumetric_samples = 64
            try:
                bpy.context.scene.eevee.shadow_pool_size = "1024"
            except Exception:
                pass
        times = {}
        for key in [c.strip() for c in args.cams.split(",") if c.strip()]:
            js.apply_shot(key)
            times[key] = RD.render_still(js.cams[js.active.label][key],
                                         os.path.join(args.render_dir, f"preview_{key}"),
                                         engine=eng)
        tf = os.path.join(args.render_dir, "render_times.json")
        prev = json.load(open(tf)) if os.path.exists(tf) else {}
        prev.update({k: {"seconds": round(v, 2), "engine": args.engine, "res": args.res,
                         "samples": args.samples,
                         "machine": js.active.name} for k, v in times.items()})
        json.dump(prev, open(tf, "w"), indent=1)
        js.apply_shot("C2")
        print(f"[render] times: {times}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
