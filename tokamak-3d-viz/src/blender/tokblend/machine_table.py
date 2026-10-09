"""Load an engineering machine bake (``tokviz.machine/1``) into Blender.

The bake is written by ``src/run_bake_machine.py`` (see
``src/tokviz/machine/build.py``): one ``<component>.npz`` per component plus a
``manifest.json``.  For part ``i`` of a component the archive holds

    v_iii (N,3) local vertices      f_iii (M,4) quads      t_iii (K,3) triangles
    x_iii (I,4,4) instance transforms (world = X @ [v, 1])   l_iii (I,) labels

and ``part_names`` / ``materials`` (the material TAG, e.g. ``inconel600``).

What this module does
---------------------
* one collection per subsystem (TF, Vessel, Bellows, Ports, PF, IronCore,
  Structure, Limiters, NBI, Plasma, FieldLines, Human, + Extras for
  anything a later variant adds that matches no rule), all under one root
  collection per machine (``JET_1975``);
* every instance becomes a LINKED DUPLICATE: one mesh datablock per part,
  shared by all of that part's objects, each object carrying its stored
  transform.  32 TF coils -> 32 objects, 1 mesh;
* custom properties on each object: component id, part, label, material tag,
  subsystem, provenance summary (sources / pages / confidence), status tag;
  the full provenance list is stored once, as JSON, on the shared mesh;
* thin shells (``meta.thin_shell``, the bellows) get a Solidify modifier with
  the recorded ``thickness_m``;
* a status tag (``tok_status_tag``) is derived from a component/part
  ``status`` text, e.g. the 1983 rotary valves "installation planned January
  1984" -> ``installed_1984``; :func:`apply_strict_year` hides such objects in
  a strict as-of-year view.

Nothing here is JET-specific except the default subsystem rules; the DIII-D
pipeline (``machine.py``) is untouched.
"""
from __future__ import annotations

import json
import math
import os
import re

import bpy
import numpy as np

from .bl import collection

# ---------------------------------------------------------------------------
# subsystem classification
# ---------------------------------------------------------------------------
#: Canonical subsystem collections, in inside -> outside "layer" order.
SUBSYSTEMS = ("Plasma", "FieldLines", "Vessel", "Bellows", "Limiters", "Ports",
              "TF", "Structure", "PF", "IronCore", "NBI", "Human", "Extras")

#: Ordered (regex on component name or type, subsystem).  First match wins.
SUBSYSTEM_RULES = (
    (r"limiter_port", "Ports"),
    (r"^plasma|lcfs|flux_surf", "Plasma"),
    (r"field_?line", "FieldLines"),
    (r"bellows", "Bellows"),
    (r"^tf|toroidal_field", "TF"),
    (r"^pf|poloidal_field", "PF"),
    (r"iron", "IronCore"),
    (r"limiter", "Limiters"),
    (r"^nbi|neutral_beam|injector", "NBI"),
    (r"port", "Ports"),
    (r"vessel|first_wall", "Vessel"),
    (r"structure|ring|shell|cylinder|mechanical", "Structure"),
    (r"human|figure|person", "Human"),
    (r"pump|valve|gas|adaptor|diagnostic", "Extras"),
)

#: Parts that belong to the central column: never removed by the cutaway
#: ("everything OUTBOARD of the central column" is cut).
CENTRAL_COLUMN_RE = re.compile(
    r"coil1|central_pillar|pillar_top_hub|centre_piece|center_piece|inner_cylinder")

_CONF_RANK = {"text": 5, "table": 5, "drawing": 4, "digitized": 3,
              "assumed": 1, None: 0}


def classify(comp_name, comp_type=""):
    for pat, sub in SUBSYSTEM_RULES:
        if re.search(pat, comp_name) or (comp_type and re.search(pat, comp_type)):
            return sub
    return "Extras"


# ---------------------------------------------------------------------------
# provenance / status helpers
# ---------------------------------------------------------------------------
def provenance_summary(entries):
    """Compress a component's provenance list into a few object properties."""
    sources, pages, conf = [], set(), {}
    for e in entries or []:
        src = e.get("source")
        if src and src not in sources:
            sources.append(src)
        c = e.get("confidence")
        conf[c] = conf.get(c, 0) + 1
        if e.get("page") is not None and src not in (None, "KERNEL"):
            try:
                pages.add(int(e["page"]))
            except (TypeError, ValueError):
                pass
    known = [c for c in conf if c in _CONF_RANK and c is not None]
    worst = min(known, key=lambda c: _CONF_RANK[c]) if known else "unknown"
    return {
        "tok_sources": ", ".join(str(s) for s in sources),
        "tok_pages": sorted(pages),
        "tok_confidence": " ".join(f"{k}:{v}" for k, v in sorted(
            conf.items(), key=lambda kv: -_CONF_RANK.get(kv[0], 0)) if k),
        "tok_confidence_min": worst,
    }


def status_tag(text):
    """``'... installation planned January 1984'`` -> ``'installed_1984'``.

    Returns '' when the text does not say the item was absent / later.
    """
    if not text:
        return ""
    t = str(text).lower()
    years = re.findall(r"\b(19[5-9]\d|20\d\d)\b", t)
    later = any(w in t for w in ("not on the machine", "planned", "to be",
                                 "installed in", "installation", "later",
                                 "not installed", "not yet"))
    if years and later:
        return f"installed_{max(years)}"
    if "removed" in t or "dismantled" in t:
        return f"removed_{max(years)}" if years else "removed"
    return ""


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def make_mesh(name, v, q, t, smooth_angle_deg=32.0):
    """Mesh datablock from vertex / quad / triangle arrays (bulk foreach_set)."""
    v = np.asarray(v, np.float32).reshape(-1, 3)
    q = np.asarray(q, np.int32).reshape(-1, 4)
    t = np.asarray(t, np.int32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.ravel())
    loops = np.concatenate([q.ravel(), t.ravel()]).astype(np.int32)
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    starts = np.concatenate([np.arange(len(q), dtype=np.int32) * 4,
                             len(q) * 4 + np.arange(len(t), dtype=np.int32) * 3])
    me.polygons.add(len(q) + len(t))
    me.polygons.foreach_set("loop_start", starts.astype(np.int32))
    me.update(calc_edges=True)
    me.validate(verbose=False)
    if len(me.polygons):
        me.shade_smooth()
        if hasattr(me, "set_sharp_from_angle"):          # 4.1+
            me.set_sharp_from_angle(angle=math.radians(smooth_angle_deg))
    return me


def _mat4(X):
    from mathutils import Matrix
    return Matrix([[float(X[r, c]) for c in range(4)] for r in range(4)])


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------
class MachineTable:
    """Everything built from one bake: collections, objects, manifest."""

    def __init__(self, bake_dir, label):
        self.bake = bake_dir
        self.label = label
        self.manifest = json.load(open(os.path.join(bake_dir, "manifest.json")))
        self.root = None
        self.colls = {}          # subsystem -> collection
        self.objects = []        # every object created
        self.by_sub = {}         # subsystem -> [objects]
        self.meshes = {}         # (component, part) -> mesh
        self.plasma_info = {}

    # -- names ---------------------------------------------------------------
    @property
    def name(self):
        return self.manifest.get("machine", {}).get("name", self.label)

    def coll_name(self, sub, primary):
        return sub if primary else f"{sub}_{self.label}"

    # -- build ---------------------------------------------------------------
    def build(self, material_for, primary=True, log=print, skip=()):
        """Create collections and linked-duplicate objects.

        ``material_for(tag, comp, part)`` returns a bpy Material for a bake
        material tag.  ``primary`` machines get the bare subsystem collection
        names (``TF``); a loaded variant gets ``TF_1983`` etc.
        """
        man = self.manifest
        self.root = collection(f"JET_{self.label}")
        self.root["tok_machine"] = self.name
        self.root["tok_variant_label"] = self.label
        comps = man.get("components", {})
        for comp_name, comp in comps.items():
            if comp_name in skip:
                continue
            sub = classify(comp_name, comp.get("type", ""))
            coll = self._coll(sub, primary)
            path = os.path.join(self.bake, comp.get("file", f"{comp_name}.npz"))
            if not os.path.exists(path):
                log(f"[machine] {comp_name}: {path} missing, skipped")
                continue
            z = np.load(path, allow_pickle=False)
            names = [str(s) for s in z["part_names"]]
            tags = [str(s) for s in z["materials"]]
            prov = comp.get("provenance", []) or []
            summ = provenance_summary(prov)
            spec_params = (comp.get("spec") or {}).get("params") or {}
            comp_status = str(spec_params.get("status", "") or
                              (comp.get("info") or {}).get("status", "") or "")
            # 1983-format fields: status ('installed' | 'not_installed'),
            # hide_in_strict_variant (bool); part meta 'installed' ('1984-01')
            comp_state = str(comp.get("status", "") or "")
            comp_hide_strict = bool(comp.get("hide_in_strict_variant", False))
            comp_variant = str(comp.get("variant", "") or "")
            prov_json = json.dumps(prov, ensure_ascii=False)
            n_obj = 0
            for i, pname in enumerate(names):
                v = z[f"v_{i:03d}"]
                q = z[f"f_{i:03d}"]
                t = z[f"t_{i:03d}"]
                X = z[f"x_{i:03d}"]
                labels = [str(s) for s in z[f"l_{i:03d}"]]
                pmeta = (comp.get("parts", {}).get(pname, {}) or {}).get("meta", {}) or {}
                me = make_mesh(f"{self.label}:{comp_name}:{pname}", v, q, t)
                me["tok_provenance_json"] = prov_json
                me["tok_component"] = comp_name
                me["tok_part"] = pname
                mat = material_for(tags[i], comp_name, pname)
                if mat is not None:
                    me.materials.append(mat)
                self.meshes[(comp_name, pname)] = me
                note = str(pmeta.get("status_note", "") or comp_status)
                state = str(pmeta.get("status", "") or comp_state)
                status = note or state
                stag = status_tag(note)
                inst = str(pmeta.get("installed", "") or "")
                if re.match(r"\d{4}", inst) and (comp_hide_strict or "not" in state):
                    stag = f"installed_{inst[:4]}"
                elif comp_hide_strict and not stag:
                    stag = "not_installed"
                nocut = bool(CENTRAL_COLUMN_RE.search(pname) or
                             CENTRAL_COLUMN_RE.search(comp_name))
                for k in range(X.shape[0]):
                    lab = labels[k] if k < len(labels) else f"{k:03d}"
                    obn = f"{self.label}_{comp_name}.{pname}.{lab}"
                    ob = bpy.data.objects.new(obn[:63], me)
                    ob.matrix_world = _mat4(X[k])
                    coll.objects.link(ob)
                    ob["tok_machine"] = self.name
                    ob["tok_component"] = comp_name
                    ob["tok_component_type"] = comp.get("type", "")
                    ob["tok_part"] = pname
                    ob["tok_label"] = lab
                    ob["tok_instance"] = k
                    ob["tok_material_tag"] = tags[i]
                    ob["tok_subsystem"] = sub
                    for kk, vv in summ.items():
                        ob[kk] = vv
                    ob["tok_status"] = status
                    ob["tok_status_tag"] = stag
                    ob["tok_state"] = state
                    ob["tok_hide_in_strict"] = 1 if (comp_hide_strict or stag) else 0
                    ob["tok_variant_of"] = comp_variant
                    ob["tok_nocut"] = 1.0 if nocut else 0.0
                    ob["tok_explode"] = 0.0
                    ob["tok_off"] = (0.0, 0.0, 0.0)
                    if pmeta.get("thin_shell"):
                        th = float(pmeta.get("thickness_m", 0.002))
                        sol = ob.modifiers.new("Solidify", "SOLIDIFY")
                        sol.thickness = th
                        sol.offset = 0.0
                        sol.use_even_offset = True
                        sol.use_rim = True
                        ob["tok_thin_shell_m"] = th
                    self.objects.append(ob)
                    self.by_sub.setdefault(sub, []).append(ob)
                    n_obj += 1
            if comp_name == "plasma" or sub == "Plasma":
                self.plasma_info = comp.get("info", {}) or {}
            log(f"[machine] {self.label} {comp_name:<12} -> {coll.name:<12} "
                f"{len(names)} parts, {n_obj} objects")
        return self

    def _coll(self, sub, primary):
        if sub not in self.colls:
            c = collection(self.coll_name(sub, primary), parent=self.root)
            c["tok_subsystem"] = sub
            c["tok_machine"] = self.name
            self.colls[sub] = c
        return self.colls[sub]

    # -- queries ---------------------------------------------------------------
    def ensure_colls(self, subs, primary=True):
        for s in subs:
            self._coll(s, primary)
        return self

    def objects_in(self, *subs):
        out = []
        for s in subs:
            out += self.by_sub.get(s, [])
        return out

    def pages_by_source(self, subs, limit=7, visible_only=True):
        """{source: [pages]} of non-assumed provenance for the given subsystems,
        most-cited first, skipping components whose objects are all hidden."""
        hidden = set()
        if visible_only:
            per = {}
            for ob in self.objects:
                per.setdefault(ob["tok_component"], []).append(ob.hide_render)
            hidden = {c for c, h in per.items() if all(h)}
        count = {}
        for comp_name, comp in self.manifest.get("components", {}).items():
            if comp_name in hidden:
                continue
            if classify(comp_name, comp.get("type", "")) not in subs:
                continue
            for e in comp.get("provenance", []) or []:
                src, pg = e.get("source"), e.get("page")
                if pg is None or src in (None, "KERNEL") or e.get("confidence") == "assumed":
                    continue
                d = count.setdefault(src, {})
                d[int(pg)] = d.get(int(pg), 0) + 1
        out = {}
        for src, d in sorted(count.items(), key=lambda kv: -sum(kv[1].values())):
            out[src] = sorted(sorted(d, key=lambda p: -d[p])[:limit])
        return out

    def pages_for(self, subs, source_prefix=None, limit=6):
        """Most-cited source pages for the given subsystems (for captions)."""
        count = {}
        for comp_name, comp in self.manifest.get("components", {}).items():
            if classify(comp_name, comp.get("type", "")) not in subs:
                continue
            for e in comp.get("provenance", []) or []:
                src, pg = e.get("source"), e.get("page")
                if pg is None or src in (None, "KERNEL"):
                    continue
                if source_prefix and not str(src).startswith(source_prefix):
                    continue
                if e.get("confidence") == "assumed":
                    continue
                count[int(pg)] = count.get(int(pg), 0) + 1
        best = sorted(count, key=lambda p: -count[p])[:limit]
        return sorted(best)


def apply_strict_year(tables, year, log=print):
    """Hide every object whose status tag says it arrived after ``year``."""
    n = 0
    for tb in tables:
        for ob in tb.objects:
            tag = ob.get("tok_status_tag", "")
            m = re.match(r"installed_(\d{4})", tag or "")
            hide = bool(m and int(m.group(1)) > int(year)) or tag == "not_installed"
            ob["tok_strict_hidden"] = 1 if hide else 0
            if hide:
                ob.hide_render = True
                ob.hide_viewport = True
                n += 1
    log(f"[machine] strict-{year}: hid {n} objects with a later status tag")
    return n


def world_bounds(objs):
    """(max radius from Z axis, zmin, zmax, max 3-D radius) over objects."""
    from mathutils import Vector
    rmax, zmin, zmax, r3 = 0.0, 1e9, -1e9, 0.0
    for ob in objs:
        if ob.type != "MESH":
            continue
        mw = ob.matrix_world
        for c in ob.bound_box:
            p = mw @ Vector(c)
            rmax = max(rmax, math.hypot(p.x, p.y))
            zmin, zmax = min(zmin, p.z), max(zmax, p.z)
            r3 = max(r3, p.length)
    return rmax, zmin, zmax, r3
