"""Run the component builders and write the bake (npz per component + manifest).

Bake layout (one directory)::

    <component>.npz   part_names (P,), materials (P,), and for part i:
                      v_iii (N,3) float32 vertices, local frame, metres
                      f_iii (M,4) int32 quads, t_iii (K,3) int32 triangles
                      x_iii (I,4,4) float32 instance transforms (world = X @ [v,1])
                      l_iii (I,) instance labels
    manifest.json     existing tokviz manifest ("objects", created_utc, ...)
                      + "machine" (name, lineage, overrides, source files)
                      + "components" (parts, counts, materials, provenance)

Instances are transforms, never copies; normals are made consistent and
outward for closed solids (``meta.closed_solid``); thin shells (bellows)
carry ``meta.thin_shell`` + thickness for a solidify modifier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import re

from ..export import Bake
from .components import RECORD_TYPES, REGISTRY
from .components.base import ComponentResult, RecordOnly
from .schema import MachineDB, Tracker


@dataclass
class MachineBuild:
    db: MachineDB
    components: dict = field(default_factory=dict)   # name -> ComponentResult
    unbuilt: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def __getitem__(self, name) -> ComponentResult:
        return self.components[name]


def _removed_by(db: MachineDB, msg: str):
    """The components_remove record that deleted the database entry named in a
    builder's KeyError message (None if the input is simply absent)."""
    m = re.search(r"'([A-Za-z0-9_./-]+)'", msg)
    if not m:
        return None
    path = m.group(1)
    for a in db.applied:
        if a.get("op") != "remove":
            continue
        tgt = str(a.get("target", ""))
        if tgt == path or path.startswith(tgt + ".") or \
                (tgt.startswith("profiles/") and Path(tgt).stem == path):
            return a
    return None


def build_machine(db: MachineDB, only=None, log=None) -> MachineBuild:
    out = MachineBuild(db)
    for spec in db.components:
        if only and spec.name not in only:
            continue
        fn = REGISTRY.get(spec.type)
        if fn is None:
            rec = spec.type in RECORD_TYPES
            out.unbuilt.append({"name": spec.name, "type": spec.type,
                                "state": "record_only" if rec else "no_builder",
                                "reason": ("record-only entry: facts kept in the manifest, "
                                           "no geometry") if rec else "no builder for this type",
                                "provenance": spec.provenance, "params": spec.params})
            continue
        t0 = time.time()
        try:
            res = fn(Tracker(db, spec), spec)
        except RecordOnly as exc:
            out.unbuilt.append({"name": spec.name, "type": spec.type, "state": "record_only",
                                "reason": f"record-only entry: {exc}",
                                "provenance": spec.provenance, "params": spec.params})
            continue
        except KeyError as exc:          # an input was removed / is absent in this variant
            msg = str(exc.args[0] if exc.args else exc)
            rm = _removed_by(db, msg)
            out.unbuilt.append({"name": spec.name, "type": spec.type,
                                "state": "removed" if rm else "missing_input",
                                "reason": (f"inputs removed by the variant "
                                           f"(components_remove '{rm['target']}'): {msg}")
                                if rm else f"missing input: {msg}",
                                "removed_by": rm, "provenance": spec.provenance})
            if log:
                log(f"  {spec.name:<10} NOT BUILT: {'removed' if rm else 'missing input'} {exc}")
            continue
        out.timings[spec.name] = time.time() - t0
        out.components[spec.name] = res
        if log:
            nv = sum(len(p.mesh.v) for p in res.parts)
            log(f"  {spec.name:<10} {len(res.parts):2d} parts {nv:7d} verts "
                f"({out.timings[spec.name]:.1f}s)")
    # entries whose params another builder consumed are realised, not unbuilt
    by = {}
    for name, res in out.components.items():
        for c in res.consumed:
            by.setdefault(c, []).append(name)
    for u in out.unbuilt:
        if u["name"] in by and u.get("state") in ("no_builder", "record_only"):
            u["state"] = "realized"
            u["realized_by"] = by[u["name"]]
            u["reason"] = (f"params realised by component(s) {', '.join(by[u['name']])}")
    return out


def _variant_of(db: MachineDB, origin: str) -> str:
    """Lineage layer (machine dir name) that introduced an entry."""
    for name in reversed(db.lineage):
        if f"machines/{name}/" in str(origin):
            return name
    return db.lineage[0]


def _status_of(spec_params: dict, res: ComponentResult | None = None) -> dict:
    """Installation status tags ('status' param, e.g. the 1983 rotary valves)."""
    st = {}
    if res is not None and res.info.get("status"):
        st = {k: res.info[k] for k in ("status", "status_note", "installed") if k in res.info}
    elif isinstance(spec_params, dict) and spec_params.get("status"):
        t = str(spec_params["status"])
        st = {"status": "not_installed" if re.search(r"not on the machine|planned", t, re.I)
              else "installed", "status_note": t}
    if not st:
        st = {"status": "installed"}
    st["hide_in_strict_variant"] = st["status"] != "installed"
    return st


def _stats(res: ComponentResult) -> dict:
    parts = {}
    for p in res.parts:
        parts[p.name] = {
            "material": p.material,
            "n_verts": int(len(p.mesh.v)), "n_quads": int(len(p.mesh.q)),
            "n_tris": int(len(p.mesh.t)), "n_instances": p.n_instances,
            "labels": list(p.labels), "meta": _json(p.meta),
            "bbox_local": [p.mesh.v.min(0).round(4).tolist(), p.mesh.v.max(0).round(4).tolist()],
        }
    return parts


def _json(v):
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_bake(mb: MachineBuild, out_dir) -> Path:
    bake = Bake(out_dir)
    comps, totals = {}, {"verts": 0, "faces": 0, "verts_instanced": 0, "faces_instanced": 0}
    for name, res in mb.components.items():
        arrays = {"part_names": np.asarray([p.name for p in res.parts]),
                  "materials": np.asarray([p.material for p in res.parts])}
        for i, p in enumerate(res.parts):
            arrays[f"v_{i:03d}"] = p.mesh.v.astype(np.float32)
            arrays[f"f_{i:03d}"] = p.mesh.q.astype(np.int32)
            arrays[f"t_{i:03d}"] = p.mesh.t.astype(np.int32)
            arrays[f"x_{i:03d}"] = p.transforms.astype(np.float32)
            arrays[f"l_{i:03d}"] = np.asarray(p.labels)
            nv, nf = len(p.mesh.v), p.mesh.n_faces
            totals["verts"] += nv
            totals["faces"] += nf
            totals["verts_instanced"] += nv * p.n_instances
            totals["faces_instanced"] += nf * p.n_instances
        bake.save(name, **arrays)
        spec = next((c for c in mb.db.components if c.name == name), None)
        origin = spec.origin if spec is not None else ""
        layers = sorted({_variant_of(mb.db, e.get("origin", "")) for e in res.provenance
                         if e.get("origin")} - {_variant_of(mb.db, origin)})
        comps[name] = {
            "file": f"{name}.npz", "type": res.type,
            "variant": _variant_of(mb.db, origin),
            "modified_by_variants": layers,
            **_status_of(spec.params if spec is not None else {}, res),
            "parts": _stats(res),
            "n_instances_total": int(sum(p.n_instances for p in res.parts)),
            "provenance": res.provenance,
            "assumed_inputs": res.assumptions,
            "spec": {"params": _json(res.params), "provenance": _json(res.spec_provenance)},
            "info": _json(res.info),
            "realizes": sorted(res.consumed),
        }
    db = mb.db
    for u in mb.unbuilt:
        spec = next((c for c in db.components if c.name == u["name"]), None)
        u["variant"] = _variant_of(db, spec.origin if spec is not None else "")
        u.update(_status_of(u.get("params") or (spec.params if spec is not None else {})))
    index = [{"name": n, "type": c["type"], "state": "built", "variant": c["variant"],
              "status": c["status"]} for n, c in comps.items()]
    index += [{"name": u["name"], "type": u["type"], "state": u.get("state", "unbuilt"),
               "variant": u["variant"], "status": u["status"],
               **({"realized_by": u["realized_by"]} if "realized_by" in u else {})}
              for u in mb.unbuilt]
    for a in db.applied:
        if a.get("op") == "remove":
            index.append({"name": a["target"], "type": "remove", "state": "removed",
                          "variant": _variant_of(db, a.get("origin", "")),
                          "removed": a.get("removed"), "reason": a.get("reason"),
                          "source": a.get("source"), "page": a.get("page")})
    bake.set("machine", {
        "name": db.name, "lineage": db.lineage, "meta": _json(db.meta),
        "scale": 1.0, "units": "m",
        "frame": "Z = machine axis, Z=0 equatorial plane; component meshes are in a "
                 "local frame, world = transform @ [v, 1]",
        "source_files_sha256_16": db.files,
        "overrides_applied": _json(db.applied),
        "warnings": db.warnings,
        "variant": db.meta.get("variant", db.name),
        "unbuilt_components": _json(mb.unbuilt),
        "component_index": _json(index),
    })
    bake.set("bake_format", {
        "kind": "tokviz.machine/1",
        "arrays": {"v_iii": "float32 (N,3) local vertices", "f_iii": "int32 (M,4) quads",
                   "t_iii": "int32 (K,3) triangles", "x_iii": "float32 (I,4,4) instance transforms",
                   "l_iii": "instance labels", "part_names": "(P,)", "materials": "(P,)"},
        "normals": "closed solids oriented outward; bellows are open thin shells",
    })
    bake.set("components", comps)
    bake.set("mesh_totals", totals)
    return bake.finish()


def load_component(npz_path) -> list:
    """Read a component archive back: list of dicts (name, material, v, q, t, X, labels)."""
    z = np.load(npz_path, allow_pickle=False)
    out = []
    for i, nm in enumerate(z["part_names"]):
        out.append({"name": str(nm), "material": str(z["materials"][i]),
                    "v": z[f"v_{i:03d}"], "q": z[f"f_{i:03d}"], "t": z[f"t_{i:03d}"],
                    "X": z[f"x_{i:03d}"], "labels": [str(s) for s in z[f"l_{i:03d}"]]})
    return out
