"""Machine description: dataclasses, validation, and a loader with inheritance.

Two kinds of machine directory are understood:

* **base** -- ``dimensions.yaml`` (sections of ``- {id, value, unit, source,
  page, confidence, note, ...}`` entries, see ``machines/jet1975``) plus
  ``profiles/*.csv`` and, optionally, ``ASSUMPTIONS_KERNEL.md`` whose fenced
  ``yaml`` block lists the geometry-kernel assumptions (kept OUT of the sourced
  database on purpose).
* **variant** -- ``deltas.yaml``::

      base: jet1975                     # name (sibling dir) or path
      meta: {machine: ...}              # optional, merged
      overrides:
        - {path: tf_coils.overall_height, value: 5.9, unit: m,
           source: JET1983, page: 12, confidence: text, note: "..."}
      components_add:
        - {name: extra_coils, type: rect_coil_set, params: {...},
           source: ..., page: ..., confidence: ..., note: ...}
      components_remove: [nbi]          # or [{name: nbi, note: ...}]

  ``path`` is ``<section>.<id>`` (replaces the value, and the provenance, of
  that entry -- or adds it if absent), ``<section>.<id>.<field>`` (sets one
  field), ``kernel.<id>`` (kernel assumption) or
  ``components.<name>.<param>`` / ``components.<name>.params.<param>``.
  A variant may also carry its own ``profiles/`` (searched first) and its own
  ``ASSUMPTIONS_KERNEL.md`` (merged over the base one).

Every value a builder reads goes through a :class:`Tracker`, so each component
carries the exact list of database entries (source, page, confidence) it was
built from into the bake manifest.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONFIDENCE = ("text", "table", "drawing", "digitized", "assumed")
RESERVED = {"meta", "components", "kernel", "base", "overrides",
            "components_add", "components_remove"}
PROV_KEYS = ("source", "page", "confidence", "note", "figure", "table", "derived", "reason")
REPO_ROOT = Path(__file__).resolve().parents[3]
MACHINES_DIR = REPO_ROOT / "machines"


class SchemaError(ValueError):
    """The machine description is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class Dim:
    """One sourced value."""
    section: str
    id: str
    value: Any
    unit: str
    source: str
    confidence: str
    page: Any = None
    note: str = ""
    derived: bool = False
    ref: str | None = None          # figure / table label
    origin: str = ""                # file (layer) the entry came from
    supersedes: dict | None = None  # provenance of the base value it replaced

    @property
    def path(self) -> str:
        return f"{self.section}.{self.id}"

    def prov(self) -> dict:
        d = {"path": self.path, "value": _plain(self.value), "unit": self.unit,
             "source": self.source, "page": self.page,
             "confidence": self.confidence, "origin": self.origin}
        if self.ref:
            d["ref"] = self.ref
        if self.derived:
            d["derived"] = True
        if self.note:
            d["note"] = self.note
        if self.supersedes:
            d["supersedes"] = self.supersedes
        return d


@dataclass
class Profile:
    name: str
    file: str
    columns: list
    header: list
    confidence: str
    data: Any                        # ndarray (n, k) or {part: ndarray}

    def prov(self) -> dict:
        return {"path": f"profiles.{self.name}", "file": self.file,
                "confidence": self.confidence,
                "source_line": next((h for h in self.header if h), "")[:200]}


@dataclass
class ComponentSpec:
    name: str
    type: str
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    origin: str = ""


@dataclass
class MachineDB:
    name: str
    root: Path
    lineage: list
    meta: dict
    dims: dict                       # section -> {id -> Dim}
    profile_dirs: list
    components: list
    applied: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    files: dict = field(default_factory=dict)       # rel path -> sha256
    removed_profiles: set = field(default_factory=set)
    _profiles: dict = field(default_factory=dict, repr=False)

    # -- access ---------------------------------------------------------------
    def dim(self, path: str) -> Dim:
        sec, _, key = path.partition(".")
        try:
            return self.dims[sec][key]
        except KeyError:
            raise KeyError(f"{self.name}: no database entry '{path}'") from None

    def value(self, path: str):
        return self.dim(path).value

    def has(self, path: str) -> bool:
        sec, _, key = path.partition(".")
        return key in self.dims.get(sec, {})

    def component(self, name: str) -> ComponentSpec:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    def profile(self, name: str) -> Profile:
        if name in self.removed_profiles:
            raise KeyError(f"{self.name}: profile '{name}' removed by a delta")
        if name not in self._profiles:
            for d in self.profile_dirs:
                p = Path(d) / f"{name}.csv"
                if p.exists():
                    self._profiles[name] = _read_profile(name, p)
                    break
            else:
                raise KeyError(f"{self.name}: no profile '{name}' in {self.profile_dirs}")
        return self._profiles[name]


class Tracker:
    """Records every database value, profile and kernel assumption a builder
    reads, so the manifest can say exactly what each mesh is made of."""

    def __init__(self, db: MachineDB, spec: ComponentSpec):
        self.db, self.spec = db, spec
        self.used: dict = {}
        self.params_used: dict = {}
        self.consumed: dict = {}        # component name -> type (params used)

    def dim(self, path: str) -> Dim:
        d = self.db.dim(path)
        self.used[d.path] = d.prov()
        return d

    def v(self, path: str, index: int | None = None) -> Any:
        val = self.dim(path).value
        if index is not None:
            return float(val[index])
        return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else val

    def k(self, key: str) -> Any:
        return self.v(f"kernel.{key}")

    def profile(self, name: str, part: str | None = None):
        pr = self.db.profile(name)
        self.used[f"profiles.{name}"] = pr.prov()
        if part is None:
            return pr.data
        return pr.data[part]

    def component(self, ctype: str, pred=None):
        """First database component of type ``ctype`` (optionally matching
        ``pred(spec)``) whose params this builder consumes -- e.g. a variant's
        layout entry.  Recorded in the provenance (``components.<name>``) and
        in ``consumed`` so the manifest can say which builder realised it.
        Returns None when the variant has no such component."""
        for c in self.db.components:
            if c.type == ctype and (pred is None or pred(c)):
                self.consumed[c.name] = c.type
                self.used[f"components.{c.name}"] = {
                    "path": f"components.{c.name}", "type": c.type, "origin": c.origin,
                    **{k: _plain(v) for k, v in c.provenance.items()}}
                return c
        return None

    def own(self) -> None:
        """Record the builder's own component entry in the provenance."""
        c = self.spec
        self.used[f"components.{c.name}"] = {
            "path": f"components.{c.name}", "type": c.type, "origin": c.origin,
            **{k: _plain(v) for k, v in c.provenance.items()}}

    def p(self, key: str, default=None):
        val = self.spec.params.get(key, default)
        self.params_used[key] = _plain(val)
        return val

    def provenance(self) -> list:
        return [self.used[k] for k in sorted(self.used)]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _yaml_load(text: str, where: str):
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SchemaError("PyYAML is required to read machine databases "
                          "(pip install pyyaml)") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SchemaError(f"{where}: invalid YAML: {exc}") from exc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _make_dim(section: str, e: dict, origin: str, where: str,
              default_source: str | None = None,
              default_conf: str | None = None) -> Dim:
    if not isinstance(e, dict):
        raise SchemaError(f"{where}: entry in '{section}' is not a mapping: {e!r}")
    missing = [k for k in ("id", "value") if k not in e]
    if missing:
        raise SchemaError(f"{where}: {section} entry {e.get('id')!r} lacks {missing}")
    src = e.get("source", default_source)
    conf = e.get("confidence", default_conf)
    if src is None or conf is None:
        raise SchemaError(f"{where}: {section}.{e['id']} has no source/confidence")
    if conf not in CONFIDENCE:
        raise SchemaError(f"{where}: {section}.{e['id']}: confidence {conf!r} "
                          f"not in {CONFIDENCE}")
    if conf != "assumed" and e.get("page") is None and src not in ("KERNEL",):
        raise SchemaError(f"{where}: {section}.{e['id']}: sourced value without page")
    unit = e.get("unit", "-")
    return Dim(section=section, id=str(e["id"]), value=e["value"],
               unit=str(unit), source=str(src), confidence=conf,
               page=e.get("page"), note=str(e.get("note", "") or ""),
               derived=bool(e.get("derived", False)),
               ref=e.get("figure") or e.get("table"), origin=origin)


def _read_sections(doc: dict, origin: str, where: str, lenient: bool = False) -> dict:
    out: dict = {}
    for sec, items in doc.items():
        if sec in RESERVED:
            continue
        if not isinstance(items, list):
            if lenient:          # variant metadata such as `variant: "..."`
                continue
            raise SchemaError(f"{where}: section '{sec}' must be a list of entries")
        out[sec] = {}
        for e in items:
            d = _make_dim(sec, e, origin, where)
            if d.id in out[sec]:
                raise SchemaError(f"{where}: duplicate id '{sec}.{d.id}'")
            out[sec][d.id] = d
    return out


_FENCE = re.compile(r"^```ya?ml[ \t]*\n(.*?)^```", re.S | re.M)


def _read_kernel(md: Path, origin: str) -> dict:
    """Kernel assumptions from the fenced yaml block(s) of ASSUMPTIONS_KERNEL.md."""
    out: dict = {}
    if not md.exists():
        return out
    for block in _FENCE.findall(md.read_text(encoding="utf-8")):
        doc = _yaml_load(block, str(md)) or {}
        for e in doc.get("kernel", []) or []:
            d = _make_dim("kernel", e, origin, str(md),
                          default_source="KERNEL", default_conf="assumed")
            out[d.id] = d
    return out


def _read_profile(name: str, path: Path) -> Profile:
    header, body = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line[1:].strip())
        elif line.strip():
            body.append(line)
    rows = list(csv.reader(io.StringIO("\n".join(body))))
    cols = [c.strip() for c in rows[0]]
    conf = "unknown"
    for h in header:
        m = re.match(r"confidence:\s*([a-z]+)", h)
        if m:
            conf = m.group(1)
    if "part" in cols:
        ip = cols.index("part")
        num = [i for i, c in enumerate(cols) if c.endswith("_m")]
        parts: dict = {}
        for r in rows[1:]:
            parts.setdefault(r[ip].strip(), []).append([float(r[i]) for i in num])
        data = {k: np.asarray(v, float) for k, v in parts.items()}
    else:
        num = [i for i, c in enumerate(cols) if c.endswith("_m")]
        data = np.asarray([[float(r[i]) for i in num] for r in rows[1:]], float)
    rel = _rel(path)
    return Profile(name=name, file=rel, columns=cols, header=header,
                   confidence=conf, data=data)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _resolve_base(base: str, variant_dir: Path) -> Path:
    cand = [variant_dir / base, variant_dir.parent / base, MACHINES_DIR / base,
            Path(base)]
    for c in cand:
        if (c / "dimensions.yaml").exists() or (c / "deltas.yaml").exists():
            return c.resolve()
    raise SchemaError(f"{variant_dir}: base machine '{base}' not found "
                      f"(looked in {[str(c) for c in cand]})")


def load_machine(path, _stack: tuple = ()) -> MachineDB:
    """Load a base machine or a delta variant (recursively resolving ``base``)."""
    from .components import DEFAULT_COMPONENTS   # late import (registry)

    root = Path(path)
    if root.is_file():
        root = root.parent
    root = root.resolve()
    if root in _stack:
        raise SchemaError(f"inheritance cycle: {[str(s) for s in _stack + (root,)]}")

    deltas, dims_file = root / "deltas.yaml", root / "dimensions.yaml"
    if deltas.exists():
        db = _load_variant(root, deltas, _stack + (root,))
    elif dims_file.exists():
        doc = _yaml_load(dims_file.read_text(encoding="utf-8"), str(dims_file)) or {}
        origin = _rel(dims_file)
        dims = _read_sections(doc, origin, str(dims_file))
        kmd = root / "ASSUMPTIONS_KERNEL.md"
        dims["kernel"] = _read_kernel(kmd, _rel(kmd))
        if doc.get("components"):
            comps = [_component_from(c, origin, str(dims_file)) for c in doc["components"]]
        else:
            comps = [ComponentSpec(name=n, type=t, params=dict(p),
                                   provenance={"source": "KERNEL",
                                               "confidence": "assumed",
                                               "note": "default component list "
                                                       "of the geometry kernel"},
                                   origin="tokviz.machine.components")
                     for n, t, p in DEFAULT_COMPONENTS]
        files = {origin: _sha(dims_file)}
        if kmd.exists():
            files[_rel(kmd)] = _sha(kmd)
        prof_dir = root / "profiles"
        for p in sorted(prof_dir.glob("*.csv")):
            files[_rel(p)] = _sha(p)
        db = MachineDB(name=root.name, root=root, lineage=[root.name],
                       meta=dict(doc.get("meta") or {}), dims=dims,
                       profile_dirs=[prof_dir], components=comps, files=files)
    else:
        raise SchemaError(f"{root}: neither dimensions.yaml nor deltas.yaml")
    validate(db)
    return db


def _component_from(c: dict, origin: str, where: str) -> ComponentSpec:
    if isinstance(c, str):
        raise SchemaError(f"{where}: component {c!r} needs at least name+type")
    name = c.get("name") or c.get("component") or c.get("id")
    if not name:
        raise SchemaError(f"{where}: component without name: {c!r}")
    ctype = c.get("type") or c.get("kind") or name
    if "params" in c:
        params = dict(c.get("params") or {})
    else:
        params = {k: v for k, v in c.items()
                  if k not in PROV_KEYS + ("name", "id", "component", "type", "kind", "unit")}
    prov = {k: c[k] for k in PROV_KEYS if k in c}
    if "source" not in prov or "confidence" not in prov:
        raise SchemaError(f"{where}: component '{name}' lacks source/confidence")
    if prov["confidence"] not in CONFIDENCE:
        raise SchemaError(f"{where}: component '{name}': bad confidence {prov['confidence']!r}")
    return ComponentSpec(name=str(name), type=str(ctype), params=params,
                         provenance=prov, origin=origin)


def _load_variant(root: Path, deltas: Path, stack: tuple) -> MachineDB:
    doc = _yaml_load(deltas.read_text(encoding="utf-8"), str(deltas)) or {}
    if "base" not in doc:
        raise SchemaError(f"{deltas}: variant without 'base:'")
    base = load_machine(_resolve_base(str(doc["base"]), root), stack)
    db = copy.deepcopy(base)
    origin = _rel(deltas)
    db.name, db.root = root.name, root
    db.lineage = base.lineage + [root.name]
    db.meta.update(doc.get("meta") or {})
    for k, v in doc.items():
        if k not in RESERVED and not isinstance(v, list):
            db.meta[k] = v
    db.meta["base"] = base.name
    db.files[origin] = _sha(deltas)
    db.applied = list(base.applied)
    if (root / "profiles").is_dir():
        db.profile_dirs = [root / "profiles"] + list(db.profile_dirs)
        for p in sorted((root / "profiles").glob("*.csv")):
            db.files[_rel(p)] = _sha(p)
        db._profiles = {}
    kmd = root / "ASSUMPTIONS_KERNEL.md"
    if kmd.exists():
        db.dims.setdefault("kernel", {}).update(_read_kernel(kmd, _rel(kmd)))
        db.files[_rel(kmd)] = _sha(kmd)

    # whole new/replacement sections written directly in the delta file
    for sec, entries in _read_sections(doc, origin, str(deltas), lenient=True).items():
        for i, d in entries.items():
            old = db.dims.setdefault(sec, {}).get(i)
            if old is not None:
                d.supersedes = old.prov()
            db.dims[sec][i] = d
            db.applied.append({"op": "set", "path": d.path, "origin": origin})

    for ov in doc.get("overrides") or []:
        _apply_override(db, ov, origin, str(deltas))

    for rm in doc.get("components_remove") or []:
        _apply_remove(db, rm, origin)

    for add in doc.get("components_add") or []:
        spec = _component_from(add, origin, str(deltas))
        if any(c.name == spec.name for c in db.components):
            db.warnings.append(f"components_add: '{spec.name}' replaces the base component")
            db.components = [c for c in db.components if c.name != spec.name]
        db.components.append(spec)
        db.applied.append({"op": "add_component", "name": spec.name,
                           "type": spec.type, "origin": origin, **spec.provenance})
    return db


def _apply_remove(db: MachineDB, rm, origin: str) -> None:
    """Remove a component (by name), a whole database section (and the
    components built from it), one ``<section>.<id>`` entry, or a base profile
    (``profiles/<file>.csv``)."""
    target = rm if isinstance(rm, str) else (rm.get("component") or rm.get("name")
                                             or rm.get("id") or rm.get("path"))
    target = str(target)
    rec = {"op": "remove", "target": target, "origin": origin,
           **({k: rm[k] for k in PROV_KEYS if k in rm} if isinstance(rm, dict) else {})}
    hit = []
    if target.startswith("profiles/"):
        stem = Path(target).stem
        db.removed_profiles.add(stem)
        db._profiles.pop(stem, None)
        hit.append("profile")
    else:
        if any(c.name == target for c in db.components):
            db.components = [c for c in db.components if c.name != target]
            hit.append("component")
        if target in db.dims:
            del db.dims[target]
            before = len(db.components)
            db.components = [c for c in db.components if c.type != target]
            hit.append("section" + ("+component" if len(db.components) < before else ""))
        elif "." in target:
            sec, _, key = target.partition(".")
            if key in db.dims.get(sec, {}):
                del db.dims[sec][key]
                hit.append("entry")
    if not hit:
        db.warnings.append(f"components_remove: '{target}' matches nothing in the base")
    rec["removed"] = hit
    db.applied.append(rec)


def _apply_override(db: MachineDB, ov: dict, origin: str, where: str) -> None:
    if not isinstance(ov, dict) or "path" not in ov:
        raise SchemaError(f"{where}: override without 'path': {ov!r}")
    parts = str(ov["path"]).split(".")
    if len(parts) < 2:
        raise SchemaError(f"{where}: override path {ov['path']!r} needs <section>.<id>")
    rec = {"op": "override", "path": ov["path"], "origin": origin,
           **{k: ov[k] for k in PROV_KEYS if k in ov}}

    if parts[0] == "components":
        name = parts[1]
        key = parts[3] if len(parts) > 3 and parts[2] == "params" else parts[-1]
        try:
            spec = db.component(name)
        except KeyError:
            raise SchemaError(f"{where}: override of unknown component '{name}'") from None
        rec["old"] = _plain(spec.params.get(key))
        spec.params[key] = ov.get("value")
        db.applied.append(rec)
        return

    sec, key = parts[0], parts[1]
    fld = parts[2] if len(parts) > 2 else "value"
    if "confidence" in ov and ov["confidence"] not in CONFIDENCE:
        raise SchemaError(f"{where}: override {ov['path']}: bad confidence")
    old = db.dims.get(sec, {}).get(key)
    if fld != "value":
        if old is None:
            raise SchemaError(f"{where}: override {ov['path']}: no entry to modify")
        rec["old"] = _plain(getattr(old, fld if fld != "figure" else "ref", None))
        setattr(old, "ref" if fld in ("figure", "table") else fld, ov.get("value"))
        db.applied.append(rec)
        return
    if "value" not in ov:
        raise SchemaError(f"{where}: override {ov['path']} has no value")
    entry = {"id": key, "value": ov["value"],
             "unit": ov.get("unit", old.unit if old else "-")}
    for k in PROV_KEYS:
        if k in ov:
            entry[k] = ov[k]
    new = _make_dim(sec, entry, origin, where)
    if old is not None:
        new.supersedes = old.prov()
        rec["old"] = _plain(old.value)
    db.dims.setdefault(sec, {})[key] = new
    db.applied.append(rec)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate(db: MachineDB) -> None:
    """Structural checks.  Raises SchemaError on the first hard problem."""
    from .components import RECORD_TYPES, REGISTRY

    names = [c.name for c in db.components]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SchemaError(f"{db.name}: duplicate component names {sorted(dup)}")
    for sec, entries in db.dims.items():
        for d in entries.values():
            if d.confidence not in CONFIDENCE:
                raise SchemaError(f"{db.name}: {d.path}: bad confidence")
            if d.unit == "m" and isinstance(d.value, (int, float)) \
                    and not isinstance(d.value, bool) and abs(d.value) > 100:
                raise SchemaError(f"{db.name}: {d.path} = {d.value} m is implausible")
    for c in db.components:
        if c.type in RECORD_TYPES:
            continue
        if c.type not in REGISTRY:
            db.warnings.append(f"component '{c.name}': type '{c.type}' has no "
                               f"builder -- it will be listed as unbuilt")
    for pdir in db.profile_dirs:
        for p in Path(pdir).glob("*.csv"):
            if p.stem in db.removed_profiles:
                continue
            pr = db.profile(p.stem)
            arrs = pr.data.values() if isinstance(pr.data, dict) else [pr.data]
            for a in arrs:
                if not np.all(np.isfinite(a)):
                    raise SchemaError(f"{db.name}: profile {p.name} has non-finite values")
