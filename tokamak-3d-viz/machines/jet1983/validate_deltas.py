#!/usr/bin/env python3
"""Validate machines/jet1983/deltas.yaml against the base machines/<base>/dimensions.yaml.

Checks
  * top-level keys: base, variant, overrides, components_remove, components_add
  * every override path "<section>.<id>" exists in the base (section list entry with that id)
  * every removed component exists: "<section>", "<section>.<id>" or "profiles/<file>" in the base folder
  * every added component id is unique and does not collide with a base "<section>.<id>" / id
  * required fields, confidence enum, int page, source file exists in library/ (warning only)
Prints tables of overrides / removes / adds and counts by confidence. Exit code 1 on any error.

Usage:  python validate_deltas.py [path/to/deltas.yaml]
Needs PyYAML (pip install pyyaml).
"""
import os
import re
import sys
from collections import Counter

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
MACHINES = os.path.dirname(HERE)
LIBRARY = os.path.normpath(os.path.join(MACHINES, "..", "..", "library"))
CONF = {"text", "table", "drawing", "digitized", "assumed"}
TOP = ["base", "variant", "overrides", "components_remove", "components_add"]


def short(v, n=28):
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def table(rows, headers):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "deltas.yaml")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    errors, warnings = [], []

    for k in TOP:
        if k not in d:
            errors.append(f"missing top-level key '{k}'")
    extra = set(d) - set(TOP)
    if extra:
        warnings.append(f"extra top-level keys: {sorted(extra)}")

    base_dir = os.path.join(MACHINES, d.get("base", ""))
    with open(os.path.join(base_dir, "dimensions.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    index = {}  # "section.id" -> entry
    for sec, items in base.items():
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and "id" in it:
                    index[f"{sec}.{it['id']}"] = it
    base_ids = {k.split(".", 1)[1] for k in index}

    # PyYAML (YAML 1.1) reads '2.8e6' (no dot-sign in exponent) as a STRING -> forbid in deltas, report in base
    num_str = re.compile(r"[-+]?\d+(\.\d*)?[eE][-+]?\d+")

    def scan_numstr(obj, where):
        if isinstance(obj, str) and num_str.fullmatch(obj):
            errors.append(f"{where}: '{obj}' parses as a string under YAML 1.1 - write it as a plain float")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                scan_numstr(v, f"{where}.{k}")
        elif isinstance(obj, list):
            for j, v in enumerate(obj):
                scan_numstr(v, f"{where}[{j}]")

    for sec in ("overrides", "components_add"):
        for i, e in enumerate(d.get(sec) or []):
            scan_numstr(e.get("value") if sec == "overrides" else e.get("params"), f"{sec}[{i}]")
    base_numstr = [k for k, it in index.items() if isinstance(it.get("value"), str) and num_str.fullmatch(it["value"])]
    if base_numstr:
        warnings.append(f"base has {len(base_numstr)} numeric values that YAML 1.1 reads as strings "
                        f"(loader must coerce): {', '.join(base_numstr)}")

    def check_common(kind, i, e, required):
        for r in required:
            if r not in e:
                errors.append(f"{kind}[{i}]: missing field '{r}'")
        if "page" in e and not isinstance(e["page"], int):
            errors.append(f"{kind}[{i}]: page must be int, got {e['page']!r}")
        if "confidence" in e and e["confidence"] not in CONF:
            errors.append(f"{kind}[{i}]: bad confidence {e['confidence']!r}")
        src = e.get("source")
        if src and not os.path.exists(os.path.join(LIBRARY, src)):
            warnings.append(f"{kind}[{i}]: source file not found in library/: {src}")

    # overrides
    rows, conf_o = [], Counter()
    for i, e in enumerate(d.get("overrides") or []):
        check_common("overrides", i, e, ["path", "value", "unit", "source", "page", "confidence", "note"])
        p = e.get("path", "")
        ok = p in index
        if not ok:
            errors.append(f"overrides[{i}]: path '{p}' not found in base")
        old = index[p]["value"] if ok else "?"
        if ok and isinstance(old, (int, float)) and not isinstance(e.get("value"), (int, float)):
            warnings.append(f"overrides[{i}]: '{p}' numeric in base but override value is {type(e.get('value')).__name__}")
        conf_o[e.get("confidence")] += 1
        rows.append([p, short(old), short(e.get("value")), e.get("unit"), e.get("confidence"),
                     f"{e.get('source', '').replace('.pdf', '')} p.{e.get('page')}", "OK" if ok else "MISSING"])
    print(f"\n== OVERRIDES ({len(rows)}) ==")
    table(rows, ["path", "base value", "new value", "unit", "conf", "source", "check"])

    # removes
    rows = []
    for i, e in enumerate(d.get("components_remove") or []):
        check_common("components_remove", i, e, ["component", "reason", "source", "page"])
        c = e.get("component", "")
        if c in index:
            what, ok = "entry", True
        elif c in base and isinstance(base[c], list):
            what, ok = f"section ({len(base[c])} entries)", True
        elif os.path.exists(os.path.join(base_dir, c)):
            what, ok = "file", True
        else:
            what, ok = "?", False
            errors.append(f"components_remove[{i}]: '{c}' not found in base")
        rows.append([c, what, f"{e.get('source', '').replace('.pdf', '')} p.{e.get('page')}", "OK" if ok else "MISSING"])
    print(f"\n== COMPONENTS_REMOVE ({len(rows)}) ==")
    table(rows, ["component", "kind", "source", "check"])

    # adds
    rows, seen, conf_a = [], set(), Counter()
    for i, e in enumerate(d.get("components_add") or []):
        check_common("components_add", i, e, ["component", "type", "params", "source", "page", "confidence", "note"])
        c = e.get("component", "")
        if c in seen:
            errors.append(f"components_add[{i}]: duplicate id '{c}'")
        if c in index or c in base_ids:
            errors.append(f"components_add[{i}]: id '{c}' collides with a base id")
        seen.add(c)
        params = e.get("params") or {}
        if not isinstance(params, dict):
            errors.append(f"components_add[{i}]: params must be a mapping")
            params = {}
        ap = params.get("assumed_params", [])
        for k in ap:
            if k not in params:
                errors.append(f"components_add[{i}]: assumed_params names unknown key '{k}'")
        conf_a[e.get("confidence")] += 1
        rows.append([c, e.get("type"), e.get("confidence"), len(params), len(ap),
                     f"{e.get('source', '').replace('.pdf', '')} p.{e.get('page')}"])
    print(f"\n== COMPONENTS_ADD ({len(rows)}) ==")
    table(rows, ["component", "type", "conf", "#params", "#assumed", "source"])

    print("\n== SUMMARY ==")
    print(f"base={d.get('base')}  variant={d.get('variant')!r}")
    print(f"overrides: {sum(conf_o.values())}  by confidence: {dict(conf_o)}")
    print(f"removes:   {len(d.get('components_remove') or [])}")
    print(f"adds:      {sum(conf_a.values())}  by confidence: {dict(conf_a)}")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("RESULT:", "FAIL" if errors else "PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
