#!/usr/bin/env python3
"""Bake the engineering model of a machine from its sourced database.

Runs OUTSIDE Blender (numpy only for the geometry; PyYAML to read the
database).  Writes one ``<component>.npz`` per component plus
``manifest.json`` (same Bake/manifest conventions as run_bake.py, extended
with "machine", "components", "mesh_totals").

    python src/run_bake_machine.py --machine machines/jet1975 --out data/bake_jet1975
    python src/run_bake_machine.py --machine machines/jet1983 --out data/bake_jet1983
    python src/run_bake_machine.py --machine machines/jet1975 --only tf_coils,vessel
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config                                       # noqa: E402
from tokviz.machine import build_machine, load_machine, write_bake  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default=str(config.REPO_ROOT / "machines" / "jet1975"))
    ap.add_argument("--out", default=None, help="default: data/bake_<machine name>")
    ap.add_argument("--only", default="", help="comma-separated component names")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda *a: print(*a, flush=True)
    db = load_machine(args.machine)
    out = args.out or str(config.BAKE_DIR / f"bake_{db.name}")
    log(f"[1/3] {db.name}: lineage {' -> '.join(db.lineage)}, "
        f"{sum(len(s) for s in db.dims.values())} entries, {len(db.components)} components")
    for w in db.warnings:
        log(f"      warning: {w}")
    log("[2/3] building")
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    mb = build_machine(db, only=only, log=log)
    for u in mb.unbuilt:
        log(f"      unbuilt: {u['name']} ({u['type']}): {u['reason']}")
    log(f"[3/3] writing {out}")
    path = write_bake(mb, out)
    import json
    tot = json.loads(Path(path).read_text())["mesh_totals"]
    log(f"      unique: {tot['verts']} verts / {tot['faces']} faces; "
        f"instanced: {tot['verts_instanced']} verts / {tot['faces_instanced']} faces")
    if "plasma" in mb.components:
        i = mb["plasma"].info
        log(f"      plasma: delta = {i['delta']:.4f} (table fit {i['fit']['delta']:.4f}, "
            f"rms {i['fit']['rms_m']*1e3:.1f} mm), min wall clearance "
            f"{i['min_wall_clearance']*1e3:.1f} mm")
    log(f"done in {time.time()-t0:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
