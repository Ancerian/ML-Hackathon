#!/usr/bin/env python3
"""Export the JET machines as web-ready GLB files + web/jet_meta.json.

    python src/run_export_web.py                  # jet1975 + jet1983 -> web/
    python src/run_export_web.py --out web --machines jet1975

No Draco / meshopt / KTX2: the size budget (<= 12 MB per file, <= ~250 k
rendered triangles) is met by geometry LOD (see tokviz.machine.export_gltf).
The plasma and field lines come from the jet1975 physics bake (the 1983
variant keeps the 1975 plasma component).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokviz import config                                   # noqa: E402
from tokviz.machine import export_gltf as EG                # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.REPO_ROOT / "web"))
    ap.add_argument("--machines", default="jet1975,jet1983")
    ap.add_argument("--physics", default=str(config.BAKE_DIR / "bake_jet1975" / "physics"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)
    results, ok = {}, True
    for name in [s.strip() for s in args.machines.split(",") if s.strip()]:
        t0 = time.time()
        path = out / f"{name}.glb"
        r = EG.export_machine(config.REPO_ROOT / "machines" / name,
                              config.BAKE_DIR / f"bake_{name}", args.physics, path, log=log)
        st = EG.glb_stats(path)
        results[name] = r
        log(f"{name}: {path}  {st['bytes'] / 1e6:.2f} MB, {st['triangles_rendered']} rendered "
            f"tris ({st['triangles_unique']} unique), {st['line_points']} line points, "
            f"{st['n_nodes']} nodes / {st['n_meshes']} meshes / {st['n_materials']} materials "
            f"[{time.time() - t0:.1f}s]")
        for g, s in st["groups"].items():
            log(f"    {g:<11} {s['triangles']:7d} tris  {s['line_points']:5d} pts  {s['nodes']:4d} nodes")
        lo, hi = st["bbox"]
        log(f"    bbox x [{lo[0]:.3f}, {hi[0]:.3f}]  y [{lo[1]:.3f}, {hi[1]:.3f}]  "
            f"z [{lo[2]:.3f}, {hi[2]:.3f}]")
        if st["bytes"] > EG.MAX_FILE_BYTES or st["triangles_rendered"] > EG.TRI_BUDGET:
            log("    BUDGET EXCEEDED")
            ok = False
        if r["stats"]["bake_mismatches"]:
            log(f"    bake structure mismatches: {r['stats']['bake_mismatches']}")
    EG.build_meta(results, out)
    log(f"meta: {out / 'jet_meta.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
