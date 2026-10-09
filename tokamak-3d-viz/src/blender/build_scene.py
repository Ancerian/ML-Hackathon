#!/usr/bin/env python3
"""Build the whole tokamak scene inside Blender from a bake directory.

    blender --background --python src/blender/build_scene.py -- \
        --bake data/bake01 --out tokamak.blend

Add ``--render eevee`` or ``--render cycles`` to also write the still set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokblend import materials as M
from tokblend import machine as MA
from tokblend import plasma as PL
from tokblend import cutaway as CU
from tokblend import cameras as CAM
from tokblend import render as RD
from tokblend import compositor as CO
from tokblend.bl import collection, clear_scene


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


#: Per-shot state: how far the cutaway is open, and which collections render.
#: A "plasma only" frame that still has the cryostat in it is not a plasma
#: frame -- with the camera inside the cryostat you photograph its inner wall.
SHOTS = {
    "01_exterior": dict(cut=0.00, show=("Machine",)),
    "02_cutaway":  dict(cut=1.00, show=("Machine", "PlasmaVolume", "FieldLines")),
    "03_plasma":   dict(cut=0.00, show=("PlasmaVolume", "FieldLines")),
    "04_poincare": dict(cut=0.00, show=("Diagnostics",)),
    # nested shells only read when they are cut open
    "05_qprofile": dict(cut=0.55, show=("FluxSurfaces", "FieldLines")),
    "06_divertor": dict(cut=1.00, show=("Machine", "PlasmaVolume")),
}


def _set_visible(names):
    names = set(names) | {"Rig"}
    for c in bpy.data.collections:
        vis = c.name in names
        c.hide_render = not vis
        c.hide_viewport = not vis


def _apply_shot(key, cams, cut_targets):
    """Put the scene into one shot's state: cutaway, collections, camera."""
    shot = SHOTS.get(key, dict(cut=1.0, show=("Machine", "PlasmaVolume")))
    CU.set_cut(cut_targets, shot["cut"])
    _set_visible(shot["show"])
    if key in cams:
        bpy.context.scene.camera = cams[key]
    return shot


def _measure(*collections):
    """Bounding radius of the machine, and of the plasma alone, in metres."""
    from mathutils import Vector

    def radius_of(coll):
        best = 0.0
        for ob in coll.objects:
            if ob.type != "MESH":
                continue
            for c in ob.bound_box:
                best = max(best, (ob.matrix_world @ Vector(c)).length)
        return best

    radii = [radius_of(c) for c in collections]
    return max(radii), (radii[-1] if len(radii) > 1 else radii[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--out", default="tokamak.blend")
    ap.add_argument("--render", choices=["none", "eevee", "cycles", "both"],
                    default="none")
    ap.add_argument("--render-dir", default="render")
    ap.add_argument("--res", type=int, nargs=2, default=[3840, 2160])
    ap.add_argument("--samples-eevee", type=int, default=128)
    ap.add_argument("--samples-cycles", type=int, default=256)
    ap.add_argument("--only-camera", default=None)
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args(argv_after_dashdash())

    bake = args.bake
    man = json.load(open(os.path.join(bake, "manifest.json")))
    scale = float(man["geometry"]["scale"])
    R0 = float(man["geometry"]["R0_scaled"])
    a_minor = R0 / 2.5

    print(f"[build] {man['provenance']['machine']} shot "
          f"{man['provenance']['shot_file']} frame {man['provenance']['frame_index']} "
          f"t={man['provenance']['time_ms']:.0f} ms  R0_scaled={R0:.3f} m", flush=True)

    clear_scene()
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0

    c_machine = collection("Machine")
    c_vol = collection("PlasmaVolume")
    c_surf = collection("FluxSurfaces")
    c_lines = collection("FieldLines")
    c_diag = collection("Diagnostics")
    c_rig = collection("Rig")

    # ---------------------------------------------------------------- shaders
    gz = np.load(os.path.join(bake, "psi_n_grid.npz"))
    R_rng = (float(gz["R_scaled"][0]), float(gz["R_scaled"][-1]))
    Z_rng = (float(gz["Z_scaled"][0]), float(gz["Z_scaled"][-1]))

    m_plasma = M.plasma_volume("PlasmaVolume", gz["psi_n"], R_rng, Z_rng)
    m_flux = M.flux_surface_material("FluxSurface", alpha=0.42, emission=1.1)
    m_line = M.emissive_material("FieldLine", (0.30, 0.72, 1.0, 1.0), 1.4)
    # Puncture colours match docs/ANALYSIS.md panel A, so the render and the
    # analysis figure can be read side by side.
    m_punct = M.emissive_material("Puncture", (1.0, 0.80, 0.28, 1.0), 2.6)
    # Emission strength must stay low here.  Above ~1.5 the AgX view transform
    # pushes these past its shoulder and desaturates them: at strengths 1.8-4.0
    # all three classes rendered identically white, which defeats the point.
    m_punct_cls = {
        "regular": M.emissive_material("Punct_regular", (0.18, 0.42, 1.00, 1.0), 0.55),
        "island":  M.emissive_material("Punct_island",  (1.00, 0.55, 0.06, 1.0), 1.10),
        "chaotic": M.emissive_material("Punct_chaotic", (1.00, 0.13, 0.13, 1.0), 0.85),
        "escaped": M.emissive_material("Punct_escaped", (0.35, 0.35, 0.35, 1.0), 0.30),
    }
    m_w = M.metal_material("Tungsten", "tungsten")
    m_be = M.metal_material("Beryllium", "beryllium")
    m_ss = M.metal_material("StainlessSteel", "steel")
    m_cu = M.metal_material("Copper", "copper")

    # ---------------------------------------------------------------- machine
    vessel, prof = MA.build_vessel(bake, c_machine, m_ss)
    pf = MA.build_pf_coils(bake, c_machine, m_cu)
    tf = MA.build_tf_coils(bake, c_machine, m_ss, n_coils=18,
                           width=0.55 * scale / 3.0, depth=0.32 * scale / 3.0)
    cryo = MA.build_cryostat(prof, c_machine, m_ss, clearance=2.1 * scale / 3.0)
    div = MA.build_divertor(prof, c_machine, m_w)
    ports = MA.build_ports(prof, c_machine, m_ss, n_ports=12,
                           r_port=0.75 * scale / 3.0, length=3.0 * scale / 3.0)
    tiles = None
    if not args.no_tiles:
        tiles = MA.build_tiles(prof, c_machine, m_be, n_tor=120)
    print(f"[build] machine: vessel + {len(pf)} PF + {len(tf)} TF + cryostat "
          f"+ divertor + {len(ports)} ports" + (" + tiles" if tiles else ""), flush=True)

    # ---------------------------------------------------------------- plasma
    vol = PL.build_plasma_volume(bake, c_vol, m_plasma)
    surfaces = PL.build_flux_surfaces(bake, c_surf, m_flux)
    lines = PL.build_fieldlines(bake, c_lines, m_line, bevel=0.006 * scale)
    punct = PL.build_punctures_classified(bake, c_diag, m_punct_cls,
                                          radius=0.0035 * scale)
    plane = PL.build_poincare_plane(bake, c_diag, mat=M.matte_material("SectionCard", color=(0.004, 0.005, 0.008, 1.0)))
    print(f"[build] plasma: volume + {len(surfaces)} flux surfaces + field lines "
          f"+ punctures ({', '.join(f'{k}:{len(v.data.vertices)}' for k, v in punct.items())})",
          flush=True)

    # ---------------------------------------------------------------- cutaway
    # Flux surfaces are cut too: nested shells are invisible unless opened.
    # The plasma VOLUME is deliberately NOT cut -- EEVEE's froxel volumes need
    # a closed bounding mesh, and an opened one renders as flat slabs.  Seeing
    # the whole plasma through the machine opening also reads better.
    cut_targets = ([vessel, cryo, div] + tf + ports
                   + ([tiles] if tiles else []) + surfaces)
    # The wedge must open TOWARD the cameras, which orbit at azimuth
    # -125 deg .. -140 deg; a wedge at -45..+45 deg is cut on the far side and
    # is invisible.  Centre it on -135 deg.
    CU.apply_wedge_cut(cut_targets, cut=0.0, start=-math.pi,
                       width=math.pi / 2)
    print(f"[build] cutaway rig on {len(cut_targets)} objects", flush=True)

    # ------------------------------------------------------------------- rig
    CAM.dark_world()
    scene_radius, plasma_radius = _measure(c_machine, c_vol)
    print(f"[build] scene radius {scene_radius:.2f} m, plasma radius "
          f"{plasma_radius:.2f} m", flush=True)
    cams = CAM.standard_cameras(scene_radius, R0, a_minor,
                                plasma_radius=plasma_radius, coll=c_rig)
    keys = CAM.key_rig(scene_radius, coll=c_rig)
    CAM.set_engine_visibility(keys, in_cycles=True, in_eevee=True)
    proxies = CAM.plasma_proxy_lights(R0, 0.0, n=8,
                                      energy=1.6 * 4.0 * math.pi * (R0 * 0.6) ** 2 * 6.0,
                                      coll=c_rig)
    CAM.set_engine_visibility(proxies, in_cycles=False, in_eevee=True)
    CO.build()
    print(f"[build] {len(cams)} cameras, {len(keys)} key lights, "
          f"{len(proxies)} EEVEE proxy lights, compositor", flush=True)

    RD.setup_common(res=tuple(args.res))

    # Save in a state that SHOWS something.  Without this the file opens with
    # the cutaway shut and all six collections visible at once: a sealed
    # cryostat with the plasma, flux surfaces and punctures hidden inside it.
    # src/blender/set_shot.py flips between the six shots interactively.
    _apply_shot("02_cutaway", cams, cut_targets)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    print(f"[build] saved {args.out}", flush=True)

    # ---------------------------------------------------------------- render
    if args.render != "none":
        os.makedirs(args.render_dir, exist_ok=True)
        engines = (["BLENDER_EEVEE", "CYCLES"] if args.render == "both"
                   else ["CYCLES"] if args.render == "cycles" else ["BLENDER_EEVEE"])
        for eng in engines:
            if eng == "CYCLES":
                RD.setup_cycles(samples=args.samples_cycles)
            else:
                RD.setup_eevee(samples=args.samples_eevee)
            for key, cam in cams.items():
                if args.only_camera and key != args.only_camera:
                    continue
                shot = SHOTS.get(key, dict(cut=1.0, show=("Machine", "Plasma")))
                CU.set_cut(cut_targets, shot["cut"])
                _set_visible(set(shot["show"]) | {"Rig"})
                tag = "cycles" if eng == "CYCLES" else "eevee"
                RD.render_still(cam, os.path.join(args.render_dir,
                                                  f"{key}_{tag}"), engine=eng)
        _apply_shot("02_cutaway", cams, cut_targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
