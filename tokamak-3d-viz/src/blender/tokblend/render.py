"""Engine configuration and still rendering.

Two paths from one scene:

* EEVEE for look-development -- seconds per frame, fully interactive.
* Cycles on Metal for the finals -- correct volume scattering, and the plasma
  actually lights the vessel.

Measured on an Apple M4 (10-core GPU), this scene's volume at 480x320 / 32 spp
takes ~1.1 s on GPU.  The FIRST Cycles render of a session additionally pays a
one-off Metal kernel compilation of roughly 100 s; it is cached afterwards, so
do not mistake it for render time.
"""
from __future__ import annotations

import os
import time

import bpy

from .bl import eevee_engine_id, enable_cycles


def setup_common(res=(3840, 2160), fmt="PNG", depth="16", transparent=False):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = fmt
    if fmt in {"PNG", "OPEN_EXR", "OPEN_EXR_MULTILAYER", "TIFF"}:
        sc.render.image_settings.color_depth = depth
    sc.render.film_transparent = bool(transparent)
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast"
    sc.render.use_compositing = True
    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_emit = True
    vl.use_pass_cryptomatte_object = True
    vl.use_pass_cryptomatte_material = True
    return sc


def setup_eevee(samples=128, volume_res="2"):
    sc = bpy.context.scene
    sc.render.engine = eevee_engine_id()
    ee = sc.eevee
    ee.taa_render_samples = int(samples)
    ee.use_shadows = True
    ee.use_volumetric_shadows = True
    ee.volumetric_samples = 128
    ee.volumetric_tile_size = volume_res        # '1' is sharpest, '16' coarsest
    ee.volumetric_start = 0.5
    ee.volumetric_end = 400.0
    try:
        ee.use_raytracing = True                 # 4.2+: replaces SSR/GTAO
    except Exception:
        pass
    # NOTE: EEVEE lost its Bloom option at 4.2.  Glare now belongs in the
    # compositor -- see tokblend.compositor.
    return sc


def setup_cycles(samples=256, threshold=0.01, volume_bounces=4):
    sc = bpy.context.scene
    if not enable_cycles():
        raise RuntimeError("Cycles add-on unavailable")
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "GPU"
    cy.samples = int(samples)
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = float(threshold)
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:
        pass
    cy.max_bounces = 12
    cy.volume_bounces = int(volume_bounces)      # default is 0 -- too low here
    cy.volume_step_rate = 0.5
    cy.volume_max_steps = 1024
    return sc


def _apply_engine_visibility(engine):
    key = "use_in_cycles" if engine == "CYCLES" else "use_in_eevee"
    for ob in bpy.data.objects:
        if key in ob.keys():
            ob.hide_render = not bool(ob[key])


def render_still(camera, out_path, engine="CYCLES"):
    sc = bpy.context.scene
    sc.camera = camera
    _apply_engine_visibility(engine)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sc.render.filepath = out_path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print(f"[render] {engine} {os.path.basename(out_path)} {dt:.1f}s", flush=True)
    return dt
