"""Render clean, presentation-ready frames from the local tokamak Blender scene.

Usage:
  blender --background tokamak-3d-viz/tokamak.blend --python submission/render_assets.py -- \
      --out submission/assets/renders/reimagined --res 1600 900 --samples 48

The saved scene is never modified. This script selects the existing physical
collections/cameras and renders without the annotation pass used by the report.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy


SHOTS = {
    "cutaway": {"camera": "CAM_02_cutaway", "visible": {"Machine", "PlasmaVolume", "FieldLines"}},
    "fieldlines": {"camera": "CAM_03_plasma", "visible": {"FieldLines"}},
    "surfaces": {"camera": "CAM_05_qprofile", "visible": {"FluxSurfaces"}},
    "poincare": {"camera": "CAM_04_poincare", "visible": {"Diagnostics"}},
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--res", type=int, nargs=2, default=(1600, 900))
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--shots", nargs="+", choices=sorted(SHOTS), default=list(SHOTS))
    parser.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    parser.add_argument("--animate", choices=sorted(SHOTS), help="Also render a looping, slow camera orbit")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--orbit-deg", type=float, default=3.0)
    return parser.parse_args(argv)


def set_render_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("PresentationWorld")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    # Keep the original low-key studio reflections on metal. The world itself
    # is rendered transparent and receives the deck's warm-white matte later.
    background.inputs["Color"].default_value = (0.02, 0.03, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.02


def configure(scene, args):
    scene.render.engine = "BLENDER_EEVEE" if args.engine == "eevee" else "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    if args.engine == "eevee":
        scene.eevee.taa_render_samples = args.samples
    else:
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    set_render_world(scene)
    # The source scene was tuned for a near-black background. A slightly
    # less reflective steel keeps its brushed color and edge detail legible
    # when the transparent render is placed on the deck's light paper.
    for material in bpy.data.materials:
        if material.name not in {"StainlessSteel", "Copper", "Beryllium", "Tungsten"}:
            continue
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Metallic"].default_value = 0.52


def select_shot(scene, shot_name):
    spec = SHOTS[shot_name]
    scene.camera = bpy.data.objects[spec["camera"]]
    visible = spec["visible"] | {"Rig"}
    for collection in bpy.data.collections:
        collection.hide_render = collection.name not in visible
        collection.hide_viewport = collection.name not in visible


def style_poincare_card():
    card = bpy.data.materials.get("SectionCard")
    if not card or not card.use_nodes:
        return
    for node in card.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Base Color"].default_value = (0.91, 0.93, 0.90, 1.0)
            node.inputs["Metallic"].default_value = 0.0
            node.inputs["Roughness"].default_value = 0.95
            node.inputs["Emission Color"].default_value = (0.91, 0.93, 0.90, 1.0)
            node.inputs["Emission Strength"].default_value = 0.65


def style_field_materials(shot_name):
    if shot_name in {"cutaway", "fieldlines"}:
        material = bpy.data.materials.get("FieldLine")
        if material:
            emission = next((node for node in material.node_tree.nodes if node.type == "EMISSION"), None)
            if emission:
                emission.inputs["Color"].default_value = (0.025, 0.24, 0.72, 1.0)
                emission.inputs["Strength"].default_value = 1.05
    if shot_name == "surfaces":
        material = bpy.data.materials.get("FluxSurface")
        if material:
            for node in material.node_tree.nodes:
                if node.type == "EMISSION":
                    node.inputs["Strength"].default_value = 0.55
                elif node.label == "Alpha":
                    node.inputs[1].default_value = 0.72
    if shot_name == "poincare":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tokamak-3d-viz" / "src" / "blender"))
        from tokblend.bl import get_modifier_input, set_modifier_input
        for obj in bpy.data.objects:
            if not obj.name.startswith("Poincare_"):
                continue
            modifier = obj.modifiers.get("Instancer")
            if not modifier or not modifier.node_group:
                continue
            socket_id = modifier.node_group.get("_sockets", {}).get("Radius")
            if socket_id:
                radius = get_modifier_input(modifier, socket_id, 0.0105)
                set_modifier_input(modifier, socket_id, float(radius) * 1.7)
        for name, color, strength in (
            ("Punct_regular", (0.035, 0.16, 0.86, 1.0), 0.65),
            ("Punct_island", (1.0, 0.30, 0.015, 1.0), 0.82),
            ("Punct_chaotic", (0.88, 0.025, 0.055, 1.0), 0.78),
            ("Punct_escaped", (0.16, 0.18, 0.20, 1.0), 0.55),
        ):
            material = bpy.data.materials.get(name)
            if material:
                emission = next((node for node in material.node_tree.nodes if node.type == "EMISSION"), None)
                if emission:
                    emission.inputs["Color"].default_value = color
                    emission.inputs["Strength"].default_value = strength


def render_camera_orbit(scene, shot_name, args):
    import mathutils

    select_shot(scene, shot_name)
    cam = scene.camera
    target = next((c.target for c in cam.constraints if c.type == "TRACK_TO" and c.target), None)
    centre = target.location.copy() if target else mathutils.Vector((0.0, 0.0, 0.0))
    original = cam.location.copy()
    offset = original - centre
    radius = math.hypot(offset.x, offset.y)
    base_angle = math.atan2(offset.y, offset.x)
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.filepath = os.path.abspath(os.path.join(args.out, f"tokamak-{shot_name}-motion_"))
    angle = math.radians(args.orbit_deg)
    for frame in range(1, args.frames + 1):
        phase = math.sin(2.0 * math.pi * (frame - 1) / args.frames)
        a = base_angle + angle * phase
        cam.location = (centre.x + radius * math.cos(a), centre.y + radius * math.sin(a), centre.z + offset.z)
        cam.keyframe_insert(data_path="location", frame=frame)
    bpy.context.view_layer.update()
    bpy.ops.render.render(animation=True)
    cam.location = original
    print(f"PRESENTATION_MOTION {shot_name} {args.frames} frames", flush=True)


def main():
    args = parse_args()
    scene = bpy.context.scene
    configure(scene, args)
    os.makedirs(args.out, exist_ok=True)
    for shot_name in args.shots:
        select_shot(scene, shot_name)
        style_field_materials(shot_name)
        if shot_name == "poincare":
            style_poincare_card()
            scene.view_settings.view_transform = "Standard"
        target = os.path.abspath(os.path.join(args.out, f"tokamak-{shot_name}-light.png"))
        scene.render.filepath = target
        bpy.ops.render.render(write_still=True)
        print(f"PRESENTATION_RENDER {shot_name} {target}", flush=True)
    if args.animate:
        render_camera_orbit(scene, args.animate, args)


if __name__ == "__main__":
    main()
