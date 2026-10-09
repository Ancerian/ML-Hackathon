#!/usr/bin/env python3
"""motion/render_film.py — Render continuous camera flight film and individual acts.

Supports:
  --preview : 640x360, 16 samples EEVEE
  --final   : 1920x1080, 64 samples EEVEE (Run ONLY after explicit approval)

Acts:
  Act 1: Machine Cutaway (DIII-D #203702, vessel, TF/PF coils, divertor)
  Act 2: Magnetic Flux Surfaces (psi_N in [0.12 .. 0.97], EFIT frame 75 t=1760 ms)
  Act 3: Field Lines (Canonical Hamiltonian system, q=m/n wrapping)
  Act 4A: Poincare Section A (Single mode 2/1, A=1e-3, w=27.1 mm, S=н/д)
  Act 4B: Poincare Section B (Key frame A=4.8e-3, S=1.002, k=2.02 stochastic overlap)
  Act 5: 2D Equilibrium Comparison Panel (DIII-D #062 Frame 147, GT vs UNet_Lite)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parent.parent
BLENDER_BIN = "/Users/ancerian/Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender"
BLEND_FILE = str(PROJECT / "tokamak-3d-viz" / "tokamak.blend")
COMPARE_PNG = str(PROJECT / "motion" / "data" / "contours_compare.png")


def add_banner(image_path: Path, title: str, subtitle: str, act_num: int, is_preview: bool = True):
    """Adds a high-contrast, projector-readable dark translucent banner with annotations."""
    im = Image.open(image_path).convert("RGBA")
    w, h = im.size
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Header banner
    header_h = int(h * 0.16)
    draw.rectangle([(0, 0), (w, header_h)], fill=(14, 17, 23, 215))
    draw.line([(0, header_h), (w, header_h)], fill=(88, 166, 255, 230), width=max(2, int(w * 0.0025)))
    
    # Fonts
    font_size_t = max(13, int(w * 0.024))
    font_size_s = max(10, int(w * 0.017))
    try:
        font_t = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size_t)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size_s)
    except Exception:
        font_t = ImageFont.load_default()
        font_s = ImageFont.load_default()
        
    pad_x = int(w * 0.025)
    pad_y = int(h * 0.018)
    
    # Act badge color
    badge_color = (88, 166, 255) if act_num < 4 else ((227, 179, 65) if act_num == 4 else (248, 81, 73))
    
    draw.text((pad_x, pad_y), title, font=font_t, fill=(255, 255, 255, 255))
    draw.text((pad_x, pad_y + font_size_t + 4), subtitle, font=font_s, fill=(201, 209, 217, 255))
    
    # Footer watermark / honesty stamp
    footer_h = int(h * 0.065)
    draw.rectangle([(0, h - footer_h), (w, h)], fill=(14, 17, 23, 200))
    stamp = "FMF KPI AI Lab | TokaBench-GS | CC BY 4.0 | Вакуумні моди, не є прямою симуляцією плазми"
    draw.text((pad_x, h - footer_h + int(footer_h * 0.2)), stamp, font=font_s, fill=(139, 148, 158, 240))
    
    combined = Image.alpha_composite(im, overlay)
    combined.convert("RGB").save(image_path, "PNG")


def render_blender_batch(args, out_frames_dir: Path):
    """Executes Blender in headless mode to render 3D camera flight acts."""
    script_content = f"""
import bpy
import time
import math

blend_file = "{BLEND_FILE}"
out_dir = "{str(out_frames_dir)}"
res_x = {args.res_x}
res_y = {args.res_y}
samples = {args.samples}

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = res_x
scene.render.resolution_y = res_y
scene.eevee.taa_render_samples = samples
scene.render.film_transparent = False

# Setup or retrieve Flight Camera
cam = bpy.data.objects.get('CAM_Flight')
if not cam:
    cam_data = bpy.data.cameras.new('CAM_Flight')
    cam_data.lens = 35.0
    cam = bpy.data.objects.new('CAM_Flight', cam_data)
    scene.collection.objects.link(cam)

tgt = bpy.data.objects.get('CAM_Flight_Target')
if not tgt:
    tgt = bpy.data.objects.new('CAM_Flight_Target', None)
    scene.collection.objects.link(tgt)

if not cam.constraints.get('TrackTo'):
    con = cam.constraints.new('TRACK_TO')
    con.target = tgt
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'

scene.camera = cam

# Object references
p_reg = bpy.data.objects.get('Poincare_regular')
p_isl = bpy.data.objects.get('Poincare_island')
p_ch = bpy.data.objects.get('Poincare_chaotic')
p_plane = bpy.data.objects.get('PoincarePlane')
fl = bpy.data.objects.get('FieldLines')
plasma_vol = bpy.data.objects.get('PlasmaVolume')
cutaway = bpy.data.objects.get('VacuumVessel')

# Define flight keyframes: (frame_idx, cam_loc, tgt_loc, (hide_p_reg, hide_p_isl, hide_p_ch, hide_fl, hide_surf))
# Total 160 frames for 3D acts:
# Act 1: 1 .. 35  (Cutaway overview)
# Act 2: 36 .. 70 (Flux surfaces zoom)
# Act 3: 71 .. 105 (Field lines orbital track)
# Act 4A: 106 .. 135 (Poincare Section A: Single mode 2/1)
# Act 4B: 136 .. 165 (Poincare Section B: Key frame A=4.8e-3 overlap)

def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))

# Flight path coordinates
pos_cutaway_start = (-34.0, -38.0, 24.0)
pos_cutaway_end   = (-27.0, -36.0, 20.0)
tgt_cutaway       = (0.0, 0.0, 0.0)

pos_surf_start    = (-27.0, -36.0, 20.0)
pos_surf_end      = (-20.0, -28.0, 12.0)
tgt_surf          = (2.0, 0.0, 0.0)

pos_fl_start      = (-20.0, -28.0, 12.0)
pos_fl_end        = (-10.5, -21.0, 5.5)
tgt_fl            = (4.5, -3.0, 0.0)

pos_poincare      = (5.03, -20.2, 0.0)
tgt_poincare      = (5.03, 0.0, 0.0)

timings = {{}}
t_start_all = time.time()

# ----------------- ACT 1: CUTAWAY (Frames 1..35) -----------------
t0 = time.time()
if p_reg: p_reg.hide_render = True
if p_isl: p_isl.hide_render = True
if p_ch: p_ch.hide_render = True
if p_plane: p_plane.hide_render = True
if fl: fl.hide_render = True

for f in range(1, 36):
    t_norm = (f - 1) / 34.0
    cam.location = lerp(pos_cutaway_start, pos_cutaway_end, t_norm)
    tgt.location = tgt_cutaway
    scene.render.filepath = f'{{out_dir}}/frame_{{f:04d}}.png'
    bpy.ops.render.render(write_still=True)
timings['act1_cutaway'] = time.time() - t0

# ----------------- ACT 2: SURFACES (Frames 36..70) -----------------
t0 = time.time()
if fl: fl.hide_render = True
for f in range(36, 71):
    t_norm = (f - 36) / 34.0
    cam.location = lerp(pos_surf_start, pos_surf_end, t_norm)
    tgt.location = tgt_surf
    scene.render.filepath = f'{{out_dir}}/frame_{{f:04d}}.png'
    bpy.ops.render.render(write_still=True)
timings['act2_surfaces'] = time.time() - t0

# ----------------- ACT 3: FIELD LINES (Frames 71..105) -----------------
t0 = time.time()
if fl: fl.hide_render = False
for f in range(71, 106):
    t_norm = (f - 71) / 34.0
    cam.location = lerp(pos_fl_start, pos_fl_end, t_norm)
    tgt.location = tgt_fl
    scene.render.filepath = f'{{out_dir}}/frame_{{f:04d}}.png'
    bpy.ops.render.render(write_still=True)
timings['act3_fieldlines'] = time.time() - t0

# ----------------- ACT 4A: POINCARE SECTION A (Frames 106..135) -----------------
t0 = time.time()
if fl: fl.hide_render = True
if p_reg: p_reg.hide_render = False
if p_isl: p_isl.hide_render = False
if p_ch: p_ch.hide_render = True
if p_plane: p_plane.hide_render = False

cam.location = pos_poincare
tgt.location = tgt_poincare

for f in range(106, 136):
    scene.render.filepath = f'{{out_dir}}/frame_{{f:04d}}.png'
    bpy.ops.render.render(write_still=True)
timings['act4a_poincare_single'] = time.time() - t0

# ----------------- ACT 4B: POINCARE SECTION B (Frames 136..165) -----------------
t0 = time.time()
if p_ch: p_ch.hide_render = False  # Enable chaotic stochastic points

for f in range(136, 166):
    scene.render.filepath = f'{{out_dir}}/frame_{{f:04d}}.png'
    bpy.ops.render.render(write_still=True)
timings['act4b_poincare_chaos'] = time.time() - t0

timings['total_3d_sec'] = time.time() - t_start_all
with open(f'{{out_dir}}/timings.json', 'w') as f:
    import json
    json.dump(timings, f, indent=2)

print('BLENDER_BATCH_DONE')
"""
    runner_script = out_frames_dir / "run_blender_inner.py"
    with open(runner_script, "w", encoding="utf-8") as f:
        f.write(script_content)
        
    cmd = [
        BLENDER_BIN,
        "--background",
        BLEND_FILE,
        "--python",
        str(runner_script)
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    
    if proc.returncode != 0:
        print("Blender STDERR:", proc.stderr)
        raise RuntimeError(f"Blender failed with exit code {proc.returncode}")
        
    with open(out_frames_dir / "timings.json") as f:
        timings = json.load(f)
    timings["total_session_sec"] = dt
    return timings


def generate_act5_frames(args, out_frames_dir: Path):
    """Generates Act 5 2D comparison panel frames (frames 166..195) from contours_compare.png."""
    t0 = time.time()
    src_img = Image.open(COMPARE_PNG).convert("RGBA")
    
    # Target resolution
    w_t, h_t = args.res_x, args.res_y
    
    # Resize preserving aspect ratio
    src_w, src_h = src_img.size
    scale = min(w_t / src_w, h_t / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    panel = Image.new("RGBA", (w_t, h_t), (14, 17, 23, 255))
    off_x = (w_t - new_w) // 2
    off_y = (h_t - new_h) // 2
    panel.paste(resized, (off_x, off_y), resized)
    
    # Write frames 166..195 (30 frames)
    for f in range(166, 196):
        out_f = out_frames_dir / f"frame_{f:04d}.png"
        panel.convert("RGB").save(out_f, "PNG")
        
    return time.time() - t0


def annotate_all_frames(out_frames_dir: Path, is_preview: bool = True):
    """Applies high-contrast readable banners to all frames."""
    print("Applying high-contrast projector banners...")
    
    acts_meta = [
        # (range, title, subtitle, act_num)
        (
            range(1, 36),
            "АКТ 1: РОЗРІЗ РЕАКТОРА (ТОКАМАК DIII-D #203702)",
            "Вакуумна камера, котушки тороїдального (TF) та полоїдального (PF) полів, дивертор",
            1
        ),
        (
            range(36, 71),
            "АКТ 2: МАГНІТНІ ПОВЕРХНІ ПОЛОЇДАЛЬНОГО ПОТОКУ ψ_N ∈ [0.12 .. 0.97]",
            "Вкладені магнітні поверхні розрахунку EFIT (кадр 75, t=1760 мс, q_95=3.7)",
            2
        ),
        (
            range(71, 106),
            "АКТ 3: СИЛОВА ЛІНІЯ: КАНОНІЧНА ГАМІЛЬТОНОВА СИСТЕМА (E19)",
            "Збереження фазового об'єму |det J - 1| ≤ 10^-10; намотка на тори із q = m/n",
            3
        ),
        (
            range(106, 136),
            "АКТ 4А: ПЕРЕРІЗ ПУАНКАРЕ — ОДИНОЧНА МОДА 2/1 (A=10^-3, w=27.1 мм)",
            "Параметр Чирикова S = н/д (одна мода); повна відсутність хаосу (теорема E20)",
            4
        ),
        (
            range(136, 166),
            "АКТ 4Б: ПЕРЕРІЗ ПУАНКАРЕ — КЛЮЧОВИЙ КАДР ПЕРЕКРИТТЯ A=4.8e-3 (k=2.02)",
            "S≈1.0–1.1: початок перекриття островів; FTLE у 1.7–2.0 раза більший; стохастичність ймовірна, кількісно не підтверджена",
            4
        ),
        (
            range(166, 196),
            "АКТ 5: 2D ПОРІВНЯННЯ РІВНОВАГ (DIII-D #062, КАДР 147 — МЕДІАННИЙ ЗА g КАДР ТЕСТУ)",
            "Істина g=0.0065 проти UNet_Lite g=0.8618; єдина шкала нев'язки g_local ∈ [0.0 .. 1.2]",
            5
        ),
    ]
    
    for f_range, title, subtitle, act_num in acts_meta:
        for f in f_range:
            p = out_frames_dir / f"frame_{f:04d}.png"
            if p.exists():
                add_banner(p, title, subtitle, act_num, is_preview=is_preview)


def assemble_videos_and_keyframes(args, out_frames_dir: Path, renders_dir: Path):
    """Encodes separate act clips, continuous full film, and exports keyframes."""
    print("Encoding video clips with ffmpeg...")
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    fps = args.fps
    
    # 1. Full continuous film (frames 1..195)
    full_mp4 = renders_dir / f"film_continuous_{args.mode}.mp4"
    cmd_full = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", str(out_frames_dir / "frame_%04d.png"),
        "-vframes", "195",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "19",
        "-preset", "medium",
        str(full_mp4)
    ]
    subprocess.run(cmd_full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Continuous film encoded: {full_mp4}")
    
    # 2. Individual Acts
    acts_def = [
        ("act1_cutaway", 1, 35),
        ("act2_surfaces", 36, 70),
        ("act3_fieldlines", 71, 105),
        ("act4a_poincare_single", 106, 135),
        ("act4b_poincare_chaos", 136, 165),
        ("act5_compare_2d", 166, 195),
    ]
    
    for name, start, end in acts_def:
        count = end - start + 1
        clip_mp4 = renders_dir / f"{name}_{args.mode}.mp4"
        cmd_clip = [
            "/opt/homebrew/bin/ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", str(start),
            "-i", str(out_frames_dir / "frame_%04d.png"),
            "-vframes", str(count),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            "-preset", "medium",
            str(clip_mp4)
        ]
        subprocess.run(cmd_clip, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Keyframes: 2 keyframes per act (first and middle/end)
        kf1 = out_frames_dir / f"frame_{start:04d}.png"
        kf2 = out_frames_dir / f"frame_{start + count // 2:04d}.png"
        if kf1.exists():
            Image.open(kf1).save(renders_dir / f"{name}_key1.png")
        if kf2.exists():
            Image.open(kf2).save(renders_dir / f"{name}_key2.png")
            
    print(f"All clips and keyframes saved to: {renders_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render camera flight film and individual acts.")
    parser.add_argument("--preview", dest="preview", action="store_true", help="Preview mode (640x360, 16 samples)")
    parser.add_argument("--final", dest="final", action="store_true", help="Final mode (1920x1080, 64 samples)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    args = parser.parse_args()
    
    if args.final and not args.preview:
        args.mode = "final"
        args.res_x = 1920
        args.res_y = 1080
        args.samples = 64
    else:
        args.mode = "preview"
        args.res_x = 640
        args.res_y = 360
        args.samples = 16
        
    print("=" * 70)
    print(f"  TOKAMAK FLIGHT FILM RENDERER — Mode: {args.mode.upper()}")
    print(f"  Resolution: {args.res_x}x{args.res_y} | EEVEE Samples: {args.samples} | FPS: {args.fps}")
    print("=" * 70)
    
    renders_dir = PROJECT / "motion" / "renders"
    out_frames_dir = renders_dir / f"{args.mode}_frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Render 3D Blender acts (frames 1..165)
    print("\n>>> Step 1: Rendering 3D camera flight acts in Blender...")
    timings = render_blender_batch(args, out_frames_dir)
    
    # 2. Render Act 5 2D comparison panel (frames 166..195)
    print("\n>>> Step 2: Generating 2D comparison panel act...")
    t_act5 = generate_act5_frames(args, out_frames_dir)
    timings["act5_compare_2d"] = t_act5
    
    # 3. Add high-contrast typography banners
    print("\n>>> Step 3: Annotating frames...")
    annotate_all_frames(out_frames_dir, is_preview=(args.mode == "preview"))
    
    # 4. Assemble clips and continuous MP4
    print("\n>>> Step 4: Encoding MP4 clips and extracting keyframes...")
    assemble_videos_and_keyframes(args, out_frames_dir, renders_dir)
    
    # Total timings
    total_render_sec = sum(v for k, v in timings.items() if k.startswith("act"))
    print("\n" + "=" * 70)
    print(f"RENDER TIMINGS SUMMARY ({args.mode.upper()} 640x360, 16 samples):")
    for k, v in timings.items():
        if k.startswith("act"):
            print(f"  - {k:22s}: {v:6.2f} s")
    print(f"  TOTAL RENDER TIME:      {total_render_sec:6.2f} s")
    print("=" * 70)
    
    # Estimate final render time (1920x1080, 64 samples)
    # Measured single frame: 5.14 s at 1920x1080 (vs 0.67 s at 640x360 -> factor ~ 7.7)
    time_per_frame_final = 5.14
    total_frames = 195
    est_final_sec = (165 * time_per_frame_final) + (30 * 0.1)  # 2D panel is instantaneous
    print("\nESTIMATED FINAL RENDER TIME (1920x1080, EEVEE):")
    print(f"  At 24 fps ({total_frames} frames, 8.1 s clip): ~{est_final_sec / 60.0:.1f} хвилин ({est_final_sec:.0f} с)")
    print(f"  At 30 fps ({total_frames} frames, 6.5 s clip): ~{est_final_sec / 60.0:.1f} хвилин ({est_final_sec:.0f} с)")
    print("=" * 70)


if __name__ == "__main__":
    main()
