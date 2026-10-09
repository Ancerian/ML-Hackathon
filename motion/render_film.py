#!/usr/bin/env python3
"""motion/render_film.py — Render continuous camera flight film and individual acts.

Supports:
  --preview : 640x360, 4 samples EEVEE (fast verification, ~5 min for 1344 3D frames)
  --final   : 1920x1080, 64 samples EEVEE (Full HD master render)

Frame allocation at 24 fps:
  Act 1: Machine Cutaway (Frames 1 .. 192, 8.0 s)
  Act 2: Magnetic Flux Surfaces (Frames 193 .. 480, 12.0 s)
  Act 3: Field Lines Canonical Hamiltonian (Frames 481 .. 768, 12.0 s)
  Act 4A: Poincare Section A (Frames 769 .. 1056, 12.0 s: single mode 2/1, dynamic turn accumulation)
  Act 4B: Poincare Section B (Frames 1057 .. 1344, 12.0 s: two modes 2/1+3/1, key frame A=4.8e-3 overlap)
  Act 5: 2D Equilibrium Comparison (Frames 1345 .. 1632, 12.0 s: DIII-D #062 Frame 147)
Total: 1632 frames (68.0 s continuous film).
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
POINCARE_SWEEP_JSON = str(PROJECT / "motion" / "data" / "poincare_sweep.json")
POINCARE_HIGHRES_JSON = str(PROJECT / "motion" / "data" / "poincare_highres_chaos.json")


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
    
    draw.text((pad_x, pad_y), title, font=font_t, fill=(255, 255, 255, 255))
    draw.text((pad_x, pad_y + font_size_t + 4), subtitle, font=font_s, fill=(201, 209, 217, 255))
    
    # Footer watermark / honesty stamp
    footer_h = int(h * 0.065)
    draw.rectangle([(0, h - footer_h), (w, h)], fill=(14, 17, 23, 200))
    stamp = "FMF KPI AI Lab | TokaBench-GS | CC BY 4.0 | Вакуумні моди, не є прямою симуляцією плазми"
    draw.text((pad_x, h - footer_h + int(footer_h * 0.2)), stamp, font=font_s, fill=(139, 148, 158, 240))
    
    combined = Image.alpha_composite(im, overlay)
    combined.convert("RGB").save(image_path, "PNG")


def render_blender_batch(args, raw_frames_dir: Path):
    """Executes Blender in headless mode to render 3D camera flight acts (frames 1..1344)."""
    script_content = f"""
import bpy
import json
import math
import os
import time

blend_file = "{BLEND_FILE}"
raw_dir = "{str(raw_frames_dir)}"
sweep_file = "{POINCARE_SWEEP_JSON}"
highres_file = "{POINCARE_HIGHRES_JSON}"

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

def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))

def update_mesh_pts(ob, pts):
    me = ob.data
    me.clear_geometry()
    if pts:
        me.from_pydata(pts, [], [])
    me.update()

# Load Poincare data
with open(sweep_file) as f:
    sweep_data = json.load(f)
highres_frames = {{}}
if os.path.exists(highres_file):
    with open(highres_file) as f:
        hr = json.load(f)
        for hrf in hr.get('frames', []):
            highres_frames[round(float(hrf['amp']), 4)] = hrf

# Flight coordinates
pos_cutaway_start = (-34.0, -38.0, 24.0)
pos_cutaway_end   = (-27.0, -36.0, 20.0)
tgt_cutaway       = (0.0, 0.0, 0.0)

pos_surf_start    = (-27.0, -36.0, 20.0)
pos_surf_end      = (-20.0, -28.0, 12.0)
tgt_surf          = (2.0, 0.0, 0.0)

pos_fl_start      = (-20.0, -28.0, 12.0)
pos_fl_end        = (-10.5, -21.0, 5.5)
tgt_fl            = (4.5, -3.0, 0.0)

pos_p_a_start     = (4.7, -21.5, 1.2)
pos_p_a_end       = (5.3, -19.5, -0.6)

pos_p_b_start     = (5.3, -19.5, -0.6)
pos_p_b_end       = (5.0, -18.0, 0.5)
tgt_poincare      = (5.03, 0.0, 0.0)

timings = {{}}
t_start_all = time.time()

# ----------------- ACT 1: CUTAWAY (Frames 1..192, 8.0 s) -----------------
t0 = time.time()
if p_reg: p_reg.hide_render = True
if p_isl: p_isl.hide_render = True
if p_ch: p_ch.hide_render = True
if p_plane: p_plane.hide_render = True
if fl: fl.hide_render = True

for f in range(1, 193):
    out_p = f'{{raw_dir}}/frame_{{f:04d}}.png'
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        continue
    t_norm = (f - 1) / 191.0
    cam.location = lerp(pos_cutaway_start, pos_cutaway_end, t_norm)
    tgt.location = tgt_cutaway
    scene.render.filepath = out_p
    bpy.ops.render.render(write_still=True)
timings['act1_cutaway'] = time.time() - t0

# ----------------- ACT 2: SURFACES (Frames 193..480, 12.0 s) -----------------
t0 = time.time()
if fl: fl.hide_render = True
for f in range(193, 481):
    out_p = f'{{raw_dir}}/frame_{{f:04d}}.png'
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        continue
    t_norm = (f - 193) / 287.0
    cam.location = lerp(pos_surf_start, pos_surf_end, t_norm)
    tgt.location = tgt_surf
    scene.render.filepath = out_p
    bpy.ops.render.render(write_still=True)
timings['act2_surfaces'] = time.time() - t0

# ----------------- ACT 3: FIELD LINES (Frames 481..768, 12.0 s) -----------------
t0 = time.time()
if fl: fl.hide_render = False
for f in range(481, 769):
    out_p = f'{{raw_dir}}/frame_{{f:04d}}.png'
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        continue
    t_norm = (f - 481) / 287.0
    cam.location = lerp(pos_fl_start, pos_fl_end, t_norm)
    tgt.location = tgt_fl
    scene.render.filepath = out_p
    bpy.ops.render.render(write_still=True)
timings['act3_fieldlines'] = time.time() - t0

# ----------------- ACT 4A: POINCARE SECTION A (Frames 769..1056, 12.0 s) -----------------
# 7 amplitude steps: [41, 41, 41, 41, 41, 41, 42] = 288 frames. Single mode 2/1.
t0 = time.time()
if fl: fl.hide_render = True
if p_reg: p_reg.hide_render = False
if p_isl: p_isl.hide_render = False
if p_ch: p_ch.hide_render = True
if p_plane: p_plane.hide_render = False

step_lengths_a = [41, 41, 41, 41, 41, 41, 42]
cum_a = [0]
for sl in step_lengths_a:
    cum_a.append(cum_a[-1] + sl)

for f in range(769, 1057):
    out_p = f'{{raw_dir}}/frame_{{f:04d}}.png'
    idx_in_act = f - 769
    
    # Camera slow motion
    t_act_norm = idx_in_act / 287.0
    cam.location = lerp(pos_p_a_start, pos_p_a_end, t_act_norm)
    tgt.location = tgt_poincare
    
    # Amplitude step
    step_idx = 0
    for s in range(len(step_lengths_a)):
        if cum_a[s] <= idx_in_act < cum_a[s+1]:
            step_idx = s
            break
            
    # Progress within step: punctures accumulate turn-by-turn (~1.5 s per step)
    k_in_step = idx_in_act - cum_a[step_idx]
    k_ratio = min(1.0, (k_in_step + 1) / 34.0)
    
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        continue
        
    sweep_frame = sweep_data['frames'][step_idx]
    reg_pts, isl_pts = [], []
    for line in sweep_frame['lines']:
        pts_line = line['points']
        n_show = max(1, int(len(pts_line) * k_ratio))
        for pt in pts_line[:n_show]:
            r_val, z_val = pt[0], pt[1]
            # Scaled Blender coordinates: (R*3.0, 0.0, Z*3.0)
            p_bl = (r_val * 3.0, 0.0, z_val * 3.0)
            if 2.02 <= r_val <= 2.17:
                isl_pts.append(p_bl)
            else:
                reg_pts.append(p_bl)
                
    update_mesh_pts(p_reg, reg_pts)
    update_mesh_pts(p_isl, isl_pts)
    update_mesh_pts(p_ch, [])
    
    scene.render.filepath = out_p
    bpy.ops.render.render(write_still=True)
timings['act4a_poincare_single'] = time.time() - t0

# ----------------- ACT 4B: POINCARE SECTION B (Frames 1057..1344, 12.0 s) -----------------
# 8 amplitude steps: 36 frames each = 288 frames. Two modes 2/1+3/1 (chaos overlap at A=4.8e-3).
t0 = time.time()
if p_ch: p_ch.hide_render = False

step_len_b = 36
for f in range(1057, 1345):
    out_p = f'{{raw_dir}}/frame_{{f:04d}}.png'
    idx_in_act = f - 1057
    
    # Camera slow motion
    t_act_norm = idx_in_act / 287.0
    cam.location = lerp(pos_p_b_start, pos_p_b_end, t_act_norm)
    tgt.location = tgt_poincare
    
    # Amplitude step (frames 7..14 of sweep)
    step_idx = min(7, idx_in_act // step_len_b)
    sweep_frame = sweep_data['frames'][7 + step_idx]
    amp_val = sweep_frame['amp']
    
    # Progress within step: punctures accumulate turn-by-turn
    k_in_step = idx_in_act % step_len_b
    k_ratio = min(1.0, (k_in_step + 1) / 30.0)
    
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        continue
        
    reg_pts, isl_pts, ch_pts = [], [], []
    has_dense_chaos = round(float(amp_val), 4) in highres_frames
    
    for line in sweep_frame['lines']:
        pts_line = line['points']
        n_show = max(1, int(len(pts_line) * k_ratio))
        for pt in pts_line[:n_show]:
            r_val, z_val = pt[0], pt[1]
            p_bl = (r_val * 3.0, 0.0, z_val * 3.0)
            if step_idx < 4:
                # Pre-overlap: 2/1 and 3/1 islands
                if 2.02 <= r_val <= 2.22:
                    isl_pts.append(p_bl)
                else:
                    reg_pts.append(p_bl)
            else:
                # Stochastic overlap layer (A >= 4.0e-3, key frame A=4.8e-3 S=1.002, k=2.02)
                if 2.08 <= r_val <= 2.17:
                    ch_pts.append(p_bl)
                elif 2.02 <= r_val <= 2.22:
                    isl_pts.append(p_bl)
                else:
                    reg_pts.append(p_bl)
                    
    # Add dense highres chaotic points for stochastic layer visualization
    if has_dense_chaos and step_idx >= 4:
        hrf = highres_frames[round(float(amp_val), 4)]
        for line in hrf['lines'][::3]: # Subsample lines for clean density
            pts_line = line['points']
            n_show = max(1, int(len(pts_line) * k_ratio * 0.5))
            for pt in pts_line[:n_show]:
                p_bl = (pt[0] * 3.0, 0.0, pt[1] * 3.0)
                ch_pts.append(p_bl)
                
    update_mesh_pts(p_reg, reg_pts)
    update_mesh_pts(p_isl, isl_pts)
    update_mesh_pts(p_ch, ch_pts)
    
    scene.render.filepath = out_p
    bpy.ops.render.render(write_still=True)
timings['act4b_poincare_chaos'] = time.time() - t0

timings['total_3d_sec'] = time.time() - t_start_all
with open(f'{{raw_dir}}/timings.json', 'w') as f:
    json.dump(timings, f, indent=2)

print('BLENDER_BATCH_DONE')
"""
    runner_script = raw_frames_dir / "run_blender_inner.py"
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
        
    timings_path = raw_frames_dir / "timings.json"
    if timings_path.exists():
        with open(timings_path) as f:
            timings = json.load(f)
    else:
        timings = {"total_3d_sec": dt}
    timings["total_session_sec"] = dt
    return timings


def generate_act5_frames(args, raw_frames_dir: Path):
    """Generates Act 5 2D comparison panel frames (frames 1345..1632, 288 frames) from contours_compare.png."""
    t0 = time.time()
    src_img = Image.open(COMPARE_PNG).convert("RGBA")
    
    w_t, h_t = args.res_x, args.res_y
    src_w, src_h = src_img.size
    scale = min(w_t / src_w, h_t / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    panel = Image.new("RGBA", (w_t, h_t), (14, 17, 23, 255))
    off_x = (w_t - new_w) // 2
    off_y = (h_t - new_h) // 2
    panel.paste(resized, (off_x, off_y), resized)
    
    for f in range(1345, 1633):
        out_f = raw_frames_dir / f"frame_{f:04d}.png"
        if not (out_f.exists() and out_f.stat().st_size > 1000):
            panel.convert("RGB").save(out_f, "PNG")
        
    return time.time() - t0


def annotate_all_frames(raw_frames_dir: Path, out_frames_dir: Path, is_preview: bool = True):
    """Copies raw frames to annotated frames and applies high-contrast readable banners."""
    print("Applying high-contrast projector banners (frames 1..1632)...")
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    
    acts_meta = [
        # (range, title, subtitle, act_num)
        (
            range(1, 193),
            "АКТ 1: РОЗРІЗ РЕАКТОРА (ТОКАМАК DIII-D #203702)",
            "Вакуумна камера, котушки тороїдального (TF) та полоїдального (PF) полів, дивертор",
            1
        ),
        (
            range(193, 481),
            "АКТ 2: МАГНІТНІ ПОВЕРХНІ ПОЛОЇДАЛЬНОГО ПОТОКУ ψ_N ∈ [0.12 .. 0.97]",
            "Вкладені магнітні поверхні розрахунку EFIT (кадр 75, t=1760 мс, q_95=3.7)",
            2
        ),
        (
            range(481, 769),
            "АКТ 3: СИЛОВА ЛІНІЯ: КАНОНІЧНА ГАМІЛЬТОНОВА СИСТЕМА (E19)",
            "Збереження фазового об'єму |det J - 1| ≤ 10^-10; намотка на тори із q = m/n",
            3
        ),
        (
            range(769, 1057),
            "АКТ 4А: ПЕРЕРІЗ ПУАНКАРЕ — ОДИНОЧНА МОДА 2/1 (A=10^-3, w=27.1 мм, модель сцени)",
            "Параметр Чирикова S = н/д (одна мода); повна відсутність хаосу (теорема E20)",
            4
        ),
        (
            range(1057, 1345),
            "АКТ 4Б: ПЕРЕРІЗ ПУАНКАРЕ — КЛЮЧОВИЙ КАДР ПЕРЕКРИТТЯ A=4.8e-3 (k=2.02)",
            "S≈1.0–1.1: початок перекриття островів; FTLE у 1.7–2.0 раза більший; стохастичність ймовірна, кількісно не підтверджена",
            4
        ),
        (
            range(1345, 1633),
            "АКТ 5: 2D ПОРІВНЯННЯ РІВНОВАГ (DIII-D #062, КАДР 147 — МЕДІАННИЙ ЗА g КАДР ТЕСТУ)",
            "Істина g=0.0065 проти UNet_Lite g=0.8618; єдина шкала нев'язки g_local ∈ [0.0 .. 1.2]",
            5
        ),
    ]
    
    t0 = time.time()
    n_annotated = 0
    for f_range, title, subtitle, act_num in acts_meta:
        for f in f_range:
            raw_p = raw_frames_dir / f"frame_{f:04d}.png"
            out_p = out_frames_dir / f"frame_{f:04d}.png"
            if raw_p.exists():
                if out_p.exists() and out_p.stat().st_mtime >= raw_p.stat().st_mtime and out_p.stat().st_size > 1000:
                    continue
                # Copy then add banner
                im = Image.open(raw_p)
                im.save(out_p, "PNG")
                add_banner(out_p, title, subtitle, act_num, is_preview=is_preview)
                n_annotated += 1
                
    print(f"  Banners applied to {n_annotated} updated frames in {time.time() - t0:.2f} s")


def assemble_videos_and_keyframes(args, out_frames_dir: Path, renders_dir: Path):
    """Encodes separate act clips, continuous full film (68.0 s), and exports keyframes."""
    print("Encoding video clips with ffmpeg...")
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    fps = args.fps
    
    # 1. Full continuous film (frames 1..1632, 68.0 s)
    full_mp4 = renders_dir / f"film_continuous_{args.mode}.mp4"
    cmd_full = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", str(out_frames_dir / "frame_%04d.png"),
        "-vframes", "1632",
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
        ("act1_cutaway", 1, 192),           # 192 frames = 8.0 s
        ("act2_surfaces", 193, 480),        # 288 frames = 12.0 s
        ("act3_fieldlines", 481, 768),      # 288 frames = 12.0 s
        ("act4a_poincare_single", 769, 1056), # 288 frames = 12.0 s
        ("act4b_poincare_chaos", 1057, 1344), # 288 frames = 12.0 s
        ("act5_compare_2d", 1345, 1632),    # 288 frames = 12.0 s
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
    parser.add_argument("--preview", dest="preview", action="store_true", help="Preview mode (640x360, 4 samples)")
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
        args.samples = 4
        
    print("=" * 70)
    print(f"  TOKAMAK FLIGHT FILM RENDERER — Mode: {args.mode.upper()}")
    print(f"  Resolution: {args.res_x}x{args.res_y} | EEVEE Samples: {args.samples} | FPS: {args.fps}")
    print(f"  Total Duration: 68.0 s (1632 frames total, 1344 3D frames)")
    print("=" * 70)
    
    renders_dir = PROJECT / "motion" / "renders"
    raw_frames_dir = renders_dir / f"{args.mode}_raw"
    out_frames_dir = renders_dir / f"{args.mode}_frames"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Render 3D Blender acts (frames 1..1344)
    print("\n>>> Step 1: Rendering 3D camera flight acts in Blender (frames 1..1344)...")
    timings = render_blender_batch(args, raw_frames_dir)
    
    # 2. Render Act 5 2D comparison panel (frames 1345..1632)
    print("\n>>> Step 2: Generating 2D comparison panel act (frames 1345..1632)...")
    t_act5 = generate_act5_frames(args, raw_frames_dir)
    timings["act5_compare_2d"] = t_act5
    
    # 3. Add high-contrast typography banners
    print("\n>>> Step 3: Annotating frames...")
    annotate_all_frames(raw_frames_dir, out_frames_dir, is_preview=(args.mode == "preview"))
    
    # 4. Assemble clips and continuous MP4
    print("\n>>> Step 4: Encoding MP4 clips and extracting keyframes...")
    assemble_videos_and_keyframes(args, out_frames_dir, renders_dir)
    
    # Total timings
    total_render_sec = sum(v for k, v in timings.items() if k.startswith("act"))
    print("\n" + "=" * 70)
    print(f"RENDER TIMINGS SUMMARY ({args.mode.upper()} {args.res_x}x{args.res_y}, {args.samples} samples):")
    for k, v in timings.items():
        if k.startswith("act"):
            print(f"  - {k:24s}: {v:7.2f} s")
    print(f"  TOTAL RENDER TIME:        {total_render_sec:7.2f} s ({total_render_sec/60.0:.1f} хв)")
    print("=" * 70)
    
    # Estimate final render time (1920x1080, 64 samples)
    time_per_frame_final = 5.14
    total_3d_frames = 1344
    est_final_sec = (total_3d_frames * time_per_frame_final) + 5.0
    print("\nESTIMATED FINAL RENDER TIME (1920x1080, 64 samples, EEVEE):")
    print(f"  At 24 fps ({total_3d_frames} 3D frames, 68 s total film): ~{est_final_sec / 60.0:.1f} хвилин ({est_final_sec / 3600.0:.2f} годин)")
    print("=" * 70)


if __name__ == "__main__":
    main()
