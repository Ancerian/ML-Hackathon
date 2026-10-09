#!/usr/bin/env python3
"""Matplotlib overlays for the JET F5 deliverables (runs OUTSIDE Blender).

    python src/overlay_jet.py stills   render/jet/final/*.json
    python src/overlay_jet.py sections render/jet/sections/*_raw.json
    python src/overlay_jet.py anim     render/jet/anim/frames   (attribution + phase titles)

stills    <name>.png + <name>.json (anchors from render_jet_finals.py)
          -> <name>_annotated.png : short title, leader-line labels, small
          attribution line.  No burnt-in stamp; the full page lists belong in
          the figure captions of the manual.
sections  <tag>_<C>_raw.png + .json (orthographic camera mapping)
          -> <tag>_<C>.pdf (vector lines / text over the embedded raster) and
          <tag>_<C>.png.  Every dimension value comes from
          machines/jet1975/dimensions.yaml and carries «[EUR 5516e, с. N]»,
          N = the PDF page recorded there.

The original drawings are never traced or overlaid: the raster underneath is
our own render, the numbers are the sourced table values.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import patheffects as pe            # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
from PIL import Image                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = os.path.join(ROOT, "machines", "jet1975", "dimensions.yaml")

plt.rcParams["font.family"] = "DejaVu Sans"         # ships with matplotlib; Cyrillic
plt.rcParams["pdf.fonttype"] = 42                    # embed TrueType (editable text)
plt.rcParams["svg.fonttype"] = "none"


def ua(x, nd=None):
    """Ukrainian decimal comma."""
    s = f"{x:.{nd}f}" if nd is not None else f"{x:g}"
    return s.replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------------------
# stills
# ---------------------------------------------------------------------------
def layout_column(items, H, top, bottom, gap):
    """Place label y positions near their anchors without overlap."""
    items.sort(key=lambda d: d["y"])
    ys = [min(max(d["y"], top), bottom) for d in items]
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= d
                ys[i] += d
                moved = True
        ys = [min(max(y, top), bottom) for y in ys]
        if not moved:
            break
    # if the column is still over-full, spread it evenly
    if len(ys) > 1 and any(b - a < gap * 0.9 for a, b in zip(ys, ys[1:])):
        ys = list(np.linspace(top, min(bottom, top + gap * (len(ys) - 1)), len(ys)))
    for d, y in zip(items, ys):
        d["ly"] = y
    return items


def annotate_still(js_path):
    meta = json.load(open(js_path))
    d = os.path.dirname(js_path)
    img = Image.open(os.path.join(d, meta["image"])).convert("RGB")
    W, H = img.size
    dpi = 200.0
    s = W / 3840.0                                    # font scale vs 4K
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    fs_title, fs_sub, fs_lab, fs_att = 17 * s, 9.5 * s, 9.2 * s, 7.2 * s
    # title block (top-left)
    ax.add_patch(FancyBboxPatch((0.018 * W, 0.022 * H), 0.43 * W, 0.085 * H,
                                boxstyle=f"round,pad=0,rounding_size={8 * s}",
                                fc=(0.05, 0.06, 0.07, 0.62), ec="none"))
    ax.text(0.03 * W, 0.058 * H, meta["title"], color="white", fontsize=fs_title,
            fontweight="bold", va="center", ha="left")
    ax.text(0.03 * W, 0.089 * H, meta["variant"], color=(0.82, 0.85, 0.88),
            fontsize=fs_sub, va="center", ha="left")

    labels = [dict(l) for l in meta.get("labels", [])]
    left = [l for l in labels if l["x"] < W / 2]
    right = [l for l in labels if l["x"] >= W / 2]
    while abs(len(left) - len(right)) > 3:          # balance the columns
        src, dst = (left, right) if len(left) > len(right) else (right, left)
        src.sort(key=lambda l: abs(l["x"] - W / 2))
        dst.append(src.pop(0))
    gap = 0.058 * H
    top, bottom = 0.16 * H, 0.90 * H
    layout_column(left, H, top, bottom, gap)
    layout_column(right, H, top, bottom, gap)
    bbox = dict(boxstyle=f"round,pad=0.35,rounding_size=0.25", fc=(0.06, 0.07, 0.08, 0.72),
                ec=(1, 1, 1, 0.35), lw=0.6 * s)
    for side, col in (("L", left), ("R", right)):
        for l in col:
            x_txt = 0.03 * W if side == "L" else 0.97 * W
            ha = "left" if side == "L" else "right"
            t = ax.text(x_txt, l["ly"], l["text"], color="white", fontsize=fs_lab,
                        ha=ha, va="center", bbox=bbox, zorder=5)
            # leader: from the box edge to the anchor
            r = fig.canvas.get_renderer()
            bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            x0 = max(bb.x0, bb.x1) + 0.004 * W if side == "L" else min(bb.x0, bb.x1) - 0.004 * W
            elbow = x0 + (0.02 * W if side == "L" else -0.02 * W)
            ax.plot([x0, elbow, l["x"]], [l["ly"], l["ly"], l["y"]], color="white", lw=1.0 * s,
                    alpha=0.95, zorder=4, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=2.6 * s, foreground=(0, 0, 0, 0.55)),
                                  pe.Normal()])
            ax.add_patch(Circle((l["x"], l["y"]), 5.5 * s * W / 3840, fc="white",
                                ec=(0, 0, 0, 0.7), lw=0.8 * s, zorder=6))
    # attribution (bottom-left, small)
    ax.text(0.018 * W, 0.975 * H, meta["attribution"], color=(0.92, 0.92, 0.92),
            fontsize=fs_att, ha="left", va="bottom",
            bbox=dict(boxstyle="square,pad=0.35", fc=(0, 0, 0, 0.45), ec="none"))
    out = os.path.join(d, os.path.splitext(meta["image"])[0] + "_annotated.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[overlay] {out}")
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def load_dims():
    import yaml
    d = yaml.safe_load(open(DIMS))
    out = {}
    for sec, rows in d.items():
        if isinstance(rows, list):
            for e in rows:
                out[f"{sec}.{e['id']}"] = e
    return out


class Sheet:
    def __init__(self, meta, img):
        self.m = meta
        self.img = img
        self.W, self.H = img.size
        o = meta["ortho"]
        self.r = np.array(o["right"])
        self.u = np.array(o["up"])
        self.c = np.array(o["centre"])
        self.scale = o["scale"]

    def px(self, P):
        P = np.asarray(P, float)
        q = P - self.c
        return (self.W / 2 + (q @ self.r) / self.scale * self.W,
                self.H / 2 - (q @ self.u) / self.scale * self.W)

    # C5: section coordinates (s along screen-right, Z)
    def sz(self, s, z):
        rh = self.r.copy()
        rh[2] = 0
        rh /= np.linalg.norm(rh)
        return self.px(s * rh + np.array([0, 0, z]))

    # C6: plan coordinates along the screen axes, centred on the machine axis
    def pl(self, a, b):
        return self.px(a * self.r + b * self.u + np.array([0, 0, 0.0]))

    @property
    def m_per_px(self):
        return self.scale / self.W


TAG = "EUR 5516e, с. {}"
PX = 300.0 / 72.0                                   # points -> sheet pixels (dpi 300)
DIMC = "#1b1f24"
EXTC = (0.25, 0.28, 0.32, 0.55)


def dim_line(ax, p0, p1, text, fs, offset_text=0.0, side=+1, ticks=(), color=DIMC, lw=0.8):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    for t in ticks:
        v = np.subtract(p1, p0)
        n = np.array([-v[1], v[0]]) / (np.hypot(*v) + 1e-9)
        a, b = np.add(t, 9 * n), np.subtract(t, 9 * n)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw * 1.3)
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    vert = abs(p1[0] - p0[0]) < abs(p1[1] - p0[1])
    kw = dict(fontsize=fs, color=color, ha="center", va="center", zorder=8,
              bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.9))
    if vert:
        ax.text(mx + side * (fs * PX * 0.95 + offset_text), my, text, rotation=90, **kw)
    else:
        ax.text(mx + offset_text, my - side * fs * PX * 0.95, text, **kw)


def ext_line(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=EXTC, lw=0.45, ls=(0, (4, 3)), zorder=3)


def centre_line(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=(0.1, 0.1, 0.1, 0.7), lw=0.6,
            ls=(0, (14, 4, 2, 4)), zorder=3)


SECTION_LEGEND = [("мідь (обмотки TF, PF)", (0.75, 0.33, 0.12)),
                  ("залізо (ярмо)", (0.28, 0.36, 0.34)),
                  ("Inconel (камера, сильфони, порти)", (0.36, 0.45, 0.62)),
                  ("сталь (корпуси, опори, циліндр)", (0.55, 0.55, 0.58)),
                  ("алюмінієвий сплав (кільце, оболонка)", (0.70, 0.70, 0.62)),
                  ("Mo / графіт / Ni (лімітери)", (0.45, 0.47, 0.55)),
                  ("плазма: ψN, тепле ядро → холодний край", (1.0, 0.62, 0.40))]


def _sheet_fig(sh, pad):
    W, H = sh.W, sh.H
    L, R, T, Bt = pad
    FW, FH = W + L + R, H + T + Bt
    dpi = 300.0
    fig = plt.figure(figsize=(FW / dpi, FH / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-L, W + R)
    ax.set_ylim(H + Bt, -T)
    ax.axis("off")
    ax.imshow(sh.img, extent=(0, W, H, 0), interpolation="bilinear", zorder=1)
    return fig, ax, dpi


def _title_block(ax, sh, title, sub, fs, pad):
    L, R, T, Bt = pad
    ax.text(-L + 40, -T + 50, title, fontsize=fs * 1.7, fontweight="bold", va="top",
            color=DIMC)
    ax.text(-L + 40, -T + 50 + fs * PX * 2.6, sub, fontsize=fs * 0.95, va="top",
            color=(0.3, 0.32, 0.35))


def _legend(ax, x, y, fs, rows=SECTION_LEGEND):
    ax.text(x, y, "Коди перерізу:", fontsize=fs, fontweight="bold", color=DIMC, va="top")
    srgb = lambda c: tuple(12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
                           for x in c)
    for i, (t, c) in enumerate(rows):
        c = srgb(c) if "плазма" not in t else c
        yy = y + (i + 1) * fs * PX * 1.75
        ax.add_patch(Rectangle((x, yy - fs * PX * 0.55), fs * PX * 2.2, fs * PX * 1.1, fc=c,
                               ec=DIMC, lw=0.4, zorder=9))
        ax.text(x + fs * PX * 2.8, yy, t, fontsize=fs * 0.9, va="center", color=DIMC)


def section_C5(meta, img, D, out_base, y83, k_fs=1.0, k_tier=1.0):
    """k_fs / k_tier enlarge labels and tier spacing for a print variant that is
    shown at text width (the full sheet shrinks ~0.4x there)."""
    sh = Sheet(meta, img)
    mpp = sh.m_per_px
    pad = tuple(int(p * k) for p, k in zip((330, 430, 520, 620), (k_fs, k_fs, k_fs, k_fs)))
    fig, ax, dpi = _sheet_fig(sh, pad)
    fs = 6.2 * k_fs
    tier = 0.40 * k_tier                        # metres between dimension tiers
    v = lambda k: D[k]["value"]
    pg = lambda k: TAG.format(D[k]["page"])

    top_iron = v("iron_core.upper_arm_Z_top")
    hub = v("iron_core.pillar_hub_Z_top")
    floor = meta.get("floor_Z", -5.72)
    Rl = v("overall.overall_diameter") / 2

    # centre lines: machine axis, midplane, plasma axis
    centre_line(ax, sh.sz(0, floor - 0.4), sh.sz(0, hub + 0.4))
    centre_line(ax, sh.sz(-Rl - 0.3, 0), sh.sz(Rl + 0.3, 0))
    for sg in (-1, 1):
        centre_line(ax, sh.sz(sg * v("plasma.R0"), -2.6), sh.sz(sg * v("plasma.R0"), 2.6))

    # ---- top tiers (right half, radii from the machine axis) --------------------
    z = hub + 0.55
    rows_top = [
        ("ring", None),
        ("R0", [v("plasma.R0")], f"R0 = {ua(v('plasma.R0'))} м  [{pg('plasma.R0')}]", 0.0),
        ("plasma", [v("plasma.R_inner_plasma"), v("plasma.R_outer_plasma")],
         f"плазма R {ua(v('plasma.R_inner_plasma'))} … {ua(v('plasma.R_outer_plasma'))} м; "
         f"2a = {ua(2 * v('plasma.a'), 2)} м  [{pg('plasma.a')}]", 0.0),
        ("vin", [v("vessel.rs_R_inner_wall_inboard"), v("vessel.rs_R_inner_wall_outboard")],
         f"камера, внутр. R {ua(v('vessel.rs_R_inner_wall_inboard'))} … "
         f"{ua(v('vessel.rs_R_inner_wall_outboard'))} м  [{pg('vessel.rs_R_inner_wall_outboard')}]", 0.0),
        ("vout", [v("vessel.rs_R_outer_wall_inboard"), v("vessel.rs_R_outer_wall_outboard")],
         f"камера, зовн. R {ua(v('vessel.rs_R_outer_wall_inboard'))} … "
         f"{ua(v('vessel.rs_R_outer_wall_outboard'))} м  [{pg('vessel.rs_R_outer_wall_outboard')}]", 0.0),
        ("tf", [v("tf_coils.bore_R_inner"), v("tf_coils.bore_R_outer")],
         f"отвір TF R {ua(v('tf_coils.bore_R_inner'))} … {ua(v('tf_coils.bore_R_outer'), 2)} м  "
         f"[{pg('tf_coils.bore_R_outer')}]", 0.0),
    ]
    # ring (full diameter) first
    ring_R = v("structure.ring_outer_diameter") / 2
    ring_z = v("structure.ring_Z_top")
    p0, p1 = sh.sz(-ring_R, z), sh.sz(ring_R, z)
    for sg in (-1, 1):
        ext_line(ax, sh.sz(sg * ring_R, ring_z), sh.sz(sg * ring_R, z + 0.12))
    dim_line(ax, p0, p1, f"кільце ø{ua(v('structure.ring_outer_diameter'))} м  "
             f"[{pg('structure.ring_outer_diameter')}]", fs)
    z += tier
    feat_z = {"R0": 0.0, "plasma": 0.0, "vin": 0.0, "vout": 0.0, "tf": 0.0}
    for key, radii, text, _ in rows_top[1:]:
        z += tier * (0.0 if key == "R0" else 0.0)
        p0, p1 = sh.sz(0, z), sh.sz(max(radii), z)
        for R in radii:
            ext_line(ax, sh.sz(R, feat_z[key]), sh.sz(R, z + 0.12))
        dim_line(ax, p0, p1, text, fs, offset_text=0.0,
                 ticks=[sh.sz(R, z) for R in radii[:-1]])
        z += tier

    # ---- bottom tiers: PF coil radii, then overall diameter -------------------
    z = floor - 0.55
    pf = [("PF 1", v("pf_coils.coil1_mean_diameter") / 2, -2.24,
           f"PF 1: R {ua(v('pf_coils.coil1_mean_diameter') / 2, 3)} м (ø{ua(v('pf_coils.coil1_mean_diameter'))})  "
           f"[{pg('pf_coils.coil1_mean_diameter')}]"),
          ("PF 2", v("pf_coils.coil2_mean_diameter") / 2, -v("pf_coils.coil2_Z"),
           f"PF 2: R {ua(v('pf_coils.coil2_mean_diameter') / 2, 2)} м (ø{ua(v('pf_coils.coil2_mean_diameter'))})  "
           f"[{pg('pf_coils.coil2_mean_diameter')}]"),
          ("PF 3", v("pf_coils.coil3_R"), -v("pf_coils.coil3_Z"),
           f"PF 3: R {ua(v('pf_coils.coil3_R'))} м  [{pg('pf_coils.coil3_R')}]"),
          ("PF 4", v("pf_coils.coil4_R"), -v("pf_coils.coil4_Z"),
           f"PF 4: R {ua(v('pf_coils.coil4_R'))} м  [{pg('pf_coils.coil4_R')}]")]
    for name, R, zc, text in pf:
        ext_line(ax, sh.sz(R, zc), sh.sz(R, z - 0.12))
        dim_line(ax, sh.sz(0, z), sh.sz(R, z), text, fs, side=-1)
        z -= tier
    z -= 0.1
    for sg in (-1, 1):
        ext_line(ax, sh.sz(sg * Rl, floor), sh.sz(sg * Rl, z - 0.12))
    dim_line(ax, sh.sz(-Rl, z), sh.sz(Rl, z),
             f"габарит ø{ua(v('overall.overall_diameter'))} м  [{pg('overall.overall_diameter')}]",
             fs * 1.1, side=-1)

    # ---- vertical dims, left: vessel inner height, plasma 2b, overall 11.5 ----
    s = -Rl - 0.55
    zt = v("vessel.rs_Z_inner_wall_top")
    ext_line(ax, sh.sz(-v("vessel.rs_centre_line_R"), zt), sh.sz(s - 0.12, zt))
    ext_line(ax, sh.sz(-v("vessel.rs_centre_line_R"), -zt), sh.sz(s - 0.12, -zt))
    dim_line(ax, sh.sz(s, -zt), sh.sz(s, zt),
             f"камера, внутр. Z ±{ua(zt)} м  [{pg('vessel.rs_Z_inner_wall_top')}]", fs, side=-1)
    s -= tier
    b = v("plasma.b")
    for sg in (-1, 1):
        ext_line(ax, sh.sz(-2.3, sg * b), sh.sz(s - 0.12, sg * b))
    dim_line(ax, sh.sz(s, -b), sh.sz(s, b),
             f"плазма 2b = {ua(2 * b, 2)} м  [{pg('plasma.b')}]", fs, side=-1)
    s -= tier
    oh = v("overall.overall_height")
    zb = top_iron - oh
    ext_line(ax, sh.sz(-Rl, top_iron), sh.sz(s - 0.12, top_iron))
    ext_line(ax, sh.sz(-Rl, zb), sh.sz(s - 0.12, zb))
    dim_line(ax, sh.sz(s, zb), sh.sz(s, top_iron),
             f"габарит {ua(oh)} м  [{pg('overall.overall_height')}]", fs * 1.1, side=-1)

    # ---- vertical dims, right: PF 2 / 3 / 4 heights ---------------------------
    s = Rl + 0.55
    for key, R, lab in (("pf_coils.coil4_Z", v("pf_coils.coil4_R"), "PF 4"),
                        ("pf_coils.coil3_Z", v("pf_coils.coil3_R"), "PF 3"),
                        ("pf_coils.coil2_Z", v("pf_coils.coil2_mean_diameter") / 2, "PF 2")):
        zz = v(key)
        for sg in (-1, 1):
            ext_line(ax, sh.sz(R, sg * zz), sh.sz(s + 0.12, sg * zz))
        dim_line(ax, sh.sz(s, -zz), sh.sz(s, zz), f"{lab}: Z ±{ua(zz)} м  [{pg(key)}]", fs,
                 side=+1)
        s += tier

    # title, legend, notes
    title = f"JET {'1983' if y83 else '1975'} — вертикальний переріз R–Z через ярмо"
    # 1983 subtitle is long: break after the variant so it clears the top dimension tier
    sub = (meta["variant"].rstrip(".") + (".\n" if y83 else ".  ") +
           "Ортографічна проекція, масштаб за розмірами; "
           "розміри — таблиці/креслення EUR 5516e (сторінки PDF у дужках)")
    if y83:
        sub += ";\nгеометрія — варіант 1983 р., розміри — проектні значення 1975 р."
    _title_block(ax, sh, title, sub, fs, pad)
    if k_fs > 1.0:      # print variant: legend in the empty top-left corner
        _legend(ax, -pad[0] + 60, -pad[2] + fs * PX * 6, fs * 0.95)
    else:
        _legend(ax, sh.W + pad[1] - 60 - fs * PX * 26, sh.H + pad[3] - 60 - fs * PX * 14.5,
                fs * 0.95)
    ax.text(-pad[0] + 40, sh.H + pad[3] - 40, meta["attribution"], fontsize=fs * 0.85,
            color=(0.35, 0.37, 0.4), va="bottom")
    _save(fig, out_base, dpi)


def section_C6(meta, img, D, out_base, y83):
    sh = Sheet(meta, img)
    pad = (700, 1000, 380, 560)
    fig, ax, dpi = _sheet_fig(sh, pad)
    fs = 6.2
    v = lambda k: D[k]["value"]
    pg = lambda k: TAG.format(D[k]["page"])
    Rl = v("overall.overall_diameter") / 2
    # centre lines + R0 circle (dash-dot)
    centre_line(ax, sh.pl(-Rl - 0.4, 0), sh.pl(Rl + 0.4, 0))
    centre_line(ax, sh.pl(0, -Rl - 0.4), sh.pl(0, Rl + 0.4))
    th = np.linspace(0, 2 * np.pi, 400)
    R0 = v("plasma.R0")
    xy = np.array([sh.pl(R0 * math.cos(t), R0 * math.sin(t)) for t in th])
    ax.plot(xy[:, 0], xy[:, 1], color=(0.1, 0.1, 0.1, 0.7), lw=0.6, ls=(0, (14, 4, 2, 4)),
            zorder=3)
    # radial dimensions from the axis, fanned over the right half (NBI is left)
    rad = [
        (R0, f"R0 = {ua(R0)} м  [{pg('plasma.R0')}]"),
        (v("plasma.R_outer_plasma"), f"плазма R {ua(v('plasma.R_outer_plasma'))} м  [{pg('plasma.R_outer_plasma')}]"),
        (v("vessel.rs_R_inner_wall_outboard"),
         f"камера, внутр. R {ua(v('vessel.rs_R_inner_wall_outboard'))} м  [{pg('vessel.rs_R_inner_wall_outboard')}]"),
        (v("vessel.rs_R_outer_wall_outboard"),
         f"камера, зовн. R {ua(v('vessel.rs_R_outer_wall_outboard'))} м  [{pg('vessel.rs_R_outer_wall_outboard')}]"),
        (v("tf_coils.bore_R_outer"), f"отвір TF R {ua(v('tf_coils.bore_R_outer'), 2)} м  [{pg('tf_coils.bore_R_outer')}]"),
        (v("pf_coils.coil4_R"), f"PF 4 (під площиною) R {ua(v('pf_coils.coil4_R'))} м  [{pg('pf_coils.coil4_R')}]"),
        (v("iron_core.limb_R_inner"), f"ярмо, внутр. R {ua(v('iron_core.limb_R_inner'))} м  [{pg('iron_core.limb_R_inner')}]"),
    ]
    inner = [
        (v("plasma.R_inner_plasma"), f"плазма R {ua(v('plasma.R_inner_plasma'))} м  [{pg('plasma.R_inner_plasma')}]"),
        (v("vessel.rs_R_inner_wall_inboard"),
         f"камера, внутр. R {ua(v('vessel.rs_R_inner_wall_inboard'))} м  [{pg('vessel.rs_R_inner_wall_inboard')}]"),
        (v("tf_coils.bore_R_inner"), f"отвір TF R {ua(v('tf_coils.bore_R_inner'))} м  [{pg('tf_coils.bore_R_inner')}]"),
        (v("pf_coils.coil1_mean_diameter") / 2,
         f"PF 1 R {ua(v('pf_coils.coil1_mean_diameter') / 2, 3)} м  [{pg('pf_coils.coil1_mean_diameter')}]"),
    ]
    R_txt = Rl + 0.9
    angs = np.radians(np.linspace(62, -62, len(rad)))
    for (R, text), a in zip(rad, angs):
        _radius(ax, sh, R, a, R_txt, text, fs)
    angs_i = np.radians([112, 128, 232, 248])
    for (R, text), a in zip(inner, angs_i):
        _radius(ax, sh, R, a, R_txt, text, fs, left=True)
    # diameters: ring (top), overall (bottom)
    rr = v("structure.ring_outer_diameter") / 2
    b = Rl + 0.7
    for sg in (-1, 1):
        ext_line(ax, sh.pl(sg * rr, 0), sh.pl(sg * rr, b + 0.12))
    dim_line(ax, sh.pl(-rr, b), sh.pl(rr, b),
             f"кільце ø{ua(v('structure.ring_outer_diameter'))} м  [{pg('structure.ring_outer_diameter')}]", fs)
    b = -Rl - 0.7
    for sg in (-1, 1):
        ext_line(ax, sh.pl(sg * Rl, 0), sh.pl(sg * Rl, b - 0.12))
    dim_line(ax, sh.pl(-Rl, b), sh.pl(Rl, b),
             f"габарит ø{ua(v('overall.overall_diameter'))} м  [{pg('overall.overall_diameter')}]",
             fs * 1.1, side=-1)
    title = f"JET {'1983' if y83 else '1975'} — план, переріз у площині Z = 0"
    sub = (meta["variant"].rstrip(".") + ".  Ортографічна проекція згори; розміри — EUR 5516e "
           "(сторінки PDF у дужках)")
    if y83:
        sub += ";\nгеометрія — варіант 1983 р., розміри — проектні значення 1975 р."
    _title_block(ax, sh, title, sub, fs, pad)
    _legend(ax, sh.W + pad[1] - 60 - fs * PX * 26, sh.H + pad[3] - 60 - fs * PX * 14.5,
            fs * 0.95)
    ax.text(-pad[0] + 40, sh.H + pad[3] - 40, meta["attribution"], fontsize=fs * 0.85,
            color=(0.35, 0.37, 0.4), va="bottom")
    _save(fig, out_base, dpi)


def _radius(ax, sh, R, a, R_txt, text, fs, left=False):
    c = sh.pl(0, 0)
    p = sh.pl(R * math.cos(a), R * math.sin(a))
    q = sh.pl(R_txt * math.cos(a), R_txt * math.sin(a))
    ax.annotate("", xy=p, xytext=c, arrowprops=dict(arrowstyle="-|>", color=DIMC, lw=0.7,
                                                    shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=7)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=EXTC, lw=0.5, ls=(0, (4, 3)), zorder=6)
    ax.add_patch(Circle(p, 4, fc=DIMC, ec="none", zorder=8))
    dx = -1 if q[0] < c[0] else 1
    q2 = (q[0] + dx * 40, q[1])
    ax.plot([q[0], q2[0]], [q[1], q2[1]], color=EXTC, lw=0.5, zorder=6)
    ax.text(q2[0] + dx * 8, q2[1], text, fontsize=fs, ha="left" if dx > 0 else "right",
            va="center", color=DIMC, zorder=9,
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.9))


def _save(fig, base, dpi):
    fig.savefig(base + ".pdf", dpi=dpi)
    fig.savefig(base + ".png", dpi=dpi)
    plt.close(fig)
    print(f"[overlay] {base}.pdf / .png")


def annotate_section(js_path, print_variant=False):
    meta = json.load(open(js_path))
    d = os.path.dirname(js_path)
    img = Image.open(os.path.join(d, meta["image"])).convert("RGB")
    D = load_dims()
    base = os.path.join(d, os.path.basename(js_path).replace("_raw.json", ""))
    y83 = "1983" in meta.get("machine", "") or "1983" in os.path.basename(js_path)
    if print_variant:
        # larger labels for a page-width figure (manual, fig. 5.1)
        if meta["key"] == "C5":
            section_C5(meta, img, D, base + "_print", y83, k_fs=K_FS_PRINT, k_tier=K_TIER_PRINT)
        return
    if meta["key"] == "C5":
        section_C5(meta, img, D, base, y83)
    else:
        section_C6(meta, img, D, base, y83)


# ---------------------------------------------------------------------------
# animation frames: attribution + phase title burnt in with PIL (fast)
# ---------------------------------------------------------------------------
def annotate_frames(fdir, fps=24, phases=((0, 10, "JET 1975 — загальний вигляд"),
                                           (10, 18, "розріз октанта"),
                                           (18, 30, "плазма: магнітні поверхні ψN і силові лінії"),
                                           (30, 40, "вертикальний переріз R–Z"))):
    from PIL import ImageDraw, ImageFont
    font_path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                             "ttf", "DejaVuSans.ttf")
    bold_path = font_path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    out = os.path.join(os.path.dirname(fdir.rstrip("/")), "frames_annotated")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(fdir, "*.png")))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        W, H = im.size
        s = W / 1920
        ft = ImageFont.truetype(bold_path, int(30 * s))
        fa = ImageFont.truetype(font_path, int(15 * s))
        t = i / fps
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for t0, t1, txt in phases:
            if t0 <= t < t1:
                a = min(1.0, (t - t0) / 0.6, (t1 - t) / 0.6)
                a = max(0.0, a)
                dr.text((int(40 * s), int(34 * s)), txt, font=ft,
                        fill=(255, 255, 255, int(235 * a)),
                        stroke_width=int(2 * s), stroke_fill=(0, 0, 0, int(140 * a)))
        att = ATTRIB
        dr.text((int(24 * s), H - int(30 * s)), att, font=fa, fill=(235, 235, 235, 220),
                stroke_width=1, stroke_fill=(0, 0, 0, 150))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        im.save(os.path.join(out, os.path.basename(f)))
    print(f"[overlay] {len(files)} frames -> {out}")


K_FS_PRINT, K_TIER_PRINT = 2.0, 1.45       # print variant of the C5 section

ATTRIB = "Власна 3D-реконструкція; побудовано за даними EUR 5516e (1975)"


def main(argv):
    mode = argv[0]
    paths = []
    for p in argv[1:]:
        paths += sorted(glob.glob(p)) or [p]
    if mode == "stills":
        for p in paths:
            if p.endswith(".json") and "render_times" not in p:
                annotate_still(p)
    elif mode == "sections":
        for p in paths:
            annotate_section(p)
    elif mode == "sections-print":
        for p in paths:
            annotate_section(p, print_variant=True)
    elif mode == "anim":
        annotate_frames(paths[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
