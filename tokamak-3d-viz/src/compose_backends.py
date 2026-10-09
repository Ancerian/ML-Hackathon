#!/usr/bin/env python3
"""Compose the two backends' Poincare renders side by side, annotated.

This is the picture of the open question in docs/COIL-BACKEND.md: the analytic
perturbation makes clean island chains, the coil field makes a chaotic sea, and
it is not yet settled how much of that difference is physics and how much is
residual field-reconstruction error.

    python src/compose_backends.py --left render/analytic.png \
        --right render/coil.png --out render/backends.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CAVEAT = ("Open question. The coil backend's chaotic fraction barely responds to "
          "amplitude (33% -> 43% over a tenfold current increase), which is a floor, "
          "not a response. Two causes remain unseparated: a broad poloidal spectrum "
          "driving all five n=1 resonances at once, or residual field-reconstruction "
          "error acting as broadband noise. Chaos statistics from the coil backend "
          "are NOT trusted -- see docs/COIL-BACKEND.md.")


def panel_text(w, h, lines, dpi=100, fontsize=17, colour="w"):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    for i, (txt, col) in enumerate(lines):
        fig.text(0.02, 0.86 - i * 0.30, txt, color=col, fontsize=fontsize,
                 family="DejaVu Sans", va="top")
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA")
    plt.close(fig)
    return img


def legend_strip(w, h, dpi=100):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    items = [("regular (KAM)", "#2e6bff"), ("island (phase-locked)", "#ff8c0f"),
             ("chaotic", "#ff2121")]
    for i, (lab, col) in enumerate(items):
        fig.text(0.03 + i * 0.30, 0.55, "●", color=col, fontsize=26, va="center")
        fig.text(0.055 + i * 0.30, 0.55, lab, color="w", fontsize=16, va="center")
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA")
    plt.close(fig)
    return img


def autocrop(img, thresh=14, pad=8):
    """Trim the near-black margins.

    The Poincare camera renders a square frame while the section itself is a
    tall poloidal cross-section, so roughly half of each render is empty.
    """
    a = np.asarray(img.convert("L"), dtype=np.int16)
    cols = np.where(a.max(axis=0) > thresh)[0]
    rows = np.where(a.max(axis=1) > thresh)[0]
    if cols.size == 0 or rows.size == 0:
        return img
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad, img.width)
    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad, img.height)
    return img.crop((x0, y0, x1, y1))


def wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width=width)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--bake-left", default="data/bake01")
    ap.add_argument("--bake-right", default="data/bake_coil")
    ap.add_argument("--out", default="render/backends.png")
    args = ap.parse_args(argv)

    L = autocrop(Image.open(args.left).convert("RGBA"))
    R = autocrop(Image.open(args.right).convert("RGBA"))
    h = min(L.height, R.height)
    L = L.resize((int(L.width * h / L.height), h))
    R = R.resize((int(R.width * h / R.height), h))

    gap = 16
    head = int(h * 0.15)
    foot = int(h * 0.26)
    W = L.width + gap + R.width
    canvas = Image.new("RGBA", (W, head + h + foot), (10, 11, 14, 255))
    canvas.alpha_composite(L, (0, head))
    canvas.alpha_composite(R, (L.width + gap, head))

    ml = json.loads((Path(args.bake_left) / "manifest.json").read_text())
    mr = json.loads((Path(args.bake_right) / "manifest.json").read_text())
    cl = ml.get("orbit_classification", {})
    cr = mr.get("orbit_classification", {})
    pr = mr.get("perturbation", {})

    def counts(c):
        return (f"regular {c.get('n_regular','?')} / island {c.get('n_island','?')}"
                f" / chaotic {c.get('n_chaotic','?')}")

    canvas.alpha_composite(panel_text(L.width, head, [
        ("ANALYTIC backend  —  perturbation is a flux function", "w"),
        ("B = curl(psi_total grad phi),  dB_phi = 0 identically", "#9fb4d8"),
        (counts(cl), "#ffcf6b"),
    ]), (0, 0))

    canvas.alpha_composite(panel_text(R.width, head, [
        ("COIL backend  —  Biot-Savart, held as a vector potential", "w"),
        (f"I = {pr.get('current_A', 0):.0f} A,  div|B| rel = "
         f"{pr.get('divergence_relative_rms', float('nan')):.1e},  dB_phi ≠ 0", "#9fb4d8"),
        (counts(cr), "#ffcf6b"),
    ]), (L.width + gap, 0))

    canvas.alpha_composite(legend_strip(W, int(foot * 0.30)), (0, head + h))

    prov = ml.get("provenance", {})
    lines = [(f"{prov.get('machine','?')} shot "
              f"{str(prov.get('shot_file','')).split('_')[-1].split('.')[0]}   "
              f"t = {prov.get('time_ms',0):.0f} ms   q95 = {prov.get('efit_q95',0):.3f}"
              f"   —   same equilibrium, same seeds, same classifier", "w")]
    canvas.alpha_composite(panel_text(W, int(foot * 0.22), lines, fontsize=16),
                           (0, head + h + int(foot * 0.28)))

    cw = wrap(CAVEAT, max(90, int(W / 13)))
    canvas.alpha_composite(panel_text(
        W, int(foot * 0.56), [(ln, "#ffa227") for ln in cw], fontsize=15),
        (0, head + h + int(foot * 0.44)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.out)
    print(f"-> {args.out}  ({canvas.width}x{canvas.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
