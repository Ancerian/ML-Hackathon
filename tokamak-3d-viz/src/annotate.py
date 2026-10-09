#!/usr/bin/env python3
"""Overlay the scientific annotation layer onto rendered stills.

Runs OUTSIDE Blender (matplotlib + Pillow).  Every number it prints is read
from the bake's ``manifest.json``, so the image and docs/PHYSICS.md cannot
disagree.

    python src/annotate.py --bake data/bake01 --render-dir render

Produces ``<name>_annotated.png`` beside each input.
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

CAVEAT = ("Vacuum approximation: a vacuum Poincare section OVERESTIMATES "
          "stochasticity, because the plasma screens resonant components. "
          "Islands are a prescribed perturbation, not a measurement.")


def _panel(fig_w_px, fig_h_px, dpi=100):
    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    return fig


def _to_rgba(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(buf, mode="RGBA")


def q_profile_panel(bake, man, w=760, h=520, dpi=100):
    """q(psi_n) with the resonant surfaces and island widths marked."""
    z = np.load(Path(bake) / "q_profile.npz")
    x, q = z["psi_n"], z["q"]
    modes = man["perturbation"]["modes"]

    fig = _panel(w, h, dpi)
    ax = fig.add_axes([0.17, 0.22, 0.79, 0.72])
    ax.plot(x, q, color="#5cc8ff", lw=2.0)
    for md in modes:
        s, W, qr = md["psi_n_res"], md["island_width_psin"], md["q_res"]
        ax.axvspan(s - W / 2, s + W / 2, color="#ffb02e", alpha=0.22, lw=0)
        ax.axhline(qr, color="#ffb02e", ls="--", lw=0.9, alpha=0.8)
        ax.annotate(f"q={md['m']}/{md['n']}", (0.02, qr), xycoords=("axes fraction", "data"),
                    color="#ffb02e", fontsize=12, va="bottom")
    ax.set_xlabel(r"$\psi_N$", color="w", fontsize=13)
    ax.set_ylabel(r"$q$", color="w", fontsize=13)
    ax.tick_params(colors="w", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#888")
    ax.set_facecolor((0, 0, 0, 0.55))
    ax.grid(alpha=0.18, color="w")
    return _to_rgba(fig)


def colorbar_panel(w=520, h=150, dpi=100, label=r"$\psi_N$   core $\rightarrow$ edge"):
    fig = _panel(w, h, dpi)
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.26])
    grad = np.linspace(0, 1, 256)[None, :]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "tok", ["#ffd973", "#f25a40", "#2659f2"])
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors="w", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#888")
    ax.set_title(label, color="w", fontsize=12, pad=8)
    return _to_rgba(fig)


def caption_panel(man, w=1900, h=250, dpi=100, fontsize=15):
    p, e, pert, qa = (man["provenance"], man["equilibrium"],
                      man["perturbation"], man["qa"])
    modes = ", ".join(f"{m['m']}/{m['n']}" for m in pert["modes"])
    shot = p["shot_file"].split("_")[-1].split(".")[0]
    lines = [
        f"{p['machine']} shot {shot}    t = {p['time_ms']:.0f} ms    "
        f"q95 = {p['efit_q95']:.3f}    "
        r"$\beta_N$ = " + f"{p['efit_beta_n']:.2f}    li = {p['efit_li']:.2f}",
        f"R_axis = {e['r_axis']:.4f} m    "
        r"B$_\phi$(axis) = " + f"{e['B_phi_on_axis']:.3f} T (calibrated to q95)    "
        f"geometry x{man['geometry']['scale']:.1f} uniform",
        f"perturbation m/n = {modes}    "
        f"amp = {pert['modes'][0]['amp_fraction']:.0e}    "
        f"Chirikov S = {pert['chirikov_S']:.3f}",
    ]
    fig = _panel(w, h, dpi)
    # scale type with the panel so 4K and preview renders both read correctly
    fs = max(8.0, fontsize * (w / 1900.0))
    for i, ln in enumerate(lines):
        fig.text(0.012, 0.88 - i * 0.20, ln, color="w", fontsize=fs,
                 family="DejaVu Sans", va="top")
    wrapped = _wrap(CAVEAT, int(max(60, w / (fs * 0.52))))
    for j, ln in enumerate(wrapped):
        fig.text(0.012, 0.30 - j * 0.145, ln, color="#ffb02e",
                 fontsize=fs * 0.82, family="DejaVu Sans", va="top")
    return _to_rgba(fig)


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width=width)


def annotate(img_path, bake, man, out_path, with_q=True, with_bar=True):
    base = Image.open(img_path).convert("RGBA")
    W, H = base.size

    cap = caption_panel(man, w=max(int(W * 0.98), 400), h=max(int(H * 0.16), 120))
    base.alpha_composite(cap, (int(W * 0.01), int(H - cap.size[1] - H * 0.015)))

    if with_q:
        qp = q_profile_panel(bake, man, w=max(int(W * 0.26), 300),
                             h=max(int(H * 0.32), 220))
        base.alpha_composite(qp, (int(W - qp.size[0] - W * 0.02), int(H * 0.03)))
    if with_bar:
        cb = colorbar_panel(w=max(int(W * 0.20), 260), h=max(int(H * 0.10), 110))
        base.alpha_composite(cb, (int(W * 0.02), int(H * 0.03)))

    base.convert("RGB").save(out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--pattern", default="*.png")
    args = ap.parse_args(argv)

    man = json.loads((Path(args.bake) / "manifest.json").read_text())
    rd = Path(args.render_dir)
    todo = [p for p in sorted(rd.glob(args.pattern))
            if not p.stem.endswith("_annotated")]
    if not todo:
        print(f"no images matching {args.pattern} in {rd}")
        return 1
    for p in todo:
        # the q-profile inset belongs on the physics frames, not the exterior
        with_q = not p.stem.startswith("01_")
        out = p.with_name(f"{p.stem}_annotated.png")
        annotate(p, args.bake, man, out, with_q=with_q)
        print(f"  {p.name} -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
