#!/usr/bin/env python3
"""Quick matplotlib preview of a machine bake (our own plots, from the meshes).

    python src/preview_machine.py --bake data/bake_jet1975

Writes into the bake directory:
  preview_rz.png    R-Z section: every part cut by the half-plane through the
                    centre of its first instance (so a TF coil, a rigid sector,
                    an iron limb ... each appear in their own mid-plane);
                    NBI injectors (not in any such plane) as a light projection
  preview_plan.png  plan: all instances cut by the plane Z = 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from tokviz.machine.build import load_component    # noqa: E402
from tokviz.machine.geom import slice_segments as slice_plane  # noqa: E402

COLORS = {
    "plasma": "#d9468f", "vessel": "#4a6fa5", "bellows": "#7fa7d9", "ports": "#2e8b8b",
    "tf_coils": "#c47a2c", "pf_coils": "#b8860b", "iron_core": "#555555",
    "structure": "#6b8e23", "limiters": "#8a2be2", "nbi": "#b22222", "human": "#000000",
}
#: components added by delta variants: coloured (and listed in the legend) by type
TYPE_COLORS = {
    "limiter_module": "#8a2be2", "vessel_restraint_ring": "#1f3f75", "pump_chamber": "#b22222",
    "nbi_adaptor": "#e8710a", "rotary_valve": "#e8710a", "gas_inlet": "#009e73",
    "port": "#2e8b8b",
}


def tris_of(q, t):
    parts = [t] if len(t) else []
    if len(q):
        parts += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    return np.vstack(parts) if parts else np.zeros((0, 3), int)


def is_z_rotation(M):
    return abs(M[2, 2] - 1) < 1e-6 and abs(M[0, 2]) < 1e-6 and abs(M[1, 2]) < 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default="data/bake_jet1975")
    args = ap.parse_args(argv)
    bake = Path(args.bake)
    man = json.loads((bake / "manifest.json").read_text())
    comps = man["components"]
    name = man["machine"]["name"]

    fig_rz, ax = plt.subplots(figsize=(11, 11))
    fig_pl, bx = plt.subplots(figsize=(12, 12))
    handles = {}
    for cname, c in comps.items():
        key = cname if cname in COLORS else c.get("type", cname)
        col = COLORS.get(cname) or TYPE_COLORS.get(c.get("type"), "#888888")
        hidden = bool(c.get("hide_in_strict_variant"))      # e.g. 1983 rotary valves
        ls = (0, (3, 2)) if hidden else "solid"
        for part in load_component(bake / c["file"]):
            V, T, X = part["v"].astype(float), tris_of(part["q"], part["t"]), part["X"].astype(float)
            # ---- R-Z section ----------------------------------------------------------
            M0 = X[0]
            if is_z_rotation(M0):
                phi = np.arctan2(M0[1, 0], M0[0, 0])
                if np.hypot(M0[0, 3], M0[1, 3]) > 1e-6:          # placed off-axis (figure)
                    phi = np.arctan2(M0[1, 3], M0[0, 3])
                segs_rz = []
                for M in X:
                    if not is_z_rotation(M):
                        continue
                    ang = (np.arctan2(M[1, 3], M[0, 3]) if np.hypot(M[0, 3], M[1, 3]) > 1e-6
                           else np.arctan2(M[1, 0], M[0, 0]))
                    if abs(np.angle(np.exp(1j * (ang - phi)))) > 1e-6:
                        continue
                    W = V @ M[:3, :3].T + M[:3, 3]
                    c_, s_ = np.cos(-phi - 1e-5), np.sin(-phi - 1e-5)   # avoid on-plane vertices
                    L = np.c_[c_ * W[:, 0] - s_ * W[:, 1], s_ * W[:, 0] + c_ * W[:, 1], W[:, 2]]
                    sg = slice_plane(L, T, L[:, 1])
                    sg = sg[(sg[:, :, 0] > -1e-9).all(1)]
                    segs_rz.append(sg[:, :, [0, 2]])
                if segs_rz:
                    sg = np.vstack(segs_rz)
                    ax.add_collection(LineCollection(sg, colors=col, linewidths=0.8,
                                                     linestyles=ls))
                    # mirror (-R) side for a full-section look
                    ax.add_collection(LineCollection(sg * [-1, 1], colors=col,
                                                     linewidths=0.4, alpha=0.35))
            else:
                for M in X:
                    W = V @ M[:3, :3].T + M[:3, 3]
                    ax.plot(np.hypot(W[:, 0], W[:, 1]), W[:, 2], ".", ms=1.2, color=col,
                            alpha=0.35)
            # ---- plan at Z = 0 ----------------------------------------------------
            segs = []
            for M in X:
                W = V @ M[:3, :3].T + M[:3, 3]
                if W[:, 2].min() > 0 or W[:, 2].max() < 0:
                    continue
                sg = slice_plane(W, T, W[:, 2] - 1e-6)
                segs.append(sg[:, :, :2])
            if segs:
                bx.add_collection(LineCollection(np.vstack(segs), colors=col, linewidths=0.6,
                                                 linestyles=ls))
        lab = key + (f" ({c.get('status')}, dashed)" if hidden else "")
        handles[lab] = plt.Line2D([], [], color=col, lw=2, label=lab,
                                  linestyle="--" if hidden else "-")

    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 9.8)
    ax.set_ylim(-6.3, 6.6)
    ax.axhline(0, color="#bbbbbb", lw=0.5)
    ax.axvline(0, color="#bbbbbb", lw=0.5)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    pl = comps.get("plasma", {}).get("info", {})
    ttl = f"{name}: R-Z section of the baked meshes (each part in its own mid-plane)"
    if pl:
        ttl += f"\nLCFS: R0={pl['R0']}, a={pl['a']}, b={pl['b']}, delta={pl['delta']:.3f}"
    ax.set_title(ttl, fontsize=10)
    ax.legend(handles=list(handles.values()), loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig_rz.tight_layout()
    fig_rz.savefig(bake / "preview_rz.png", dpi=150)

    bx.set_aspect("equal")
    lim = 10.5
    bx.set_xlim(-lim, lim)
    bx.set_ylim(-lim, lim)
    bx.set_xlabel("X [m]")
    bx.set_ylabel("Y [m]")
    bx.set_title(f"{name}: plan, all instances cut at Z = 0", fontsize=10)
    bx.legend(handles=list(handles.values()), loc="upper right", fontsize=8)
    bx.grid(alpha=0.2)
    fig_pl.tight_layout()
    fig_pl.savefig(bake / "preview_plan.png", dpi=150)
    print(bake / "preview_rz.png")
    print(bake / "preview_plan.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
