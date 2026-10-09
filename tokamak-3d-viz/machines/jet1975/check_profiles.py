#!/usr/bin/env python3
"""Consistency checks for the JET-1975 dimension database and R-Z profiles.

Loads dimensions.yaml + profiles/*.csv, checks closure / tangency / extents against the
sourced values, prints a mismatch table and writes a diagnostic overlay PNG to _check/.

Dependencies: numpy, matplotlib (PyYAML used if available, else a minimal reader for the
single-line flow-mapping format used in dimensions.yaml).

Usage:  python check_profiles.py [--no-plot]
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(HERE, "profiles")
OUTDIR = os.path.join(HERE, "_check")


# ----------------------------------------------------------------------------- loading
def _mini_yaml(path):
    """Minimal reader: top-level 'key:' sections containing '- {k: v, ...}' items, plus a
    flat 'meta:' mapping. Enough for dimensions.yaml; not a general YAML parser."""
    data, sec = {}, None
    for raw in open(path, encoding="utf-8"):
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("- {") else raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            sec = line[:-1]
            data[sec] = None
            continue
        s = line.strip()
        if s.startswith("- {") and s.endswith("}"):
            data[sec] = data[sec] or []
            data[sec].append(_flow_map(s[2:]))
        elif ":" in s and sec:
            k, v = s.split(":", 1)
            data[sec] = data[sec] or {}
            data[sec][k.strip()] = v.strip().strip('"')
    return data


def _flow_map(s):
    body, out, depth, cur, q = s.strip()[1:-1], [], 0, "", False
    for ch in body:
        if ch == '"':
            q = not q
        if not q and ch in "[{":
            depth += 1
        if not q and ch in "]}":
            depth -= 1
        if ch == "," and depth == 0 and not q:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    d = {}
    for item in out:
        k, v = item.split(":", 1)
        v = v.strip()
        try:
            d[k.strip()] = ast.literal_eval(v)
        except Exception:
            d[k.strip()] = {"true": True, "false": False}.get(v, v.strip('"'))
    return d


def load_yaml(path):
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return _mini_yaml(path)


def load_csv(name):
    rows, cols = [], None
    for line in open(os.path.join(PROF, name), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(",")
        if cols is None:
            cols = parts
            continue
        rows.append(parts)
    out = {}
    for i, c in enumerate(cols):
        vals = [r[i] for r in rows]
        try:
            out[c] = np.array([float(v) for v in vals])
        except ValueError:
            out[c] = np.array(vals)
    return out


D = load_yaml(os.path.join(HERE, "dimensions.yaml"))
IDX = {}
for sec, items in D.items():
    if isinstance(items, list):
        for it in items:
            IDX[(sec, it["id"])] = it


def V(sec, key):
    return IDX[(sec, key)]["value"]


# ----------------------------------------------------------------------------- geometry helpers
def seg_dist(p, poly):
    """min distance from points p (N,2) to polyline poly (M,2)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-30)
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((p[:, None, :] - proj) ** 2).sum(2)).min(1)


def inside(p, poly):
    x, y = p[:, 0], p[:, 1]
    xs, ys = poly[:, 0], poly[:, 1]
    res = np.zeros(len(p), bool)
    for i in range(len(poly) - 1):
        x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1)
        res ^= cond
    return res


def max_turn(poly):
    """largest direction change (deg) between successive segments; big values = kinks."""
    d = np.diff(poly, axis=0)
    d = d[np.hypot(d[:, 0], d[:, 1]) > 1e-9]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dt = (np.diff(np.r_[ang, ang[0]]) + 180) % 360 - 180
    return np.abs(dt).max()


def densify(poly, n=4000):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])


def rect(Rc, Zc, w, h):
    return np.array([[Rc - w / 2, Zc - h / 2], [Rc + w / 2, Zc - h / 2], [Rc + w / 2, Zc + h / 2],
                     [Rc - w / 2, Zc + h / 2], [Rc - w / 2, Zc - h / 2]])


# ----------------------------------------------------------------------------- data
vin = load_csv("vessel_inner.csv")
vout = load_csv("vessel_outer.csv")
tfi = load_csv("tf_coil_inner.csv")
tfo = load_csv("tf_coil_outer.csv")
pb = load_csv("plasma_boundary.csv")
iron = load_csv("iron_core_outline.csv")
ring = load_csv("ring_structure.csv")
lim = load_csv("limiter_outer.csv")

VI = np.column_stack([vin["R_m"], vin["Z_m"]])
VO = np.column_stack([vout["R_m"], vout["Z_m"]])
TI = np.column_stack([tfi["R_m"], tfi["Z_m"]])
TO = np.column_stack([tfo["R_m"], tfo["Z_m"]])
PBq = np.column_stack([pb["R_m"], pb["Z_m"]])
PB = np.vstack([PBq[::-1] * [1, -1], PBq])  # mirrored: outboard half of the D (Z<0 and Z>0)

# PF coil boxes from the database
R1 = V("pf_coils", "coil1_mean_diameter") / 2
coil_boxes = {}
w1 = V("pf_coils", "coil1_winding_radial")
R1o = V("pf_coils", "coil1_outer_diameter") / 2
pitch = V("pf_coils", "coil1_pitch_height")
for k in range(10):
    zc = (k - 4.5) * pitch
    coil_boxes[f"PF1.{k+1}"] = rect(R1o - w1 / 2, zc, w1, pitch - 0.02)
w2, h2 = V("pf_coils", "coil2_section_assumed")
for s in (1, -1):
    coil_boxes[f"PF2{'+' if s > 0 else '-'}"] = rect(V("pf_coils", "coil2_mean_diameter") / 2, s * V("pf_coils", "coil2_Z"), w2, h2)
    coil_boxes[f"PF3{'+' if s > 0 else '-'}"] = rect(V("pf_coils", "coil3_R"), s * V("pf_coils", "coil3_Z"),
                                                    V("pf_coils", "coil3_width"), V("pf_coils", "coil3_height"))
    for j, dz in enumerate((-0.325, 0.325)):
        coil_boxes[f"PF4{'+' if s > 0 else '-'}{j}"] = rect(V("pf_coils", "coil4_R"), s * (V("pf_coils", "coil4_Z") + dz),
                                                           V("pf_coils", "coil4_width"), V("pf_coils", "coil4_block_height"))

rows = []


def check(name, expected, measured, tol, src="", note=""):
    diff = measured - expected
    ok = abs(diff) <= tol
    rows.append((name, expected, measured, diff, tol, "OK" if ok else "MISMATCH", src, note))


def info(name, measured, note=""):
    rows.append((name, np.nan, measured, np.nan, np.nan, "INFO", "", note))


# ----------------------------------------------------------------------------- vessel
for nm, P in (("vessel_inner", VI), ("vessel_outer", VO)):
    check(f"{nm}: closed (|first-last|)", 0.0, float(np.hypot(*(P[0] - P[-1]))), 1e-6, "construction")
    check(f"{nm}: max turn between segments [deg]", 0.0, float(max_turn(P)), 2.0, "tangency", "arc steps <=1 deg -> <=2 deg means G1")
check("vessel_inner R_min", V("vessel", "rs_R_inner_wall_inboard"), VI[:, 0].min(), 1e-4, "p298")
check("vessel_inner R_max", V("vessel", "rs_R_inner_wall_outboard"), VI[:, 0].max(), 1e-4, "p298")
check("vessel_inner Z_max", V("vessel", "rs_Z_inner_wall_top"), VI[:, 1].max(), 1e-4, "p298")
check("vessel_outer R_min", V("vessel", "rs_R_outer_wall_inboard"), VO[:, 0].min(), 1e-4, "p298")
check("vessel_outer R_max", V("vessel", "rs_R_outer_wall_outboard"), VO[:, 0].max(), 1e-4, "p298")
check("vessel_outer Z_max", V("vessel", "rs_half_height_outer"), VO[:, 1].max(), 1e-4, "p298")
check("vessel_outer width vs 2965", V("vessel", "rs_overall_width"), np.ptp(VO[:, 0]), 1e-4, "p298")
check("vessel inner width vs Table IV.1-2 'horizontal 2.63'", V("vessel", "horizontal_diameter_table"), np.ptp(VI[:, 0]), 0.02,
      "p310", "2.63 = 4.33-1.70 (bellows shields), not the wall")
check("vessel inner height vs Table IV.1-2 'vertical 4.18'", V("vessel", "vertical_diameter_table"), np.ptp(VI[:, 1]), 0.02,
      "p310", "wall gives 4.27; bore inside shields?")
check("vessel inner R_min vs Table I.3-1 Ri 1.66", V("vessel", "rs_vessel_Ri_table"), VI[:, 0].min(), 0.01, "p83")
check("vessel inner R_max vs Table I.3-1 Re 4.37", V("vessel", "rs_vessel_Re_table"), VI[:, 0].max(), 0.01, "p83")
wall = seg_dist(densify(VO, 3000), VI)
check("vessel wall thickness min (Ri, Ra)", V("vessel", "wall_thickness_Ri_Ra"), wall.min(), 0.002, "p310")
check("vessel wall thickness max", V("vessel", "wall_thickness_max"), wall.max(), 0.002, "p310")

# ----------------------------------------------------------------------------- TF coil
for nm, P in (("tf_coil_inner", TI), ("tf_coil_outer", TO)):
    check(f"{nm}: closed", 0.0, float(np.hypot(*(P[0] - P[-1]))), 1e-6, "fit")
    check(f"{nm}: max turn [deg]", 0.0, float(max_turn(P)), 2.0, "fit", "G1 arc chain")
check("TF overall width (casing) vs 3857", V("tf_coils", "overall_width_fig"), np.ptp(TO[:, 0]), 0.02, "p325")
check("TF overall height (casing) vs 5680", V("tf_coils", "overall_height_fig"), np.ptp(TO[:, 1]), 0.02, "p325/327",
      "5680 is over the support pads")
check("TF height incl. support pads vs 5680", V("tf_coils", "overall_height_fig"), 2 * V("tf_coils", "support_pad_Z_outer"), 0.03, "p325")
check("TF bore R_in vs Rci 1.49", V("tf_coils", "bore_R_inner"), TI[:, 0].min(), 0.005, "p83", "anchor")
check("TF bore R_out vs Rce 4.60", V("tf_coils", "bore_R_outer"), TI[:, 0].max(), 0.015, "p83")
check("TF inner leg thickness vs 373.5", V("tf_coils", "inner_leg_radial_thickness"), TI[:, 0].min() - TO[:, 0].min(), 0.015, "p325")
check("TF outer leg thickness vs 371", V("tf_coils", "outer_leg_radial_thickness"), TO[:, 0].max() - TI[:, 0].max(), 0.015, "p325")
straight = TI[(np.abs(TI[:, 0] - TI[:, 0].min()) < 1e-6)]
check("TF straight inner-leg length vs 2872", V("tf_coils", "straight_inner_leg"), np.ptp(straight[:, 1]), 0.20, "p325",
      "fit: straight bore longer than dimension")

# ----------------------------------------------------------------------------- plasma
check("plasma table R(Z->0) vs R0+a = 4.21", V("plasma", "R0") + V("plasma", "a"),
      float(np.polyval(np.polyfit(PBq[:5, 1] ** 2, PBq[:5, 0], 1), 0.0)), 0.015, "p332/p83")
check("plasma table Z_max vs b = 2.10", V("plasma", "b"), PBq[:, 1].max(), 0.02, "p332/p83", "table reaches 2.195")
info("plasma table R at Z_max [m]", PBq[np.argmax(PBq[:, 1]), 0], "R of top point (D triangularity: 2.31 < R0)")
check("plasma table points count (task said 31)", 31, len(PBq), 0, "p332", "table has 30 rows")
gap_pv = seg_dist(PB, VI)
check("plasma (tabulated) inside vessel inner wall", 1.0, float(np.all(inside(PB, VI))), 0, "p332/p298")
info("min gap plasma boundary -> vessel inner wall [m]", gap_pv.min())
check("Re(plasma)->Re(rigid vessel) gap vs '16 cm' (Table I.3-1)", 0.16, VI[:, 0].max() - 4.21, 0.02, "p83")
check("Ri(plasma)->Ri(rigid vessel) gap vs '5 cm'", 0.05, 1.71 - VI[:, 0].min(), 0.01, "p83")

# ----------------------------------------------------------------------------- clearances
check("vessel outer inside TF bore", 1.0, float(np.all(inside(densify(VO, 2000), TI))), 0, "p298/p325")
gap_vt = seg_dist(densify(VO, 2000), TI)
info("min gap vessel outer -> TF bore [m]", gap_vt.min(), "needs >= 0.025 Kaowool + expansion 0.030")
gap_top = TI[:, 1].max() - VO[:, 1].max()
info("vertical gap vessel top -> TF bore top [m]", gap_top)
tf_in_face = TO[:, 0].min()
check("PF1 outer radius < TF inner-leg outer face (inner cylinder 4 cm)", 0.04, tf_in_face - R1o, 0.02, "p402/p374")
for nm, box in coil_boxes.items():
    if nm.startswith("PF1"):
        continue
    hit = inside(densify(box, 200), TO).any()
    check(f"{nm} box outside TF casing", 0.0, float(hit), 0, "p401-406/p325")
check("PF1 mean radius from drawing (d2170-327) vs table 1.816/2", R1, R1o - w1 / 2, 0.02, "p401/p402",
      "winding centre 0.9215 vs 0.908")
check("PF1 stack (10 x 497) fits below centre pieces (|Z|<2.98)", 1.0, float(10 * pitch / 2 < 2.98), 0, "p402/p80",
      "stack half-height 2.485 m")

# ----------------------------------------------------------------------------- iron
parts = {}
for p, r, z in zip(iron["part"], iron["R_m"], iron["Z_m"]):
    parts.setdefault(p, []).append((r, z))
parts = {k: np.array(v) for k, v in parts.items()}
limb = parts["outer_limb"]
check("iron limb outer R vs overall diameter/2 = 7.40", V("overall", "overall_diameter") / 2, limb[:, 0].max(), 0.05, "p4/p80")
Ztop = parts["upper_radial_arm"][:, 1].max()
Zbot = parts["foot_plate"][:, 1].min()
check("iron top-to-foot height vs 11.5", V("overall", "overall_height"), Ztop - Zbot, 0.05, "p4/p80")
ld = limb[:, 0].max() - limb[:, 0].min()
check("limb section: depth x width vs 1.40 m2 (Table IV.4-5 / 8)", V("iron_core", "area_vertical_arms_basic") / 8,
      ld * V("iron_core", "limb_toroidal_width"), 0.05, "p412", "width derived -> tautological, shows derivation")
cp = parts["centre_piece_upper"]
check("centre core area pi*Rcp^2 vs 9.9 m2", V("iron_core", "area_central_core"), np.pi * cp[:, 0].max() ** 2, 1.0, "p412/p80",
      "non-saturated region at top/bottom")
masses = [8 * 71, 8 * 71, 8 * 88, 200, 2 * 90, 43]
check("iron masses sum vs total 2263 t", 2263, sum(masses), 0, "p415")
check("coil 4 outer R < limb inner R", 1.0, float(V("pf_coils", "coil4_R") + V("pf_coils", "coil4_width") / 2 < limb[:, 0].min()), 0, "p406/p80")
check("coil 3 top < upper arm bottom", 1.0, float(V("pf_coils", "coil3_Z") + V("pf_coils", "coil3_height") / 2
                                                   < parts["upper_radial_arm"][:, 1].min()), 0, "p401/p80")
check("coil 2 top < upper arm bottom", 1.0, float(V("pf_coils", "coil2_Z") + h2 / 2 < parts["upper_radial_arm"][:, 1].min()), 0,
      "p401/p80")
check("coil 2 inner edge > centre-piece R", 1.0, float(V("pf_coils", "coil2_mean_diameter") / 2 - w2 / 2 > cp[:, 0].max()), 0,
      "p401/p80", "coil 2 surrounds the large-diameter part of the core")

# ----------------------------------------------------------------------------- report
def fmt(x):
    return "" if (isinstance(x, float) and np.isnan(x)) else f"{x:9.4f}"


print(f"{'check':66s} {'expected':>9s} {'measured':>9s} {'diff':>9s} {'tol':>8s}  status    source     note")
print("-" * 150)
for r in rows:
    print(f"{r[0][:66]:66s} {fmt(r[1]):>9s} {fmt(r[2]):>9s} {fmt(r[3]):>9s} {fmt(r[4]):>8s}  {r[5]:8s}  {r[6]:9s}  {r[7]}")
nbad = sum(r[5] == "MISMATCH" for r in rows)
print("-" * 150)
print(f"{len(rows)} checks, {nbad} mismatches (mismatches against source are expected where the source itself is inconsistent; see README)")

# ----------------------------------------------------------------------------- plot
if "--no-plot" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = {"central_pillar": "#8c8c8c", "pillar_top_hub": "#8c8c8c", "upper_radial_arm": "#b07d4f",
              "lower_radial_arm": "#b07d4f", "outer_limb": "#b07d4f", "foot_plate": "#8c6d4f",
              "centre_piece_upper": "#c9a36b", "centre_piece_lower": "#c9a36b"}
    for k, P in parts.items():
        if k.startswith("model"):
            Pm = np.vstack([P, P[::-1] * [1, -1]])
            ax.plot(Pm[:, 0], Pm[:, 1], ":", color="#6a3d9a", lw=1, label="iron: axisym. model p.394" if k.endswith("inner_edge") else None)
        else:
            ax.fill(P[:, 0], P[:, 1], color=colors.get(k, "#aaa"), alpha=0.35, lw=0.8, ec="#5a4630")
    ax.fill([], [], color="#b07d4f", alpha=0.35, label="iron core (p.80, digitized)")
    for p in ("upper_ring", "lower_ring"):
        m = ring["part"] == p
        ax.fill(ring["R_m"][m], ring["Z_m"][m], color="#1f9e89", alpha=0.35, label="ring (p.378, Z assumed)" if p == "upper_ring" else None)
    ax.fill(TO[:, 0], TO[:, 1], color="#d95f02", alpha=0.25, label="TF coil casing (p.325 fit)")
    ax.fill(TI[:, 0], TI[:, 1], color="white")
    ax.plot(TO[:, 0], TO[:, 1], color="#d95f02", lw=1.2)
    ax.plot(TI[:, 0], TI[:, 1], color="#d95f02", lw=1.2, ls="--")
    ax.fill(VO[:, 0], VO[:, 1], color="#377eb8", alpha=0.45, label="vacuum vessel rigid sector (p.298)")
    ax.fill(VI[:, 0], VI[:, 1], color="white")
    ax.plot(VI[:, 0], VI[:, 1], color="#377eb8", lw=0.8)
    ax.plot(PB[:, 0], PB[:, 1], "o-", ms=2.5, color="#e41a1c", lw=1, label="plasma boundary Table IV.2-4 (mirrored)")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2.96 + 1.25 * np.cos(th), 2.10 * np.sin(th), color="#e41a1c", lw=0.6, ls=":", label="ellipse R0=2.96,a=1.25,b=2.10")
    ax.fill(lim["R_m"], lim["Z_m"], color="k", label="outer rail limiter (assumed)")
    for nm, box in coil_boxes.items():
        ax.fill(box[:, 0], box[:, 1], color="#4daf4a", alpha=0.7, ec="k", lw=0.5)
    ax.fill([], [], color="#4daf4a", label="PF coils 1-4 (Table IV.4-3, Figs IV.4-13/14)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(2.96, color="k", lw=0.4, ls=":")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 8.4)
    ax.set_ylim(-6.3, 6.5)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("JET 1975 design (EUR 5516e) - R-Z diagnostic overlay of sourced profiles\n(our reconstruction; not a copy of any figure)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25)
    fn = os.path.join(OUTDIR, "jet1975_overlay.png")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    print("overlay written:", fn)
