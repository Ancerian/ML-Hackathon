"""Vacuum vessel: 32 rigid double-wall sectors (8 octants) + double bellows.

Both skins come from ``profiles/vessel_inner.csv`` / ``vessel_outer.csv``
(Fig. IV.1-3 arc chain).  The two profiles are resampled piecewise between
common breakpoints (port edges), so inner and outer skins, end plates, port
jambs and the bellows all share one row structure.

Port openings are cut out of the skins (grid cells removed, a jamb joins the
two skins around the hole); the port stubs themselves are the ``ports``
component.  Sector variants: plain, hport (middle sector of the octant),
vport_large, vport_small (see ASSUMPTIONS_KERNEL.md).
"""
from __future__ import annotations

import numpy as np

from ..geom import Mesh, crossings, orient, resample_pieces, rot_z
from .base import Layout, Part, finish


# ---------------------------------------------------------------------------
# shared geometry
# ---------------------------------------------------------------------------
def port_geometry(tr) -> dict:
    """Opening sizes/positions (m) used by the vessel, the ports and the ring."""
    hw = tr.v("ports.horizontal_port_width")
    hh = tr.v("ports.horizontal_port_height")
    vl = tr.v("ports.vertical_port_large")        # [toroidal, radial]
    vlr = tr.v("ports.vertical_port_large_R_range")
    vs = tr.v("ports.vertical_port_small")
    vsr = tr.v("ports.vertical_port_small_R_range")
    # size from the text (higher confidence), centre from the digitized R-range
    cl, cs = 0.5 * (vlr[0] + vlr[1]), 0.5 * (vsr[0] + vsr[1])
    return {
        "h_half_w": hw / 2, "h_half_h": hh / 2,
        "vl_half_w": vl[0] / 2, "vl_R": (cl - vl[1] / 2, cl + vl[1] / 2),
        "vs_half_w": vs[0] / 2, "vs_R": (cs - vs[1] / 2, cs + vs[1] / 2),
    }


def _labelled_breaks(p: np.ndarray, pg: dict) -> dict:
    out = {}
    zc = pg["h_half_h"]
    up = crossings(p, 1, +zc, where=lambda q: q[0] > 3.0)
    dn = crossings(p, 1, -zc, where=lambda q: q[0] > 3.0)
    assert len(up) == 1 and len(dn) == 1, "outboard port edge crossings"
    out["h_up"], out["h_dn"] = up[0], dn[0]
    for tag, (r0, r1) in (("vl", pg["vl_R"]), ("vs", pg["vs_R"])):
        for rr, nm in ((r0, "0"), (r1, "1")):
            top = crossings(p, 0, rr, where=lambda q: q[1] > 1.0)
            bot = crossings(p, 0, rr, where=lambda q: q[1] < -1.0)
            assert len(top) == 1 and len(bot) == 1, f"vertical port edge R={rr}"
            out[f"{tag}{nm}_top"], out[f"{tag}{nm}_bot"] = top[0], bot[0]
    return out


def vessel_rows(tr, n_pol: int) -> dict:
    """Inner/outer profiles resampled in correspondence + break row indices."""
    inner = np.asarray(tr.profile("vessel_inner"), float)
    outer = np.asarray(tr.profile("vessel_outer"), float)
    if np.allclose(inner[0], inner[-1]):
        inner = inner[:-1]
    if np.allclose(outer[0], outer[-1]):
        outer = outer[:-1]
    pg = port_geometry(tr)
    bo, bi = _labelled_breaks(outer, pg), _labelled_breaks(inner, pg)
    labels = sorted(bo, key=lambda k: bo[k])
    si = [bi[k] for k in labels]
    if np.any(np.diff(si) <= 0):
        raise ValueError("inner/outer breakpoints are not in the same order")
    (out_rs, in_rs), bidx = resample_pieces([outer, inner], n_pol,
                                            [[bo[k] for k in labels], si])
    return {"outer": out_rs, "inner": in_rs, "row": dict(zip(labels, bidx)),
            "n": len(out_rs), "ports": pg}


def _cells_between(r0: int, r1: int, n: int) -> list:
    """Cell rows from vertex row r0 up to r1 (exclusive), wrapping."""
    out, r = [], r0
    while r != r1:
        out.append(r)
        r = (r + 1) % n
    return out


def sector_mesh(rows: dict, half_angle: float, n_tor: int, holes: list,
                edge_cols: list, span=None) -> Mesh:
    """Double-skin rigid sector centred on phi = 0.

    ``holes``: list of (cell_rows, (col0, col1)) removed from both skins.
    ``edge_cols``: list of (half_width_m, rows) -- columns placed at a fixed
    perpendicular distance from the centre plane on those rows (parallel-sided
    port edges); column angles are merged into the uniform grid.
    ``span``: optional callable R -> (lo, hi) angle arrays giving the end-plate
    angles per profile row (sector ends on planes parallel to the bellows
    centre plane, 1983 reading); ``half_angle`` is then only the nominal grid
    half-width.  Columns inside a port band keep their angle on the port rows
    (identity), the rest are stretched to the row's end angles."""
    O, I, n = rows["outer"], rows["inner"], rows["n"]
    # ---- column angles ---------------------------------------------------
    base = np.linspace(-half_angle, half_angle, n_tor + 1)
    ins, bands = [], []
    for hw, rws in edge_cols:
        R_all = np.concatenate([O[rws, 0], I[rws, 0]])
        a = np.arcsin(hw / R_all)
        ins.append(float(np.mean(a)))
        bands.append((a.min(), a.max()))
    spacing = base[1] - base[0]
    keep = []
    for b in base:
        ok = True
        if abs(abs(b) - half_angle) > 1e-12:          # always keep the end plates
            for lo, hi in bands:
                if lo - 0.35 * spacing <= abs(b) <= hi + 0.35 * spacing:
                    ok = False
        if ok:
            keep.append(b)
    cols = np.array(sorted(set(np.round(keep + [s * a for a in ins for s in (-1, 1)], 12))))
    nc = len(cols)
    col_of = {}
    for k, a in enumerate(ins):
        col_of[k] = (int(np.argmin(abs(cols + a))), int(np.argmin(abs(cols - a))))

    A_o = np.broadcast_to(cols, (n, nc)).copy()
    A_i = A_o.copy()
    if span is not None:
        bf = np.zeros(n)
        for (hw, rws), (lo_b, hi_b) in zip(edge_cols, bands):
            rr = np.unique(np.concatenate([rws, np.asarray(rws) + 1]) % n)
            bf[rr] = np.maximum(bf[rr], hi_b + 0.35 * spacing)

        def remap(R):
            lo, hi = span(R)
            lo, hi = np.broadcast_to(lo, R.shape), np.broadcast_to(hi, R.shape)
            frac = (cols + half_angle) / (2 * half_angle)
            A = lo[:, None] + frac[None, :] * (hi - lo)[:, None]
            for r in np.flatnonzero(bf > 0):
                if abs(lo[r] + hi[r]) > 1e-12 or not (bf[r] < min(hi[r], half_angle)):
                    raise ValueError("sector_mesh: port band does not fit the sector span")
                ab = np.abs(cols)
                st = np.where(ab <= bf[r], ab,
                              bf[r] + (ab - bf[r]) * (hi[r] - bf[r]) / (half_angle - bf[r]))
                A[r] = np.sign(cols) * st
            return A
        A_o, A_i = remap(O[:, 0]), remap(I[:, 0])
    for k, (hw, rws) in enumerate(edge_cols):
        c0, c1 = col_of[k]
        rr = np.unique(np.concatenate([rws, np.asarray(rws) + 1]) % n)
        A_o[rr, c0], A_o[rr, c1] = -np.arcsin(hw / O[rr, 0]), np.arcsin(hw / O[rr, 0])
        A_i[rr, c0], A_i[rr, c1] = -np.arcsin(hw / I[rr, 0]), np.arcsin(hw / I[rr, 0])

    def skin(P, A):
        return np.stack([P[:, 0:1] * np.cos(A), P[:, 0:1] * np.sin(A),
                         np.broadcast_to(P[:, 1:2], A.shape)], -1).reshape(-1, 3)

    V = np.vstack([skin(O, A_o), skin(I, A_i)])
    off = n * nc
    vo = lambda r, c: (r % n) * nc + c
    vi = lambda r, c: off + (r % n) * nc + c

    hole_cells = set()
    for cell_rows, cc in holes:
        (c0, c1) = col_of[cc] if isinstance(cc, int) else cc
        for r in cell_rows:
            for c in range(c0, c1):
                hole_cells.add((r, c))
    Q = []
    for r in range(n):
        for c in range(nc - 1):
            if (r, c) in hole_cells:
                continue
            Q.append([vo(r, c), vo(r + 1, c), vo(r + 1, c + 1), vo(r, c + 1)])
            Q.append([vi(r, c), vi(r, c + 1), vi(r + 1, c + 1), vi(r + 1, c)])
    for c in (0, nc - 1):                               # end plates
        for r in range(n):
            Q.append([vo(r, c), vi(r, c), vi(r + 1, c), vo(r + 1, c)])
    for cell_rows, cc in holes:                         # jambs
        (c0, c1) = col_of[cc] if isinstance(cc, int) else cc
        vr = list(cell_rows) + [(cell_rows[-1] + 1) % n]
        loop = [(vr[0], c) for c in range(c0, c1 + 1)]
        loop += [(r, c1) for r in vr[1:]]
        loop += [(vr[-1], c) for c in range(c1 - 1, c0 - 1, -1)]
        loop += [(r, c0) for r in vr[-2:0:-1]]
        for (r_a, c_a), (r_b, c_b) in zip(loop, loop[1:] + loop[:1]):
            Q.append([vo(r_a, c_a), vo(r_b, c_b), vi(r_b, c_b), vi(r_a, c_a)])
    return orient(Mesh(V, np.asarray(Q)))


# ---------------------------------------------------------------------------
# 1983 octant layout: parallel-flange bellows of constant toroidal length
# ---------------------------------------------------------------------------
def octant_layout(tr, L) -> dict | None:
    """The variant's ``vessel_sector`` layout entry (e.g. jet1983
    vessel_octant_layout_1983), checked against the Layout, or None (1975)."""
    spec = tr.component("vessel_sector")
    if spec is None:
        return None
    p = spec.params
    if abs(float(p.get("pitch_deg", np.rad2deg(L.pitch))) - np.rad2deg(L.pitch)) > 1e-9:
        raise ValueError(f"{spec.name}: pitch {p['pitch_deg']} != 360/{L.n_tf}")
    if int(p.get("rigid_total", L.n_pieces)) != L.n_pieces or \
            int(p.get("bellows_total", L.n_tf)) != L.n_tf:
        raise ValueError(f"{spec.name}: counts do not match the database overrides")
    if int(p.get("rigid_per_octant", 0)) != L.n_pieces // L.n_oct or \
            int(p.get("bellows_per_octant", 0)) != L.n_tf // L.n_oct:
        raise ValueError(f"{spec.name}: per-octant counts do not match the layout")
    d = 0.5 * float(p["bellows_toroidal_length"])
    half = L.pitch / 2

    def end_angle(R):                       # sector end plane at |y| = d from the bellows plane
        return half - np.arcsin(d / np.asarray(R, float))

    return {"spec": spec, "d": d, "end_angle": end_angle,
            "sequence": list(p.get("sequence_in_octant", []))}


def _width_table(lay, L) -> dict:
    out = {}
    for R in (4.389, 1.664):
        e = float(np.rad2deg(lay["end_angle"](R)))
        out[f"R{R:.3f}"] = {"rigid_full_deg": 2 * e, "rigid_end_deg": e,
                            "bellows_deg": float(np.rad2deg(L.pitch)) - 2 * e}
    return out


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_vessel(tr, spec):
    L = Layout(tr)
    n_pol = int(tr.p("n_pol", 280))
    n_tor = int(tr.p("n_tor_sector", 16))
    openings = bool(tr.p("openings", True))
    rows = vessel_rows(tr, n_pol)
    tr.v("vessel.rs_R_inner_wall_inboard"), tr.v("vessel.rs_R_outer_wall_outboard")
    tr.v("vessel.wall_thickness_Ri_Ra"), tr.v("vessel.wall_thickness_max")
    tr.v("vessel.bellows_per_section"), tr.v("vessel.octant_angle")
    bel = np.deg2rad(tr.k("vessel_bellows_angle_deg"))
    half = (L.pitch - bel) / 2
    n, row, pg = rows["n"], rows["row"], rows["ports"]

    h_rows = _cells_between(row["h_dn"], row["h_up"], n)
    vl_top = _cells_between(row["vl1_top"], row["vl0_top"], n)
    vl_bot = _cells_between(row["vl0_bot"], row["vl1_bot"], n)
    vs_top = _cells_between(row["vs1_top"], row["vs0_top"], n)
    vs_bot = _cells_between(row["vs0_bot"], row["vs1_bot"], n)
    edge_cols = [(pg["h_half_w"], h_rows),
                 (pg["vl_half_w"], vl_top + vl_bot),
                 (pg["vs_half_w"], vs_top + vs_bot)]
    variants = {"plain": []}
    if openings:
        variants.update({"hport": [(h_rows, 0)],
                         "vport_large": [(vl_top, 1), (vl_bot, 1)],
                         "vport_small": [(vs_top, 2), (vs_bot, 2)]})
    material = f"inconel600 ({tr.v('vessel.material_rigid_sectors')})"
    parts = []
    lay = octant_layout(tr, L) if L.split_end else None
    span_full = None
    if lay is not None:                  # 1983: parallel-flange bellows, wedge sectors
        ea = lay["end_angle"]
        half = float(ea(4.389))          # nominal grid half-width (outboard plasma-side wall)
        span_full = lambda R: (-ea(R), ea(R))
        m_b = sector_mesh(rows, half / 2, max(2, n_tor // 2), [], [],
                          span=lambda R: (np.zeros_like(R), ea(R)))
        m_a = sector_mesh(rows, half / 2, max(2, n_tor // 2), [], [],
                          span=lambda R: (-ea(R), np.zeros_like(R)))
        idx = [i for i in range(L.n_sec) if i % L.per_oct == 0]
        # piece "a" closes octant j-1, piece "b" opens octant j (weld plane between them)
        parts.append(Part("rigid_sector_end_a", m_a, "inconel600", L.rot_sectors(idx),
                          [f"sector_{i:02d}a_oct{(L.octant_of(i) - 1) % L.n_oct}" for i in idx],
                          meta={"sector_indices": idx, "closed_solid": True,
                                "role": "octant end piece (half sector), -phi side of the weld",
                                "material_note": material}))
        parts.append(Part("rigid_sector_end_b", m_b, "inconel600", L.rot_sectors(idx),
                          [f"sector_{i:02d}b_oct{L.octant_of(i)}" for i in idx],
                          meta={"sector_indices": idx, "closed_solid": True,
                                "role": "octant end piece (half sector), +phi side of the weld",
                                "material_note": material}))
    elif L.split_end:                    # weld sectors -> two half-width end pieces
        idx = [i for i in range(L.n_sec) if i % L.per_oct == 0]
        m = sector_mesh(rows, half / 2, max(2, n_tor // 2), [], edge_cols)
        T, labs = [], []
        for i in idx:
            for sgn, nm in ((-1, "a"), (+1, "b")):
                T.append(rot_z(L.sector_phi(i) + sgn * half / 2))
                labs.append(f"sector_{i:02d}{nm}_oct{L.octant_of(i) if sgn > 0 else (L.octant_of(i) - 1) % L.n_oct}")
        parts.append(Part("rigid_sector_end_half", m, "inconel600", np.stack(T), labs,
                          meta={"sector_indices": idx, "half_angle_deg": float(np.rad2deg(half / 2)),
                                "closed_solid": True}))
    for kind, holes in variants.items():
        idx = [i for i in range(L.n_sec)
               if (L.sector_kind(i) if openings else "plain") == kind
               and not (L.split_end and i % L.per_oct == 0)]
        if not idx:
            continue
        m = sector_mesh(rows, half, n_tor, holes, edge_cols, span=span_full)
        parts.append(Part(
            name=f"rigid_sector_{kind}", mesh=m, material="inconel600",
            transforms=L.rot_sectors(idx),
            labels=[f"sector_{i:02d}_oct{L.octant_of(i)}" for i in idx],
            meta={"sector_indices": idx, "half_angle_deg": float(np.rad2deg(half)),
                  "material_note": material, "closed_solid": True}))
    info = {"sector_angle_deg": float(np.rad2deg(2 * half)),
            "rigid_pieces": int(sum(p.n_instances for p in parts)),
            "split_end_sectors": bool(L.split_end),
            "bellows_angle_deg": float(np.rad2deg(bel)),
            "n_pol": n, "openings": openings,
            "octant_boundaries_deg": [float(np.rad2deg(L.sector_phi(j * L.per_oct)))
                                      for j in range(L.n_oct)]}
    if lay is not None:
        info.update({
            "layout": lay["spec"].name, "sequence_in_octant": lay["sequence"],
            "bellows_flanges": "parallel planes |y| = d from the TF/bellows plane",
            "bellows_toroidal_length_m": 2 * lay["d"],
            "bellows_angle_deg": None, "sector_angle_deg": None,
            "widths_deg": _width_table(lay, L),
            "rigid_pieces_per_octant": L.n_pieces // L.n_oct,
            "pitch_deg": float(np.rad2deg(L.pitch))})
    return finish(tr, parts, info)


def build_bellows(tr, spec):
    L = Layout(tr)
    n_pol = int(tr.p("n_pol", 280))
    per_conv = int(tr.p("samples_per_convolution", 8))
    rows = vessel_rows(tr, n_pol)
    O, I = rows["outer"], rows["inner"]
    frac = float(tr.k("bellows_ply_depth_fraction"))
    n_ply = int(tr.v("vessel.bellows_per_section"))
    t_wall = tr.v("vessel.bellows_wall_thickness")
    tr.v("vessel.material_bellows")
    lay = octant_layout(tr, L) if L.split_end else None
    extra = {}
    if lay is not None:
        # 1983: parallel flanges |y| = d; convolutions from the 1983 text
        # (10 mm per convolution between two 4 mm flanges) over the assumed length
        b83 = tr.component("vessel_bellows", lambda c: "count_total" in c.params)
        tr.component("vessel_bellows", lambda c: "uncertainty" in c.params)
        if b83 is not None and int(b83.params["count_total"]) != L.n_tf:
            raise ValueError(f"{b83.name}: count_total != {L.n_tf}")
        bp = b83.params if b83 is not None else {}
        Lb = 2 * lay["d"]
        if "convolution_width" in bp:
            n_conv = int(round((Lb - 2 * float(bp.get("flange_thickness", 0.0)))
                               / float(bp["convolution_width"])))
        else:
            n_conv = int(tr.k("bellows_convolutions"))
        t = np.linspace(-1.0, 1.0, n_conv * per_conv + 1)
        wave = 0.5 - 0.5 * np.cos(np.pi * n_conv * (t + 1))
        bel = None
        extra = {"flanges": "parallel", "toroidal_length_m": Lb,
                 "convolution_width_m": bp.get("convolution_width"),
                 "convolution_height_m_text": bp.get("convolution_height"),
                 "ply_depth_note": "two concentric plies share the double-wall gap; each "
                                   "oscillates over kernel.bellows_ply_depth_fraction of it "
                                   "(the 120 mm convolution height does not fit twice)"}
    else:
        bel = np.deg2rad(tr.k("vessel_bellows_angle_deg"))
        n_conv = int(tr.k("bellows_convolutions"))
        phis = np.linspace(-bel / 2, bel / 2, n_conv * per_conv + 1)
        wave = 0.5 - 0.5 * np.cos(2 * np.pi * n_conv * (phis - phis[0]) / bel)
    parts = []
    plies = {"outer_ply": 1.0 - frac * wave, "inner_ply": frac * wave}
    if n_ply == 1:
        plies = {"ply": 0.5 + 0.5 * frac * (wave - 0.5)}
    for nm, u in plies.items():
        P = I[None, :, :] + u[:, None, None] * (O - I)[None, :, :]      # (nt, np, 2)
        if lay is not None:
            ph = np.arcsin(t[:, None] * lay["d"] / P[..., 0])
        else:
            ph = np.broadcast_to(phis[:, None], P.shape[:2])
        V = np.stack([P[..., 0] * np.cos(ph), P[..., 0] * np.sin(ph),
                      P[..., 1]], -1).reshape(-1, 3)
        nt, npol = P.shape[:2]
        ia = np.arange(nt - 1)[:, None]
        ib = np.arange(npol)[None, :]
        b1 = (ib + 1) % npol
        Q = np.stack([(ia * npol + ib).ravel(), (ia * npol + b1).ravel(),
                      ((ia + 1) * npol + b1).ravel(), ((ia + 1) * npol + ib).ravel()], -1)
        m = Mesh(V, Q)
        idx = list(range(L.n_tf))
        parts.append(Part(name=f"bellows_{nm}", mesh=m, material="inconel625",
                          transforms=L.rot_tf(idx),
                          labels=[f"bellows_{i:02d}_{nm}" for i in idx],
                          meta={"thin_shell": True, "thickness_m": t_wall,
                                "convolutions": n_conv}))
    return finish(tr, parts, {"bellows_angle_deg": None if bel is None else float(np.rad2deg(bel)),
                              "convolutions": n_conv, "plies": n_ply, **extra})
