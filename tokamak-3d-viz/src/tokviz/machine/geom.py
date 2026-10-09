"""Mesh primitives for the engineering model (pure numpy).

A :class:`Mesh` holds float vertices plus quad faces ``q`` (M, 4) and triangle
faces ``t`` (K, 3).  Meshes are built in a component-local frame; instances are
4x4 transforms applied by the consumer (Blender) -- never baked copies.

Conventions: metres; machine axis = Z; local frame of a toroidally repeated
part has its centre plane at phi = 0 (the +X half of the XZ plane).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..surfaces import resample_closed, revolve  # noqa: F401  (re-exported helpers)


# ---------------------------------------------------------------------------
# mesh container
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    v: np.ndarray
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.int64))
    t: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))

    def __post_init__(self):
        self.v = np.asarray(self.v, float).reshape(-1, 3)
        self.q = np.asarray(self.q, np.int64).reshape(-1, 4)
        self.t = np.asarray(self.t, np.int64).reshape(-1, 3)

    @property
    def n_faces(self) -> int:
        return len(self.q) + len(self.t)

    def tris(self) -> np.ndarray:
        """All faces as triangles (quads split along 0-2)."""
        q = self.q
        return np.vstack([self.t, q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) if len(q) else self.t

    def transformed(self, M: np.ndarray) -> "Mesh":
        return Mesh(apply(M, self.v), self.q.copy(), self.t.copy())

    def flipped(self) -> "Mesh":
        return Mesh(self.v, self.q[:, ::-1], self.t[:, ::-1])


def merge(meshes) -> Mesh:
    vs, qs, ts, off = [], [], [], 0
    for m in meshes:
        if m is None:
            continue
        vs.append(m.v)
        qs.append(m.q + off)
        ts.append(m.t + off)
        off += len(m.v)
    if not vs:
        return Mesh(np.zeros((0, 3)))
    return Mesh(np.vstack(vs), np.vstack(qs), np.vstack(ts))


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def rot_z(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    return M


def translate(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = [x, y, z]
    return M


def frame(origin, xdir, zdir_hint=(0, 0, 1)) -> np.ndarray:
    """Rigid transform whose local +X maps to ``xdir`` (unit), local Z as close
    to ``zdir_hint`` as possible, placed at ``origin``."""
    x = np.asarray(xdir, float)
    x = x / np.linalg.norm(x)
    zh = np.asarray(zdir_hint, float)
    y = np.cross(zh, x)
    if np.linalg.norm(y) < 1e-9:
        y = np.cross([0, 1, 0], x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


def apply(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------------------
# topology utilities
# ---------------------------------------------------------------------------
def weld(m: Mesh, tol: float = 1e-9) -> Mesh:
    """Merge coincident vertices; quads that collapse become triangles and
    fully degenerate faces are dropped (used for surfaces touching the axis)."""
    key = np.round(m.v / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(first)                    # keep original vertex order
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    v = m.v[first[order]]
    q = remap[inv[m.q]] if len(m.q) else m.q
    t = remap[inv[m.t]] if len(m.t) else m.t
    new_q, new_t = [], [t] if len(t) else []
    if len(q):
        uniq = np.array([len(set(r)) for r in q])
        new_q.append(q[uniq == 4])
        for r in q[uniq == 3]:
            rr = [r[0]] + [r[i] for i in range(1, 4) if r[i] != r[i - 1]]
            if rr[-1] == rr[0]:
                rr = rr[:-1]
            new_t.append(np.array([rr[:3]]))
    tt = np.vstack(new_t) if new_t else np.zeros((0, 3), np.int64)
    if len(tt):
        tt = tt[np.array([len(set(r)) for r in tt]) == 3]
    return Mesh(v, np.vstack(new_q) if new_q else np.zeros((0, 4), np.int64), tt)


def orient(m: Mesh, outward: bool = True) -> Mesh:
    """Make face winding consistent across shared edges (BFS), then, for a
    closed mesh, make the normals point outward (positive signed volume)."""
    faces = [list(r) for r in m.q] + [list(r) for r in m.t]
    n = len(faces)
    if n == 0:
        return m
    edge_faces: dict = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    flip = np.zeros(n, bool)
    seen = np.zeros(n, bool)

    def directed(fi):
        f = faces[fi][::-1] if flip[fi] else faces[fi]
        return set(zip(f, f[1:] + f[:1]))

    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            fi = stack.pop()
            de = directed(fi)
            for a, b in de:
                for fj in edge_faces[(min(a, b), max(a, b))]:
                    if fj == fi or seen[fj]:
                        continue
                    # consistent neighbour traverses the shared edge as (b, a)
                    if (a, b) in directed(fj):
                        flip[fj] = True
                    seen[fj] = True
                    stack.append(fj)
    nq = len(m.q)
    q = np.array([f[::-1] if flip[i] else f for i, f in enumerate(faces[:nq])],
                 np.int64).reshape(-1, 4)
    t = np.array([f[::-1] if flip[nq + i] else f for i, f in enumerate(faces[nq:])],
                 np.int64).reshape(-1, 3)
    out = Mesh(m.v, q, t)
    if outward and signed_volume(out) < 0:
        out = out.flipped()
    return out


def signed_volume(m: Mesh) -> float:
    T = m.tris()
    if not len(T):
        return 0.0
    a, b, c = m.v[T[:, 0]], m.v[T[:, 1]], m.v[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_closed(m: Mesh) -> bool:
    T = m.tris()
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2))


# ---------------------------------------------------------------------------
# 2-D helpers
# ---------------------------------------------------------------------------
def poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ccw(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p if poly_area(p) > 0 else p[::-1]


def ear_clip(p: np.ndarray) -> np.ndarray:
    """Triangulate a simple 2-D polygon (any winding). Returns (n-2, 3) indices
    into ``p`` with CCW winding in the polygon's own plane."""
    p = np.asarray(p, float)
    idx = list(range(len(p)))
    if poly_area(p) < 0:
        idx = idx[::-1]
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * len(p) ** 2:
        guard += 1
        n = len(idx)
        clipped = False
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = p[i0], p[i1], p[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = p[j]
                if (cross(a, b, q) >= -1e-14 and cross(b, c, q) >= -1e-14
                        and cross(c, a, q) >= -1e-14):
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                clipped = True
                break
        if not clipped:          # degenerate remainder: fan it
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            idx = idx[:3]
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.asarray(tris, np.int64)


def arclen(p: np.ndarray, closed: bool = True) -> np.ndarray:
    pp = np.vstack([p, p[:1]]) if closed else p
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pp, axis=0).T))])


def sample_at(p: np.ndarray, s_targets, closed: bool = True) -> np.ndarray:
    pp = np.vstack([p, p[:1]]) if closed else p
    s = arclen(p, closed)
    st = np.mod(s_targets, s[-1]) if closed else np.asarray(s_targets)
    return np.stack([np.interp(st, s, pp[:, j]) for j in range(p.shape[1])], -1)


def crossings(p: np.ndarray, axis: int, value: float, where=None) -> list:
    """Arc-length positions where a closed polyline crosses ``coord[axis] = value``;
    optional ``where(point)`` filter."""
    pp = np.vstack([p, p[:1]])
    s = arclen(p)
    out = []
    for i in range(len(p)):
        a, b = pp[i, axis] - value, pp[i + 1, axis] - value
        if a == 0.0 or a * b < 0:
            f = 0.0 if a == 0.0 else a / (a - b)
            pt = pp[i] + f * (pp[i + 1] - pp[i])
            if where is None or where(pt):
                out.append(s[i] + f * (s[i + 1] - s[i]))
    return sorted(set(np.round(out, 12)))


def resample_pieces(profiles, n: int, breaks_list) -> tuple:
    """Resample several closed profiles to ``n`` points each, with the SAME
    number of points in every piece between corresponding breakpoints (so the
    profiles stay in index correspondence).  ``breaks_list[k]`` are arc-length
    positions on ``profiles[k]`` (same count/order for all).  Returns
    ``(list of (n,2) arrays, break_indices)``."""
    ref = profiles[0]
    L0 = arclen(ref)[-1]
    b0 = list(breaks_list[0])
    edges0 = [0.0] + b0 + [L0]
    lens = np.diff(edges0)
    counts = np.maximum(1, np.round(lens / L0 * n).astype(int))
    counts[np.argmax(counts)] += n - counts.sum()
    out = []
    for p, br in zip(profiles, breaks_list):
        L = arclen(p)[-1]
        edges = [0.0] + list(br) + [L]
        s = np.concatenate([edges[i] + (edges[i + 1] - edges[i]) * np.arange(c) / c
                            for i, c in enumerate(counts)])
        out.append(sample_at(p, s))
    bidx = list(np.cumsum(counts)[:-1])
    return out, bidx


def clip_halfplane(poly: np.ndarray, axis: int, value: float, keep_less: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of a closed polygon against coord[axis] <=/>= value."""
    out = []
    P = np.asarray(poly, float)
    n = len(P)

    def inside(pt):
        return pt[axis] <= value if keep_less else pt[axis] >= value

    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        ia, ib = inside(a), inside(b)
        if ia:
            out.append(a)
        if ia != ib:
            f = (value - a[axis]) / (b[axis] - a[axis])
            out.append(a + f * (b - a))
    return np.asarray(out) if out else np.zeros((0, 2))


def point_seg_dist(pts: np.ndarray, poly: np.ndarray, closed: bool = True) -> np.ndarray:
    """Distance from each 2-D point to a polyline."""
    P = np.asarray(poly, float)
    A = P
    B = np.roll(P, -1, axis=0) if closed else P[1:]
    if not closed:
        A = P[:-1]
    d = np.full(len(pts), np.inf)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-30)
    for k in range(0, len(pts), 2048):
        X = pts[k:k + 2048, None, :]
        t = np.clip(((X - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
        C = A[None] + t[..., None] * AB[None]
        d[k:k + 2048] = np.sqrt(((X - C) ** 2).sum(-1)).min(1)
    return d


# ---------------------------------------------------------------------------
# solids
# ---------------------------------------------------------------------------
def grid_faces(n_a: int, n_b: int, closed_a: bool, closed_b: bool, off: int = 0) -> np.ndarray:
    """Quads of an (n_a x n_b) vertex grid, index = a*n_b + b."""
    ia = np.arange(n_a if closed_a else n_a - 1)[:, None]
    ib = np.arange(n_b if closed_b else n_b - 1)[None, :]
    a1, b1 = (ia + 1) % n_a, (ib + 1) % n_b
    q = np.stack([(ia * n_b + ib).ravel(), (a1 * n_b + ib).ravel(),
                  (a1 * n_b + b1).ravel(), (ia * n_b + b1).ravel()], -1)
    return q + off


def loft(sections: np.ndarray, closed_s: bool = True, cap: bool = False) -> Mesh:
    """Skin a sequence of closed 3-D rings ``sections`` (Ns, K, 3).  With
    ``closed_s`` the sequence wraps (torus-like); otherwise ``cap`` closes both
    ends with ear-clipped planar-ish polygons."""
    S = np.asarray(sections, float)
    ns, k = S.shape[:2]
    m = Mesh(S.reshape(-1, 3), grid_faces(ns, k, closed_s, True))
    if cap and not closed_s:
        tris = []
        for si in (0, ns - 1):
            ring = S[si]
            c = ring.mean(0)
            _, _, vt = np.linalg.svd(ring - c)
            uv = (ring - c) @ vt[:2].T
            tris.append(ear_clip(uv) + si * k)
        m = Mesh(m.v, m.q, np.vstack(tris))
    return orient(weld(m)) if cap or closed_s else m


def hexa(c: np.ndarray) -> Mesh:
    """Hexahedron from 8 corners ordered (x0y0z0, x1y0z0, x1y1z0, x0y1z0, then z1)."""
    q = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                  [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return orient(Mesh(np.asarray(c, float), q))


def box(lo, hi) -> Mesh:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return hexa([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                 [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def wedge_box(r0, r1, z0, z1, half_w, n_wedge: int | None = None, n_r: int = 1) -> Mesh:
    """Radial block in the local XZ plane (x = R), toroidal half width
    ``half_w`` clamped to the radial wedge ``R tan(pi/n_wedge)`` so that
    ``n_wedge`` copies never overlap near the axis."""
    xs = np.linspace(r0, r1, n_r + 1)
    if n_wedge:
        hw = np.minimum(half_w, xs * np.tan(np.pi / n_wedge) * (1 - 1e-6))
    else:
        hw = np.full_like(xs, half_w)
    # section rings along x: 4 corners (y-, z0), (y+, z0), (y+, z1), (y-, z1)
    S = np.array([[[x, -h, z0], [x, h, z0], [x, h, z1], [x, -h, z1]]
                  for x, h in zip(xs, hw)])
    return loft(S, closed_s=False, cap=True)


def solid_of_revolution(poly_rz: np.ndarray, n_tor: int = 128,
                        phi0: float = 0.0, phi1: float = 2 * np.pi) -> Mesh:
    """Revolve a closed (R, Z) polygon; full turn gives a closed solid (points
    on the axis are welded), a partial turn is capped at both ends."""
    p = ccw(poly_rz)
    full = abs((phi1 - phi0) - 2 * np.pi) < 1e-9
    if full:
        v, q = revolve(p, n_tor=n_tor, phi0=phi0, phi1=phi1, closed_tor=True)
        return orient(weld(Mesh(v, q)))
    phis = np.linspace(phi0, phi1, n_tor)
    S = np.stack([np.c_[p[:, 0] * np.cos(f), p[:, 0] * np.sin(f), p[:, 1]] for f in phis])
    return loft(S, closed_s=False, cap=True)


def slotted_sector(poly_rz: np.ndarray, pitch: float, half_clear, n_tor: int = 12) -> Mesh:
    """Revolve a closed (R, Z) polygon between two neighbouring flat-sided
    slots.  The part is centred on phi = 0; the slots are centred on
    +-pitch/2 and have half width ``half_clear`` (scalar or callable of R,Z),
    measured perpendicular to the slot centre plane (parallel-sided slot, like
    a TF coil casing)."""
    p = ccw(poly_rz)
    R = p[:, 0]
    hc = half_clear(p) if callable(half_clear) else np.full(len(p), float(half_clear))
    # angle (from the slot centre plane) of the point at perpendicular distance hc
    a = np.arcsin(np.clip(hc / np.maximum(R, 1e-9), -1, 1))
    lo = -pitch / 2 + a
    hi = pitch / 2 - a
    if np.any(hi - lo <= 1e-6):
        raise ValueError("slotted_sector: profile reaches radii where the slots meet")
    fr = np.linspace(0, 1, n_tor)
    phis = lo[None, :] + (hi - lo)[None, :] * fr[:, None]            # (n_tor, n_pol)
    S = np.stack([R[None, :] * np.cos(phis), R[None, :] * np.sin(phis),
                  np.broadcast_to(p[:, 1], phis.shape)], -1)
    return loft(S, closed_s=False, cap=True)


def rect_ring(w: float, h: float, n_w: int = 1, n_h: int = 1) -> np.ndarray:
    """Closed CCW rectangle (u in [-w/2, w/2], v in [-h/2, h/2]) with extra
    points along the sides; returns (K, 2)."""
    u0, u1, v0, v1 = -w / 2, w / 2, -h / 2, h / 2
    pts = []
    pts += [(u, v0) for u in np.linspace(u0, u1, n_w + 1)[:-1]]
    pts += [(u1, v) for v in np.linspace(v0, v1, n_h + 1)[:-1]]
    pts += [(u, v1) for u in np.linspace(u1, u0, n_w + 1)[:-1]]
    pts += [(u0, v) for v in np.linspace(v1, v0, n_h + 1)[:-1]]
    return np.asarray(pts)


def cylinder(radius: float, z0: float, z1: float, n: int = 32, r_in: float = 0.0) -> Mesh:
    """Solid (or hollow if r_in > 0) cylinder along local Z."""
    if r_in > 0:
        poly = np.array([[r_in, z0], [radius, z0], [radius, z1], [r_in, z1]])
    else:
        poly = np.array([[0, z0], [radius, z0], [radius, z1], [0, z1]])
    return solid_of_revolution(poly, n_tor=n)


def ellipsoid(a: float, b: float, c: float, n_u: int = 24, n_v: int = 12) -> Mesh:
    th = np.linspace(0, np.pi, n_v + 1)
    poly = np.c_[np.sin(th), -np.cos(th)]          # half-meridian, poles on axis
    v, q = revolve(poly, n_tor=n_u)
    v = v * np.array([a, b, c])
    return orient(weld(Mesh(v, q)))


def sample_surface(m: Mesh, M_list=None, k: int = 3) -> np.ndarray:
    """Points on the surface: vertices plus interior points of every face at
    barycentric/bilinear parameters (densified for collision sampling)."""
    pts = [m.v]
    if len(m.q):
        V = m.v[m.q]                                   # (F,4,3)
        for u in np.linspace(0, 1, k + 2)[1:-1]:
            for w in np.linspace(0, 1, k + 2)[1:-1]:
                pts.append((1 - u) * (1 - w) * V[:, 0] + u * (1 - w) * V[:, 1]
                           + u * w * V[:, 2] + (1 - u) * w * V[:, 3])
            for w in (0.0,):
                pts.append((1 - u) * V[:, 0] + u * V[:, 1])
                pts.append((1 - u) * V[:, 1] + u * V[:, 2])
    if len(m.t):
        V = m.v[m.t]
        pts.append(V.mean(1))
    P = np.vstack(pts)
    if M_list is None:
        return P
    return np.vstack([apply(M, P) for M in M_list])


# ---------------------------------------------------------------------------
# sections (used by the preview and the tests)
# ---------------------------------------------------------------------------
def slice_segments(V: np.ndarray, T: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Segments (n, 2, 3) where the triangle mesh (V, T) crosses the level
    set d(V) = 0 (d given per vertex; avoid exact zeros by offsetting d)."""
    D = d[T]
    hits = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        m = (D[:, a] * D[:, b]) < 0
        den = np.where(m, D[:, a] - D[:, b], 1.0)
        f = np.where(m, D[:, a] / den, 0.0)
        hits.append((m, V[T[:, a]] + f[:, None] * (V[T[:, b]] - V[T[:, a]])))
    segs = []
    for (ma, Pa), (mb, Pb) in ((hits[0], hits[1]), (hits[1], hits[2]), (hits[2], hits[0])):
        k = ma & mb
        if k.any():
            segs.append(np.stack([Pa[k], Pb[k]], 1))
    return np.vstack(segs) if segs else np.zeros((0, 2, 3))


def chain_loops(segs: np.ndarray, decimals: int = 7) -> list:
    """Join 2-D segments (n, 2, 2) into closed/open polylines."""
    key = lambda p: tuple(np.round(p, decimals))
    adj: dict = {}
    for i, (a, b) in enumerate(segs):
        adj.setdefault(key(a), []).append((i, 1))
        adj.setdefault(key(b), []).append((i, 0))
    used = np.zeros(len(segs), bool)
    loops = []
    for i0 in range(len(segs)):
        if used[i0]:
            continue
        used[i0] = True
        pts = [segs[i0][0], segs[i0][1]]
        cur = key(segs[i0][1])
        while True:
            nxt = [(j, e) for j, e in adj.get(cur, []) if not used[j]]
            if not nxt:
                break
            j, e = nxt[0]
            used[j] = True
            p = segs[j][e]
            pts.append(p)
            cur = key(p)
        loops.append(np.asarray(pts))
    return loops
