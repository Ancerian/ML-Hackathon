"""Turn equilibrium/tracing results into 3-D geometry for Blender.

Everything here is plain numpy.  Meshes are emitted as ``(verts, faces)`` with
``verts`` in metres and ``faces`` as quad index arrays, ready for
``Mesh.from_pydata``.  A uniform ``scale`` is applied on the way out: it
multiplies R and Z alike, so the aspect ratio -- and therefore q(psi) and the
whole field-line topology -- is unchanged.
"""
from __future__ import annotations

import numpy as np

from .equilibrium import Equilibrium, TWO_PI


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to ``n`` points, evenly in arc length."""
    p = np.asarray(pts, dtype=float)
    closed = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("degenerate polygon")
    target = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, p.shape[1]))
    for j in range(p.shape[1]):
        out[:, j] = np.interp(target, s, closed[:, j])
    return out


# ---------------------------------------------------------------------------
# surfaces of revolution
# ---------------------------------------------------------------------------
def revolve(profile_rz: np.ndarray, n_tor: int = 128, scale: float = 1.0,
            phi0: float = 0.0, phi1: float = TWO_PI, closed_tor: bool = True):
    """Sweep a closed (R, Z) profile around the Z axis.

    Returns ``(verts (n_pol*n_tor, 3), faces (n_pol*n_seg, 4))``.
    """
    prof = np.asarray(profile_rz, dtype=float) * float(scale)
    n_pol = prof.shape[0]
    if closed_tor:
        phis = np.linspace(phi0, phi1, n_tor, endpoint=False)
    else:
        phis = np.linspace(phi0, phi1, n_tor)
    c, s = np.cos(phis), np.sin(phis)

    X = prof[:, 0][None, :] * c[:, None]      # (n_tor, n_pol)
    Y = prof[:, 0][None, :] * s[:, None]
    Z = np.repeat(prof[:, 1][None, :], phis.size, axis=0)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    n_seg = n_tor if closed_tor else n_tor - 1
    i = np.arange(n_seg)[:, None]
    j = np.arange(n_pol)[None, :]
    i1 = (i + 1) % n_tor
    j1 = (j + 1) % n_pol
    faces = np.stack([
        (i * n_pol + j).ravel(),
        (i1 * n_pol + j).ravel(),
        (i1 * n_pol + j1).ravel(),
        (i * n_pol + j1).ravel(),
    ], axis=-1)
    return verts, faces.astype(np.int32)


def flux_surface(eq: Equilibrium, psi_n: float, n_pol: int = 128,
                 n_tor: int = 128, scale: float = 1.0):
    """Closed torus mesh for one axisymmetric flux surface."""
    prof = eq.surface(float(psi_n))
    if prof is None:
        return None
    prof = resample_closed(prof, n_pol)
    return revolve(prof, n_tor=n_tor, scale=scale)


def flux_surface_set(eq: Equilibrium, levels, n_pol=128, n_tor=128, scale=1.0):
    """``{psi_n: (verts, faces)}`` for every level that has a closed surface."""
    out = {}
    for s in levels:
        m = flux_surface(eq, float(s), n_pol=n_pol, n_tor=n_tor, scale=scale)
        if m is not None:
            out[float(s)] = m
    return out


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------
def polyline_set(xyz: np.ndarray, alive: np.ndarray | None = None,
                 scale: float = 1.0):
    """Split a traced ``(N, M, 3)`` block into a list of ``(M, 3)`` polylines."""
    arr = np.asarray(xyz, dtype=float) * float(scale)
    if alive is None:
        alive = np.ones(arr.shape[0], dtype=bool)
    return [arr[i] for i in range(arr.shape[0]) if alive[i]]


def punctures_xyz(R: np.ndarray, Z: np.ndarray, phi: float = 0.0,
                  scale: float = 1.0) -> np.ndarray:
    """Poincare punctures as Cartesian points on the phi = const plane."""
    r = np.asarray(R, dtype=float).ravel() * float(scale)
    z = np.asarray(Z, dtype=float).ravel() * float(scale)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


# ---------------------------------------------------------------------------
# machine geometry from the real coil table
# ---------------------------------------------------------------------------
def coil_profile(coil: dict) -> np.ndarray:
    """Cross-section polygon of one PF coil, in (R, Z).

    The dataset stores each coil as a centre, a width and height, and two
    shear angles, following the EFIT ``AC``/``AC2`` convention used for the
    DIII-D F-coils:

    * ``angle1`` (AC) tilts the horizontal edges away from the R axis;
    * ``angle2`` (AC2) is the angle the vertical edges make with the R axis,
      so 90 degrees is an unsheared rectangle.

    ``angle2 == 0`` means "not specified", NOT a zero-degree shear -- most
    DIII-D F-coils carry ``angle1 = angle2 = 0`` and are plain rectangles.
    """
    w, h = float(coil["width"]), float(coil["height"])
    a1 = float(coil.get("angle1", 0.0))
    a2 = float(coil.get("angle2", 0.0))

    dr = np.array([-w / 2, +w / 2, +w / 2, -w / 2])
    dz = np.array([-h / 2, -h / 2, +h / 2, +h / 2])

    if abs(a1) > 1e-9:
        dz = dz + dr * np.tan(np.deg2rad(a1))
    if abs(a2) > 1e-9:
        t = np.tan(np.deg2rad(a2))
        if abs(t) > 1e-9:
            dr = dr + dz / t

    return np.stack([float(coil["R"]) + dr, float(coil["Z"]) + dz], axis=-1)


def coil_torus(coil: dict, n_tor: int = 96, scale: float = 1.0):
    """Full toroidal ring for one PF coil."""
    return revolve(coil_profile(coil), n_tor=n_tor, scale=scale)


def d_shape(R0: float, a: float, kappa: float = 1.8, delta: float = 0.45,
            n: int = 160) -> np.ndarray:
    """Miller-parametrised D cross-section, in (R, Z).

        R(t) = R0 + a cos(t + arcsin(delta) sin t)
        Z(t) = kappa a sin(t)

    This is the standard shaping parametrisation (elongation ``kappa``,
    triangularity ``delta``) used throughout tokamak physics.  It is NOT the
    constant-tension 'Princeton D' -- that is a different curve, defined by a
    force-balance condition rather than by shaping parameters.
    """
    t = np.linspace(0.0, TWO_PI, int(n), endpoint=False)
    x = np.arcsin(np.clip(delta, -1.0, 1.0))
    R = R0 + a * np.cos(t + x * np.sin(t))
    Z = kappa * a * np.sin(t)
    return np.stack([R, Z], axis=-1)


def offset_closed(profile_rz: np.ndarray, d: float) -> np.ndarray:
    """Offset a closed (R, Z) polygon outward by ``d`` along its normals."""
    p = np.asarray(profile_rz, float)
    nxt = np.roll(p, -1, axis=0)
    prv = np.roll(p, 1, axis=0)
    tang = nxt - prv
    L = np.hypot(tang[:, 0], tang[:, 1])[:, None]
    L = np.where(L < 1e-12, 1e-12, L)
    tang = tang / L
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=-1)   # outward for CCW
    if _poly_area(p) < 0:
        nrm = -nrm
    return p + d * nrm


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def tf_coil_ring(profile_rz: np.ndarray, n_coils: int, thickness: float,
                 width: float, scale: float = 1.0):
    """``n_coils`` box-section TF coils placed evenly in phi.

    Returns a list of ``(verts, faces)``; each coil is the profile swept a
    short toroidal distance, so the set reads as discrete coils rather than a
    continuous shell.
    """
    out = []
    half = 0.5 * width / max(np.mean(profile_rz[:, 0]), 1e-6)
    for i in range(n_coils):
        phi_c = TWO_PI * i / n_coils
        v, f = revolve(profile_rz, n_tor=6, scale=scale,
                       phi0=phi_c - half, phi1=phi_c + half, closed_tor=False)
        out.append((v, f))
    return out
