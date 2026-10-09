"""Toroidal-field ripple of the JET 1975 TF set by Biot-Savart.

32 D-shaped coils (tf_coils.n_coils, p.77), each represented by filaments
along the WINDING-PACK CENTRELINE: the digitized bore contour
(profiles/tf_coil_inner.csv, Fig. IV.2-1 p.325) offset outward by half the
leg radial thickness (0.3735 m inner / 0.371 m outer, p.325 -> 0.186 m).
Total current Ic = 41 MA basic / 51 MA extended (Table I.3-1 p.83), i.e.
1.28 / 1.59 MA per coil.

Two ripple conventions are reported, because the source and the usual modern
definition differ by a factor of two:

    delta   = (Bmax - Bmin) / (Bmax + Bmin)        modern / this task
    eps_src = 2 (Bmax - Bmin) / (Bmax + Bmin)      JET 1975, Fig. IV.2-4 p.330
                                                   caption ("E = 2 x (...)")

Table I.3-1's "TF ripple 3.6 %" and Table IV.2-4 (p.332) use eps_src.

Bmax is on a coil plane (phi = 0), Bmin midway between coils (phi = pi/32).
Both ratios are independent of the current (B is linear in I).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..rmp_coils import biot_savart, cart_to_cyl_vec, cyl_to_cart
from ..surfaces import offset_closed, resample_closed

PROFILE_DIR = Path(__file__).resolve().parents[3] / "machines" / "jet1975" / "profiles"

#: Fig. IV.2-4 (p.330) ripple curve, digitized by us from a 150-dpi render of
#: the page (axis ticks located in pixel space: 2.03 px/cm in R, 30.6 px per 1 %
#: in eps).  Values are eps_src (x2 convention).  Estimated reading error
#: +-2 cm in R, +-0.1 % absolute (+-5 % relative on the steep flank).  The
#: figure is for I = 66.4 kA/turn (= 51 MA extended); ratios are
#: current-independent.
FIG_IV24_EPS_PCT = np.array([
    # R [m], eps_src [%]
    [3.80, 0.20],
    [3.90, 0.52],
    [4.00, 1.05],
    [4.10, 1.95],
    [4.20, 3.30],
    [4.25, 4.45],
    [4.30, 6.4],
    [4.40, 16.6],
    [4.45, 22.2],
    [4.50, 28.0],
])


def load_csv(name, profile_dir=PROFILE_DIR, cols=2):
    rows = []
    for line in (Path(profile_dir) / f"{name}.csv").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append([float(v) for v in s.split(",")[:cols]])
        except ValueError:
            continue                              # header line
    return np.asarray(rows)


def winding_centreline(half_thickness: float = 0.186, n: int = 240,
                       profile_dir=PROFILE_DIR) -> np.ndarray:
    """Closed (R, Z) centreline of the winding pack, ``n`` points."""
    bore = load_csv("tf_coil_inner", profile_dir)
    bore = resample_closed(bore, 1200)
    c = offset_closed(bore, half_thickness)
    return resample_closed(c, n)


def tf_filaments(centreline: np.ndarray, n_coils: int = 32, current_total: float = 4.1e7,
                 n_rad: int = 1, n_tor: int = 1, pack_rad: float = 0.30,
                 pack_tor: float = 0.26, phi0: float = 0.0) -> list:
    """``(vertices, current)`` loops for Biot-Savart.

    ``n_rad x n_tor`` filaments per coil, spread uniformly over a rectangular
    pack ``pack_rad`` (normal to the contour) x ``pack_tor`` (toroidal) and
    sharing the coil current equally.  The positive current direction is
    chosen so B_phi > 0.
    """
    c = np.asarray(centreline, float)
    I_coil = current_total / n_coils
    # the contour is CCW in (R, Z); outward normal of each vertex
    nxt, prv = np.roll(c, -1, 0), np.roll(c, 1, 0)
    t = nxt - prv
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    nrm = np.stack([t[:, 1], -t[:, 0]], -1)
    offs_r = (np.arange(n_rad) - (n_rad - 1) / 2) * (pack_rad / n_rad if n_rad > 1 else 0.0)
    offs_t = (np.arange(n_tor) - (n_tor - 1) / 2) * (pack_tor / n_tor if n_tor > 1 else 0.0)
    loops = []
    for k in range(n_coils):
        ph = phi0 + 2 * np.pi * k / n_coils
        e_r = np.array([np.cos(ph), np.sin(ph), 0.0])
        e_p = np.array([-np.sin(ph), np.cos(ph), 0.0])
        for dr in offs_r:
            rz = c + dr * nrm
            for dt in offs_t:
                v = (rz[:, 0:1] * e_r[None, :] + dt * e_p[None, :]
                     + rz[:, 1:2] * np.array([0.0, 0.0, 1.0])[None, :])
                v = np.vstack([v, v[:1]])
                # CCW in (R,Z) seen from +phi: current flows up the outer leg ->
                # B_phi < 0 inside; reverse to get B_phi > 0.
                loops.append((v[::-1].copy(), I_coil / (n_rad * n_tor)))
    return loops


def field_rz_phi(loops, R, Z, phi):
    """(B_R, B_phi, B_Z) at matching arrays of (R, Z, phi)."""
    R, Z, phi = np.broadcast_arrays(np.asarray(R, float), np.asarray(Z, float),
                                    np.asarray(phi, float))
    pts = cyl_to_cart(R.ravel(), phi.ravel(), Z.ravel())
    B = biot_savart(loops, pts)
    BR, BP, BZ = cart_to_cyl_vec(B, phi.ravel())
    return BR.reshape(R.shape), BP.reshape(R.shape), BZ.reshape(R.shape)


def ripple(loops, R, Z=0.0, n_coils: int = 32, phi0: float = 0.0):
    """Return ``(delta, eps_src, Bmax, Bmin)`` at (R, Z) arrays.

    Uses |B| on the coil plane and midway between coils.  (On the midplane
    B_R = B_Z = 0 by symmetry, so |B| = |B_phi| there.)
    """
    R = np.atleast_1d(np.asarray(R, float))
    Z = np.broadcast_to(np.asarray(Z, float), R.shape)
    out = []
    for ph in (phi0, phi0 + np.pi / n_coils):
        BR, BP, BZ = field_rz_phi(loops, R, Z, np.full(R.shape, ph))
        out.append(np.sqrt(BR ** 2 + BP ** 2 + BZ ** 2))
    B_on, B_off = out
    Bmax, Bmin = np.maximum(B_on, B_off), np.minimum(B_on, B_off)
    d = (Bmax - Bmin) / (Bmax + Bmin)
    return d, 2 * d, Bmax, Bmin


def kernel_tf_phi0_rad(machine_dir=None) -> float:
    """Azimuth of TF coil 0 (``kernel.tf_phi0_deg``) from the machine's
    ASSUMPTIONS_KERNEL.md yaml block, in radians (0 if not found).

    The ripple RATIO does not depend on it (coil plane vs. mid-gap are taken
    relative to it), but absolute angles -- e.g. placing ripple markers on the
    3-D machine, whose variants may rotate the coil set -- must use it.
    """
    import re
    root = Path(machine_dir) if machine_dir else PROFILE_DIR.parent
    for fn in ("deltas.yaml", "ASSUMPTIONS_KERNEL.md"):
        p = root / fn
        if p.exists():
            m = re.search(r"tf_phi0_deg\D*?value:\s*([-+0-9.eE]+)", p.read_text(encoding="utf-8"))
            if m:
                return float(np.deg2rad(float(m.group(1))))
    return 0.0
