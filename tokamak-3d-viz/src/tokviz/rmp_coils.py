"""Vacuum field of a resonant-magnetic-perturbation coil array, by Biot-Savart.

Used to test VC1: is the analytic envelope g(psi_n) = psi_n**(m/2) a fair stand-in
for the radial profile that a real external coil set produces?

Geometry
--------
A DIII-D-like internal (I-coil) array: two toroidal rows of single-turn
"picture-frame" loops sitting on a cylinder of radius ``R_coil``, each spanning
``span_deg`` in toroidal angle, with alternating current sign so the array
carries a chosen toroidal mode number.

The published geometry varies between sources, so it is PARAMETERISED here and
the conclusion is checked for robustness against it (see run_coil_compare.py)
rather than resting on one quoted number.
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi


def picture_frame_loop(R, z_lo, z_hi, phi0, phi1, n_arc=40, n_leg=6):
    """Closed loop on a cylinder: arc at z_hi, leg down, arc back at z_lo, leg up.

    Returns ``(N, 3)`` Cartesian vertices, closed (last point == first).
    """
    a = np.linspace(phi0, phi1, n_arc)
    top = np.stack([R * np.cos(a), R * np.sin(a), np.full_like(a, z_hi)], axis=-1)
    legz = np.linspace(z_hi, z_lo, n_leg)[1:]
    leg1 = np.stack([np.full_like(legz, R * np.cos(phi1)),
                     np.full_like(legz, R * np.sin(phi1)), legz], axis=-1)
    b = a[::-1]
    bot = np.stack([R * np.cos(b), R * np.sin(b), np.full_like(b, z_lo)], axis=-1)[1:]
    legz2 = np.linspace(z_lo, z_hi, n_leg)[1:]
    leg2 = np.stack([np.full_like(legz2, R * np.cos(phi0)),
                     np.full_like(legz2, R * np.sin(phi0)), legz2], axis=-1)
    return np.vstack([top, leg1, bot, leg2])


def icoil_array(R_coil=2.184, z_centre=0.754, height=0.5, n_coils=6,
                span_deg=57.6, n_toroidal=3, current=1.0, rows=("upper", "lower"),
                lower_phase_deg=60.0):
    """DIII-D-like I-coil array.

    ``n_toroidal`` sets the sign pattern around the torus: coil j carries
    ``cos(n * phi_j)``-like polarity, which for ``n_coils = 2 n`` gives a clean
    alternating +/- array.  ``lower_phase_deg`` is the toroidal phase shift
    between the upper and lower rows, which in experiment selects between
    resonant and non-resonant configurations.

    Returns a list of ``(vertices, current)`` pairs.
    """
    loops = []
    for row in rows:
        zc = z_centre if row == "upper" else -z_centre
        dphi = 2.0 * np.pi / n_coils
        span = np.deg2rad(span_deg)
        shift = 0.0 if row == "upper" else np.deg2rad(lower_phase_deg)
        for j in range(n_coils):
            c = j * dphi + shift
            # Sinusoidal current weighting, NOT the binary sign of the cosine.
            # A binary +/- pattern is itself rich in harmonics and distorts the
            # poloidal spectrum the array couples to, which is exactly what is
            # being measured here.
            sign = float(np.cos(n_toroidal * c))
            if row == "lower":
                sign = -sign            # opposite rows drive the odd-n spectrum
            v = picture_frame_loop(R_coil, zc - height / 2, zc + height / 2,
                                   c - span / 2, c + span / 2)
            loops.append((v, sign * current))
    return loops


def biot_savart(loops, points, chunk=20000):
    """B at ``points`` (N, 3) from a list of ``(vertices, current)`` loops.

    Straight-segment Biot-Savart, summed over every segment of every loop.
    """
    pts = np.asarray(points, float)
    B = np.zeros_like(pts)
    for v, I in loops:
        a = v[:-1]
        b = v[1:]
        dl = b - a
        for s in range(0, pts.shape[0], chunk):
            P = pts[s:s + chunk][:, None, :]          # (C,1,3)
            r1 = P - a[None, :, :]                    # (C,S,3)
            r2 = P - b[None, :, :]
            n1 = np.linalg.norm(r1, axis=-1)
            n2 = np.linalg.norm(r2, axis=-1)
            cross = np.cross(dl[None, :, :], r1)
            denom = (n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=-1)))
            denom = np.where(np.abs(denom) < 1e-18, np.nan, denom)
            fac = (n1 + n2) / denom
            contrib = np.nansum(cross * fac[..., None], axis=1)
            B[s:s + chunk] += (MU0 * I / (4.0 * np.pi)) * contrib
    return B


def vector_potential(loops, points, chunk=20000):
    """A at ``points`` (N, 3) from a list of ``(vertices, current)`` loops.

    For a straight segment from a to b,

        A = (mu0 I / 4pi) * Lhat * ln[ (|r2| + |r1| + L) / (|r2| + |r1| - L) ]

    Representing the field through A and taking the curl analytically is what
    makes the reconstructed B divergence-free STRUCTURALLY, rather than to
    within interpolation error -- the same reason the analytic perturbation
    backend writes its field as curl(psi grad phi).
    """
    pts = np.asarray(points, float)
    A = np.zeros_like(pts)
    for v, I in loops:
        a = v[:-1]
        b = v[1:]
        dl = b - a
        L = np.linalg.norm(dl, axis=-1)
        good = L > 1e-14
        a, b, dl, L = a[good], b[good], dl[good], L[good]
        Lhat = dl / L[:, None]
        for s0 in range(0, pts.shape[0], chunk):
            P = pts[s0:s0 + chunk][:, None, :]
            n1 = np.linalg.norm(P - a[None, :, :], axis=-1)
            n2 = np.linalg.norm(P - b[None, :, :], axis=-1)
            num = n2 + n1 + L[None, :]
            den = n2 + n1 - L[None, :]
            den = np.where(den < 1e-14, 1e-14, den)
            f = np.log(num / den)
            A[s0:s0 + chunk] += (MU0 * I / (4.0 * np.pi)) * (
                f[..., None] * Lhat[None, :, :]).sum(axis=1)
    return A


def cyl_to_cart(R, phi, Z):
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)


def cart_to_cyl_vec(Bxyz, phi):
    """Cartesian vector field -> (B_R, B_phi, B_Z) at the given toroidal angles."""
    c, s = np.cos(phi), np.sin(phi)
    B_R = Bxyz[..., 0] * c + Bxyz[..., 1] * s
    B_phi = -Bxyz[..., 0] * s + Bxyz[..., 1] * c
    return B_R, B_phi, Bxyz[..., 2]
