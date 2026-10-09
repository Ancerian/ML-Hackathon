"""Quantitative analysis of Poincare sections.

Three measurements, each independent of the equilibrium solver that produced
the field, so each is a real test rather than a restatement:

1. **Rotation number** nu = 1/q, measured by unwrapping the poloidal angle
   along the CONTINUOUS trajectory.  The section alone is not enough: at q < 2
   a puncture advances the poloidal angle by more than pi, so unwrapping the
   punctures is ambiguous.
2. **Orbit classification** into regular / island / chaotic.  Islands are
   found by LOCKING of nu on a low-order rational, in a band shared with
   neighbouring seeds -- NOT by radial excursion (VR9: excursion grows toward
   the edge for regular orbits too).  Unlocked orbits whose weighted-Birkhoff
   nu fails to converge are chaotic; the rest regular (see :func:`classify`).
3. **Island width measured from the section**, to compare against the pendulum
   prediction W = 4 sqrt(eps q / |dq/dpsi_N|).
"""
from __future__ import annotations

import numpy as np

from .equilibrium import Equilibrium, TWO_PI
from .fieldline import FieldLines


def trace_theta(fl: FieldLines, r0, z0, n_turns=120, pts_per_turn=30,
                rtol=1e-9, atol=1e-11):
    """Continuous trace, returning unwrapped poloidal angle and psi_n.

    Returns ``(theta_unwrapped (N, M), psi_n (N, M), phi (M), alive (N,))``.
    """
    r0 = np.atleast_1d(np.asarray(r0, float))
    z0 = np.atleast_1d(np.asarray(z0, float))
    phi = np.linspace(0.0, TWO_PI * n_turns, int(n_turns * pts_per_turn) + 1)
    R, Z, alive = fl.trace(r0, z0, phi, rtol=rtol, atol=atol)
    eq = fl.eq
    th = np.unwrap(np.arctan2(Z - eq.z_axis, R - eq.r_axis), axis=1)
    sn = eq.psi_n(R.ravel(), Z.ravel()).reshape(R.shape)
    return th, sn, phi, alive


def _birkhoff_weights(n):
    """Normalised bump weights w(t) = exp(-1/(t(1-t))), t in (0, 1).

    The weight and all its derivatives vanish at both ends, which is what makes
    the weighted average converge super-polynomially on a quasiperiodic orbit
    instead of at the plain-average rate of O(1/N).
    """
    t = (np.arange(n) + 0.5) / n
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return w / w.sum()


def weighted_birkhoff(vals):
    """Weighted Birkhoff average along axis 1."""
    v = np.atleast_2d(np.asarray(vals, float))
    return v @ _birkhoff_weights(v.shape[1])


def rotation_number(theta, phi):
    """nu = 1/q by weighted Birkhoff average of dtheta/dphi.

    A PLAIN average over a finite window is not good enough here.  theta
    advances non-uniformly around a shaped flux surface, so the finite-window
    mean carries an O(1/N) oscillatory error -- at 20 turns that is several
    percent, which swamps any sane chaos threshold and labels perfectly
    regular orbits chaotic.  (Measured: with the plain estimator, half the
    orbits came out "chaotic" at ZERO perturbation, where the system is exactly
    integrable.)

    Weighting with a C-infinity bump that vanishes at both ends removes the
    oscillation and converges to machine precision on quasiperiodic orbits,
    while failing to converge on chaotic ones -- which is exactly the contrast
    the classifier needs.  (Das & Yorke; Sander & Meiss.)
    """
    d = np.diff(theta, axis=1)
    dphi = float(phi[1] - phi[0])
    return np.abs(weighted_birkhoff(d)) / dphi


def rotation_number_digits(theta, phi):
    """Convergence of the weighted Birkhoff average, in decimal digits.

    Compares the average over the whole orbit with the average over its first
    half.  Quasiperiodic orbits -- regular AND island, since an island orbit is
    phase-locked -- agree to many digits; chaotic orbits do not.
    """
    d = np.diff(theta, axis=1)
    m = d.shape[1] // 2
    dphi = float(phi[1] - phi[0])
    nu_full = np.abs(weighted_birkhoff(d)) / dphi
    nu_half = np.abs(weighted_birkhoff(d[:, :m])) / dphi
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(nu_full - nu_half) / np.where(nu_full > 0, nu_full, np.nan)
        dig = -np.log10(np.where(rel > 0, rel, 1e-16))
    return np.nan_to_num(dig, nan=0.0, posinf=16.0), nu_full


def find_locked_bands(nu, psi_n0, rationals, lock_tol=0.02, min_orbits=2,
                     mutual_tol=2e-3, solo_tol=3e-4, nu_profile=None,
                     flatness=0.2):
    """Bands of neighbouring seeds whose rotation number locks on one rational.

    An island chain is phase-LOCKED: every orbit inside it shares the SAME
    nu = n/m across a finite band of psi_n.  A KAM orbit has irrational nu that
    varies smoothly from seed to seed.

    Proximity to a rational is therefore not enough on its own: with a 2%
    window, neighbouring KAM orbits routinely fall inside it by chance.
    (Measured: at zero perturbation, where no island can exist, a
    proximity-only test reported two.)  So a band must additionally be
    MUTUALLY consistent -- its members' nu must agree with each other to
    ``mutual_tol``, which locking gives for free and a smooth profile does not.

    A single orbit counts only if it is locked far more tightly (``solo_tol``),
    which lets a narrow island sampled by one seed still register.

    ``mutual_tol`` alone is not enough once the seeding is dense: neighbouring
    seeds then have nearly equal nu simply because the profile is smooth.
    (Measured: with 150 seeds, a mutual-consistency test alone again reported
    two islands at zero perturbation.)  So when ``nu_profile`` -- the
    equilibrium (psi_n, 1/q) -- is supplied, a band must additionally be FLAT
    compared with how much nu would have changed across its own psi_n span
    had it not been locked.  That comparison is scale-free and does not care
    how densely the region was sampled.
    """
    prof_x = prof_y = None
    if nu_profile is not None:
        prof_x, prof_y = (np.asarray(a, float) for a in nu_profile)
    order = [int(i) for i in np.argsort(psi_n0)]
    bands = []

    def emit(m, n, run):
        if not run:
            return
        v = nu[run]
        mean = float(np.mean(v))
        if mean <= 0:
            return
        spread = float(np.max(np.abs(v - mean)) / mean)

        flat_ok = True
        expected = None
        if prof_x is not None and len(run) >= 2:
            lo = float(np.min(psi_n0[run]))
            hi = float(np.max(psi_n0[run]))
            expected = abs(float(np.interp(hi, prof_x, prof_y))
                           - float(np.interp(lo, prof_x, prof_y)))
            observed = float(np.max(v) - np.min(v))
            # only decisive when the profile would have moved appreciably
            if expected > 5.0 * mutual_tol * mean:
                flat_ok = observed < flatness * expected

        if len(run) >= min_orbits and spread < mutual_tol and flat_ok:
            bands.append(dict(m=m, n=n, q=m / n, nu=n / m, members=list(run),
                              nu_mean=mean, nu_spread=spread,
                              nu_expected_change=expected, kind="band"))
        elif len(run) == 1 and abs(v[0] - n / m) < solo_tol * (n / m):
            bands.append(dict(m=m, n=n, q=m / n, nu=n / m, members=list(run),
                              nu_mean=mean, nu_spread=0.0, kind="solo"))

    for (m, n) in rationals:
        target = n / m
        hit = np.abs(nu - target) < lock_tol * target
        run = []
        for idx in order:
            if hit[idx]:
                run.append(idx)
            else:
                emit(m, n, run)
                run = []
        emit(m, n, run)
    return bands


def classify(theta, psi_n, phi, alive, min_digits=5.0, rationals=None,
             lock_tol=0.02, psi_n0=None, stability_digits=2.0,
             nu_profile=None):
    """Label every orbit regular / island / chaotic / escaped.

    * **island**  -- nu locked on a low-order rational, in a band shared with a
      neighbouring seed.  Islands are identified by LOCKING, not by radial
      excursion.  They are checked before chaos, because an island orbit
      carries a second, slow libration frequency about its O-point and so
      converges under weighted Birkhoff much more slowly than a KAM orbit;
      judged on convergence alone it would be misread as chaotic.
    * **chaotic** -- not locked, and the weighted Birkhoff average fails to
      converge to ``min_digits``.
    * **regular** -- everything else.
    """
    if rationals is None:
        rationals = [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1), (7, 2), (4, 1)]
    digits, nu = rotation_number_digits(theta, phi)
    exc = psi_n.max(axis=1) - psi_n.min(axis=1)
    n_orb = theta.shape[0]

    if psi_n0 is None:
        psi_n0 = psi_n[:, 0]
    bands = find_locked_bands(nu, np.asarray(psi_n0), rationals,
                              lock_tol=lock_tol, nu_profile=nu_profile)

    island = np.zeros(n_orb, bool)
    band_of = [None] * n_orb
    for b in bands:
        for i in b["members"]:
            if alive[i] and digits[i] > stability_digits:
                island[i] = True
                band_of[i] = (b["m"], b["n"])

    chaotic = alive & ~island & (digits < min_digits)
    regular = alive & ~island & ~chaotic

    labels = np.full(n_orb, "escaped", dtype=object)
    labels[regular] = "regular"
    labels[island] = "island"
    labels[chaotic] = "chaotic"
    return dict(labels=labels, nu=nu, digits=digits, conv=digits, excursion=exc,
                bands=bands, band_of=band_of,
                n_regular=int(regular.sum()), n_island=int(island.sum()),
                n_chaotic=int(chaotic.sum()), n_escaped=int((~alive).sum()),
                chaos_fraction=float(chaotic.sum() / max(alive.sum(), 1)))


def measured_island_width(psi_n, cls, q_res, tol=1e-9):
    """Full width in psi_n that the island chain at ``q_res`` actually fills.

    Taken as the union of the radial excursions of the orbits locked on that
    resonance -- i.e. the band the section really shows, which is what the
    pendulum formula predicts.
    """
    members = []
    for b in cls.get("bands", []):
        if abs(b["q"] - q_res) < 1e-9:
            members.extend(b["members"])
    members = [i for i in members if cls["labels"][i] == "island"]
    if not members:
        return None
    lo = float(psi_n[members].min())
    hi = float(psi_n[members].max())
    return dict(width=hi - lo, lo=lo, hi=hi, centre=0.5 * (lo + hi),
                n_orbits=len(members))


def plateau_report(nu, psi_n0, rationals, tol=5e-3):
    """Where the rotation number locks onto a low-order rational.

    A plateau in nu(psi_n) is the section's own evidence of a resonance; it
    needs no input from the equilibrium's q profile.
    """
    out = []
    for m, n in rationals:
        target = n / m                      # nu = 1/q = n/m
        sel = np.abs(nu - target) < tol * max(target, 1e-9)
        if sel.sum() < 2:
            continue
        out.append(dict(m=m, n=n, q=m / n, nu=target,
                        n_orbits=int(sel.sum()),
                        psi_n_lo=float(psi_n0[sel].min()),
                        psi_n_hi=float(psi_n0[sel].max()),
                        psi_n_span=float(psi_n0[sel].max() - psi_n0[sel].min())))
    return out
