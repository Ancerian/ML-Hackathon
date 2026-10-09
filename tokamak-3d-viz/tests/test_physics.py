#!/usr/bin/env python3
"""Physics checks, runnable with or without pytest.

    python tests/test_physics.py            # standalone
    pytest tests/test_physics.py            # if pytest is installed

Needs a shot parquet; point TOKVIZ_DATA_DIR at one.  Skips cleanly if absent,
so the suite is safe in CI where the CC BY data is not redistributed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokviz.config import DATA_DIR, DEFAULT_SHOT
from tokviz.equilibrium import load_shot, select_frame, Equilibrium, TWO_PI
from tokviz.fieldline import FieldLines, invariant_drift, q_from_tracing
from tokviz.perturbation import (Mode, Perturbation, resonant_surfaces,
                                 island_width_psin, chirikov)

SHOT = DATA_DIR / DEFAULT_SHOT
_cache = {}


def have_data():
    return SHOT.exists()


def setup():
    """Build the reference equilibrium once and memoise it."""
    if "eq" in _cache:
        return _cache
    shot = load_shot(SHOT)
    k = select_frame(shot)
    eq = Equilibrium(shot.psirz[k], shot.grid_R, shot.grid_Z, machine=shot.source)
    eq.find_axis()
    n = int(shot.lcfs_n[k])
    eq.set_boundary_from_contour(shot.lcfs_r[k][:n], shot.lcfs_z[k][:n])
    eq.calibrate_F(float(shot.q95[k]))
    _cache.update(shot=shot, k=k, eq=eq)
    return _cache


# ---------------------------------------------------------------------------
def test_axis_matches_efit():
    """Our O-point finder must land on EFIT's published axis."""
    c = setup()
    eq, shot, k = c["eq"], c["shot"], c["k"]
    dR = abs(eq.r_axis - float(shot.r_axis[k]))
    dZ = abs(eq.z_axis - float(shot.z_axis[k]))
    assert dR < 0.05 * eq.dR, f"dR={dR*1e3:.4f} mm"
    assert dZ < 0.05 * eq.dZ, f"dZ={dZ*1e3:.4f} mm"


def test_lcfs_is_a_psi_contour():
    """psi must be near-constant along the dataset's own LCFS polygon."""
    c = setup()
    eq, shot, k = c["eq"], c["shot"], c["k"]
    n = int(shot.lcfs_n[k])
    vals = eq.psi_at(shot.lcfs_r[k][:n], shot.lcfs_z[k][:n])
    rel = vals.std() / abs(eq.psi_bdy - eq.psi_axis)
    assert rel < 1e-3, f"psi spread on LCFS = {rel:.2e} of the flux range"


def test_F_calibration_recovers_toroidal_field():
    """F pinned to q95 alone should reproduce DIII-D's real B_phi ~ 1.9-2.1 T."""
    c = setup()
    eq = c["eq"]
    b = eq.F / eq.r_axis
    assert 1.7 < b < 2.3, f"B_phi(axis) = {b:.3f} T"


def test_q_contour_matches_q_traced():
    """Two independent q calculations must agree."""
    c = setup()
    eq = c["eq"]
    fl = FieldLines(eq, None)
    for level in (0.30, 0.50, 0.70, 0.85):
        pts = eq.surface(level)
        assert pts is not None
        i = int(np.argmax(pts[:, 0]))
        q_c = eq.q_at(level)
        q_t = q_from_tracing(fl, pts[i, 0], pts[i, 1], n_turns=120)
        rel = abs(q_c - q_t) / q_c
        assert rel < 5e-3, f"psi_n={level}: contour {q_c:.4f} vs traced {q_t:.4f}"


def test_psi_is_invariant_without_perturbation():
    """Axisymmetry theorem, doubling as an integrator accuracy test."""
    c = setup()
    eq = c["eq"]
    r = eq.r_axis + np.array([0.15, 0.35])
    drift, alive = invariant_drift(eq, r, np.full(2, eq.z_axis), n_turns=40)
    assert alive.all()
    assert drift.max() < 1e-5, f"psi drift = {drift.max():.2e}"


def test_island_width_has_no_m_dependence():
    """W = 4 sqrt(eps q / |dq/dpsi_N|) -- the m-dependence cancels.

    (The canonical pair is (theta*, psi_t) with H = psi, so W grows as
    sqrt(q), not as q; see island_width_psin.)
    """
    a = island_width_psin(1e-3, 2.0, 5.0)
    b = island_width_psin(1e-3, 2.0, 5.0)
    assert a == b
    assert abs(a - 4.0 * np.sqrt(1e-3 * 2.0 / 5.0)) < 1e-15
    # scaling: quadrupling eps doubles W
    assert abs(island_width_psin(4e-3, 2.0, 5.0) / a - 2.0) < 1e-9
    # scaling: W ~ sqrt(q) at fixed eps and shear
    assert abs(island_width_psin(1e-3, 8.0, 5.0) / a - 2.0) < 1e-9


def test_island_width_matches_traced_separatrix():
    """The pendulum width must match the width the field lines actually fill.

    Single-mode perturbation (no overlap, no stochastic layer to speak of),
    seeds across the resonance on the outboard midplane.  An orbit is trapped
    iff its resonant phase xi = m theta* - n phi librates (range < 2 pi); the
    measured width is the psi_N extent of all trapped trajectories.

    This is the regression test for the sqrt(q) error: the old
    4 sqrt(eps q^2/|dq/dpsi|) is 41 % too wide at q = 2 and 73 % at q = 3.
    """
    c = setup()
    eq = c["eq"]
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    dq = np.gradient(q, x)
    tsf = eq.theta_star_field()
    rr = np.linspace(eq.r_axis + 0.03, eq.r_axis + 0.62, 800)
    ss = eq.psi_n(rr, np.full_like(rr, eq.z_axis))
    ok = np.isfinite(ss)
    n_turns = 50
    phi = np.linspace(0.0, TWO_PI * n_turns, n_turns * 24 + 1)
    for m, n in ((2, 1), (3, 1)):
        md = Mode(m=m, n=n, amp=1e-3)
        md.psin_res = resonant_surfaces(x, q, md.q_res)[0]
        pert = Perturbation(eq, [md], tsf=tsf)
        eps = pert.amplitude_at(md)
        dqds = float(np.interp(md.psin_res, x, dq))
        W = island_width_psin(eps, md.q_res, dqds)

        targets = md.psin_res + np.linspace(-0.75, 0.75, 20) * W
        r0 = np.interp(targets, ss[ok], rr[ok])
        R, Z, alive = FieldLines(eq, pert).trace(
            r0, np.full_like(r0, eq.z_axis), phi, rtol=1e-8, atol=1e-10)
        th = tsf.value(R.ravel(), Z.ravel()).reshape(R.shape)
        xi = np.unwrap(m * th, axis=1) - n * phi[None, :]
        trapped = alive & ((xi.max(axis=1) - xi.min(axis=1)) < TWO_PI)
        assert trapped.sum() >= 3, f"{m}/{n}: only {trapped.sum()} trapped orbits"
        assert not trapped[0] and not trapped[-1], f"{m}/{n}: seeds do not bracket the island"
        sn = eq.psi_n(R[trapped].ravel(), Z[trapped].ravel())
        W_meas = float(sn.max() - sn.min())
        assert abs(W_meas / W - 1.0) < 0.10, (
            f"{m}/{n}: measured {W_meas:.4f} vs predicted {W:.4f}")
        W_old = W * np.sqrt(md.q_res)          # the retired q**2 formula
        assert abs(W_meas / W_old - 1.0) > 0.20, "test cannot tell sqrt(q) apart"


def test_perturbation_derivatives_match_finite_difference():
    """The chain rule through theta* must be right."""
    c = setup()
    eq = c["eq"]
    x, q = eq.q_profile(np.linspace(0.05, 0.99, 60))
    md = Mode(m=2, n=1, amp=1e-3)
    md.psin_res = resonant_surfaces(x, q, md.q_res)[0]
    pert = Perturbation(eq, [md])
    rng = np.random.default_rng(0)
    r = eq.r_axis + rng.uniform(-0.25, 0.35, 6)
    z = eq.z_axis + rng.uniform(-0.35, 0.35, 6)
    h = 1e-6
    for phi in (0.0, 1.1):
        aR, aZ = pert.dpsi_derivs(r, z, phi)
        fR = (pert.dpsi(r + h, z, phi) - pert.dpsi(r - h, z, phi)) / (2 * h)
        fZ = (pert.dpsi(r, z + h, phi) - pert.dpsi(r, z - h, phi)) / (2 * h)
        scale = max(np.abs(aR).max(), np.abs(aZ).max())
        assert np.abs(aR - fR).max() / scale < 1e-4
        assert np.abs(aZ - fZ).max() / scale < 1e-4


def test_perturbation_spectrum_is_clean():
    """A mode m must not leak significant power into other harmonics.

    This is the regression test for the bug that manufactured a fake island
    chain at q = 1 from a perturbation containing only m = 2 and m = 3.
    """
    c = setup()
    eq = c["eq"]
    x, q = eq.q_profile(np.linspace(0.05, 0.99, 60))
    md = Mode(m=2, n=1, amp=1e-3)
    md.psin_res = resonant_surfaces(x, q, md.q_res)[0]
    pert = Perturbation(eq, [md])
    for level in (0.335, 0.752):
        pts = eq.surface(level)
        cum, tot = eq._loop_integral(pts)
        th = TWO_PI * cum[:-1] / tot
        vals = pert.dpsi(pts[:, 0], pts[:, 1], 0.0)
        w = np.diff(np.r_[th, th[0] + TWO_PI])
        amp = {}
        for m in range(0, 6):
            cc = np.sum(vals * np.cos(m * th) * w) / np.pi
            ss = np.sum(vals * np.sin(m * th) * w) / np.pi
            amp[m] = float(np.hypot(cc, ss))
        peak = amp[2]
        for m in (0, 1, 3, 4, 5):
            assert amp[m] / peak < 0.05, (
                f"psi_n={level}: m={m} leakage {amp[m]/peak:.4f} of the m=2 peak")


def test_resonant_surfaces_are_ordered_and_inside():
    c = setup()
    eq = c["eq"]
    x, q = eq.q_profile(np.linspace(0.05, 0.99, 80))
    s1 = resonant_surfaces(x, q, 1.0)
    s2 = resonant_surfaces(x, q, 2.0)
    s3 = resonant_surfaces(x, q, 3.0)
    assert s1 and s2 and s3
    assert 0.0 < s1[0] < s2[0] < s3[0] < 1.0, (s1, s2, s3)


def test_chirikov_is_symmetric():
    assert abs(chirikov(0.1, 0.7, 0.1, 0.9) - chirikov(0.1, 0.9, 0.1, 0.7)) < 1e-12


def test_vector_potential_curl_matches_biot_savart():
    """curl A must reproduce the Biot-Savart field.

    This is what lets the coil backend be divergence-free structurally.
    """
    from tokviz.rmp_coils import vector_potential, biot_savart
    t = np.linspace(0.0, 2 * np.pi, 400)
    loop = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)
    P = np.array([0.3, 0.2, 0.5])
    h = 1e-5
    J = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3); d[j] = h
        J[:, j] = (vector_potential([(loop, 1.0)], (P + d)[None, :])[0]
                   - vector_potential([(loop, 1.0)], (P - d)[None, :])[0]) / (2 * h)
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    B = biot_savart([(loop, 1.0)], P[None, :])[0]
    assert np.max(np.abs(curl - B)) / np.max(np.abs(B)) < 1e-8


def test_biot_savart_matches_analytic_loop():
    """On-axis field of a circular loop: mu0 I a^2 / (2 (a^2+z^2)^{3/2})."""
    from tokviz.rmp_coils import biot_savart, MU0
    a = 1.0
    t = np.linspace(0.0, 2 * np.pi, 400)
    loop = np.stack([a * np.cos(t), a * np.sin(t), np.zeros_like(t)], axis=-1)
    zs = np.array([0.0, 0.5, 1.0, 2.0])
    P = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=-1)
    Bz = biot_savart([(loop, 1.0)], P)[:, 2]
    ana = MU0 * a ** 2 / (2 * (a ** 2 + zs ** 2) ** 1.5)
    assert np.max(np.abs(Bz / ana - 1.0)) < 1e-3


def test_coil_backend_is_divergence_free():
    """The coil backend must keep div B at roundoff.

    Holding B on a grid instead of the vector potential gave a relative
    |div B| of 3.2e-2 overall and 2.1 at the plasma edge -- and that error
    does not look like an error, it looks like chaos.
    """
    from tokviz.rmp_coils import icoil_array
    from tokviz.coilfield import CoilPerturbation
    c = setup()
    cp = CoilPerturbation(c["eq"], icoil_array(n_toroidal=1, current=1.0),
                          n_keep=(1, 3, 5, 7), n_phi=48, grid_shape=(97, 97))
    rep = cp.divergence_report(n_sample=12)
    assert rep["relative_rms"] < 1e-3, rep


# ---------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not have_data():
        print(f"SKIP: no shot data at {SHOT}")
        print("      set TOKVIZ_DATA_DIR to a directory holding the parquet files")
        return 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
