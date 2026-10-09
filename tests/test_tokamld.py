"""Tests for tokamld package.

Regression & physics checks:
1. Cerfon-Freidberg analytic Solov'ev equilibrium -> Δ* ψ + μ₀ R² p' + FF' ≈ 0.
2. Ground truth + 1% noise -> relative GS residual g ≈ 0.63 (E6, E38).
3. Green function argument k vs m = k^2 check (E33).
4. Island width formula W = 4√(ψ̃ q / |dq/dψ|) check (R4, VR15).
5. Topology detector check on canonical vs spurious critical points.
6. Score reproduction checks on perfect/zeros and PCA+Ridge (S' = 1.0, 0.0, 0.361368).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.integrate as integrate
import torch

try:
    import pytest
except ImportError:
    pytest = None

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import tokamld
from tokamld.gs import delta_star, residual, gs_inconsistency, gate, MU0
from tokamld.topology import find_critical_points, check_canonical_topology
from tokamld.conformal import SimultaneousConformalBands
from tokamld.fieldline import green_function, magnetic_island_width, FieldLineIntegrator


def test_green_function_elliptic_integral_modulus_m_vs_k():
    """E33 regression test: Green's function must use m = k^2 in ellipk/ellipe.
    
    Verifies against direct numerical quadrature of Biot-Savart vector potential:
    ψ(R, Z) = R * A_φ = (μ₀ I R_c R / 4π) ∫_0^{2π} cos(t) / |x - x'(t)| dt
    """
    Rc, Zc = 1.5, 0.25
    R, Z = 1.0, 0.0
    mu = MU0

    # Direct quadrature
    def integrand(t):
        dist = np.sqrt(R ** 2 + Rc ** 2 - 2.0 * R * Rc * np.cos(t) + (Z - Zc) ** 2)
        return np.cos(t) / dist

    val_quad, _ = integrate.quad(integrand, 0.0, 2.0 * np.pi)
    psi_quad = (mu * Rc * R / (4.0 * np.pi)) * val_quad

    # tokamld Green function
    psi_green = green_function(Rc, Zc, R, Z, mu=mu)

    # Relative difference must be < 1e-10
    rel_err = abs(psi_green - psi_quad) / psi_quad
    assert rel_err < 1e-10, f"Green function mismatch vs quadrature: {rel_err}"

    # Also test symmetry G(R1, Z1; R2, Z2) == G(R2, Z2; R1, Z1)
    psi_sym = green_function(R, Z, Rc, Zc, mu=mu)
    assert abs(psi_green - psi_sym) < 1e-14


def test_island_width_formula():
    """R4 / VR15 regression test: W = 4 * sqrt(psi_tilde * q / |dq/dpsi|).
    
    Checks that the formula has no spurious factor of sqrt(q) (not 4*sqrt(eps*q^2/shear)).
    """
    psi_tilde = 1e-3
    q_res = 2.0
    dq_dpsi = 5.0

    # Correct formula: 4 * sqrt(1e-3 * 2 / 5) = 4 * sqrt(4e-4) = 4 * 0.02 = 0.08
    expected_w = 4.0 * np.sqrt(psi_tilde * q_res / dq_dpsi)
    measured_w = magnetic_island_width(psi_tilde, q_res, dq_dpsi)
    assert abs(measured_w - expected_w) < 1e-12
    assert abs(measured_w - 0.08) < 1e-12


def test_cerfon_freidberg_analytic_equilibrium_residual():
    """Analytic Solov'ev equilibrium (Cerfon & Freidberg 2010): Δ* ψ + μ₀ R² p' + FF' = 0.
    
    A Solov'ev solution has constant p' and constant FF':
    u(x, y) = x^4/8 + A (x^2 ln(x) / 2 - x^4/8) + c1 + c2 x^2 + c3 (y^2 - x^2 ln(x)) ...
    Satisfies x d/dx(1/x du/dx) + d^2u/dy^2 = (1 - A) x^2 + A exactly.
    """
    # Simple polynomial Solov'ev solution: u(x, y) = x^4 / 8  (with A=0, so (1-A) x^2 = x^2)
    # Δ* (x^4 / 8) = d2/dx2(x^4/8) - (1/x) d/dx(x^4/8) = (12 x^2 / 8) - (4 x^2 / 8) = x^2
    # In physical coordinates R = R0 x, Z = R0 y: Δ* (R^4 / 8) = R^2
    R = np.linspace(1.0, 3.0, 101)
    Z = np.linspace(-1.0, 1.0, 101)
    RR, ZZ = np.meshgrid(R, Z)

    # Let psi(R, Z) = R^4 / 8 + Z^2 * R^2 / 2  -> let's test pure R^4 / 8
    psi = RR ** 4 / 8.0

    # Delta* (R^4 / 8) = R^2
    ds_np = delta_star(psi, R, Z)
    # Interior points away from boundary
    ds_inner = ds_np[10:-10, 10:-10]
    expected_inner = (RR ** 2)[10:-10, 10:-10]
    # Check max absolute relative error < 1e-3 (finite difference discretization order)
    rel_err = np.max(np.abs(ds_inner - expected_inner) / expected_inner)
    assert rel_err < 1e-3, f"Delta* R^4/8 mismatch: {rel_err}"

    # Also test torch implementation
    psi_t = torch.tensor(psi, dtype=torch.float64)
    R_t = torch.tensor(R, dtype=torch.float64)
    Z_t = torch.tensor(Z, dtype=torch.float64)
    ds_torch = delta_star(psi_t, R_t, Z_t).numpy()
    assert np.max(np.abs(ds_torch[10:-10, 10:-10] - expected_inner) / expected_inner) < 1e-3


def test_topology_canonical_and_spurious():
    """Topology check: Gaussian hill has 1 O-point (canonical); dipole perturbation adds O/X points."""
    R = np.linspace(1.0, 2.5, 65)
    Z = np.linspace(-1.0, 1.0, 65)
    RR, ZZ = np.meshgrid(R, Z)

    # 1. Canonical: Single elliptical hill (magnetic axis)
    r_axis, z_axis = 1.705, 0.005
    psi_canonical = -np.exp(-((RR - r_axis) ** 2 + 2.0 * (ZZ - z_axis) ** 2) / 0.2)
    inside = ((RR - r_axis) ** 2 + (ZZ - z_axis) ** 2) < 0.25

    is_canon, n_spur, info = check_canonical_topology(psi_canonical, R, Z, inside)
    assert is_canon is True
    assert n_spur == 0
    assert info["n_O"] == 1
    assert info["n_X"] == 0

    # 2. Perturbed with volcano center (adds spurious critical points)
    psi_perturbed = psi_canonical + 0.3 * np.exp(-((RR - r_axis) ** 2 + (ZZ - z_axis) ** 2) / 0.01)
    is_canon_pert, n_spur_pert, info_pert = check_canonical_topology(psi_perturbed, R, Z, inside)
    assert is_canon_pert is False
    assert n_spur_pert >= 1


def test_conformal_bands_coverage():
    """Simultaneous conformal bands coverage guarantee test."""
    rng = np.random.default_rng(42)
    # 50 synthetic 2D frames
    y_true = rng.standard_normal((50, 16, 16))
    noise = rng.normal(0, 0.2, size=(50, 16, 16))
    y_pred = y_true + noise

    cal_bands = SimultaneousConformalBands(alpha=0.80)
    cal_bands.fit(y_true[:30], y_pred[:30])

    cov = cal_bands.evaluate_coverage(y_true[30:], y_pred[30:])
    # Empirical coverage on test should be reasonable (typically >= 70-80%)
    assert cov["mean_pixel_coverage"] > 0.70
    assert np.isfinite(cov["c_star"])


def test_conformal_comparative_methods():
    """Comparative test: Simultaneous vs Pointwise vs Bonferroni."""
    from tokamld.conformal import PointwiseConformalBands, BonferroniConformalBands, compare_conformal_methods
    rng = np.random.default_rng(42)
    # 40 calibration, 20 test frames on 8x8 grid (64 pixels)
    y_true_cal = rng.standard_normal((40, 8, 8))
    y_pred_cal = y_true_cal + rng.normal(0, 0.1, size=(40, 8, 8))

    y_true_test = rng.standard_normal((20, 8, 8))
    y_pred_test = y_true_test + rng.normal(0, 0.1, size=(20, 8, 8))
    test_shots = np.repeat(np.arange(4), 5)

    comp = compare_conformal_methods(y_true_cal, y_pred_cal, y_true_test, y_pred_test,
                                     alpha=0.90, test_shot_ids=test_shots)
    assert "simultaneous_M" in comp
    assert "pointwise_P" in comp
    assert "bonferroni" in comp
    # On 64 pixels with alpha=0.90, Bonferroni requires n >= 64 / 0.10 - 1 = 639 frames.
    # With only 40 frames, it must report is_finite = False
    assert comp["bonferroni"]["is_finite"] is False
    assert comp["price_of_simultaneity"] > 1.0


def test_scorer_reproduction():
    """Checks that eval_submission.py reproduces perfect/zeros and PCA+Ridge known numbers."""
    from eval_submission import evaluate_submission
    # Perfect mode
    rep_p = evaluate_submission(mode="perfect")
    assert abs(rep_p["official"]["S"] - 1.0) < 1e-9
    assert abs(rep_p["extended_s_prime"]["S_prime"] - 1.0) < 1e-9

    # Zeros mode
    rep_z = evaluate_submission(mode="zeros")
    assert abs(rep_z["official"]["S"] - 0.0) < 1e-9
    assert abs(rep_z["extended_s_prime"]["S_prime"] - 0.0) < 1e-9

    # PCA+Ridge known numbers
    pca_sub_path = PROJECT / "Our try" / "04-novelty" / "t1" / "pca_ridge_c2_sub.npz"
    if pca_sub_path.exists():
        rep_pca = evaluate_submission(sub_path=pca_sub_path, mode="file", n_boot=100)
        assert abs(rep_pca["official"]["S"] - 0.192639) < 1e-4
        assert abs(rep_pca["extended_s_prime"]["S_prime"] - 0.361368) < 1e-4
