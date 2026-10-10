"""Unit tests for tokamld.tracer package."""
from pathlib import Path
import json
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tokamld.tracer.bfield import bicubic_eval_scalar, bicubic_eval_and_grad_scalar, MagneticField2D
from tokamld.tracer.perturbation import HelicalPerturbation, ResonantMode, smooth_envelope
from tokamld.tracer.integrators import (
    rk4_step_cartesian,
    midpoint_step_canonical,
    jacobian_canonical,
    rhs_cartesian,
)
from tokamld.tracer.tracer import (
    trace_poincare_cartesian,
    trace_poincare_canonical,
    compute_ftle_cartesian,
)
from tokamld.tracer.export_web import export_poincare_json
from tokamld.tracer.wall import WallBoundary


def test_bicubic_eval_and_grad_analytic():
    """Bicubic patch evaluates quadratic test function and derivatives with machine precision."""
    nr, nz = 30, 30
    r0, z0 = 1.0, -1.0
    dr, dz = 0.1, 0.1
    r_grid = np.linspace(r0, r0 + (nr - 1) * dr, nr)
    z_grid = np.linspace(z0, z0 + (nz - 1) * dz, nz)
    rr, zz = np.meshgrid(r_grid, z_grid)
    grid = jnp.array(rr**2 + 2.0 * zz**2, dtype=jnp.float64)

    test_r = 1.85
    test_z = 0.25
    val, dr_num, dz_num = bicubic_eval_and_grad_scalar(grid, test_r, test_z, r0, z0, dr, dz, nr, nz)

    val_exact = test_r**2 + 2.0 * test_z**2
    dr_exact = 2.0 * test_r
    dz_exact = 4.0 * test_z

    assert abs(float(val) - val_exact) < 1e-10
    assert abs(float(dr_num) - dr_exact) < 1e-10
    assert abs(float(dz_num) - dz_exact) < 1e-10


def test_perturbation_smooth_envelope():
    """Perturbation envelope is identically 1 for s <= 0.90, smoothly tapers, and is 0 for s >= 0.95."""
    s_core = jnp.array([0.1, 0.5, 0.7, 0.90])
    s_trans = jnp.array([0.925])
    s_edge = jnp.array([0.95, 0.99, 1.05])

    env_core = smooth_envelope(s_core)
    env_trans = smooth_envelope(s_trans)
    env_edge = smooth_envelope(s_edge)

    assert np.allclose(env_core, 1.0)
    assert 0.0 < float(env_trans[0]) < 1.0
    assert np.allclose(env_edge, 0.0)


def test_canonical_midpoint_symplecticity():
    """Implicit Midpoint preserves symplectic volume: |det J - 1| <= 1e-10."""
    # Build minimal synthetic MagneticField2D
    nr, nz = 20, 20
    r0, z0 = 1.0, -1.0
    dr, dz = 0.1, 0.1
    r_arr = np.linspace(r0, r0 + (nr - 1) * dr, nr)
    z_arr = np.linspace(z0, z0 + (nz - 1) * dz, nz)
    rr, zz = np.meshgrid(r_arr, z_arr)
    psi_synthetic = -0.5 * ((rr - 1.7)**2 + (zz)**2)

    s_q = np.linspace(0.01, 0.99, 50)
    q_synthetic = 1.0 + 2.0 * s_q**2
    psi_t_synthetic = np.linspace(0.0, 1.0, 50)

    bf = MagneticField2D(
        psi_grid=jnp.array(psi_synthetic, dtype=jnp.float64),
        cos_th_grid=jnp.array(np.ones_like(psi_synthetic), dtype=jnp.float64),
        sin_th_grid=jnp.array(np.zeros_like(psi_synthetic), dtype=jnp.float64),
        r_grid=jnp.array(r_arr, dtype=jnp.float64),
        z_grid=jnp.array(z_arr, dtype=jnp.float64),
        f_pol=3.0,
        psi_axis=-0.0,
        psi_bdy=-0.5,
        r_axis=1.7,
        z_axis=0.0,
        s_q_tab=jnp.array(s_q, dtype=jnp.float64),
        q_tab=jnp.array(q_synthetic, dtype=jnp.float64),
        psi_t_tab=jnp.array(psi_t_synthetic, dtype=jnp.float64),
        r0=r0,
        z0=z0,
        dr=dr,
        dz=dz,
        nr=nr,
        nz=nz,
        dpsi=-0.5,
    )

    pert = HelicalPerturbation([ResonantMode(m=2, n=1, amp=1e-3, phase=0.0)])
    z0_test = jnp.array([0.4, 0.5], dtype=jnp.float64)
    dphi = 2.0 * np.pi / 60.0

    z_next, res = midpoint_step_canonical(z0_test, 0.0, dphi, bf, pert)
    assert float(res) <= 1e-12

    _, det_j = jacobian_canonical(z0_test, 0.0, dphi, bf, pert)
    assert abs(det_j - 1.0) <= 1e-10


def test_cartesian_rk4_alive_masking():
    """Field lines leaving computational domain are frozen without producing NaNs."""
    nr, nz = 20, 20
    r0, z0 = 1.0, -1.0
    dr, dz = 0.1, 0.1
    r_arr = np.linspace(r0, r0 + (nr - 1) * dr, nr)
    z_arr = np.linspace(z0, z0 + (nz - 1) * dz, nz)
    rr, zz = np.meshgrid(r_arr, z_arr)
    psi_synthetic = -0.5 * ((rr - 1.7)**2 + (zz)**2)

    bf = MagneticField2D(
        psi_grid=jnp.array(psi_synthetic, dtype=jnp.float64),
        cos_th_grid=jnp.array(np.ones_like(psi_synthetic), dtype=jnp.float64),
        sin_th_grid=jnp.array(np.zeros_like(psi_synthetic), dtype=jnp.float64),
        r_grid=jnp.array(r_arr, dtype=jnp.float64),
        z_grid=jnp.array(z_arr, dtype=jnp.float64),
        f_pol=3.0,
        psi_axis=-0.0,
        psi_bdy=-0.5,
        r_axis=1.7,
        z_axis=0.0,
        s_q_tab=jnp.array(np.linspace(0.01, 0.99, 10), dtype=jnp.float64),
        q_tab=jnp.array(np.linspace(1.0, 3.0, 10), dtype=jnp.float64),
        psi_t_tab=jnp.array(np.linspace(0.0, 1.0, 10), dtype=jnp.float64),
        r0=r0,
        z0=z0,
        dr=dr,
        dz=dz,
        nr=nr,
        nz=nz,
        dpsi=-0.5,
    )

    # Initial condition outside domain
    out_state = jnp.array([0.5, 5.0], dtype=jnp.float64)
    next_st, alive = rk4_step_cartesian(out_state, 0.0, 0.1, bf, pert=None)

    assert bool(alive) is False
    assert not np.isnan(next_st).any()
    assert np.allclose(next_st, out_state)


def test_export_poincare_json(tmp_path):
    """Web JSON export creates structured file with machine tag and surfaces."""
    out_file = tmp_path / "test_poincare.json"
    surfaces = [{"index": 0, "r_init": 1.8, "z_init": 0.0, "punctures": [[1.8, 0.0], [1.79, 0.01]]}]
    p = export_poincare_json(
        output_path=out_file,
        machine="DIII-D",
        shot_id="shot_123",
        q95=3.5,
        f_pol=2.0,
        r_axis=1.7,
        z_axis=0.0,
        surfaces=surfaces,
    )
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["format"] == "tokamld-poincare-v1"
    assert data["machine"] == "DIII-D"
    assert len(data["surfaces"]) == 1


if __name__ == "__main__":
    import tempfile
    test_bicubic_eval_and_grad_analytic()
    print("test_bicubic_eval_and_grad_analytic: PASS")
    test_perturbation_smooth_envelope()
    print("test_perturbation_smooth_envelope: PASS")
    test_canonical_midpoint_symplecticity()
    print("test_canonical_midpoint_symplecticity: PASS")
    test_cartesian_rk4_alive_masking()
    print("test_cartesian_rk4_alive_masking: PASS")
    with tempfile.TemporaryDirectory() as td:
        test_export_poincare_json(Path(td))
    print("test_export_poincare_json: PASS")
    print("All tracer unit tests PASSED successfully!")
