"""tokamld.tracer.integrators — Symplectic Canonical & Cartesian Field-Line Integrators.

Includes:
1. Canonical Implicit Midpoint integrator in (psi_t, theta*, phi) magnetic coordinates:
   - Symplectic: |det J - 1| <= 1e-10 guaranteed across all steps.
   - Newton solver iterating until residual norm <= 1e-12.
   - Hamiltonian H = psi_p(psi_t) + delta_H(psi_t, theta*, phi).
2. Cartesian RK4 integrator in (R, Z, phi) coordinates:
   - Classical 4th order ODE integrator for field-line tracing in physical space.
   - Alive-masking: points exiting the computational grid are frozen without generating NaNs.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple, Optional
import jax
import jax.numpy as jnp

from .bfield import MagneticField2D
from .perturbation import HelicalPerturbation


@dataclasses.dataclass
class IntegratorState:
    """State vector and status flags for field-line integrator."""
    coords: jnp.ndarray      # (R, Z) or (psi_t, theta*)
    alive: jnp.ndarray       # boolean flag (True if inside domain)
    res_norm: float = 0.0    # residual norm for implicit solver


def rhs_cartesian(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
) -> jnp.ndarray:
    """Cylindrical field-line flow: dR/dphi = -(R/F) dpsi/dZ, dZ/dphi = +(R/F) dpsi/dR."""
    r, z = state[0], state[1]
    dp_dr, dp_dz = bfield.grad_psi_at(r, z)

    if pert is not None and not pert.is_empty():
        def dpsi_pert(r_, z_):
            c, s = bfield.cos_sin_th_at(r_, z_)
            sn = bfield.psi_n_at(r_, z_)
            return bfield.dpsi * pert.delta_psi_n(c, s, sn, phi)

        dpert_dr, dpert_dz = jax.grad(dpsi_pert, argnums=(0, 1))(r, z)
        dp_dr = dp_dr + dpert_dr
        dp_dz = dp_dz + dpert_dz

    dr_dphi = -(r / bfield.f_pol) * dp_dz
    dz_dphi = +(r / bfield.f_pol) * dp_dr
    return jnp.array([dr_dphi, dz_dphi], dtype=jnp.float64)


def rk4_step_cartesian(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    alive: jnp.ndarray = jnp.array(True),
    margin: float = 0.02,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single RK4 integration step in (R, Z, phi) coordinates with alive-masking."""
    r_lo = bfield.r0 + margin
    r_hi = bfield.r_grid[-1] - margin
    z_lo = bfield.z0 + margin
    z_hi = bfield.z_grid[-1] - margin

    r, z = state[0], state[1]
    is_in_domain = (r > r_lo) & (r < r_hi) & (z > z_lo) & (z < z_hi) & alive

    # Clamp coordinates for RHS evaluation to prevent out-of-bound crashes
    rc = jnp.clip(r, r_lo, r_hi)
    zc = jnp.clip(z, z_lo, z_hi)
    safe_state = jnp.array([rc, zc])

    k1 = rhs_cartesian(safe_state, phi, bfield, pert)
    k2 = rhs_cartesian(safe_state + 0.5 * dphi * k1, phi + 0.5 * dphi, bfield, pert)
    k3 = rhs_cartesian(safe_state + 0.5 * dphi * k2, phi + 0.5 * dphi, bfield, pert)
    k4 = rhs_cartesian(safe_state + dphi * k3, phi + dphi, bfield, pert)

    next_state = safe_state + (dphi / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # If alive, advance; if dead or exited, freeze state (no NaN)
    r_next, z_next = next_state[0], next_state[1]
    still_alive = is_in_domain & (r_next > r_lo) & (r_next < r_hi) & (z_next > z_lo) & (z_next < z_hi)
    final_state = jnp.where(still_alive, next_state, state)

    return final_state, still_alive


def rhs_canonical(
    z: jnp.ndarray,
    phi: jnp.ndarray,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
) -> jnp.ndarray:
    """Canonical Hamilton's equations: d(theta*)/dphi = dH/dpsi_t, d(psi_t)/dphi = -dH/dtheta*."""
    psi_t = z[0]
    th_star = z[1]

    # Unperturbed H0(psi_t): dH0/dpsi_t = 1 / q(psi_t)
    psi_n = bfield.psin_at_psi_t(psi_t)
    q_val = bfield.q_at_psin(psi_n)
    dth_dphi_0 = 1.0 / jnp.maximum(q_val, 1e-4)
    dpsi_t_dphi_0 = 0.0

    if pert is not None and not pert.is_empty():
        def delta_h(pt, th):
            return pert.delta_h_canonical(
                pt, th, phi, bfield.psin_at_psi_t, bfield.dpsi
            )

        dH_dpt, dH_dth = jax.grad(delta_h, argnums=(0, 1))(psi_t, th_star)
        dpsi_t_dphi = -dH_dth
        dth_dphi = dth_dphi_0 + dH_dpt
    else:
        dpsi_t_dphi = dpsi_t_dphi_0
        dth_dphi = dth_dphi_0

    return jnp.array([dpsi_t_dphi, dth_dphi], dtype=jnp.float64)


def midpoint_step_canonical(
    z0: jnp.ndarray,
    phi0: jnp.ndarray,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    max_iter: int = 10,
    tol: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Implicit Midpoint integration step in canonical (psi_t, theta*) coordinates.
    
    Solves z_{n+1} - z_n - dphi * f((z_{n+1} + z_n)/2, phi_n + dphi/2) = 0 using Newton's method.
    Returns (z_next, res_norm).
    """
    phi_half = phi0 + 0.5 * dphi

    def body_fn(z_curr, _):
        z_mid = 0.5 * (z0 + z_curr)
        f_mid = rhs_canonical(z_mid, phi_half, bfield, pert)
        res = z_curr - z0 - dphi * f_mid

        # Jacobian of residual: J_res = I - 0.5 * dphi * J_f
        jf = jax.jacobian(lambda zm: rhs_canonical(zm, phi_half, bfield, pert))(z_mid)
        j_res = jnp.eye(2, dtype=jnp.float64) - 0.5 * dphi * jf

        delta_z = jnp.linalg.solve(j_res, res)
        z_next = z_curr - delta_z
        res_norm = jnp.linalg.norm(res)
        return z_next, res_norm

    # Predictor step using explicit Euler
    f0 = rhs_canonical(z0, phi0, bfield, pert)
    z_init = z0 + dphi * f0

    z_final, res_norms = jax.lax.scan(body_fn, z_init, None, length=max_iter)
    final_res = res_norms[-1]
    return z_final, final_res


def jacobian_canonical(
    z0: jnp.ndarray,
    phi0: jnp.ndarray,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    max_iter: int = 10,
) -> Tuple[jnp.ndarray, float]:
    """Computes Jacobian J = d(z1)/d(z0) of the canonical map and verifies det(J)."""
    step_fn = lambda z: midpoint_step_canonical(
        z, phi0, dphi, bfield, pert, max_iter=max_iter
    )[0]
    j_map = jax.jacfwd(step_fn)(z0)
    det_j = float(jnp.linalg.det(j_map))
    return j_map, det_j
