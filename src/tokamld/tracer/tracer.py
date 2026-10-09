"""tokamld.tracer.tracer — Batched field-line tracing, Poincare punctures, and FTLE diagnostics in JAX."""
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp

from .bfield import MagneticField2D
from .perturbation import HelicalPerturbation
from .integrators import rk4_step_cartesian, midpoint_step_canonical

TWO_PI = 2.0 * np.pi


def trace_poincare_cartesian(
    states_0: jnp.ndarray,
    n_turns: int,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    margin: float = 0.02,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched Poincare puncture mapping in Cartesian (R, Z, phi) coordinates.

    Args:
        states_0: Initial points shape (N_lines, 2), float64.
        n_turns: Number of toroidal circuits (turns).
        dphi: Toroidal step size (radians).
        bfield: MagneticField2D instance.
        pert: Optional HelicalPerturbation instance.
        margin: Vessel domain boundary margin (metres).

    Returns:
        punctures: shape (N_lines, n_turns, 2), (R, Z) coordinates at phi = 2*pi*k.
        alive: shape (N_lines, n_turns), boolean flags indicating alive status.
    """
    n_steps_per_turn = int(round(TWO_PI / abs(dphi)))
    dphi_adj = TWO_PI / float(n_steps_per_turn)

    def trace_single_line(s0):
        def turn_step(carry, turn_idx):
            state, alive, phi_base = carry

            def inner_step(inner_carry, step_idx):
                curr_state, curr_alive = inner_carry
                phi_curr = phi_base + step_idx * dphi_adj
                next_state, next_alive = rk4_step_cartesian(
                    curr_state, phi_curr, dphi_adj, bfield, pert, curr_alive, margin
                )
                return (next_state, next_alive), None

            (final_state, final_alive), _ = jax.lax.scan(
                inner_step, (state, alive), jnp.arange(n_steps_per_turn)
            )
            next_phi_base = phi_base + TWO_PI
            return (final_state, final_alive, next_phi_base), (final_state, final_alive)

        init_carry = (s0, jnp.array(True), 0.0)
        _, (punctures, alives) = jax.lax.scan(
            turn_step, init_carry, jnp.arange(n_turns)
        )
        return punctures, alives

    return jax.vmap(trace_single_line)(states_0)


def trace_poincare_canonical(
    z_0: jnp.ndarray,
    n_turns: int,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    max_iter: int = 8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched Poincare puncture mapping in canonical (psi_t, theta*) coordinates using Implicit Midpoint."""
    n_steps_per_turn = int(round(TWO_PI / abs(dphi)))
    dphi_adj = TWO_PI / float(n_steps_per_turn)

    def trace_single_line(z0):
        def turn_step(carry, turn_idx):
            z_curr, phi_base = carry

            def inner_step(inner_carry, step_idx):
                z_st = inner_carry
                phi_curr = phi_base + step_idx * dphi_adj
                z_next, res = midpoint_step_canonical(
                    z_st, phi_curr, dphi_adj, bfield, pert, max_iter=max_iter
                )
                return z_next, res

            final_z, res_list = jax.lax.scan(
                inner_step, z_curr, jnp.arange(n_steps_per_turn)
            )
            next_phi_base = phi_base + TWO_PI
            max_res = jnp.max(res_list)
            return (final_z, next_phi_base), (final_z, max_res)

        init_carry = (z0, 0.0)
        _, (punctures, residuals) = jax.lax.scan(
            turn_step, init_carry, jnp.arange(n_turns)
        )
        return punctures, residuals

    return jax.vmap(trace_single_line)(z_0)


def trace_trajectories_cartesian(
    states_0: jnp.ndarray,
    n_total_steps: int,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    margin: float = 0.02,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Trace continuous 3D field-line trajectories (R(phi), Z(phi), phi) for rendering."""
    def trace_single(s0):
        def step(carry, step_idx):
            state, alive = carry
            phi_curr = step_idx * dphi
            next_state, next_alive = rk4_step_cartesian(
                state, phi_curr, dphi, bfield, pert, alive, margin
            )
            return (next_state, next_alive), (next_state, next_alive)

        _, (traj, alives) = jax.lax.scan(
            step, (s0, jnp.array(True)), jnp.arange(n_total_steps)
        )
        return traj, alives

    return jax.vmap(trace_single)(states_0)


def compute_ftle_cartesian(
    state_0: jnp.ndarray,
    n_turns: int,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Finite-Time Lyapunov Exponent (FTLE) for a single initial condition in Cartesian coords."""
    d0 = jnp.array([eps, 0.0], dtype=jnp.float64)
    states = jnp.stack([state_0, state_0 + d0], axis=0)

    punctures, alive = trace_poincare_cartesian(states, n_turns, dphi, bfield, pert)
    p0 = punctures[0, -1]
    p1 = punctures[1, -1]

    dist_t = jnp.linalg.norm(p1 - p0)
    t_tot = float(n_turns * TWO_PI)
    ftle = (1.0 / t_tot) * jnp.log(jnp.maximum(dist_t / eps, 1e-15))
    return float(ftle)


def compute_ftle_canonical(
    z_0: jnp.ndarray,
    n_turns: int,
    dphi: float,
    bfield: MagneticField2D,
    pert: Optional[HelicalPerturbation] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Finite-Time Lyapunov Exponent (FTLE) in canonical coordinates (psi_t, theta*)."""
    d0 = jnp.array([eps, 0.0], dtype=jnp.float64)
    z_pair = jnp.stack([z_0, z_0 + d0], axis=0)

    punctures, res = trace_poincare_canonical(z_pair, n_turns, dphi, bfield, pert)
    p0 = punctures[0, -1]
    p1 = punctures[1, -1]

    # Handle periodic theta* angle difference
    d_pt = p1[0] - p0[0]
    d_th = jnp.remainder(p1[1] - p0[1] + jnp.pi, TWO_PI) - jnp.pi
    dist_t = jnp.hypot(d_pt, d_th)

    t_tot = float(n_turns * TWO_PI)
    ftle = (1.0 / t_tot) * jnp.log(jnp.maximum(dist_t / eps, 1e-15))
    return float(ftle)
