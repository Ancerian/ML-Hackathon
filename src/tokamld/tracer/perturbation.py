"""tokamld.tracer.perturbation — Resonant helical magnetic perturbations in JAX.

Implements non-axisymmetric resonant helical perturbations:
    delta_psi_N(R, Z, phi) = sum_k A_k * f_env(psi_N) * cos(m_k * theta* - n_k * phi + alpha_k)

Using smooth de Moivre expansion:
    cos(m*theta* - n*phi + alpha) = cos(m*theta*) * cos(n*phi - alpha) + sin(m*theta*) * sin(n*phi - alpha)
with cos(m*theta*) and sin(m*theta*) constructed directly from (cos theta*, sin theta*) fields,
completely avoiding the 2pi branch cut of theta*(R, Z).

The perturbation envelope f_env(psi_N) is smoothly tapered to 0 for psi_N >= 0.95
to avoid logarithmic singularities at the divertor X-point.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple
import jax
import jax.numpy as jnp


@jax.jit
def smooth_envelope(psi_n: jnp.ndarray, s_cutoff: float = 0.90, s_max: float = 0.95) -> jnp.ndarray:
    """C1 smooth envelope tapering to 0 at s_max to protect against X-point divergence."""
    # psi_n <= s_cutoff -> 1.0
    # s_cutoff < psi_n < s_max -> cos^2(pi/2 * (psi_n - s_cutoff) / (s_max - s_cutoff))
    # psi_n >= s_max -> 0.0
    xi = (psi_n - s_cutoff) / (s_max - s_cutoff)
    taper = jnp.cos(0.5 * jnp.pi * jnp.clip(xi, 0.0, 1.0)) ** 2
    return jnp.where(psi_n <= s_cutoff, 1.0, jnp.where(psi_n >= s_max, 0.0, taper))


@dataclasses.dataclass(frozen=True)
class ResonantMode:
    """Single resonant Fourier harmonic (m, n, amp, phase)."""

    m: int              # Poloidal mode number
    n: int              # Toroidal mode number
    amp: float          # Amplitude as fraction of total poloidal flux range |psi_bdy - psi_axis|
    phase: float = 0.0  # Phase offset alpha (radians)


class HelicalPerturbation:
    """Multi-mode resonant helical perturbation container in JAX."""

    def __init__(self, modes: List[ResonantMode], s_cutoff: float = 0.90, s_max: float = 0.95):
        self.modes = tuple(modes)
        self.s_cutoff = float(s_cutoff)
        self.s_max = float(s_max)
        self.m_arr = jnp.array([m.m for m in modes], dtype=jnp.int32)
        self.n_arr = jnp.array([m.n for m in modes], dtype=jnp.int32)
        self.amp_arr = jnp.array([m.amp for m in modes], dtype=jnp.float64)
        self.phase_arr = jnp.array([m.phase for m in modes], dtype=jnp.float64)

    def is_empty(self) -> bool:
        return len(self.modes) == 0

    def delta_psi_n(
        self,
        cos_th: jnp.ndarray,
        sin_th: jnp.ndarray,
        psi_n: jnp.ndarray,
        phi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate delta_psi_N (normalized flux perturbation) at given angle and toroidal phi."""
        if self.is_empty():
            return jnp.zeros_like(psi_n)

        env = smooth_envelope(psi_n, self.s_cutoff, self.s_max)
        # Complex exponential z = cos_th + i*sin_th
        z = cos_th + 1j * sin_th

        tot = 0.0
        for m, n, amp, phase in zip(self.m_arr, self.n_arr, self.amp_arr, self.phase_arr):
            zm = z ** m
            c_m = jnp.real(zm)
            s_m = jnp.imag(zm)
            v = n * phi - phase
            c_v = jnp.cos(v)
            s_v = jnp.sin(v)
            tot = tot + amp * (c_m * c_v + s_m * s_v)

        return env * tot

    def delta_h_canonical(
        self,
        psi_t: jnp.ndarray,
        theta_star: jnp.ndarray,
        phi: jnp.ndarray,
        psi_n_of_psi_t_fn,
        dpsi: float,
    ) -> jnp.ndarray:
        """Evaluate delta_H(psi_t, theta*, phi) in canonical magnetic coordinates."""
        if self.is_empty():
            return jnp.zeros_like(psi_t)

        psi_n = psi_n_of_psi_t_fn(psi_t)
        env = smooth_envelope(psi_n, self.s_cutoff, self.s_max)

        tot = 0.0
        for m, n, amp, phase in zip(self.m_arr, self.n_arr, self.amp_arr, self.phase_arr):
            angle = m * theta_star - n * phi + phase
            tot = tot + amp * jnp.cos(angle)

        return abs(dpsi) * env * tot
