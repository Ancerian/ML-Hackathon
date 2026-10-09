"""tokamld.tracer.bfield — Bicubic magnetic field evaluator and geometric profiles in JAX.

Evaluates poloidal flux psi(R, Z), grad(psi), and straight-field-line angle
trigonometric fields (cos theta*, sin theta*) using C1 bicubic Catmull-Rom patches.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def bicubic_eval_scalar(
    grid: jnp.ndarray,
    r: jnp.ndarray,
    z: jnp.ndarray,
    r0: float,
    z0: float,
    dr: float,
    dz: float,
    nr: int,
    nz: int,
) -> jnp.ndarray:
    """Evaluate 2D scalar field on regular grid using Catmull-Rom bicubic interpolation."""
    r_norm = (r - r0) / dr
    z_norm = (z - z0) / dz
    ir = jax.lax.stop_gradient(jnp.clip(jnp.floor(r_norm).astype(jnp.int32), 1, nr - 3))
    iz = jax.lax.stop_gradient(jnp.clip(jnp.floor(z_norm).astype(jnp.int32), 1, nz - 3))
    u = r_norm - ir
    v = z_norm - iz

    bu = 0.5 * jnp.array([
        -u + 2.0 * u**2 - u**3,
        2.0 - 5.0 * u**2 + 3.0 * u**3,
        u + 4.0 * u**2 - 3.0 * u**3,
        -u**2 + u**3
    ])
    bv = 0.5 * jnp.array([
        -v + 2.0 * v**2 - v**3,
        2.0 - 5.0 * v**2 + 3.0 * v**3,
        v + 4.0 * v**2 - 3.0 * v**3,
        -v**2 + v**3
    ])

    patch = jax.lax.dynamic_slice(grid, (iz - 1, ir - 1), (4, 4))
    return jnp.dot(bv, jnp.dot(patch, bu))


@jax.jit
def bicubic_eval_and_grad_scalar(
    grid: jnp.ndarray,
    r: jnp.ndarray,
    z: jnp.ndarray,
    r0: float,
    z0: float,
    dr: float,
    dz: float,
    nr: int,
    nz: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate 2D scalar field and its analytic (d/dR, d/dZ) spatial derivatives."""
    r_norm = (r - r0) / dr
    z_norm = (z - z0) / dz
    ir = jax.lax.stop_gradient(jnp.clip(jnp.floor(r_norm).astype(jnp.int32), 1, nr - 3))
    iz = jax.lax.stop_gradient(jnp.clip(jnp.floor(z_norm).astype(jnp.int32), 1, nz - 3))
    u = r_norm - ir
    v = z_norm - iz

    bu = 0.5 * jnp.array([
        -u + 2.0 * u**2 - u**3,
        2.0 - 5.0 * u**2 + 3.0 * u**3,
        u + 4.0 * u**2 - 3.0 * u**3,
        -u**2 + u**3
    ])
    dbu = 0.5 * jnp.array([
        -1.0 + 4.0 * u - 3.0 * u**2,
        -10.0 * u + 9.0 * u**2,
        1.0 + 8.0 * u - 9.0 * u**2,
        -2.0 * u + 3.0 * u**2
    ]) / dr

    bv = 0.5 * jnp.array([
        -v + 2.0 * v**2 - v**3,
        2.0 - 5.0 * v**2 + 3.0 * v**3,
        v + 4.0 * v**2 - 3.0 * v**3,
        -v**2 + v**3
    ])
    dbv = 0.5 * jnp.array([
        -1.0 + 4.0 * v - 3.0 * v**2,
        -10.0 * v + 9.0 * v**2,
        1.0 + 8.0 * v - 9.0 * v**2,
        -2.0 * v + 3.0 * v**2
    ]) / dz

    patch = jax.lax.dynamic_slice(grid, (iz - 1, ir - 1), (4, 4))
    val = jnp.dot(bv, jnp.dot(patch, bu))
    dval_dr = jnp.dot(bv, jnp.dot(patch, dbu))
    dval_dz = jnp.dot(dbv, jnp.dot(patch, bu))
    return val, dval_dr, dval_dz


@dataclasses.dataclass
class MagneticField2D:
    """Axisymmetric equilibrium magnetic field container with straight-field-line angle grids."""

    psi_grid: jnp.ndarray       # shape (Nz, Nr), float64
    cos_th_grid: jnp.ndarray    # shape (Nz, Nr), float64
    sin_th_grid: jnp.ndarray    # shape (Nz, Nr), float64
    r_grid: jnp.ndarray         # shape (Nr,), float64
    z_grid: jnp.ndarray         # shape (Nz,), float64
    f_pol: float                # F = R * B_phi (scalar)
    psi_axis: float
    psi_bdy: float
    r_axis: float
    z_axis: float
    # q profile spline interpolation tables
    s_q_tab: jnp.ndarray        # shape (Nq,), psi_n grid
    q_tab: jnp.ndarray          # shape (Nq,), q values
    psi_t_tab: jnp.ndarray      # shape (Nq,), psi_t values (toroidal flux)

    r0: float
    z0: float
    dr: float
    dz: float
    nr: int
    nz: int
    dpsi: float

    def psi_at(self, r: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate psi(R, Z) at coordinate (R, Z)."""
        return bicubic_eval_scalar(
            self.psi_grid, r, z, self.r0, self.z0, self.dr, self.dz, self.nr, self.nz
        )

    def psi_n_at(self, r: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate normalized poloidal flux psi_N(R, Z) in [0, 1]."""
        p = self.psi_at(r, z)
        return (p - self.psi_axis) / self.dpsi

    def grad_psi_at(self, r: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate analytic (dpsi/dR, dpsi/dZ) at (R, Z)."""
        _, dp_dr, dp_dz = bicubic_eval_and_grad_scalar(
            self.psi_grid, r, z, self.r0, self.z0, self.dr, self.dz, self.nr, self.nz
        )
        return dp_dr, dp_dz

    def cos_sin_th_at(self, r: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate (cos theta*, sin theta*) at (R, Z) via bicubic patches (no branch cuts!)."""
        c = bicubic_eval_scalar(
            self.cos_th_grid, r, z, self.r0, self.z0, self.dr, self.dz, self.nr, self.nz
        )
        s = bicubic_eval_scalar(
            self.sin_th_grid, r, z, self.r0, self.z0, self.dr, self.dz, self.nr, self.nz
        )
        norm = jnp.sqrt(c**2 + s**2 + 1e-15)
        return c / norm, s / norm

    def q_at_psin(self, psi_n: jnp.ndarray) -> jnp.ndarray:
        """Evaluate q(psi_N) using 1D linear interpolation."""
        return jnp.interp(psi_n, self.s_q_tab, self.q_tab)

    def psi_t_at_psin(self, psi_n: jnp.ndarray) -> jnp.ndarray:
        """Evaluate toroidal flux psi_t(psi_N) using 1D interpolation."""
        return jnp.interp(psi_n, self.s_q_tab, self.psi_t_tab)

    def psin_at_psi_t(self, psi_t: jnp.ndarray) -> jnp.ndarray:
        """Invert toroidal flux to find normalized flux psi_N(psi_t)."""
        return jnp.interp(psi_t, self.psi_t_tab, self.s_q_tab)

    @classmethod
    def from_tokviz_equilibrium(
        cls, eq, n_s: int = 50, n_theta: int = 128
    ) -> "MagneticField2D":
        """Construct MagneticField2D from a tokviz.equilibrium.Equilibrium instance."""
        from scipy.interpolate import RectBivariateSpline

        s_tab, th_tab, d_tab = eq.theta_star_geometry(
            levels=np.linspace(0.05, 0.95, n_s), n_theta=n_theta
        )

        pad = 8
        th = np.asarray(th_tab, float)
        dth = float(th[1] - th[0])
        left = th[:1] - dth * np.arange(pad, 0, -1)
        right = th[-1] + dth * np.arange(1, pad + 1)
        th_ext = np.concatenate([left, th, right])
        d_ext = np.concatenate([d_tab[:, -pad:], d_tab, d_tab[:, :pad]], axis=1)
        spl_delta = RectBivariateSpline(s_tab, th_ext, d_ext, kx=3, ky=3, s=0)

        rr, zz = np.meshgrid(eq.R, eq.Z)
        psi_grid = eq.psi_at(rr, zz)
        dpsi_val = eq.psi_bdy - eq.psi_axis
        psi_n_grid = (psi_grid - eq.psi_axis) / dpsi_val
        tg = np.mod(np.arctan2(zz - eq.z_axis, rr - eq.r_axis), 2.0 * np.pi)

        sn_clip = np.clip(psi_n_grid, s_tab[0], s_tab[-1])
        delta_grid = spl_delta.ev(sn_clip, tg)
        th_star = tg + delta_grid

        cos_th_grid = np.cos(th_star)
        sin_th_grid = np.sin(th_star)

        # Tabulate q(psi_n) and compute toroidal flux psi_t = int q(psi) dpsi
        s_q_arr = np.linspace(0.02, 0.98, 100)
        q_arr = np.array([float(eq.q_at(float(s))) for s in s_q_arr])
        # Toroidal flux per radian: psi_t = int_0^s q(s') * dpsi ds'
        # in units of (psi_bdy - psi_axis)
        q_mid = 0.5 * (q_arr[:-1] + q_arr[1:])
        ds = s_q_arr[1] - s_q_arr[0]
        psi_t_cum = np.concatenate([[0.0], np.cumsum(q_mid * ds)]) * abs(dpsi_val)

        return cls(
            psi_grid=jnp.array(psi_grid, dtype=jnp.float64),
            cos_th_grid=jnp.array(cos_th_grid, dtype=jnp.float64),
            sin_th_grid=jnp.array(sin_th_grid, dtype=jnp.float64),
            r_grid=jnp.array(eq.R, dtype=jnp.float64),
            z_grid=jnp.array(eq.Z, dtype=jnp.float64),
            f_pol=float(eq.F),
            psi_axis=float(eq.psi_axis),
            psi_bdy=float(eq.psi_bdy),
            r_axis=float(eq.r_axis),
            z_axis=float(eq.z_axis),
            s_q_tab=jnp.array(s_q_arr, dtype=jnp.float64),
            q_tab=jnp.array(q_arr, dtype=jnp.float64),
            psi_t_tab=jnp.array(psi_t_cum, dtype=jnp.float64),
            r0=float(eq.R[0]),
            z0=float(eq.Z[0]),
            dr=float(eq.R[1] - eq.R[0]),
            dz=float(eq.Z[1] - eq.Z[0]),
            nr=int(len(eq.R)),
            nz=int(len(eq.Z)),
            dpsi=float(dpsi_val),
        )
