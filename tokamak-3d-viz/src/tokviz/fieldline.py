"""Field-line tracing and the Poincare puncture map.

Field lines of ``B = grad(psi_total) x grad(phi) + (F/R) phi_hat`` satisfy, with
the toroidal angle phi as the independent variable,

    dR/dphi = R B_R / B_phi = -(R / F) d psi_total / dZ
    dZ/dphi = R B_Z / B_phi = +(R / F) d psi_total / dR

which is a one-and-a-half degree of freedom Hamiltonian system with phi as
time, the straight-field-line angle theta* as the coordinate, the TOROIDAL flux
per radian psi_t as the conjugate momentum, and the poloidal flux psi_total as
the Hamiltonian (d psi = d psi_t / q).  psi is the Hamiltonian, not the
momentum; confusing the two overestimates island widths by sqrt(q) -- see
:func:`tokviz.perturbation.island_width_psin`.

In the AXISYMMETRIC case psi_total = psi(R,Z) has no phi dependence, the system
is integrable, and psi is conserved exactly along every line.  That is used
below as an integrator accuracy test (:func:`invariant_drift`) and it is also
the reason a section of an axisymmetric equilibrium reproduces the psi contours
and nothing more.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .equilibrium import Equilibrium, TWO_PI
from .perturbation import Perturbation


class FieldLines:
    """Traceable field-line system: an equilibrium plus an optional perturbation."""

    def __init__(self, eq: Equilibrium, pert: Perturbation | None = None,
                 margin: float = 0.02):
        self.eq = eq
        self.pert = pert
        self.F = eq.F
        self.r_lo = float(eq.R[0]) + margin
        self.r_hi = float(eq.R[-1]) - margin
        self.z_lo = float(eq.Z[0]) + margin
        self.z_hi = float(eq.Z[-1]) - margin

    # -- field --------------------------------------------------------------
    def psi_total(self, r, z, phi):
        v = self.eq.psi_at(r, z)
        if self.pert is not None:
            v = v + self.pert.dpsi(r, z, phi)
        return v

    def B(self, r, z, phi):
        """Cylindrical components ``(B_R, B_Z, B_phi)`` in tesla."""
        r = np.asarray(r, dtype=float)
        dR = self.eq.dpsi_dR(r, z)
        dZ = self.eq.dpsi_dZ(r, z)
        BR, BZ, BP = -dZ / r, dR / r, self.F / r
        if self.pert is not None:
            bR, bZ, bP = self._pert_B(r, z, phi, dR, dZ)
            BR, BZ, BP = BR + bR, BZ + bZ, BP + bP
        return BR, BZ, BP

    def _pert_B(self, r, z, phi, dpsi_dR=None, dpsi_dZ=None):
        """Perturbation field from whichever backend is attached."""
        try:
            return self.pert.delta_B(r, z, phi, dpsi_dR=dpsi_dR, dpsi_dZ=dpsi_dZ)
        except TypeError:
            return self.pert.delta_B(r, z, phi)

    # -- ODE ----------------------------------------------------------------
    def _rhs(self, phi, y):
        n = y.size // 2
        r = y[:n]
        z = y[n:]
        alive = (r > self.r_lo) & (r < self.r_hi) & (z > self.z_lo) & (z < self.z_hi)
        rc = np.clip(r, self.r_lo, self.r_hi)
        zc = np.clip(z, self.z_lo, self.z_hi)

        dR = self.eq.dpsi_dR(rc, zc)
        dZ = self.eq.dpsi_dZ(rc, zc)
        BR, BZ, BP = -dZ / rc, dR / rc, self.F / rc
        if self.pert is not None:
            bR, bZ, bP = self._pert_B(rc, zc, phi, dR, dZ)
            BR, BZ, BP = BR + bR, BZ + bZ, BP + bP

        # dR/dphi = R B_R / B_phi.  Writing it through B rather than through
        # psi alone is what lets a backend contribute a TOROIDAL perturbation:
        # a flux-function perturbation cannot, a real coil array does.
        BP = np.where(np.abs(BP) < 1e-12, np.sign(BP) * 1e-12 + 1e-30, BP)
        drdp = rc * BR / BP
        dzdp = rc * BZ / BP
        # freeze escaped lines so one runaway cannot destabilise the shared
        # adaptive step for every other seed in the batch
        drdp = np.where(alive, drdp, 0.0)
        dzdp = np.where(alive, dzdp, 0.0)
        return np.concatenate([drdp, dzdp])

    def _in_domain(self, r, z):
        return (r > self.r_lo) & (r < self.r_hi) & (z > self.z_lo) & (z < self.z_hi)

    # -- integration --------------------------------------------------------
    def trace(self, r0, z0, phi_eval, rtol=1e-10, atol=1e-12, method="DOP853"):
        """Integrate a batch of seeds to the requested toroidal angles.

        Parameters
        ----------
        r0, z0 : (N,) seed coordinates
        phi_eval : (M,) increasing toroidal angles, starting at 0

        Returns
        -------
        R, Z : (N, M) arrays
        alive : (N,) bool, False where the line left the grid
        """
        r0 = np.atleast_1d(np.asarray(r0, dtype=float))
        z0 = np.atleast_1d(np.asarray(z0, dtype=float))
        phi_eval = np.asarray(phi_eval, dtype=float)
        n = r0.size

        sol = solve_ivp(
            self._rhs,
            (float(phi_eval[0]), float(phi_eval[-1])),
            np.concatenate([r0, z0]),
            t_eval=phi_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        R = sol.y[:n, :]
        Z = sol.y[n:, :]
        alive = self._in_domain(R, Z).all(axis=1)
        return R, Z, alive

    def poincare(self, r0, z0, n_punctures=400, rtol=1e-10, atol=1e-12):
        """Puncture map on the phi = 0 plane.

        Returns ``(R, Z, alive)`` with shape ``(N, n_punctures)``.
        """
        phi = TWO_PI * np.arange(n_punctures, dtype=float)
        return self.trace(r0, z0, phi, rtol=rtol, atol=atol)

    def trace_xyz(self, r0, z0, n_turns=6, pts_per_turn=180, rtol=1e-10, atol=1e-12):
        """Trace and return Cartesian ``(N, M, 3)`` for direct use as curves."""
        phi = np.linspace(0.0, TWO_PI * n_turns, int(n_turns * pts_per_turn) + 1)
        R, Z, alive = self.trace(r0, z0, phi, rtol=rtol, atol=atol)
        X = R * np.cos(phi)[None, :]
        Y = R * np.sin(phi)[None, :]
        return np.stack([X, Y, Z], axis=-1), alive


# ---------------------------------------------------------------------------
# accuracy diagnostics
# ---------------------------------------------------------------------------
def invariant_drift(eq: Equilibrium, r0, z0, n_turns=200, rtol=1e-10, atol=1e-12):
    """Relative drift of psi along axisymmetric field lines.

    psi is an exact invariant when there is no perturbation, so this measures
    integrator error alone.  Returns max |psi - psi_0| / |psi_bdy - psi_axis|.
    """
    fl = FieldLines(eq, None)
    phi = np.linspace(0.0, TWO_PI * n_turns, 40 * n_turns + 1)
    R, Z, alive = fl.trace(r0, z0, phi, rtol=rtol, atol=atol)
    psi = eq.psi_at(R.ravel(), Z.ravel()).reshape(R.shape)
    scale = abs(eq.psi_bdy - eq.psi_axis)
    drift = np.abs(psi - psi[:, :1]).max(axis=1) / scale
    return drift, alive


def q_from_tracing(fl: FieldLines, r_seed, z_seed, n_turns=400):
    """Safety factor measured directly from a traced line.

    q = (toroidal turns) / (poloidal turns).  Independent of the contour-
    integral q in :mod:`tokviz.equilibrium`, so the two cross-check each other.
    """
    phi = np.linspace(0.0, TWO_PI * n_turns, 60 * n_turns + 1)
    R, Z, alive = fl.trace(np.atleast_1d(r_seed), np.atleast_1d(z_seed), phi)
    if not alive[0]:
        return np.nan
    th = np.unwrap(np.arctan2(Z[0] - fl.eq.z_axis, R[0] - fl.eq.r_axis))
    d_theta = th[-1] - th[0]
    if d_theta == 0:
        return np.nan
    return float(TWO_PI * n_turns / abs(d_theta))
