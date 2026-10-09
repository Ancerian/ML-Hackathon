"""Non-axisymmetric resonant perturbation.

Why this module exists
----------------------
In strict axisymmetry the field-line system is integrable, so a Poincare
section is EXACTLY the level sets of psi and carries no information the
contours do not already show.  Magnetic islands therefore cannot be a
by-product of tracing an EFIT psi -- they must be put in deliberately.  This
module puts them in, explicitly and reproducibly, and records the amplitude so
the render can state it.

Construction
------------
The perturbation is applied to the poloidal flux itself::

    psi_total(R, Z, phi) = psi(R, Z) + dpsi(R, Z, phi)

with the field taken as ``B = curl(psi_total grad phi)``, i.e.

    B_R = -(1/R) d psi_total / dZ
    B_Z = +(1/R) d psi_total / dR
    B_phi = F / R

which is divergence-free EXACTLY and for any dpsi, because it is a curl.

Each mode is written on the straight-field-line angle so that it resonates at
exactly q = m/n::

    dpsi_k = A_k g_k(psi_n) cos(m_k theta* - n_k phi + alpha_k)

and expanded as ``cos(m theta*) cos(v) + sin(m theta*) sin(v)``, ``v = n phi -
alpha``, so the stored grids are smooth (theta* itself has a 2pi branch cut
that would wreck spline derivatives).

Honesty note
------------
This is a PRESCRIBED vacuum-like perturbation, not the self-consistent plasma
response to real coils.  A vacuum field systematically OVERESTIMATES
stochasticity, because the plasma screens resonant components.  Renders built
from this module must say so.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from .equilibrium import Equilibrium, ThetaStarField, TWO_PI


@dataclasses.dataclass
class Mode:
    """One resonant harmonic.

    Attributes
    ----------
    m, n : poloidal / toroidal mode numbers; resonant where q = m/n
    amp : amplitude as a FRACTION of the total flux range |psi_bdy - psi_axis|
    phase : alpha, radians
    envelope : 'vacuum' -> psi_n**(m/2), growing toward the edge the way an
        externally applied vacuum field does; 'resonant' -> Gaussian centred on
        the resonant surface, width ``width``
    width : Gaussian width in psi_n, used only by the 'resonant' envelope
    psin_res : normalised flux of the q = m/n surface.  Required by the
        'resonant' envelope; get it from :func:`resonant_surfaces`.
    """

    m: int
    n: int
    amp: float
    phase: float = 0.0
    envelope: str = "vacuum"
    width: float = 0.08
    psin_res: float | None = None
    #: width of the C1 taper applied just outside the separatrix
    taper: float = 0.05

    @property
    def q_res(self) -> float:
        return self.m / self.n

    # -- radial envelope and its derivative ---------------------------------
    def envelope_and_deriv(self, sn):
        """``(g, dg/d psi_n)`` for this mode's radial envelope."""
        s = np.clip(np.asarray(sn, dtype=float), 0.0, None)

        if self.envelope == "vacuum":
            e = 0.5 * self.m
            base = np.power(s, e)
            # d/ds s**e, guarding s = 0 for e < 1
            with np.errstate(divide="ignore", invalid="ignore"):
                dbase = np.where(s > 1e-12, e * np.power(s, e - 1.0), 0.0)
        elif self.envelope == "resonant":
            if self.psin_res is None:
                raise ValueError(
                    "Mode.psin_res must be set for the 'resonant' envelope; "
                    "compute it with resonant_surfaces(psin, q, m/n)"
                )
            u = (s - self.psin_res) / self.width
            base = np.exp(-u * u)
            dbase = base * (-2.0 * u / self.width)
        else:
            raise ValueError(f"unknown envelope {self.envelope!r}")

        # C1 taper just outside the separatrix: max(0, s-1)**2 has zero
        # derivative on both sides of s = 1, so g stays continuously
        # differentiable there.
        u = np.maximum(s - 1.0, 0.0)
        w2 = self.taper * self.taper
        T = np.exp(-(u * u) / w2)
        dT = T * (-2.0 * u / w2)
        return base * T, dbase * T + base * dT


class Perturbation:
    """Sum of :class:`Mode` harmonics, evaluated analytically.

    Nothing is rasterised onto the EFIT grid: theta* comes from a
    :class:`~tokviz.equilibrium.ThetaStarField` and the envelope is closed
    form, so the poloidal spectrum stays clean and a mode m only ever drives
    its own q = m/n resonance.
    """

    def __init__(self, eq: Equilibrium, modes: list[Mode],
                 tsf: ThetaStarField | None = None):
        if not modes:
            raise ValueError("need at least one mode")
        self.eq = eq
        self.modes = list(modes)
        self.tsf = tsf if tsf is not None else eq.theta_star_field()
        self.dpsi_scale = abs(eq.psi_bdy - eq.psi_axis)
        self.q_res = [md.q_res for md in modes]

    def dpsi(self, r, z, phi):
        th, _, _, sn, _, _ = self.tsf.value_and_grad(r, z)
        out = np.zeros_like(np.asarray(r, dtype=float))
        for md in self.modes:
            g, _ = md.envelope_and_deriv(sn)
            out += (md.amp * self.dpsi_scale) * g * np.cos(md.m * th - md.n * phi + md.phase)
        return out

    def dpsi_derivs(self, r, z, phi, psi=None, dpsi_dR=None, dpsi_dZ=None):
        """``(d dpsi/dR, d dpsi/dZ)`` by the chain rule.

        The equilibrium quantities may be supplied by the caller to avoid
        re-evaluating the psi spline.
        """
        th, dth_dR, dth_dZ, sn, dsn_dR, dsn_dZ = self.tsf.value_and_grad(
            r, z, psi=psi, dpsi_dR=dpsi_dR, dpsi_dZ=dpsi_dZ)
        dR = np.zeros_like(np.asarray(r, dtype=float))
        dZ = np.zeros_like(dR)
        for md in self.modes:
            g, dg = md.envelope_and_deriv(sn)
            arg = md.m * th - md.n * phi + md.phase
            c, s_ = np.cos(arg), np.sin(arg)
            a = md.amp * self.dpsi_scale
            dR += a * (dg * dsn_dR * c - g * md.m * dth_dR * s_)
            dZ += a * (dg * dsn_dZ * c - g * md.m * dth_dZ * s_)
        return dR, dZ

    def delta_B(self, r, z, phi, psi=None, dpsi_dR=None, dpsi_dZ=None):
        """``(dB_R, dB_Z, dB_phi)`` -- the backend interface.

        This perturbation is a flux function, so its field is PURELY POLOIDAL:
        ``dB_phi`` is identically zero.  A real coil array is not like this; see
        :class:`tokviz.coilfield.CoilPerturbation`.
        """
        dR, dZ = self.dpsi_derivs(r, z, phi, psi=psi,
                                  dpsi_dR=dpsi_dR, dpsi_dZ=dpsi_dZ)
        r = np.asarray(r, dtype=float)
        return -dZ / r, dR / r, np.zeros_like(r)

    def amplitude_at(self, md: Mode) -> float:
        """Effective harmonic amplitude at this mode's resonant surface,
        in normalised-flux units -- the ``eps`` of :func:`island_width_psin`."""
        if md.psin_res is None:
            raise ValueError("Mode.psin_res is not set")
        g, _ = md.envelope_and_deriv(np.array([md.psin_res]))
        return float(md.amp * g[0])


# ---------------------------------------------------------------------------
# island geometry
# ---------------------------------------------------------------------------
def resonant_surfaces(psin: np.ndarray, q: np.ndarray, q_target: float):
    """All psi_n where a monotone-in-between q profile crosses ``q_target``."""
    out = []
    for i in range(len(q) - 1):
        a, b = q[i] - q_target, q[i + 1] - q_target
        if a == 0.0:
            out.append(float(psin[i]))
        elif a * b < 0:
            t = a / (a - b)
            out.append(float(psin[i] + t * (psin[i + 1] - psin[i])))
    return out


def island_width_psin(amp_psin: float, q_res: float, dq_dpsin: float) -> float:
    """Full island width in normalised flux, from the pendulum separatrix.

    Canonical pair.  With B = grad(psi) x grad(phi) + F grad(phi) and psi the
    poloidal flux per radian, the field-line equations are Hamiltonian with
    phi as time, the straight-field-line angle theta* as the coordinate, the
    TOROIDAL flux per radian psi_t as its conjugate momentum, and the POLOIDAL
    flux as the Hamiltonian, H = psi(psi_t, theta*, phi); dpsi = iota dpsi_t,
    iota = 1/q.  (psi itself is NOT the momentum.)

    Pendulum.  For H = psi_0(psi_t) + A cos(m theta* - n phi + alpha) and
    xi = m theta* - n phi + alpha, near the resonance

        d xi / d phi    = m iota' (psi_t - psi_t,res),   iota' = d iota/d psi_t
        d psi_t / d phi = -dH/d theta* = m A sin xi

    so the separatrix full width in psi_t is 4 sqrt(A / |iota'|) (m cancels).
    With |iota'| = |dq/dpsi| / q**3 and a width in psi being 1/q of the width
    in psi_t:

        W_psi = 4 sqrt( A q / |dq/dpsi| )

    Both A and dq/dpsi rescale with the flux range, so the same expression
    holds in normalised flux with eps = A / |psi_bdy - psi_axis| (the value of
    :meth:`Perturbation.amplitude_at`) and ``dq_dpsin = dq/dpsi_N``:

        W = 4 sqrt( eps q / |dq/dpsi_N| )

    (Until 2026-09-29 this returned 4 sqrt(eps q**2 / |dq/dpsi|), which treats
    psi as the momentum and overestimates W by sqrt(q).)
    """
    if dq_dpsin == 0 or amp_psin <= 0:
        return 0.0
    return 4.0 * float(np.sqrt(abs(amp_psin) * abs(q_res) / abs(dq_dpsin)))


def chirikov(w1: float, s1: float, w2: float, s2: float) -> float:
    """Chirikov overlap parameter S = (W1 + W2) / (2 |s1 - s2|).

    S >= 1 is a HEURISTIC for the onset of large-scale stochasticity, not a
    theorem.  Report it; do not present it as a criterion that has been proved.
    """
    d = abs(s1 - s2)
    if d == 0:
        return float("inf")
    return float((w1 + w2) / (2.0 * d))
