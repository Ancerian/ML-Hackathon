"""Analytic (Solov'ev) Grad-Shafranov equilibrium for the JET 1975 D plasma.

Physics
-------
Grad-Shafranov with ``B = grad psi x grad phi + F grad phi`` (psi in Wb/rad,
the convention of :mod:`tokviz.equilibrium`)::

    Delta* psi = R d/dR (1/R dpsi/dR) + d2psi/dZ2 = -mu0 R^2 p'(psi) - F F'(psi)

Solov'ev: p' and FF' are constants.  Following Cerfon & Freidberg,
"One size fits all" analytic solutions to the Grad-Shafranov equation,
Phys. Plasmas 17, 032502 (2010), write ``R = R0 x``, ``Z = R0 y``,
``psi = psi0 * u(x, y)``.  Then

    x d/dx(1/x du/dx) + d2u/dy2 = (1 - A) x^2 + A
    -mu0 p'  = (1 - A) psi0 / R0^4          (C1)
    -F F'    =      A  psi0 / R0^2          (C2)

    u = u_p + sum_i c_i u_i,   u_p = x^4/8 + A (x^2 ln x / 2 - x^4/8)

where the ``u_i`` are homogeneous (Delta* u_i = 0) polynomial/log solutions,
even in y (up-down symmetric plasma).  The coefficients ``c_i`` are fixed by
the boundary ``u = 0`` on the prescribed D, so the LCFS is exactly psi = 0
(up to the fit residual reported by :meth:`SolovevSolution.boundary_residual`).

With the boundary value psi_b = 0:

    p(psi)   = -C1 psi / mu0                 (>= 0 inside for A < 1)
    F(psi)^2 = F0^2 - 2 C2 psi,   F0 = R0 B0 (vacuum field, Table I.3-1 p.83)

Two free scalars remain: ``psi0`` (sets I_p linearly) and ``A`` (sets the
split between pressure-driven and FF'-driven current, i.e. beta_p; beta_p is
independent of psi0 because p and B_pol^2 both scale as psi0^2).

Interface
---------
:func:`build_equilibrium` returns a :class:`SolovevEquilibrium`, a subclass of
:class:`tokviz.equilibrium.Equilibrium` built from a psi(R, Z) GRID exactly as
the parquet frames are, so ``surface``, ``theta_star_field``, ``psi_n`` and
friends work unchanged.  Because Solov'ev F varies (weakly) with psi, q and
B_phi use F(psi): :meth:`SolovevEquilibrium.q_at` is overridden, and
:class:`FieldLinesFpsi` is a drop-in :class:`~tokviz.fieldline.FieldLines`
that evaluates B_phi = F(psi)/R instead of a constant F.

Nothing here imports bpy.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from ..equilibrium import Equilibrium, TWO_PI
from ..fieldline import FieldLines

MU0 = 4.0e-7 * np.pi

# ---------------------------------------------------------------------------
# JET 1975 design numbers (machines/jet1975/dimensions.yaml, Table I.3-1 p.83)
# ---------------------------------------------------------------------------
JET_R0 = 2.96        # m      plasma.R0            p.83
JET_A = 1.25         # m      plasma.a             p.83
JET_B = 2.10         # m      plasma.b             p.83
JET_DELTA = 0.3476   # -      decision G2: triangularity fitted to the SHAPE of
#                               Table IV.2-4 (p.332), components/plasma.py;
#                               manifest components.plasma.info.delta
JET_B0 = 2.77        # T      plasma.Btor_R0_basic p.83
JET_IP = 3.8e6       # A      plasma.Ip_D_basic    p.83 ("for q(a) = 6 basic")
JET_BETAP = 0.9      # -      Table I.3-1 footnotes p.83: "beta_POL = 0.9,
#                               q(a)/q(0) = 5.3" (peaked-current computation) and
#                               "D-shaped plasma with q(a)/q(0) = 6, beta_POL = 0.9"


def load_jet_design(machine_dir=None) -> dict:
    """The few design numbers used here, read from dimensions.yaml when present.

    Falls back to the module constants (identical values) if PyYAML or the
    file is unavailable.  Returns ``{name: (value, page)}``.
    """
    out = {"R0": (JET_R0, 83), "a": (JET_A, 83), "b": (JET_B, 83),
           "B0": (JET_B0, 83), "Ip": (JET_IP, 83),
           "Ic_basic": (4.1e7, 83), "Ic_extended": (5.1e7, 83),
           "ripple": (0.036, 83), "n_tf": (32, 77)}
    try:
        import yaml
        root = Path(machine_dir) if machine_dir else (
            Path(__file__).resolve().parents[3] / "machines" / "jet1975")
        doc = yaml.safe_load((root / "dimensions.yaml").read_text(encoding="utf-8"))
        idx = {(sec, e["id"]): e for sec, lst in doc.items() if isinstance(lst, list)
               for e in lst if isinstance(e, dict) and "id" in e}
        pick = {"R0": ("plasma", "R0"), "a": ("plasma", "a"), "b": ("plasma", "b"),
                "B0": ("plasma", "Btor_R0_basic"), "Ip": ("plasma", "Ip_D_basic"),
                "Ic_basic": ("tf_coils", "total_TF_current_basic"),
                "Ic_extended": ("tf_coils", "total_TF_current_extended"),
                "ripple": ("plasma", "TF_ripple"), "n_tf": ("tf_coils", "n_coils")}
        for k, key in pick.items():
            if key in idx:
                e = idx[key]
                out[k] = (float(e["value"]), e.get("page"))
    except Exception:           # pragma: no cover - fallback path
        pass
    return out


# ---------------------------------------------------------------------------
# a tiny exact algebra for sums of  c x^p y^q (ln x)^r
# ---------------------------------------------------------------------------
class Poly:
    """Sum of terms ``c * x**p * y**q * ln(x)**r`` with exact differentiation."""

    def __init__(self, terms):
        acc: dict = {}
        for c, p, q, r in terms:
            if c != 0.0:
                acc[(p, q, r)] = acc.get((p, q, r), 0.0) + float(c)
        self.t = {k: v for k, v in acc.items() if v != 0.0}

    def __add__(self, o):
        return Poly([(c, *k) for k, c in self.t.items()] + [(c, *k) for k, c in o.t.items()])

    def __mul__(self, s: float):
        return Poly([(c * s, *k) for k, c in self.t.items()])

    __rmul__ = __mul__

    def dx(self):
        out = []
        for (p, q, r), c in self.t.items():
            if p != 0:
                out.append((c * p, p - 1, q, r))
            if r >= 1:
                out.append((c * r, p - 1, q, r - 1))
        return Poly(out)

    def dy(self):
        return Poly([(c * q, p, q - 1, r) for (p, q, r), c in self.t.items() if q != 0])

    def div_x(self):
        return Poly([(c, p - 1, q, r) for (p, q, r), c in self.t.items()])

    def gs(self):
        """Delta* in normalised coordinates: u_xx - u_x / x + u_yy."""
        ux = self.dx()
        return ux.dx() + ux.div_x() * -1.0 + self.dy().dy()

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        lx = np.log(x)
        out = np.zeros(np.broadcast(x, y).shape)
        for (p, q, r), c in self.t.items():
            out = out + c * x ** p * y ** q * lx ** r
        return out


def _P(*terms):
    return Poly(terms)


#: Cerfon-Freidberg up-down-symmetric homogeneous solutions psi_1..psi_7.
CF_HOMOGENEOUS = [
    _P((1, 0, 0, 0)),
    _P((1, 2, 0, 0)),
    _P((1, 0, 2, 0), (-1, 2, 0, 1)),
    _P((1, 4, 0, 0), (-4, 2, 2, 0)),
    _P((2, 0, 4, 0), (-9, 2, 2, 0), (3, 4, 0, 1), (-12, 2, 2, 1)),
    _P((1, 6, 0, 0), (-12, 4, 2, 0), (8, 2, 4, 0)),
    _P((8, 0, 6, 0), (-140, 2, 4, 0), (75, 4, 2, 0), (-15, 6, 0, 1),
       (180, 4, 2, 1), (-120, 2, 4, 1)),
]


def homogeneous_basis(max_deg: int = 10) -> list:
    """All even-in-y solutions of Delta* u = 0 of the form x^p y^q ln^r x
    (p + q <= max_deg, r in {0, 1}), as an orthonormal null-space basis.

    Used by the least-squares boundary fit; the 7 Cerfon-Freidberg functions
    span the max_deg = 6 subspace.
    """
    mons = [(p, q, r) for p in range(max_deg + 1) for q in range(0, max_deg + 1, 2)
            for r in (0, 1) if p + q <= max_deg]
    images = [Poly([(1.0, *m)]).gs() for m in mons]
    rows = sorted({k for im in images for k in im.t})
    M = np.zeros((len(rows), len(mons)))
    ri = {k: i for i, k in enumerate(rows)}
    for j, im in enumerate(images):
        for k, c in im.t.items():
            M[ri[k], j] = c
    _, s, vt = np.linalg.svd(M)
    rank = int((s > 1e-9 * s.max()).sum())
    null = vt[rank:]
    return [Poly([(c, *m) for c, m in zip(vec, mons) if abs(c) > 1e-13]) for vec in null]


def miller_boundary(R0, a, kappa, delta, n=400):
    """Same D as components/plasma.py (Miller form)."""
    t = np.linspace(0.0, TWO_PI, int(n), endpoint=False)
    al = np.arcsin(np.clip(delta, -0.999, 0.999))
    return np.stack([R0 + a * np.cos(t + al * np.sin(t)), kappa * a * np.sin(t)], -1)


# ---------------------------------------------------------------------------
# the solution
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SolovevSolution:
    R0: float
    a: float
    kappa: float
    delta: float
    A: float
    coeffs: np.ndarray
    basis: list
    method: str
    psi0: float = 1.0           # Wb/rad, set by scale_to_current()
    B0: float = JET_B0

    def __post_init__(self):
        up = _P((1 / 8, 4, 0, 0)) + self.A * _P((0.5, 2, 0, 1), (-1 / 8, 4, 0, 0))
        u = up
        for c, h in zip(self.coeffs, self.basis):
            u = u + h * float(c)
        self.u = u
        self.ux = u.dx()
        self.uy = u.dy()

    # -- psi and derivatives in SI ------------------------------------------
    def psi(self, R, Z):
        return self.psi0 * self.u(np.asarray(R) / self.R0, np.asarray(Z) / self.R0)

    def dpsi_dR(self, R, Z):
        return self.psi0 / self.R0 * self.ux(np.asarray(R) / self.R0, np.asarray(Z) / self.R0)

    def dpsi_dZ(self, R, Z):
        return self.psi0 / self.R0 * self.uy(np.asarray(R) / self.R0, np.asarray(Z) / self.R0)

    # -- profiles -----------------------------------------------------------
    @property
    def C1(self):   # = -mu0 p'
        return (1.0 - self.A) * self.psi0 / self.R0 ** 4

    @property
    def C2(self):   # = -F F'
        return self.A * self.psi0 / self.R0 ** 2

    @property
    def F0(self):
        return self.R0 * self.B0

    def pressure(self, psi):
        return np.maximum(-self.C1 * np.asarray(psi, float) / MU0, 0.0)

    def F_of_psi(self, psi):
        return np.sqrt(self.F0 ** 2 - 2.0 * self.C2 * np.asarray(psi, float))

    def j_phi(self, R, Z):
        """Toroidal current density (A/m^2): mu0 j_phi = -Delta* psi / R."""
        R = np.asarray(R, float)
        return (self.C1 * R + self.C2 / R) / MU0

    def boundary(self, n=400):
        return miller_boundary(self.R0, self.a, self.kappa, self.delta, n)

    def boundary_residual(self, n=720):
        """Distance-like residual of the LCFS: max |psi| / |grad psi| on the
        prescribed D (metres) -- how far the psi = 0 contour sits from it."""
        b = self.boundary(n)
        ps = self.psi(b[:, 0], b[:, 1])
        g = np.hypot(self.dpsi_dR(b[:, 0], b[:, 1]), self.dpsi_dZ(b[:, 0], b[:, 1]))
        d = np.abs(ps) / np.maximum(g, 1e-30)
        return float(d.max()), float(np.sqrt(np.mean(d ** 2)))


def solve_shape(R0, a, kappa, delta, A, method="lsq", max_deg=10, n_bdy=720):
    """Coefficients c_i so that u = 0 on the Miller D.

    method 'cf7'  -- Cerfon-Freidberg: 7 point/slope/curvature conditions.
    method 'lsq'  -- least squares of u on ``n_bdy`` boundary points with the
                     full homogeneous basis up to ``max_deg`` (more flexible;
                     reproduces the D to sub-millimetre).
    """
    eps = a / R0
    al = np.arcsin(delta)
    up = _P((1 / 8, 4, 0, 0)) + A * _P((0.5, 2, 0, 1), (-1 / 8, 4, 0, 0))
    if method == "cf7":
        basis = CF_HOMOGENEOUS
        N1 = -(1 + al) ** 2 / (eps * kappa ** 2)
        N2 = (1 - al) ** 2 / (eps * kappa ** 2)
        N3 = -kappa / (eps * np.cos(al) ** 2)
        xo, xi, xt, yt = 1 + eps, 1 - eps, 1 - delta * eps, kappa * eps

        def rows(f):
            fx, fy = f.dx(), f.dy()
            fxx, fyy = fx.dx(), fy.dy()
            return np.array([
                f(xo, 0.0), f(xi, 0.0), f(xt, yt), fx(xt, yt),
                fyy(xo, 0.0) + N1 * fx(xo, 0.0),
                fyy(xi, 0.0) + N2 * fx(xi, 0.0),
                fxx(xt, yt) + N3 * fy(xt, yt)], float)

        M = np.stack([rows(h) for h in basis], axis=1)
        c = np.linalg.solve(M, -rows(up))
    elif method == "lsq":
        basis = homogeneous_basis(max_deg)
        b = miller_boundary(1.0, eps, kappa, delta, n_bdy)
        # up-down symmetric: fit the upper half only (incl. both midplane points)
        b = b[b[:, 1] >= -1e-12]
        M = np.stack([h(b[:, 0], b[:, 1]) for h in basis], axis=1)
        rhs = -up(b[:, 0], b[:, 1])
        c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    else:
        raise ValueError(method)
    return SolovevSolution(R0, a, kappa, delta, A, np.asarray(c), basis, method)


# ---------------------------------------------------------------------------
# Equilibrium adapter
# ---------------------------------------------------------------------------
class SolovevEquilibrium(Equilibrium):
    """:class:`Equilibrium` built from a Solov'ev psi GRID, with F = F(psi).

    Everything grid-based (spline, contours, axis, theta*) is inherited.  The
    constant ``F`` of the parent is set to the VACUUM value R0 B0 for code that
    insists on a scalar; q and B_phi use F(psi).
    """

    def __init__(self, sol: SolovevSolution, R, Z):
        RR, ZZ = np.meshgrid(R, Z)                      # rows = Z, cols = R
        super().__init__(sol.psi(RR, ZZ), R, Z, machine="JET1975")
        self.sol = sol
        # psi0 > 0 -> the axis is the psi MINIMUM (like DIII-D's EFIT sign)
        self.axis_sign = -1.0 if sol.psi0 > 0 else 1.0
        self.set_F(sol.F0)
        self.find_axis()
        self.psi_bdy = 0.0                               # exact by construction

    def F_of_psi(self, psi):
        return self.sol.F_of_psi(psi)

    def B_phi(self, r, z=None):
        r = np.asarray(r, float)
        if z is None:
            return self._F / r
        return self.sol.F_of_psi(self.psi_at(r, z)) / r

    def q_at(self, psi_n_level: float):
        pts = self.surface(psi_n_level)
        if pts is None:
            return None
        _, tot = self._loop_integral(pts)
        F = float(self.sol.F_of_psi(self.psi_of_psi_n(psi_n_level)))
        return float(F * tot / TWO_PI)

    def q_axis(self) -> float:
        """q on the magnetic axis from the local ellipse of psi:
        q0 = F(psi_axis) / (R_axis sqrt(psi_RR psi_ZZ))   (exact analytic psi)."""
        r, z = self.r_axis, self.z_axis
        u = self.sol.u
        R0 = self.sol.R0
        uxx, uyy = u.dx().dx(), u.dy().dy()
        prr = self.sol.psi0 / R0 ** 2 * float(uxx(r / R0, z / R0))
        pzz = self.sol.psi0 / R0 ** 2 * float(uyy(r / R0, z / R0))
        F = float(self.sol.F_of_psi(self.psi_axis))
        return F / (r * np.sqrt(abs(prr * pzz)))


class FieldLinesFpsi(FieldLines):
    """FieldLines with B_phi = F(psi)/R (Solov'ev F varies across the plasma)."""

    def _Fp(self, r, z):
        return self.eq.F_of_psi(self.eq.psi_at(r, z))

    def B(self, r, z, phi):
        r = np.asarray(r, dtype=float)
        dR = self.eq.dpsi_dR(r, z)
        dZ = self.eq.dpsi_dZ(r, z)
        BR, BZ, BP = -dZ / r, dR / r, self._Fp(r, z) / r
        if self.pert is not None:
            bR, bZ, bP = self._pert_B(r, z, phi, dR, dZ)
            BR, BZ, BP = BR + bR, BZ + bZ, BP + bP
        return BR, BZ, BP

    def _rhs(self, phi, y):
        n = y.size // 2
        r, z = y[:n], y[n:]
        alive = self._in_domain(r, z)
        rc = np.clip(r, self.r_lo, self.r_hi)
        zc = np.clip(z, self.z_lo, self.z_hi)
        BR, BZ, BP = self.B(rc, zc, phi)
        BP = np.where(np.abs(BP) < 1e-12, np.sign(BP) * 1e-12 + 1e-30, BP)
        drdp = np.where(alive, rc * BR / BP, 0.0)
        dzdp = np.where(alive, rc * BZ / BP, 0.0)
        return np.concatenate([drdp, dzdp])


# ---------------------------------------------------------------------------
# global quantities
# ---------------------------------------------------------------------------
def _inside_mask(sol: SolovevSolution, RR, ZZ):
    from matplotlib.path import Path as MplPath
    b = sol.boundary(1440)
    return MplPath(b).contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)


def global_quantities(sol: SolovevSolution, n=801) -> dict:
    """I_p, beta_p, beta_t, l_i, volume, ... by quadrature on a fine (R,Z) mesh
    clipped to the prescribed boundary, plus Ampere's law on the boundary.

    Definitions (stated because several are in use):
      B_pa    = mu0 I_p / L_bdy                (L = LCFS poloidal circumference)
      beta_p  = 2 mu0 <p>_V / B_pa^2           (<.>_V = volume average)
      beta_t  = 2 mu0 <p>_V / B0^2
      l_i(1)  = <B_pol^2>_V / B_pa^2           (EFIT 'li')
      l_i(3)  = 2 W_pol / (mu0 R0 I_p^2) = 2 <B_pol^2>_V V / (mu0^2 R0 I_p^2)
    """
    R = np.linspace(sol.R0 - sol.a - 0.01, sol.R0 + sol.a + 0.01, n)
    Z = np.linspace(-sol.kappa * sol.a - 0.01, sol.kappa * sol.a + 0.01, int(n * 1.3))
    RR, ZZ = np.meshgrid(R, Z)
    dA = (R[1] - R[0]) * (Z[1] - Z[0])
    m = _inside_mask(sol, RR, ZZ)
    r, z = RR[m], ZZ[m]
    ps = sol.psi(r, z)
    Ip_area = float(np.sum(sol.j_phi(r, z)) * dA)
    dV = TWO_PI * r * dA
    V = float(dV.sum())
    p = sol.pressure(ps)
    p_avg = float(np.sum(p * dV) / V)
    Bp2 = (sol.dpsi_dR(r, z) ** 2 + sol.dpsi_dZ(r, z) ** 2) / r ** 2
    Bp2_avg = float(np.sum(Bp2 * dV) / V)
    # Ampere on the boundary
    b = sol.boundary(4000)
    bc = np.vstack([b, b[:1]])
    dl = np.diff(bc, axis=0)
    mid = 0.5 * (bc[:-1] + bc[1:])
    L = float(np.hypot(dl[:, 0], dl[:, 1]).sum())
    BR = -sol.dpsi_dZ(mid[:, 0], mid[:, 1]) / mid[:, 0]
    BZ = sol.dpsi_dR(mid[:, 0], mid[:, 1]) / mid[:, 0]
    Ip_ampere = float(np.sum(BR * dl[:, 0] + BZ * dl[:, 1]) / MU0)
    Ip = Ip_ampere
    Bpa = MU0 * abs(Ip) / L
    area = float(m.sum() * dA)
    return {
        "Ip_A": Ip, "Ip_area_A": Ip_area, "L_bdy_m": L, "area_m2": area, "volume_m3": V,
        "p_axis_Pa": float(sol.pressure(np.min(ps) if sol.psi0 > 0 else np.max(ps))),
        "p_avg_Pa": p_avg, "B_pa_T": Bpa,
        "beta_p": 2 * MU0 * p_avg / Bpa ** 2,
        "beta_t": 2 * MU0 * p_avg / sol.B0 ** 2,
        "li1": Bp2_avg / Bpa ** 2,
        "li3": 2 * Bp2_avg * V / (MU0 ** 2 * sol.R0 * Ip ** 2),
        "F_axis_over_F0": float(sol.F_of_psi(np.min(ps) if sol.psi0 > 0 else np.max(ps)) / sol.F0),
    }


def beta_p_of_A(R0, a, kappa, delta, A, method="cf7", n=301):
    sol = solve_shape(R0, a, kappa, delta, A, method=method)
    return global_quantities(sol, n=n)["beta_p"]


def solve_A_for_beta_p(R0, a, kappa, delta, beta_p, method="cf7",
                       lo=-0.6, hi=0.6, tol=1e-5) -> float:
    """Bisection on A (beta_p decreases monotonically as A grows)."""
    f = lambda A: beta_p_of_A(R0, a, kappa, delta, A, method) - beta_p
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"beta_p={beta_p} not bracketed: f({lo})={flo:.3f}, f({hi})={fhi:.3f}")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(hi - lo) < tol:
            break
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def scale_to_current(sol: SolovevSolution, Ip: float) -> SolovevSolution:
    """Set psi0 so the plasma current equals ``Ip`` (I_p is linear in psi0)."""
    sol.psi0 = 1.0
    I1 = global_quantities(sol, n=401)["Ip_A"]
    sol.psi0 = float(Ip / I1)
    return sol


# ---------------------------------------------------------------------------
# one-call builder
# ---------------------------------------------------------------------------
def build_equilibrium(R0=JET_R0, a=JET_A, b=JET_B, delta=JET_DELTA, B0=JET_B0,
                      Ip=JET_IP, beta_p=JET_BETAP, A=None, method="cf7",
                      grid=(193, 257), margin=(0.24, 0.30)):
    """Solov'ev JET equilibrium: shape from G2, B0 and I_p from Table I.3-1.

    ``A`` overrides the beta_p solve.  ``grid = (nR, nZ)``; the grid extends
    ``margin = (dR, dZ)`` metres beyond the plasma so the psi_n = 1 contour
    and field lines on it stay well inside the spline domain.

    Returns ``(eq, sol, info)``.
    """
    kappa = b / a
    if A is None:
        A = solve_A_for_beta_p(R0, a, kappa, delta, beta_p, method=method)
    sol = solve_shape(R0, a, kappa, delta, A, method=method)
    sol.B0 = B0
    scale_to_current(sol, Ip)
    R = np.linspace(R0 - a - margin[0], R0 + a + margin[0], grid[0])
    Z = np.linspace(-b - margin[1], b + margin[1], grid[1])
    eq = SolovevEquilibrium(sol, R, Z)
    gq = global_quantities(sol)
    res_max, res_rms = sol.boundary_residual()
    info = {"A": float(A), "psi0_Wb_per_rad": sol.psi0, "method": method,
            "n_basis": len(sol.basis), "kappa": kappa, "delta": delta,
            "boundary_residual_max_m": res_max, "boundary_residual_rms_m": res_rms,
            "C1_minus_mu0_pprime": sol.C1, "C2_minus_FFprime": sol.C2,
            "F0_Tm": sol.F0, **gq}
    return eq, sol, info


def q_traced(fl: FieldLines, r_seed: float, z_seed: float, n_turns: int = 200,
             pts_per_turn: int = 60) -> float:
    """q from a traced line, by a least-squares slope of the unwrapped
    geometric poloidal angle against phi.

    Same idea as :func:`tokviz.fieldline.q_from_tracing` (independent of the
    contour integral), but a slope fit instead of end points, so the periodic
    theta_geo - theta* wobble averages out rather than setting the error.
    """
    phi = np.linspace(0.0, TWO_PI * n_turns, pts_per_turn * n_turns + 1)
    R, Z, alive = fl.trace(np.atleast_1d(r_seed), np.atleast_1d(z_seed), phi)
    if not alive[0]:
        return float("nan")
    th = np.unwrap(np.arctan2(Z[0] - fl.eq.z_axis, R[0] - fl.eq.r_axis))
    slope = np.polyfit(phi, th, 1)[0]
    return float(abs(1.0 / slope))
