"""Axisymmetric equilibrium: psi(R,Z), the poloidal field, q(psi) and
straight-field-line coordinates.

Conventions (SI, psi in Wb/rad as EFIT stores it)
-------------------------------------------------
    B_R   = -(1/R) dpsi/dZ
    B_Z   = +(1/R) dpsi/dR
    B_phi = F(psi) / R
    B_pol = |grad psi| / R

``F(psi) = R * B_phi`` is NOT shipped by the Fusion Equilibrium Challenge
dataset, so it is *calibrated* against the dataset's own ``efit_q95``; see
:meth:`Equilibrium.calibrate_F`.

Nothing here imports bpy.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
from scipy.optimize import minimize

from .config import AXIS_SIGN

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Shot loading
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ShotData:
    """One shot as stored in the challenge parquet (one row, nested arrays)."""

    source: str
    times: np.ndarray            # (T,) ms
    psirz: np.ndarray            # (T, nZ, nR)  -- rows are Z, cols are R
    grid_R: np.ndarray           # (nR,) metres
    grid_Z: np.ndarray           # (nZ,) metres
    q95: np.ndarray              # (T,)
    beta_n: np.ndarray           # (T,)
    li: np.ndarray               # (T,)
    r_axis: np.ndarray           # (T,)
    z_axis: np.ndarray           # (T,)
    lcfs_r: np.ndarray           # (T, P)
    lcfs_z: np.ndarray           # (T, P)
    lcfs_n: np.ndarray           # (T,) valid point count
    coils: dict                  # name -> dict(R, Z, width, height, angle1, angle2)

    @property
    def n_frames(self) -> int:
        return int(self.psirz.shape[0])


def load_shot(parquet_path) -> ShotData:
    """Read one challenge parquet shot file into a :class:`ShotData`."""
    import pyarrow.parquet as pq

    tbl = pq.ParquetFile(str(parquet_path)).read()

    def col(name, dtype=float):
        return np.asarray(tbl.column(name)[0].as_py(), dtype=dtype)

    names = [str(x) for x in tbl.column("coil_name")[0].as_py()]
    coils = {}
    for i, nm in enumerate(names):
        coils[nm] = dict(
            R=float(col("coil_R")[i]),
            Z=float(col("coil_Z")[i]),
            width=float(col("coil_width")[i]),
            height=float(col("coil_height")[i]),
            angle1=float(col("coil_angle1")[i]),
            angle2=float(col("coil_angle2")[i]),
        )

    return ShotData(
        source=str(tbl.column("source")[0].as_py()),
        times=col("efit_times"),
        psirz=col("efit_psirz"),
        grid_R=col("efit_grid_R"),
        grid_Z=col("efit_grid_Z"),
        q95=col("efit_q95"),
        beta_n=col("efit_beta_n"),
        li=col("efit_li"),
        r_axis=col("efit_r_axis"),
        z_axis=col("efit_z_axis"),
        lcfs_r=col("efit_lcfs_r"),
        lcfs_z=col("efit_lcfs_z"),
        lcfs_n=col("efit_lcfs_n", dtype=int),
        coils=coils,
    )


def select_frame(shot: ShotData, q95_target: float = 3.7,
                 t_window: tuple[float, float] = (1000.0, 3500.0)) -> int:
    """Pick the working frame: finite psi, inside the flat-top window, q95
    closest to ``q95_target``.

    Low q95 is wanted because the low-order rational surfaces (q = 3, 10/3,
    11/3) then sit well inside the plasma, where islands are large and legible.
    """
    finite = np.isfinite(shot.psirz).all(axis=(1, 2))
    finite &= np.isfinite(shot.q95)
    in_win = (shot.times >= t_window[0]) & (shot.times <= t_window[1])
    ok = finite & in_win
    if not ok.any():
        ok = finite
    if not ok.any():
        raise ValueError("no usable frame in shot")
    idx = np.where(ok)[0]
    best = idx[np.argmin(np.abs(shot.q95[idx] - q95_target))]
    return int(best)


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------
class Equilibrium:
    """One axisymmetric equilibrium frame.

    Parameters
    ----------
    psi : (nZ, nR) array, rows = Z, cols = R  (EFIT ordering)
    R, Z : 1-D uniform grids in metres
    machine : key into :data:`tokviz.config.AXIS_SIGN`
    """

    def __init__(self, psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                 machine: str = "DIII-D"):
        psi = np.asarray(psi, dtype=float)
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if psi.shape != (Z.size, R.size):
            raise ValueError(f"psi {psi.shape} does not match (nZ,nR)=({Z.size},{R.size})")
        # The challenge stores the axes as float32, so an "even" grid arrives
        # with ~1e-7 m of round-off.  Snap to an exact linspace when the
        # deviation is a negligible fraction of a cell; reject a grid that is
        # genuinely non-uniform, since the spline and the index<->metre
        # conversions in surface() both assume even spacing.
        R, Z = _regularize(R, "R"), _regularize(Z, "Z")

        self.psi = psi
        self.R = R
        self.Z = Z
        self.machine = machine
        self.axis_sign = AXIS_SIGN.get(machine, -1.0)
        self.dR = float(R[1] - R[0])
        self.dZ = float(Z[1] - Z[0])

        # RectBivariateSpline wants strictly increasing (x, y) with values
        # indexed [x, y].  Our psi is [Z, R], so transpose to [R, Z].
        self._spl = RectBivariateSpline(R, Z, psi.T, kx=3, ky=3, s=0)

        self._F: float | None = None
        self.psi_axis: float | None = None
        self.psi_bdy: float | None = None
        self.r_axis: float | None = None
        self.z_axis: float | None = None

    # -- raw psi and derivatives -------------------------------------------
    def psi_at(self, r, z):
        return self._spl.ev(r, z)

    def dpsi_dR(self, r, z):
        return self._spl.ev(r, z, dx=1, dy=0)

    def dpsi_dZ(self, r, z):
        return self._spl.ev(r, z, dx=0, dy=1)

    def grad_psi_mag(self, r, z):
        return np.hypot(self.dpsi_dR(r, z), self.dpsi_dZ(r, z))

    # -- axisymmetric field -------------------------------------------------
    def B_R(self, r, z):
        return -self.dpsi_dZ(r, z) / r

    def B_Z(self, r, z):
        return self.dpsi_dR(r, z) / r

    def B_pol(self, r, z):
        return self.grad_psi_mag(r, z) / r

    def B_phi(self, r, z=None):
        if self._F is None:
            raise RuntimeError("F not set -- call calibrate_F() or set_F() first")
        return self._F / np.asarray(r, dtype=float)

    def set_F(self, F: float) -> None:
        self._F = float(F)

    @property
    def F(self) -> float:
        if self._F is None:
            raise RuntimeError("F not set")
        return self._F

    # -- magnetic axis ------------------------------------------------------
    def find_axis(self) -> tuple[float, float, float]:
        """Locate the O-point by maximising ``axis_sign * psi``.

        Coarse grid argmax, then Nelder-Mead on the spline for sub-cell
        accuracy.  Returns ``(R_axis, Z_axis, psi_axis)``.
        """
        phi = self.axis_sign * self.psi
        # restrict to the interior so an edge cell cannot win
        interior = np.full(phi.shape, -np.inf)
        interior[2:-2, 2:-2] = phi[2:-2, 2:-2]
        iz, ir = np.unravel_index(np.argmax(interior), phi.shape)
        r0, z0 = float(self.R[ir]), float(self.Z[iz])

        def neg(p):
            r, z = p
            if not (self.R[0] <= r <= self.R[-1] and self.Z[0] <= z <= self.Z[-1]):
                return 1e9
            return -float(self.axis_sign * self._spl.ev(r, z))

        res = minimize(neg, np.array([r0, z0]), method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-12, maxiter=2000))
        r, z = float(res.x[0]), float(res.x[1])
        self.r_axis, self.z_axis = r, z
        self.psi_axis = float(self._spl.ev(r, z))
        return r, z, self.psi_axis

    def set_boundary_from_contour(self, lcfs_r, lcfs_z) -> float:
        """Take psi_bdy as the median psi along a supplied LCFS polygon.

        Using the dataset's own ``efit_lcfs_r/z`` keeps our normalisation
        identical to EFIT's rather than re-deriving a separatrix.
        """
        vals = self.psi_at(np.asarray(lcfs_r, float), np.asarray(lcfs_z, float))
        self.psi_bdy = float(np.median(vals))
        return self.psi_bdy

    def psi_n(self, r, z):
        """Normalised flux: 0 on axis, 1 on the boundary."""
        if self.psi_axis is None or self.psi_bdy is None:
            raise RuntimeError("call find_axis() and set_boundary_from_contour() first")
        return (self.psi_at(r, z) - self.psi_axis) / (self.psi_bdy - self.psi_axis)

    def psi_n_grid(self) -> np.ndarray:
        return (self.psi - self.psi_axis) / (self.psi_bdy - self.psi_axis)

    def psi_of_psi_n(self, psi_n) -> np.ndarray:
        return self.psi_axis + np.asarray(psi_n, float) * (self.psi_bdy - self.psi_axis)

    # -- flux surface contours ---------------------------------------------
    def surface(self, psi_n_level: float) -> np.ndarray | None:
        """Closed flux-surface polygon at a normalised flux level.

        Returns an ``(N, 2)`` array of ``(R, Z)``, oriented counter-clockwise,
        with the first point NOT repeated -- or ``None`` if no closed contour
        encircling the axis exists at that level.
        """
        from skimage import measure

        level = float(self.psi_of_psi_n(psi_n_level))
        cands = measure.find_contours(self.psi, level)
        best = None
        for c in cands:
            # c is (row, col) = (Z index, R index), fractional
            rr = self.R[0] + c[:, 1] * self.dR
            zz = self.Z[0] + c[:, 0] * self.dZ
            closed = (abs(rr[0] - rr[-1]) < 1e-9) and (abs(zz[0] - zz[-1]) < 1e-9)
            if not closed:
                continue
            pts = np.column_stack([rr[:-1], zz[:-1]])
            if pts.shape[0] < 16:
                continue
            if not _encircles(pts, self.r_axis, self.z_axis):
                continue
            if best is None or pts.shape[0] > best.shape[0]:
                best = pts
        if best is None:
            return None
        if _signed_area(best) < 0:
            best = best[::-1]
        return best

    # -- q profile and straight-field-line angle ---------------------------
    def _loop_integral(self, pts: np.ndarray) -> tuple[np.ndarray, float]:
        """Cumulative and total of ``dl / (R |grad psi|)`` around a surface.

        ``q = (F / 2pi) * total`` and ``theta* = 2pi * cumulative / total``,
        so the straight-field-line angle is independent of F.
        """
        r, z = pts[:, 0], pts[:, 1]
        rc = np.r_[r, r[0]]
        zc = np.r_[z, z[0]]
        dl = np.hypot(np.diff(rc), np.diff(zc))               # (N,)
        rm = 0.5 * (rc[:-1] + rc[1:])
        zm = 0.5 * (zc[:-1] + zc[1:])
        g = self.grad_psi_mag(rm, zm)
        g = np.where(g < 1e-12, 1e-12, g)
        w = dl / (rm * g)
        cum = np.concatenate([[0.0], np.cumsum(w)])           # (N+1,)
        return cum, float(cum[-1])

    def q_at(self, psi_n_level: float) -> float | None:
        pts = self.surface(psi_n_level)
        if pts is None:
            return None
        _, tot = self._loop_integral(pts)
        return float(self.F * tot / TWO_PI)

    def calibrate_F(self, q95_target: float, psi_n: float = 0.95) -> float:
        """Solve ``F`` so the computed q at ``psi_n`` equals the dataset's q95.

        The challenge dataset withholds ``fpol``, so B_phi cannot be derived
        from psi alone.  Since q is linear in F, this is a single division --
        no iteration, no assumed vacuum field.
        """
        pts = self.surface(psi_n)
        if pts is None:
            raise RuntimeError(f"no closed surface at psi_n={psi_n}")
        _, tot = self._loop_integral(pts)
        if tot <= 0:
            raise RuntimeError("degenerate loop integral")
        self._F = float(TWO_PI * abs(q95_target) / tot)
        return self._F

    def q_profile(self, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """q on a list of normalised-flux levels.  Skips levels with no
        closed surface; returns ``(psi_n_kept, q_kept)``."""
        keep_x, keep_q = [], []
        for s in levels:
            q = self.q_at(float(s))
            if q is not None and np.isfinite(q):
                keep_x.append(float(s))
                keep_q.append(q)
        return np.asarray(keep_x), np.asarray(keep_q)

    def theta_star_geometry(self, levels=None, n_theta: int = 512,
                            max_level: float = 0.985):
        """Tabulate ``Delta = theta* - theta_geo`` on a regular (psi_n, theta_geo) grid.

        theta* is the STRAIGHT-FIELD-LINE angle: a field line advances
        ``d theta*/d phi = 1/q`` uniformly, so a perturbation ~ cos(m theta* -
        n phi) resonates at exactly q = m/n.  The geometric angle does not have
        that property and would misplace the islands.

        We tabulate the DIFFERENCE rather than theta* itself for two reasons:
        Delta is single-valued and 2pi-periodic in theta_geo (both angles gain
        exactly 2pi per poloidal circuit), and it is small and smooth, so
        interpolating it introduces far less spurious poloidal harmonic content
        than interpolating cos(m theta*) over scattered contour points.  That
        matters: a spurious m' component seeds a fake island at q = m'/n.

        Surfaces whose geometric angle is not monotonic about the magnetic axis
        (near an X-point) are skipped.

        Returns ``(psin_tab, theta_tab, Delta)`` with ``Delta`` of shape
        ``(len(psin_tab), n_theta)`` and ``theta_tab`` spanning [0, 2pi).
        """
        if levels is None:
            levels = np.linspace(0.02, max_level, 200)

        th_uniform = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
        keep_s, keep_d = [], []

        for s_lvl in levels:
            pts = self.surface(float(s_lvl))
            if pts is None:
                continue
            cum, tot = self._loop_integral(pts)
            if tot <= 0:
                continue
            th_star = TWO_PI * cum[:-1] / tot                   # (N,), 0 -> 2pi
            tg = np.arctan2(pts[:, 1] - self.z_axis, pts[:, 0] - self.r_axis)
            tgu = np.unwrap(tg)

            # One strictly monotonic poloidal circuit.  Fails near an X-point,
            # where the surface is no longer star-shaped about the axis.
            if np.any(np.diff(tgu) <= 0):
                continue
            gain = tgu[-1] - tgu[0]
            if not (0.75 * TWO_PI < gain < TWO_PI):
                continue

            # Delta = theta* - theta_geo is single-valued and 2pi-periodic,
            # because both angles gain exactly 2pi per circuit.
            delta = th_star - tgu
            a = np.mod(tgu, TWO_PI)
            order = np.argsort(a)
            a_s, d_s = a[order], delta[order]
            a_ext = np.r_[a_s[-1] - TWO_PI, a_s, a_s[0] + TWO_PI]
            d_ext = np.r_[d_s[-1], d_s, d_s[0]]
            keep_d.append(np.interp(th_uniform, a_ext, d_ext))
            keep_s.append(float(s_lvl))

        if not keep_s:
            raise RuntimeError("no usable flux surfaces for theta* construction")
        return np.asarray(keep_s), th_uniform, np.asarray(keep_d)

    def theta_star_field(self, levels=None, n_theta: int = 512, pad: int = 8):
        """An analytic straight-field-line angle evaluator.

        Returns a :class:`ThetaStarField` giving ``theta*`` and its (R, Z)
        derivatives at ARBITRARY points, without ever rasterising an
        oscillatory field onto the coarse 65x65 EFIT grid.  That matters: a
        rasterised ``cos(m theta*)`` leaks a few percent of spurious poloidal
        harmonics, and a spurious m' seeds a fake island at q = m'/n.
        """
        s_tab, th_tab, D = self.theta_star_geometry(levels=levels, n_theta=n_theta)
        return ThetaStarField(self, s_tab, th_tab, D, pad=pad)


class ThetaStarField:
    """theta*(R, Z) = theta_geo + Delta(psi_n, theta_geo), with derivatives.

    Delta is held on a well-resolved (psi_n, theta_geo) table where it is
    smooth and slowly varying -- unlike cos(m theta*), which oscillates.  The
    (R, Z) derivatives follow by the chain rule::

        d theta*/dR = (d theta_g/dR)(1 + dDelta/d theta_g) + (dDelta/d psi_n)(d psi_n/dR)

    so the only interpolation is of a smooth function.
    """

    def __init__(self, eq: "Equilibrium", s_tab, th_tab, D, pad: int = 8):
        self.eq = eq
        self.s_lo, self.s_hi = float(s_tab[0]), float(s_tab[-1])
        # periodic padding so the spline is smooth across the theta seam
        th = np.asarray(th_tab, float)
        dth = float(th[1] - th[0])
        left = th[:1] - dth * np.arange(pad, 0, -1)
        right = th[-1] + dth * np.arange(1, pad + 1)
        th_ext = np.concatenate([left, th, right])
        D_ext = np.concatenate([D[:, -pad:], D, D[:, :pad]], axis=1)
        self._spl = RectBivariateSpline(np.asarray(s_tab, float), th_ext, D_ext,
                                        kx=3, ky=3, s=0)
        self._dpsi = eq.psi_bdy - eq.psi_axis

    def _sn_theta(self, r, z, psi=None):
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        if psi is None:
            psi = self.eq.psi_at(r, z)
        sn = (psi - self.eq.psi_axis) / self._dpsi
        dr = r - self.eq.r_axis
        dz = z - self.eq.z_axis
        tg = np.mod(np.arctan2(dz, dr), TWO_PI)
        return r, z, sn, tg, dr, dz

    def value(self, r, z):
        r, z, sn, tg, _, _ = self._sn_theta(r, z)
        sq = np.clip(sn, self.s_lo, self.s_hi)
        return tg + self._spl.ev(sq, tg)

    def value_and_grad(self, r, z, psi=None, dpsi_dR=None, dpsi_dZ=None):
        """Return ``(theta*, d/dR, d/dZ, psi_n, d psi_n/dR, d psi_n/dZ)``.

        ``psi``/``dpsi_dR``/``dpsi_dZ`` may be passed in when the caller has
        already evaluated them, which halves the spline work in the tracer's
        inner loop.
        """
        r, z, sn, tg, dr, dz = self._sn_theta(r, z, psi=psi)
        sq = np.clip(sn, self.s_lo, self.s_hi)
        inside = (sn > self.s_lo) & (sn < self.s_hi)

        D = self._spl.ev(sq, tg)
        dD_ds = np.where(inside, self._spl.ev(sq, tg, dx=1, dy=0), 0.0)
        dD_dt = self._spl.ev(sq, tg, dx=0, dy=1)

        d2 = dr * dr + dz * dz
        d2 = np.where(d2 < 1e-12, 1e-12, d2)
        dtg_dR = -dz / d2
        dtg_dZ = dr / d2

        if dpsi_dR is None:
            dpsi_dR = self.eq.dpsi_dR(r, z)
        if dpsi_dZ is None:
            dpsi_dZ = self.eq.dpsi_dZ(r, z)
        dsn_dR = dpsi_dR / self._dpsi
        dsn_dZ = dpsi_dZ / self._dpsi

        dth_dR = dtg_dR * (1.0 + dD_dt) + dD_ds * dsn_dR
        dth_dZ = dtg_dZ * (1.0 + dD_dt) + dD_ds * dsn_dZ
        return tg + D, dth_dR, dth_dZ, sn, dsn_dR, dsn_dZ


def _regularize(g: np.ndarray, name: str, tol_frac: float = 1e-3) -> np.ndarray:
    """Snap a nominally-uniform axis to an exact linspace.

    ``tol_frac`` is the allowed deviation as a fraction of one cell.
    """
    if g.size < 2:
        raise ValueError(f"{name} grid needs at least 2 points")
    lin = np.linspace(g[0], g[-1], g.size)
    cell = abs(lin[1] - lin[0])
    dev = float(np.abs(g - lin).max())
    if dev > tol_frac * cell:
        raise ValueError(
            f"{name} grid is not uniform: max deviation {dev:.3e} m "
            f"= {dev / cell * 100:.3f}% of a cell (limit {tol_frac * 100:.3f}%)"
        )
    return lin

# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------
def _signed_area(pts: np.ndarray) -> float:
    r, z = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def _encircles(pts: np.ndarray, r0: float, z0: float) -> bool:
    """True if the closed polygon winds once around (r0, z0)."""
    ang = np.unwrap(np.arctan2(pts[:, 1] - z0, pts[:, 0] - r0))
    return abs(ang[-1] - ang[0]) > 1.5 * np.pi
