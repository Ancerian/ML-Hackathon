"""Perturbation backend driven by a real coil array, via Biot-Savart.

This is the second of two interchangeable backends:

* :class:`tokviz.perturbation.Perturbation` -- analytic.  The perturbation is a
  flux function, ``B = curl(psi_total grad phi)``, so it is divergence-free
  EXACTLY and purely POLOIDAL: it has no delta B_phi at all.
* :class:`CoilPerturbation` -- this one.  The field of an actual coil set, which
  is divergence-free exactly as Biot-Savart produces it, and does carry a
  toroidal component.

The difference is not cosmetic.  delta B_phi changes the denominator of the
field-line equations ``dR/dphi = R B_R / B_phi``, which the analytic backend
cannot represent at any amplitude.

Why a Fourier representation
----------------------------
Calling Biot-Savart inside the ODE right-hand side would mean summing ~1000
segments per seed per step.  Instead the VECTOR POTENTIAL A (not B) is evaluated
once on an (R, Z) grid at a set of toroidal angles, Fourier-transformed in phi,
and the dominant harmonics kept::

    A(R, Z, phi) = sum_n [ C_n(R,Z) cos(n phi) + S_n(R,Z) sin(n phi) ]

Each C_n, S_n becomes a spline and B = curl A is taken analytically from the
spline derivatives -- a handful of lookups, the analytic backend's cost structure.

Truncation therefore does NOT break div B = 0: div(curl A) vanishes structurally.
(Holding B itself on the grid did break it -- see the class docstring, U3.)
:meth:`CoilPerturbation.divergence_report` still measures it, because losing
flux preservation silently is the failure this project has been bitten by once.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .equilibrium import Equilibrium, TWO_PI
from .rmp_coils import vector_potential, cyl_to_cart, cart_to_cyl_vec
from .perturbation import island_width_psin


class CoilPerturbation:
    """Vacuum field of a coil array, as a drop-in perturbation backend.

    The field is held as the VECTOR POTENTIAL, Fourier in phi and splined in
    (R, Z), and B is taken as its analytic curl.  That makes ``div B`` vanish
    structurally: spline mixed partials commute exactly, so ``div(curl A) = 0``
    to roundoff no matter how coarse the grid is.

    Holding B directly instead does not work.  Measured on a 65x65 grid with
    five harmonics, the relative ``|div B|`` reached 2.1 at the plasma edge --
    larger than the field itself -- and that error masquerades as chaos: psi
    stops being conserved, the weighted-Birkhoff convergence collapses, and the
    classifier reports a chaotic sea where there are islands.
    """

    def __init__(self, eq: Equilibrium, loops, n_keep=(1, 3, 5, 7, 9, 11, 13, 15),
                 n_phi=96, pad=0.0, current=1.0, grid_shape=(129, 129),
                 gain=1.0):
        self.eq = eq
        #: Linear scale applied at evaluation time.  The field is linear in the
        #: coil current, so a current scan needs one build, not one per value.
        self.gain = float(gain)
        self.loops = [(v, I * current) for v, I in loops]
        self.n_keep = tuple(int(n) for n in n_keep)
        self.current = float(current)

        nR = grid_shape[1] if grid_shape else max(eq.R.size, 129)
        nZ = grid_shape[0] if grid_shape else max(eq.Z.size, 129)
        self.R = np.linspace(eq.R[0] - pad, eq.R[-1] + pad, nR)
        self.Z = np.linspace(eq.Z[0] - pad, eq.Z[-1] + pad, nZ)
        RR, ZZ = np.meshgrid(self.R, self.Z)            # (nZ, nR)

        phis = np.linspace(0.0, TWO_PI, n_phi, endpoint=False)
        AR = np.empty((n_phi, nZ, nR))
        AP = np.empty_like(AR)
        AZ = np.empty_like(AR)
        flatR, flatZ = RR.ravel(), ZZ.ravel()
        for j, ph in enumerate(phis):
            P = cyl_to_cart(flatR, np.full_like(flatR, ph), flatZ)
            A = vector_potential(self.loops, P)
            ar, ap, az = cart_to_cyl_vec(A, np.full_like(flatR, ph))
            AR[j] = ar.reshape(nZ, nR)
            AP[j] = ap.reshape(nZ, nR)
            AZ[j] = az.reshape(nZ, nR)

        self._spl = {}
        self._power = {}
        for name, arr in (("R", AR), ("P", AP), ("Z", AZ)):
            F = np.fft.rfft(arr, axis=0) / n_phi
            self._power[name] = np.abs(F).sum(axis=(1, 2))
            for n in self.n_keep:
                if n >= F.shape[0]:
                    continue
                c = 2.0 * F[n].real if n > 0 else F[0].real
                sN = -2.0 * F[n].imag if n > 0 else np.zeros_like(F[0].real)
                self._spl[(name, n, "c")] = RectBivariateSpline(
                    self.R, self.Z, c.T, kx=3, ky=3, s=0)
                self._spl[(name, n, "s")] = RectBivariateSpline(
                    self.R, self.Z, sN.T, kx=3, ky=3, s=0)

    # -- internals ---------------------------------------------------------
    def _A_and_derivs(self, rc, zc, phi):
        """A and its (R, Z, phi) derivatives, all analytic."""
        A = {}
        dA_dR = {}
        dA_dZ = {}
        dA_dP = {}
        for name in ("R", "P", "Z"):
            v = np.zeros_like(rc)
            vR = np.zeros_like(rc)
            vZ = np.zeros_like(rc)
            vP = np.zeros_like(rc)
            for n in self.n_keep:
                kc, ks = (name, n, "c"), (name, n, "s")
                if kc not in self._spl:
                    continue
                sc, ss = self._spl[kc], self._spl[ks]
                cn, sn_ = np.cos(n * phi), np.sin(n * phi)
                c0, s0 = sc.ev(rc, zc), ss.ev(rc, zc)
                v = v + c0 * cn + s0 * sn_
                vR = vR + sc.ev(rc, zc, dx=1, dy=0) * cn + ss.ev(rc, zc, dx=1, dy=0) * sn_
                vZ = vZ + sc.ev(rc, zc, dx=0, dy=1) * cn + ss.ev(rc, zc, dx=0, dy=1) * sn_
                vP = vP + n * (-c0 * sn_ + s0 * cn)
            A[name], dA_dR[name], dA_dZ[name], dA_dP[name] = v, vR, vZ, vP
        return A, dA_dR, dA_dZ, dA_dP

    # -- the backend interface --------------------------------------------
    def delta_B(self, r, z, phi, **_ignored):
        """``(dB_R, dB_Z, dB_phi)`` as the analytic curl of A, in cylindrical.

            B_R   = (1/R) dA_Z/dphi - dA_phi/dZ
            B_phi = dA_R/dZ - dA_Z/dR
            B_Z   = A_phi/R + dA_phi/dR - (1/R) dA_R/dphi
        """
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        rc = np.clip(r, self.R[0], self.R[-1])
        zc = np.clip(z, self.Z[0], self.Z[-1])
        A, dR, dZ, dP = self._A_and_derivs(rc, zc, phi)
        g = self.gain
        BR = g * (dP["Z"] / rc - dZ["P"])
        BP = g * (dZ["R"] - dR["Z"])
        BZ = g * (A["P"] / rc + dR["P"] - dP["R"] / rc)
        return BR, BZ, BP

    # -- diagnostics -------------------------------------------------------
    def harmonic_power(self):
        """Toroidal power spectrum of each vector-potential component."""
        return {k: v.tolist() for k, v in self._power.items()}

    def retained_fraction(self):
        out = {}
        for name, p in self._power.items():
            tot = float(p[1:].sum())
            kept = float(sum(p[n] for n in self.n_keep if 0 < n < len(p)))
            out[name] = kept / tot if tot > 0 else 1.0
        return out

    def divergence_report(self, n_sample=24, psi_n_max=0.95, seed=0):
        """Relative |div B| of the reconstructed field, inside the plasma.

        Should now be at roundoff: B is the analytic curl of a spline field, and
        spline mixed partials commute, so div(curl A) vanishes identically. This
        is the gate that says so rather than assuming it.
        """
        rng = np.random.default_rng(seed)
        r, z = [], []
        guard = 0
        while len(r) < n_sample and guard < 20000:
            guard += 1
            rr = rng.uniform(self.eq.R[0], self.eq.R[-1])
            zz = rng.uniform(self.eq.Z[0], self.eq.Z[-1])
            sn = float(self.eq.psi_n(np.array([rr]), np.array([zz]))[0])
            if np.isfinite(sn) and sn <= psi_n_max:
                r.append(rr); z.append(zz)
        r = np.asarray(r); z = np.asarray(z)
        ph = rng.uniform(0.0, TWO_PI, r.size)
        h = 1e-4
        bR_p, _, _ = self.delta_B(r + h, z, ph)
        bR_m, _, _ = self.delta_B(r - h, z, ph)
        _, bZ_p, _ = self.delta_B(r, z + h, ph)
        _, bZ_m, _ = self.delta_B(r, z - h, ph)
        _, _, bP_p = self.delta_B(r, z, ph + h)
        _, _, bP_m = self.delta_B(r, z, ph - h)
        bR, bZ, bP = self.delta_B(r, z, ph)
        d_rBr = ((r + h) * bR_p - (r - h) * bR_m) / (2 * h)
        div = d_rBr / r + (bP_p - bP_m) / (2 * h) / r + (bZ_p - bZ_m) / (2 * h)
        scale = np.sqrt(bR ** 2 + bZ ** 2 + bP ** 2).mean() / np.mean(r)
        return dict(max_abs=float(np.max(np.abs(div))),
                    rms=float(np.sqrt(np.mean(div ** 2))),
                    relative_rms=float(np.sqrt(np.mean(div ** 2)) / max(scale, 1e-30)),
                    psi_n_max=float(psi_n_max), n_sample=int(r.size))

    def calibrate_to_island_width(self, target_W, m, n, psi_n_res, dq_dpsin,
                                  q_res, tsf, n_theta=128):
        """Scale the coil current so the (m, n) resonant harmonic gives ``target_W``.

        Lets a coil-driven run be compared with an analytic run at matched island
        size rather than at matched, and meaningless, coil amperage.
        """
        pts = self.eq.surface(float(psi_n_res))
        if pts is None:
            raise RuntimeError(f"no closed surface at psi_n={psi_n_res}")
        th = np.mod(tsf.value(pts[:, 0], pts[:, 1]), TWO_PI)
        order = np.argsort(th)
        th_s = th[order]
        R_s, Z_s = pts[order, 0], pts[order, 1]
        tu = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
        Ri = np.interp(tu, th_s, R_s, period=TWO_PI)
        Zi = np.interp(tu, th_s, Z_s, period=TWO_PI)

        gR = self.eq.dpsi_dR(Ri, Zi)
        gZ = self.eq.dpsi_dZ(Ri, Zi)
        g = np.hypot(gR, gZ)
        g = np.where(g < 1e-12, 1e-12, g)
        bR, bZ, _ = self.delta_B(Ri, Zi, 0.0)
        bn = (bR * gR + bZ * gZ) / g
        amp_norm = 2.0 * np.abs(np.mean(bn * np.exp(-1j * m * tu)))

        # delta psi_mn ~ R |grad psi| b_mn / m  (the normal field of a flux
        # perturbation A g cos(m theta* - n phi) is ~ m |grad psi| A g / R)
        R0 = float(np.mean(Ri))
        g0 = float(np.mean(g))
        dpsi_mn = amp_norm * R0 / max(m, 1) * 1.0
        eps_now = dpsi_mn / abs(self.eq.psi_bdy - self.eq.psi_axis) / max(g0, 1e-12)
        # same pendulum width as perturbation.island_width_psin (W ~ sqrt(eps q))
        W_now = island_width_psin(max(eps_now, 0.0), q_res, dq_dpsin)
        if W_now <= 0:
            raise RuntimeError("degenerate calibration")
        scale = (target_W / W_now) ** 2          # W ~ sqrt(amplitude)
        return float(scale), float(W_now)
