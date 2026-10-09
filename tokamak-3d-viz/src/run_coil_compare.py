#!/usr/bin/env python3
"""VC1: is the analytic envelope g(psi_n) = psi_n**(m/2) a fair stand-in for the
radial profile a REAL external coil array produces?

Method
------
Compute the vacuum field of a DIII-D-like I-coil array by Biot-Savart, project
it onto the flux-surface normal, and Fourier-decompose in the straight-field-line
angle and the toroidal angle.  The resulting |b_mn(psi_n)| profile is compared
with the same quantity derived from the stand's analytic perturbation.

Both profiles are normalised, because only the SHAPE is under test -- the
absolute amplitude is a free parameter of the stand either way.

The published I-coil geometry differs between sources, so the comparison is
repeated over a range of coil radii and heights and the conclusion is reported
with that spread rather than from one quoted number.

    python src/run_coil_compare.py --out data/coil_compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokviz import config
from tokviz.equilibrium import TWO_PI
from tokviz.perturbation import Mode, Perturbation, resonant_surfaces
from tokviz.rmp_coils import icoil_array, biot_savart, cyl_to_cart, cart_to_cyl_vec
from run_analysis import build_equilibrium


def surface_grid(eq, tsf, level, n_theta, n_phi):
    """Points on one flux surface, uniform in theta*, with the poloidal normal."""
    pts = eq.surface(float(level))
    if pts is None or pts.shape[0] < 16:
        return None
    th = np.mod(tsf.value(pts[:, 0], pts[:, 1]), TWO_PI)
    order = np.argsort(th)
    th_s, r_s, z_s = th[order], pts[order, 0], pts[order, 1]
    # periodic interpolation onto a uniform theta* grid
    tu = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    th_ext = np.r_[th_s[-1] - TWO_PI, th_s, th_s[0] + TWO_PI]
    R = np.interp(tu, th_ext, np.r_[r_s[-1], r_s, r_s[0]])
    Z = np.interp(tu, th_ext, np.r_[z_s[-1], z_s, z_s[0]])

    gR = eq.dpsi_dR(R, Z)
    gZ = eq.dpsi_dZ(R, Z)
    g = np.hypot(gR, gZ)
    g = np.where(g < 1e-12, 1e-12, g)
    nR, nZ = gR / g, gZ / g
    return tu, R, Z, nR, nZ


def resonant_harmonic(field_fn, eq, tsf, levels, m, n, n_theta=64, n_phi=32):
    """|b_mn(psi_n)| of the normal field, for a callable giving (B_R, B_Z)."""
    phis = np.linspace(0.0, TWO_PI, n_phi, endpoint=False)
    out = np.full(len(levels), np.nan)
    for i, lv in enumerate(levels):
        g = surface_grid(eq, tsf, lv, n_theta, n_phi)
        if g is None:
            continue
        tu, R, Z, nR, nZ = g
        bn = np.zeros((n_phi, n_theta))
        for j, ph in enumerate(phis):
            BR, BZ = field_fn(R, Z, ph)
            bn[j] = BR * nR + BZ * nZ
        # (m, n) component: average of bn * exp(-i(m theta* - n phi))
        ker = np.exp(-1j * (m * tu[None, :] - n * phis[:, None]))
        out[i] = np.abs(np.mean(bn * ker))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(config.DATA_DIR / config.DEFAULT_SHOT))
    ap.add_argument("--out", default="data/coil_compare")
    ap.add_argument("--modes", default="2/1,3/1")
    ap.add_argument("--n-levels", type=int, default=26)
    ap.add_argument("--n-theta", type=int, default=64)
    ap.add_argument("--n-phi", type=int, default=32)
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shot, k, eq = build_equilibrium(args.shot)
    x, q = eq.q_profile(np.linspace(0.03, 0.99, 140))
    tsf = eq.theta_star_field()
    levels = np.linspace(0.20, 0.95, args.n_levels)
    print(f"{shot.source} frame {k}; {args.n_levels} surfaces x "
          f"{args.n_theta} theta* x {args.n_phi} phi", flush=True)

    mspec = []
    for tok in args.modes.split(","):
        a, _, b = tok.strip().partition("/")
        mspec.append((int(a), int(b)))

    # --- geometry variants, since the published numbers differ ------------
    variants = [
        dict(name="nominal", R_coil=2.184, z_centre=0.754, height=0.50),
        dict(name="R -10%", R_coil=2.184 * 0.90, z_centre=0.754, height=0.50),
        dict(name="R +10%", R_coil=2.184 * 1.10, z_centre=0.754, height=0.50),
        dict(name="z -25%", R_coil=2.184, z_centre=0.754 * 0.75, height=0.50),
        dict(name="z +25%", R_coil=2.184, z_centre=0.754 * 1.25, height=0.50),
        dict(name="tall coils", R_coil=2.184, z_centre=0.754, height=0.80),
    ]

    results = {}
    for var in variants:
        loops = icoil_array(R_coil=var["R_coil"], z_centre=var["z_centre"],
                            height=var["height"], n_toroidal=1)

        def coil_field(R, Z, ph, loops=loops):
            P = cyl_to_cart(R, np.full_like(R, ph), Z)
            B = biot_savart(loops, P)
            BR, _, BZ = cart_to_cyl_vec(B, np.full_like(R, ph))
            return BR, BZ

        for (m, n) in mspec:
            key = f"{var['name']}|{m}/{n}"
            prof = resonant_harmonic(coil_field, eq, tsf, levels, m, n,
                                     args.n_theta, args.n_phi)
            results[key] = prof.tolist()
        print(f"  {var['name']:12s} done ({time.time()-t0:.0f}s)", flush=True)

    # --- the stand's own analytic envelope, same measurement --------------
    analytic = {}
    for (m, n) in mspec:
        roots = resonant_surfaces(x, q, m / n)
        md = Mode(m=m, n=n, amp=1.0e-3)
        md.psin_res = roots[0] if roots else 0.5
        pert = Perturbation(eq, [md], tsf=tsf)

        def model_field(R, Z, ph, pert=pert):
            dR, dZ = pert.dpsi_derivs(R, Z, ph)
            return -dZ / R, dR / R

        analytic[f"{m}/{n}"] = resonant_harmonic(model_field, eq, tsf, levels,
                                                 m, n, args.n_theta,
                                                 args.n_phi).tolist()
    print(f"  analytic envelope done ({time.time()-t0:.0f}s)", flush=True)

    # --- compare SHAPES ---------------------------------------------------
    def norm(a):
        a = np.asarray(a, float)
        mx = np.nanmax(np.abs(a))
        return a / mx if mx > 0 else a

    def fit_exponent(prof, lo=0.45):
        """Best-fit p in |b| ~ psi_n**p over the outer region.

        This is the number the stand actually uses: its envelope is
        psi_n**(m/2), so p should come out near m/2 if the analytic form is a
        fair stand-in.
        """
        a = np.asarray(prof, float)
        ok = np.isfinite(a) & (levels >= lo) & (a > 0)
        if ok.sum() < 4:
            return np.nan
        return float(np.polyfit(np.log(levels[ok]), np.log(a[ok]), 1)[0])

    print(f"\n{'mode':>6} {'variant':>12} {'corr':>8} {'p_coil':>8} "
          f"{'p_model':>8} {'m/2':>6} {'monotonic?':>11}")
    summary = []
    for (m, n) in mspec:
        am = norm(analytic[f"{m}/{n}"])
        p_model = fit_exponent(am)
        for var in variants:
            cp = norm(results[f"{var['name']}|{m}/{n}"])
            ok = np.isfinite(cp) & np.isfinite(am)
            corr = float(np.corrcoef(cp[ok], am[ok])[0, 1]) if ok.sum() > 3 else np.nan
            p_coil = fit_exponent(cp)
            outer = cp[np.isfinite(cp) & (levels >= 0.45)]
            mono = bool(np.all(np.diff(outer) > -0.02)) if outer.size > 2 else False
            summary.append(dict(m=m, n=n, variant=var["name"], corr=corr,
                                p_coil=p_coil, p_model=p_model,
                                m_over_2=m / 2.0, monotonic_outer=mono))
            print(f"{m}/{n:<4} {var['name']:>12} {corr:>8.3f} {p_coil:>8.2f} "
                  f"{p_model:>8.2f} {m/2.0:>6.1f} {str(mono):>11}")

    # --- figure -----------------------------------------------------------
    fig, axs = plt.subplots(1, len(mspec), figsize=(7 * len(mspec), 5))
    axs = np.atleast_1d(axs)
    for ax, (m, n) in zip(axs, mspec):
        for var in variants:
            ax.plot(levels, norm(results[f"{var['name']}|{m}/{n}"]),
                    lw=1.2, alpha=0.75, label=f"coils, {var['name']}")
        ax.plot(levels, norm(analytic[f"{m}/{n}"]), "k--", lw=2.4,
                label=r"stand: $\psi_N^{m/2}$")
        ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel(r"$|b_{mn}|$, normalised")
        ax.set_title(f"m/n = {m}/{n}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("VC1: radial profile of the resonant normal field — "
                 "real coil array vs the stand's analytic envelope")
    fig.tight_layout()
    fig.savefig(out / "coil_compare.png", dpi=130)

    (out / "coil_compare.json").write_text(json.dumps(dict(
        levels=levels.tolist(), coil_profiles=results, analytic=analytic,
        variants=variants, summary=summary,
        total_seconds=time.time() - t0), indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}/coil_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
