#!/usr/bin/env python3
"""Числа й рисунки для «Гайду по геометрії токамака» (geometry/main.tex).

Два приклади:
  * DIII-D #203702, робочий кадр tokamak-3d-viz (q95 ~ 3,7): реальна EFIT-рівновага;
  * JET 1975: D-межа Міллера і рівновага Соловйова з tokviz.machine.

Запуск (з кореня проєкту; потрібне середовище конкурсу):
    PY="fusion equilibrium challenge/starter/.venv/bin/python"
    "$PY" geometry/code/geom_tour.py            # числа -> geometry/code/geom_tour.json
    "$PY" geometry/code/geom_tour.py --figs     # + рисунки в geometry/figures/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tokamak-3d-viz" / "src"))

from tokviz.config import DATA_DIR                                   # noqa: E402
from tokviz.equilibrium import (Equilibrium, ThetaStarField, load_shot,  # noqa: E402
                                select_frame)
from tokviz.surfaces import d_shape                                  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
FIG = ROOT / "geometry" / "figures"


# ---------------------------------------------------------------------------
# параметри форми за крайніми точками замкненого контуру
# ---------------------------------------------------------------------------
def shape_params(R, Z):
    """R_geo, a, A, kappa, delta_u, delta_l, delta за крайніми точками контуру."""
    R, Z = np.asarray(R, float), np.asarray(Z, float)
    Rmax, Rmin = R.max(), R.min()
    R_geo = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    iu, il = np.argmax(Z), np.argmin(Z)
    kappa = (Z[iu] - Z[il]) / (2 * a)
    d_u = (R_geo - R[iu]) / a                   # верхня трикутність
    d_l = (R_geo - R[il]) / a                   # нижня трикутність
    return {"R_geo": R_geo, "a": a, "A": R_geo / a, "kappa": kappa,
            "delta_u": d_u, "delta_l": d_l, "delta": 0.5 * (d_u + d_l),
            "Z_geo": 0.5 * (Z[iu] + Z[il]), "R_top": R[iu], "Z_top": Z[iu],
            "R_bot": R[il], "Z_bot": Z[il]}


def area_volume(R, Z):
    """Площа перерізу (формула Гаусса) і об'єм тора (теорема Паппа--Гульдіна)."""
    R, Z = np.asarray(R, float), np.asarray(Z, float)
    S = 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
    Rc = np.sum((R + np.roll(R, -1)) * (R * np.roll(Z, -1) - np.roll(R, -1) * Z)) / (6 * S)
    return abs(S), 2 * np.pi * Rc * abs(S), Rc


# ---------------------------------------------------------------------------
# критичні точки psi: O-точка (вісь) і X-точка (сідло)
# ---------------------------------------------------------------------------
def ip_ampere(eq: Equilibrium, R, Z):
    """Струм плазми за законом Ампера: I_p = (1/mu0) * циркуляція B_pol уздовж замкненого контуру."""
    Rc, Zc = np.r_[R, R[0]], np.r_[Z, Z[0]]
    dR, dZ = np.diff(Rc), np.diff(Zc)
    rm, zm = 0.5 * (Rc[:-1] + Rc[1:]), 0.5 * (Zc[:-1] + Zc[1:])
    return float(np.sum(eq.B_R(rm, zm) * dR + eq.B_Z(rm, zm) * dZ) / (4e-7 * np.pi))


def find_x_point(eq: Equilibrium, r0, z0):
    """Сідло psi біля (r0, z0): мінімум |grad psi|^2 на сплайні."""
    f = lambda p: float(eq.dpsi_dR(*p) ** 2 + eq.dpsi_dZ(*p) ** 2)
    res = minimize(f, [r0, z0], method="Nelder-Mead",
                   options=dict(xatol=1e-8, fatol=1e-16, maxiter=4000))
    return float(res.x[0]), float(res.x[1])


def hessian_det(eq: Equilibrium, r, z):
    s = eq._spl
    prr = float(s.ev(r, z, dx=2, dy=0))
    pzz = float(s.ev(r, z, dx=0, dy=2))
    prz = float(s.ev(r, z, dx=1, dy=1))
    return prr * pzz - prz ** 2


# ---------------------------------------------------------------------------
def diiid(figs=False):
    shot = load_shot(DATA_DIR / "d3d_shot_203702.parquet")
    k = select_frame(shot)
    eq = Equilibrium(shot.psirz[k], shot.grid_R, shot.grid_Z, machine="DIII-D")
    ra, za, pa = eq.find_axis()
    n = int(shot.lcfs_n[k])
    lr, lz = shot.lcfs_r[k][:n], shot.lcfs_z[k][:n]
    eq.set_boundary_from_contour(lr, lz)
    F = eq.calibrate_F(float(shot.q95[k]))
    sp = shape_params(lr, lz)
    S, V, Rc = area_volume(lr, lz)
    # X-точка: старт біля найнижчої точки межі
    xr, xz = find_x_point(eq, sp["R_bot"], sp["Z_bot"])
    levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    lv, qv = eq.q_profile(levels)
    # theta* - theta_geo. Початок відліку theta* у theta_star_geometry -- перша точка контуру
    # skimage, тобто довільний і різний на сусідніх поверхнях; віднімаємо Delta(theta_geo = 0),
    # щоб theta* = 0 на зовнішньому екваторі кожної поверхні (див. розд. 3 гайду).
    s_all, th_all, D_all = eq.theta_star_geometry()
    off = D_all[:, 0]
    D_fix = D_all - off[:, None]
    pick = [int(np.argmin(np.abs(s_all - x))) for x in (0.2, 0.5, 0.9)]
    grad = {}
    for lev in (0.3, 0.6, 0.9):
        pts = eq.surface(lev)
        for name, Dt in (("raw", D_all), ("fixed", D_fix)):
            _, gr, gz, *_ = ThetaStarField(eq, s_all, th_all, Dt).value_and_grad(pts[:, 0], pts[:, 1])
            grad[f"{lev:.1f}_{name}"] = float(np.sqrt(np.mean(gr * gr + gz * gz)))
    out = {
        "frame": k, "t_ms": float(shot.times[k]), "q95_efit": float(shot.q95[k]),
        "grid": [int(eq.R.size), int(eq.Z.size)], "dR_m": eq.dR, "dZ_m": eq.dZ,
        "axis_RZ": [ra, za], "r_axis_efit": float(shot.r_axis[k]), "z_axis_efit": float(shot.z_axis[k]),
        "psi_axis": pa, "psi_bdy": eq.psi_bdy, "axis_sign": eq.axis_sign,
        "F_Tm": F, "Bphi_axis_T": F / ra, "Bphi_Rgeo_T": F / sp["R_geo"],
        "Ip_ampere_A": ip_ampere(eq, lr, lz), "shape": sp, "area_m2": S, "volume_m3": V, "R_centroid": Rc,
        "x_point_RZ": [xr, xz], "psiN_x": float(eq.psi_n(xr, xz)),
        "hess_det_axis": hessian_det(eq, ra, za), "hess_det_x": hessian_det(eq, xr, xz),
        "q_profile": dict(zip([f"{x:.2f}" for x in lv], [float(x) for x in qv])),
        "theta_star_max_dev_deg": {f"{s_all[i]:.1f}": float(np.degrees(np.abs(D_fix[i]).max()))
                                   for i in pick},
        "theta_star_origin_deg": [float(np.degrees(off.min())), float(np.degrees(off.max()))],
        "theta_star_origin_max_jump_deg": float(np.degrees(np.abs(np.diff(off)).max())),
        "theta_star_rms_grad_rad_per_m": grad,
        "Bpol_x": float(eq.B_pol(xr, xz)), "Bpol_outboard": float(eq.B_pol(sp["R_geo"] + sp["a"], za)),
    }
    if figs:
        _fig_diiid(eq, lr, lz, sp, (xr, xz))
        _fig_thetastar(eq, ThetaStarField(eq, s_all, th_all, D_fix), lr, lz)
    return out


def jet(figs=False):
    from tokviz.machine.equilibrium_analytic import (JET_A, JET_B, JET_DELTA, JET_R0,
                                                     build_equilibrium, miller_boundary)
    from tokviz.machine.ripple import ripple, tf_filaments, winding_centreline
    kappa = JET_B / JET_A
    b = miller_boundary(JET_R0, JET_A, kappa, JET_DELTA, 2000)
    sp = shape_params(b[:, 0], b[:, 1])
    S, V, _ = area_volume(b[:, 0], b[:, 1])
    eq, sol, info = build_equilibrium()
    lv, qv = eq.q_profile([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
    xs = sol.boundary(4000)
    loops = tf_filaments(winding_centreline(), current_total=4.1e7)
    d, eps, Bmax, _ = ripple(loops, np.array([JET_R0, 4.21]))
    out = {"miller": {"R0": JET_R0, "a": JET_A, "b": JET_B, "kappa": kappa, "delta": JET_DELTA},
           "shape_from_contour": sp, "area_m2": S, "volume_m3": V,
           "solovev": {k: info[k] for k in ("A", "psi0_Wb_per_rad", "Ip_A", "beta_p", "beta_t",
                                             "li1", "li3", "volume_m3", "area_m2",
                                             "boundary_residual_max_m")},
           "axis_RZ": [eq.r_axis, eq.z_axis], "shafranov_shift_m": eq.r_axis - JET_R0,
           "q_axis": eq.q_axis(),
           "q_profile": dict(zip([f"{x:.2f}" for x in lv], [float(x) for x in qv])),
           "ripple_delta": {"R0": float(d[0]), "4.21": float(d[1])},
           "ripple_eps_src": {"R0": float(eps[0]), "4.21": float(eps[1])},
           "B_R0_T": float(Bmax[0])}
    if figs:
        _fig_jet(eq, sol, lv, qv)
    return out


# ---------------------------------------------------------------------------
# рисунки
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    return plt


def _fig_diiid(eq, lr, lz, sp, xp):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    RR, ZZ = np.meshgrid(eq.R, eq.Z)
    pn = eq.psi_n_grid()
    ax.contour(RR, ZZ, pn, levels=np.arange(0.1, 1.0, 0.1), colors="0.55", linewidths=0.7)
    ax.contour(RR, ZZ, pn, levels=[1.03, 1.08, 1.15], colors="0.75", linewidths=0.6,
               linestyles="--")
    ax.plot(np.r_[lr, lr[0]], np.r_[lz, lz[0]], color="C0", lw=2, label="LCFS (EFIT)")
    ax.plot(eq.r_axis, eq.z_axis, "ko", ms=6, label="O-точка (вісь)")
    ax.plot(*xp, "x", color="C3", ms=11, mew=2.5, label="X-точка")
    ax.axvline(sp["R_geo"], color="C2", lw=0.8, ls=":")
    for Rv in (sp["R_geo"] - sp["a"], sp["R_geo"] + sp["a"]):
        ax.plot([Rv], [sp["Z_geo"]], "s", color="C2", ms=5)
    ax.plot([sp["R_top"], sp["R_bot"]], [sp["Z_top"], sp["Z_bot"]], "^", color="C1", ms=7,
            label=r"$Z_{\max}$, $Z_{\min}$")
    ax.annotate("", xy=(sp["R_geo"] - sp["a"], sp["Z_geo"] - 0.05),
                xytext=(sp["R_geo"] + sp["a"], sp["Z_geo"] - 0.05),
                arrowprops=dict(arrowstyle="<->", color="C2"))
    ax.text(sp["R_geo"] + 0.05, sp["Z_geo"] - 0.14, "2a", color="C2")
    ax.text(sp["R_geo"] + 0.02, sp["Z_top"] + 0.06, r"$R_{\rm geo}$", color="C2")
    ax.set_xlabel("R, м"); ax.set_ylabel("Z, м"); ax.set_aspect("equal")
    ax.set_xlim(0.95, 2.45); ax.set_ylim(-1.45, 1.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(r"DIII-D #203702: $\psi_N$ = 0,1…0,9 і межа", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "geo_diiid_shape.jpg", dpi=200, pil_kwargs={"quality": 88})
    plt.close(fig)


def _fig_thetastar(eq, tsf, lr, lz):
    from matplotlib.path import Path as MplPath
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.9), sharey=True)
    RR, ZZ = np.meshgrid(np.linspace(eq.R[0], eq.R[-1], 400), np.linspace(eq.Z[0], eq.Z[-1], 560))
    pn = eq.psi_n(RR, ZZ)
    in_lcfs = MplPath(np.column_stack([lr, lz])).contains_points(
        np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
    inside = (pn > 0.02) & (pn < 0.985) & in_lcfs     # без приватної області під X-точкою
    tg = np.mod(np.arctan2(ZZ - eq.z_axis, RR - eq.r_axis), 2 * np.pi)
    ts = np.mod(tsf.value(RR, ZZ), 2 * np.pi)
    for ax, th, title in ((axs[0], tg, r"геометричний кут $\theta$"),
                          (axs[1], ts, r"кут прямих ліній $\theta^*$")):
        ax.contour(RR, ZZ, pn, levels=np.arange(0.1, 1.0, 0.1), colors="0.6", linewidths=0.6)
        # лінії сталого кута: контури cos/sin, щоб уникнути розриву 2pi
        for c0 in np.arange(0, 2 * np.pi, np.pi / 8):
            d = np.angle(np.exp(1j * (th - c0)))
            d = np.where(inside & (np.abs(d) < 0.5), d, np.nan)
            ax.contour(RR, ZZ, d, levels=[0.0], colors="C3", linewidths=0.9)
        ax.plot(eq.r_axis, eq.z_axis, "k.", ms=5)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xlabel("R, м")
        ax.set_xlim(0.95, 2.4); ax.set_ylim(-1.3, 1.25)
    axs[0].set_ylabel("Z, м")
    fig.tight_layout()
    fig.savefig(FIG / "geo_thetastar.jpg", dpi=200, pil_kwargs={"quality": 88})
    plt.close(fig)


def fig_miller():
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.4), sharey=True)
    R0, a = 3.0, 1.0
    for ax, var, vals, fixed, title in (
            (axs[0], "kappa", [1.0, 1.4, 1.8], dict(delta=0.0), r"$\delta=0$, змінюємо $\kappa$"),
            (axs[1], "delta", [0.0, 0.3, 0.6], dict(kappa=1.7), r"$\kappa=1{,}7$, $\delta>0$"),
            (axs[2], "delta", [0.0, -0.3, -0.6], dict(kappa=1.7), r"$\kappa=1{,}7$, $\delta<0$")):
        for i, v in enumerate(vals):
            kw = dict(fixed); kw[var] = v
            p = d_shape(R0, a, n=400, **kw)
            lab = (r"$\kappa=$" if var == "kappa" else r"$\delta=$") + f"{v:g}".replace(".", ",")
            ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], color=f"C{i}", label=lab)
        ax.axvline(R0, color="0.7", lw=0.6, ls=":")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.set_xlabel("R, м")
        ax.legend(fontsize=7, loc="center", framealpha=0.8)
    axs[0].set_ylabel("Z, м")
    fig.tight_layout()
    fig.savefig(FIG / "geo_miller.jpg", dpi=200, pil_kwargs={"quality": 88})
    plt.close(fig)


def _fig_jet(eq, sol, lv, qv):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 5.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    RR, ZZ = np.meshgrid(eq.R, eq.Z)
    pn = eq.psi_n_grid()
    ax.contour(RR, ZZ, pn, levels=np.arange(0.1, 1.0, 0.1), colors="0.55", linewidths=0.7)
    b = sol.boundary(800)
    ax.plot(np.r_[b[:, 0], b[0, 0]], np.r_[b[:, 1], b[0, 1]], "C0", lw=2, label="межа Міллера")
    ax.plot(eq.r_axis, eq.z_axis, "ko", ms=5, label="магнітна вісь")
    ax.axvline(sol.R0, color="C2", lw=0.8, ls=":", label=r"$R_0=2{,}96$ м")
    ax.set_aspect("equal"); ax.set_xlabel("R, м"); ax.set_ylabel("Z, м")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_title("JET 1975: Соловйов", fontsize=10)
    ax = axs[1]
    s = np.linspace(0.02, 0.99, 60)
    x, q = eq.q_profile(s)
    ax.plot(x, q, "C0")
    ax.plot([0.0], [eq.q_axis()], "ko", ms=4)
    ax.set_xlabel(r"$\psi_N$"); ax.set_ylabel("q"); ax.grid(alpha=0.3)
    ax.set_title(r"$q(\psi_N)$: плаский струм, $q_0\approx2{,}8$", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "geo_jet_solovev.jpg", dpi=200, pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    figs = "--figs" in sys.argv
    if figs:
        FIG.mkdir(parents=True, exist_ok=True)
        fig_miller()
    res = {"DIII-D": diiid(figs), "JET1975": jet(figs)}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
