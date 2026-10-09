#!/usr/bin/env python3
"""Figure for the letter to M. Landreman: our independent reconstruction of the iota = 2 family
(eps = 1/2, the setting of his Fig. 1) and the force-balance residual for both families.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/99-bibliography/landreman2026/fig_email.py"
"""
import io, contextlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec((HERE / "verify_landreman.py").read_text(), ns)
fam1, fam2, jac, grad = ns["fam1"], ns["fam2"], ns["jac"], ns["grad"]

C = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", ink="#0b0b0b", ink2="#52514e",
         muted="#8a8984", grid="#e6e5e0", surf="#fcfcfb")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.linewidth": 0.8})

# ---------- (a) flux-surface cross-sections, iota = 2 family, eps = 1/2 ----------
eps, delta = 0.5, 1 / 64
a, b, B, psi, p, pos = fam1(eps)
zeta = np.linspace(0, 2 * np.pi, 4001)


def section(psival, phi0, n_alpha=241):
    R, Z = [], []
    for al in np.linspace(0, 2 * np.pi, n_alpha):
        u, v = -eps / 2 + np.sqrt(psival) * np.cos(al), np.sqrt(psival) * np.sin(al)
        r = np.array([pos(u, v, z) for z in zeta]).T
        ph = np.unwrap(np.arctan2(r[1], r[0]))
        tgt = ph[0] + np.mod(phi0 - ph[0], 2 * np.pi)       # bring phi0 into [ph[0], ph[0] + 2pi): no clamping
        z0 = np.interp(tgt, ph, zeta)                        # phi(zeta) is monotone (B.grad phi > 0)
        rr = pos(u, v, z0)
        R.append(np.hypot(rr[0], rr[1])); Z.append(rr[2])
    return np.array(R), np.array(Z)


def axis_point(phi0):
    g = np.array([np.sqrt(1 - eps ** 2) * np.cos(zeta), np.sqrt(1 - eps ** 2) * np.sin(zeta), eps / 2 * np.sin(2 * zeta)])
    ph = np.unwrap(np.arctan2(g[1], g[0]))
    z0 = np.interp(ph[0] + np.mod(phi0 - ph[0], 2 * np.pi), ph, zeta)
    return np.sqrt(1 - eps ** 2), eps / 2 * np.sin(2 * z0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), dpi=200, gridspec_kw=dict(width_ratios=[1, 1.15]))
fig.patch.set_facecolor(C["surf"])
phis = [(0.0, "φ = 0", C["s1"]), (np.pi / 4, "φ = π/4", C["s2"]), (np.pi / 2, "φ = π/2", C["s3"])]
for phi0, lab, col in phis:
    for k, psival in enumerate((delta, 4 * delta / 9, delta / 9)):
        R, Z = section(psival, phi0)
        ax1.plot(R, Z, color=col, lw=2.0 if k == 0 else 1.2, label=lab if k == 0 else None,
                 solid_capstyle="round")
        if k == 0 and phi0 in (0.0, np.pi / 2):              # stellarator symmetry: these sections are up-down symmetric
            asym = abs(Z.max() + Z.min()) / (Z.max() - Z.min())
            assert asym < 1e-3, f"section phi={phi0} not up-down symmetric ({asym:.3f})"
    Ra, Za = axis_point(phi0)
    ax1.plot(Ra, Za, "o", ms=5, color=col, mec=C["surf"], mew=1.5)
ax1.set_aspect("equal")
ax1.set_xlabel("R"); ax1.set_ylabel("Z")
ax1.set_title("(a) ι = 2, ϵ = 1/2: ψ = δ, 4δ/9, δ/9 (δ = 1/64)", loc="left", fontsize=10.5, color=C["ink"])
leg = ax1.legend(frameon=False, loc="upper right", fontsize=9.5, handlelength=1.6)
for t in leg.get_texts(): t.set_color(C["ink2"])

# ---------- (b) relative force-balance residual vs normalised radius ----------
rng = np.random.default_rng(3)


def residuals_f1(n=150):
    xs, rs = [], []
    for _ in range(n):
        k = np.sqrt(delta) * rng.uniform(0.15, 1.0)
        al, ze = rng.uniform(0, 2 * np.pi, 2)
        r = pos(-eps / 2 + k * np.cos(al), k * np.sin(al), ze)
        J = jac(B, r); bb = B(r)
        curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
        gp = grad(p, r)
        xs.append(k / np.sqrt(delta)); rs.append(np.linalg.norm(np.cross(curl, bb) - gp) / np.linalg.norm(gp))
    return np.array(xs), np.array(rs)


def residuals_f2(n=150, e2=1.08, S=3, d2=0.245, lam=3.5):
    B2, psi2, p2, pos2 = fam2(e2, S, lam); kb = np.sqrt(2 * d2)
    xs, rs = [], []
    for _ in range(n):
        k = kb * rng.uniform(0.15, 1.0)
        chi, ze = rng.uniform(0, 2 * np.pi, 2)
        r = pos2(-k * np.cos(chi), k * np.sin(chi), ze)
        J = jac(B2, r); bb = B2(r)
        curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
        gp = grad(p2, r)
        xs.append(k / kb); rs.append(np.linalg.norm(np.cross(curl, bb) - gp) / np.linalg.norm(gp))
    return np.array(xs), np.array(rs)


x1, r1 = residuals_f1(); x2, r2 = residuals_f2()
ax2.scatter(x1, r1, s=16, color=C["s1"], edgecolor=C["surf"], linewidth=0.6, label="ι = 2 family (ϵ = 1/2)", zorder=3)
ax2.scatter(x2, r2, s=16, color=C["s2"], edgecolor=C["surf"], linewidth=0.6, label="sheared-ι family (Fig. 2A)", zorder=3)
ax2.set_yscale("log"); ax2.set_ylim(1e-14, 1e-8); ax2.set_xlim(0, 1.05)
ax2.set_xlabel("normalised minor radius, √(ψ/δ)")
ax2.set_ylabel("|(∇×B)×B − ∇p| / |∇p|")
ax2.grid(True, axis="y", color=C["grid"], lw=0.8); ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False); ax1.spines[s].set_visible(False)
ax2.set_title("(b) force-balance residual, 4th-order finite differences", loc="left", fontsize=10.5, color=C["ink"])
leg = ax2.legend(frameon=False, loc="upper right", fontsize=9.5, markerscale=1.4)
for t in leg.get_texts(): t.set_color(C["ink2"])
fig.text(0.01, 0.005, "Independent numerical check by the AI Laboratory, Faculty of Physics and Mathematics, Igor Sikorsky KPI "
         "(formulas of arXiv:2609.26742 v2; our code).", fontsize=8, color=C["muted"])
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = HERE / "fig_email.png"
fig.savefig(out, facecolor=C["surf"])
print(out, f"median residual: F1 {np.median(r1):.1e} (max {r1.max():.1e}), F2 {np.median(r2):.1e} (max {r2.max():.1e})")
