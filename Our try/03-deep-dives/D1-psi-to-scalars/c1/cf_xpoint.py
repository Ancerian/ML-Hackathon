#!/usr/bin/env python3
"""C1 step 3 — single-null Solov'ev equilibria (Cerfon & Freidberg 2010) through the scorer.

Cerfon & Freidberg, "One size fits all" analytic solutions to the Grad-Shafranov equation,
Phys. Plasmas 17, 032502 (2010): u = u_p + sum_{i=1..12} c_i u_i, with the 7 up-down symmetric
homogeneous functions (reused from tokamak-3d-viz, equilibrium_analytic.CF_HOMOGENEOUS) plus
5 odd-in-y ones, fixed by 12 conditions for a LOWER single null:
    u = 0 at the outer/inner equatorial points, the top point and the X-point;
    u_y = 0 at both equatorial points, u_x = 0 at the top; grad u = 0 at the X-point;
    three curvature conditions (N1, N2, N3) at the equatorial points and the top.
X-point at (1 - 1.1 delta eps, -1.1 kappa eps), as in the paper.

Every odd function is verified to satisfy Delta* u = 0 exactly (Poly.gs()) before use, every
solution is checked (X-point is a saddle, the scorer's LCFS reaches it), and then the same
perturbation sweep as for real frames is run on a family of controlled shapes.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/c1/cf_xpoint.py"
Output: cf_xpoint.json, cf_xpoint.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc                                                     # noqa: E402
import alpha_tools as at                                                    # noqa: E402

sys.path.insert(0, str(sc.PROJECT / "tokamak-3d-viz" / "src"))
from tokviz.machine.equilibrium_analytic import CF_HOMOGENEOUS, _P         # noqa: E402

#: Cerfon-Freidberg odd-in-y homogeneous solutions psi_8 .. psi_12 (paper, eq. 8).
CF_ODD = [
    _P((1, 0, 1, 0)),
    _P((1, 2, 1, 0)),
    _P((1, 0, 3, 0), (-3, 2, 1, 1)),
    _P((3, 4, 1, 0), (-4, 2, 3, 0)),
    _P((8, 0, 5, 0), (-45, 4, 1, 0), (-80, 2, 3, 1), (60, 4, 1, 1)),
]


def check_homogeneous():
    xs, ys = np.linspace(0.6, 1.4, 7), np.linspace(-0.6, 0.6, 7)
    X, Y = np.meshgrid(xs, ys)
    return [float(np.max(np.abs(f.gs()(X, Y)))) for f in CF_HOMOGENEOUS + CF_ODD]


def solve_single_null(eps, kappa, delta, A):
    al = np.arcsin(delta)
    up = _P((1 / 8, 4, 0, 0)) + A * _P((0.5, 2, 0, 1), (-1 / 8, 4, 0, 0))
    N1 = -(1 + al) ** 2 / (eps * kappa ** 2)
    N2 = (1 - al) ** 2 / (eps * kappa ** 2)
    N3 = -kappa / (eps * np.cos(al) ** 2)
    xo, xi, xt, yt = 1 + eps, 1 - eps, 1 - delta * eps, kappa * eps
    xs, ys = 1 - 1.1 * delta * eps, -1.1 * kappa * eps

    def rows(f):
        fx, fy = f.dx(), f.dy()
        fxx, fyy = fx.dx(), fy.dy()
        return np.array([
            f(xo, 0.0), f(xi, 0.0), f(xt, yt), f(xs, ys),
            fy(xo, 0.0), fy(xi, 0.0), fx(xt, yt), fx(xs, ys), fy(xs, ys),
            fyy(xo, 0.0) + N1 * fx(xo, 0.0),
            fyy(xi, 0.0) + N2 * fx(xi, 0.0),
            fxx(xt, yt) + N3 * fy(xt, yt)], float)

    basis = CF_HOMOGENEOUS + CF_ODD
    M = np.stack([rows(h) for h in basis], axis=1)
    c = np.linalg.solve(M, -rows(up))
    u = up
    for ci, h in zip(c, basis):
        u = u + h * float(ci)
    return u, (xs, ys)


def frame(u, R0, R, Z):
    """psi on the DIII-D grid with DIII-D's sign convention (axis = minimum of psi)."""
    RR, ZZ = np.meshgrid(R, Z)
    psi = u(RR / R0, ZZ / R0)
    if psi[np.argmin(np.abs(Z)), np.argmin(np.abs(R - R0))] > 0:   # make the axis a minimum
        psi = -psi
    return psi


def validate(u, xsep, R0, a, kappa, grid):
    """Saddle at the X-point; scorer LCFS closes near the X-point and spans the shape."""
    ux, uy = u.dx(), u.dy()
    H = np.array([[ux.dx()(*xsep), ux.dy()(*xsep)], [uy.dx()(*xsep), uy.dy()(*xsep)]])
    saddle = bool(np.linalg.det(H) < 0)
    R, Z, mc, mf = grid
    psi = frame(u, R0, R, Z)
    from lcfs import extract_lcfs
    C = extract_lcfs(psi, R, Z, "DIII-D", mc, mf, axis_sign=-1.0)
    if C is None:
        return dict(saddle=saddle, lcfs=False)
    zx = xsep[1] * R0
    return dict(saddle=saddle, lcfs=True,
                Zmin=float(C[:, 1].min()), Zx=float(zx),
                Zmin_err_cells=float(abs(C[:, 1].min() - zx) / (Z[1] - Z[0])),
                Zmax=float(C[:, 1].max()), Zmax_pred=float(kappa * a),
                Rmin=float(C[:, 0].min()), Rmax=float(C[:, 0].max()),
                R_pred=[R0 - a, R0 + a])


if __name__ == "__main__":
    res = {"homogeneity_max_residual": check_homogeneous()}
    print("Delta* residual of the 12 homogeneous functions:",
          ["%.1e" % v for v in res["homogeneity_max_residual"]])
    grid = sc.load_grid("DIII-D")
    R, Z = grid[0], grid[1]
    R0, a = 1.69, 0.60
    shapes = [(k, d, A) for k in (1.6, 1.8) for d in (0.2, 0.4, 0.6) for A in (-0.155, 0.3)]
    rng = np.random.default_rng(11)
    frames, tags, checks = [], [], {}
    for kappa, delta, A in shapes:
        u, xsep = solve_single_null(a / R0, kappa, delta, A)
        tag = f"k={kappa},d={delta},A={A}"
        v = validate(u, xsep, R0, a, kappa, grid)
        checks[tag] = v
        ok = v["saddle"] and v["lcfs"] and v["Zmin_err_cells"] < 1.0
        print(f"  {tag:22s} saddle={v['saddle']} lcfs={v['lcfs']} "
              + (f"Zmin={v['Zmin']:.3f} vs Zx={v['Zx']:.3f} ({v['Zmin_err_cells']:.2f} cells) "
                 f"Zmax={v['Zmax']:.3f}/{v['Zmax_pred']:.3f} R=[{v['Rmin']:.3f},{v['Rmax']:.3f}]"
                 if v["lcfs"] else "") + ("" if ok else "   -> REJECTED"))
        if not ok:
            continue
        for _ in range(4):                               # sub-pixel shifts of the whole shape
            r0 = R0 + rng.uniform(-.5, .5) * (R[1] - R[0])
            frames.append(frame(u, r0, R, Z) if True else None)
            tags.append(tag)
    res["checks"] = checks
    psi = np.array(frames)
    tags = np.array(tags)
    ids = np.array([list(dict.fromkeys(tags)).index(t) for t in tags])
    print(f"accepted shapes: {len(set(tags))}/{len(shapes)}, frames: {len(psi)}")
    sw = sc.run_sweep(psi, "DIII-D", seeds=3)
    np.savez_compressed(HERE / "cf_xpoint.npz", **sw, shot_ids=ids, tags=tags)
    names = list(sw["scalars"])
    res["usable"] = int(np.isfinite(sw["base"]).all(axis=1).sum())
    res["n_frames"] = int(len(psi))
    for fam in sw["families"]:
        r = at.analyse(sw, fam, d1_eps=sc.D1_EPS)
        b = at.bootstrap(sw, fam, ids, n_boot=1000, d1_eps=sc.D1_EPS)
        res[fam] = {"alpha_global": r["alpha_global"].tolist(), "alpha_regime": r["alpha_regime"].tolist(),
                    "alpha_global_part": r["alpha_global_part"], "alpha_regime_part": r["alpha_regime_part"],
                    "slopes": r["slopes"].tolist(), "bootstrap": b}
        print(f"\n[{fam}] alpha_global " + " ".join(f"{n}={x:.3f}" for n, x in zip(names, r["alpha_global"])))
        print(f"[{fam}] alpha_regime " + " ".join(f"{n}={x:.3f}" for n, x in zip(names, r["alpha_regime"])))
        for key in ("alpha_global", "alpha_regime"):
            p = r[key + "_part"]
            bb = b[key]
            print(f"   {key}: best k={p['k']} sil={p['silhouette']:.2f} {p['partition']} | k=3 {p['k3'][0]}")
            print(f"     bootstrap over shapes: C1 {bb['k3_partition_freq'].get(at.C1_PARTITION, 0):.1%}, "
                  f"H-alt {bb['best_partition_freq'].get(at.HALT_PARTITION, 0):.1%}; "
                  f"top {list(bb['best_partition_freq'].items())[:2]}")
    (HERE / "cf_xpoint.json").write_text(json.dumps(res, indent=1, default=str))
