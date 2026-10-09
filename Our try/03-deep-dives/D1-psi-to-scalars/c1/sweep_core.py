#!/usr/bin/env python3
"""C1 — shared perturbation-sweep engine (steps 1, 3, 4 of the C1 test).

Same physics and normalisation as ../conditioning.py (D1), which is imported, not copied:
perturb a psi frame by a unit-L2 direction times eps * per_frame_scale and recompute the seven
consistency scalars with the scorer's own code. Differences from D1, all for the C1 test:

  * machine-aware (DIII-D / MAST grids and masks; the flux sign is pinned via axis_sign, the
    lesson of R9);
  * a wider eps grid (EPS: six points below D1's plus D1's own six, see THEORY.md §4);
  * RAW per-frame, per-seed absolute errors are stored, so alpha can be bootstrapped by shot;
  * frames run in parallel (one process per core).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D1 = HERE.parent
PROJECT = HERE.parents[3]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(D1))

from common import AXIS_SIGN, CONS_SCALARS                               # noqa: E402
from derive import derive_frame                                          # noqa: E402
from lcfs import extract_lcfs                                            # noqa: E402
import conditioning as d1                                                # noqa: E402

D1_EPS = list(d1.EPS_GRID)                                   # 0.003 ... 0.3 (global alpha)
EPS = [1e-4, 2e-4, 4e-4, 7e-4, 1.2e-3, 2e-3] + D1_EPS         # 12 points, THEORY.md §4
SMOOTH_BAND = (1, 4)                                          # |k| < 4 on the 65x65 grid
N_POINTS = d1.N_POINTS
GRIDS = {"DIII-D": "d3d_envelope.npz", "MAST": "mast_envelope.npz"}


def load_grid(machine: str):
    z = np.load(STARTER / "fusion_scoring" / "masks" / GRIDS[machine])
    m = z["mask_coarse"].astype(bool)
    return z["grid_R"], z["grid_Z"], m, m.astype(np.float64)


def scalars(psi2d, grid, machine):
    R, Z, mc, mf = grid
    s = AXIS_SIGN[machine]
    c = extract_lcfs(psi2d, R, Z, machine, mc, mf, n_points=N_POINTS, axis_sign=s)
    v = derive_frame(psi2d, R, Z, machine, mc, mf, contour=c, axis_sign=s)
    return np.array([v[k] for k in CONS_SCALARS], dtype=float)


def load_shots(paths, n_per_shot: int):
    """Evenly spaced finite frames from each parquet file -> (psi[n,65,65], shot_id[n])."""
    import pyarrow.parquet as pq
    psis, ids = [], []
    for si, p in enumerate(paths):
        tbl = pq.ParquetFile(str(p)).read(columns=["efit_psirz"])
        arr = np.asarray(tbl.column("efit_psirz")[0].as_py(), dtype=np.float64)
        arr = arr[np.isfinite(arr).all(axis=(1, 2))]
        if len(arr) == 0:
            continue
        idx = np.unique(np.linspace(0, len(arr) - 1, min(n_per_shot, len(arr))).astype(int))
        psis.append(arr[idx])
        ids += [si] * len(idx)
    return np.concatenate(psis), np.array(ids)


# ---------------------------------------------------------------------------- the sweep
_W = {}


def _init(psi, grid, machine, scale, tail, families, eps, seeds):
    _W.update(psi=psi, grid=grid, machine=machine, scale=scale, tail=tail,
              families=families, eps=eps, seeds=seeds)


def _direction(fam, rng, shape):
    if fam == "white":
        return d1.pert_white(rng, shape)
    if fam == "smooth":
        return d1.pert_band(rng, shape, *SMOOTH_BAND)
    if fam == "pca-tail":
        return d1.pert_pca_tail(rng, shape, _W["tail"])
    raise ValueError(fam)


def _frame_job(fi):
    """All families x eps x seeds for one frame -> (base[7], err[F, E, S, 7])."""
    psi, grid, machine = _W["psi"], _W["grid"], _W["machine"]
    base = scalars(psi[fi], grid, machine)
    F, E, S = len(_W["families"]), len(_W["eps"]), _W["seeds"]
    err = np.full((F, E, S, len(CONS_SCALARS)), np.nan)
    if not np.isfinite(base).all():
        return fi, base, err
    shape = psi.shape[1:]
    for a, fam in enumerate(_W["families"]):
        for b, eps in enumerate(_W["eps"]):
            for sd in range(S):
                # same seeding rule as D1 (conditioning.py), so D1's points are reproduced
                rng = np.random.default_rng(10_000 * fi + 97 * sd + int(eps * 1e6))
                d = _direction(fam, rng, shape) * (eps * _W["scale"])
                err[a, b, sd] = np.abs(scalars(psi[fi] + d, grid, machine) - base)
    return fi, base, err


def run_sweep(psi, machine, families=("white", "smooth", "pca-tail"), eps=EPS, seeds=2,
              pca_keep=20, procs=None):
    """Returns dict(base[N,7], err[F,E,N,S,7], eps, families, scale)."""
    grid = load_grid(machine)
    psi_bar = psi.mean()
    scale = np.sqrt(((psi - psi_bar) ** 2).sum()) / np.sqrt(len(psi))   # D1's per-frame scale
    _, tail = d1.build_pca(psi, pca_keep)
    args = (psi, grid, machine, scale, tail, list(families), list(eps), seeds)
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_init, initargs=args) as pool:
        out = pool.map(_frame_job, range(len(psi)), chunksize=1)
    base = np.full((len(psi), len(CONS_SCALARS)), np.nan)
    err = np.full((len(families), len(eps), len(psi), seeds, len(CONS_SCALARS)), np.nan)
    for fi, b, e in out:
        base[fi] = b
        err[:, :, fi] = e
    return dict(base=base, err=err, eps=np.array(eps), families=np.array(families),
                scale=scale, scalars=np.array(CONS_SCALARS))
