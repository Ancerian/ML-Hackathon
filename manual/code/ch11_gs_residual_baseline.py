#!/usr/bin/env python3
"""Розд. 11, практикум: GS-нерев'язка передбачень бейзлайну PCA+Ridge.

Запуск (потрібне середовище starter kit; системний python3 не має numpy/sklearn):

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python my_experiments/baseline_pca_ridge.py --train --n-shots 40 --n-pca 50  # якщо немає artifacts/
    .venv/bin/python ../../manual/code/ch11_gs_residual_baseline.py

Для 12 кадрів кожного з трьох демо-розрядів DIII-D (parquet_data/, без мережі) рахує:
  * R^2 передбаченого psi відносно EFIT (на кадр);
  * відносну GS-нерев'язку g(psi) з Our try/04-novelty/gs_residual_probe.py
    (найкращий МНК-підбір p', FF' — поліноми psi_N, 4+4 коефіцієнти) для EFIT і для передбачення;
  * те саме для EFIT, стиснутого до 50 головних компонент тієї ж PCA (L^2-проєкція).
Невідомо, чи входять демо-розряди до 40 навчальних розрядів — трактуйте як ілюстрацію.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "my_experiments"))
sys.path.insert(0, str(ROOT / "Our try" / "04-novelty"))

import pyarrow.parquet as pq                                   # noqa: E402
from gs_residual_probe import gs_inconsistency                # noqa: E402
import baseline_pca_ridge as bl                               # noqa: E402


def r2(a, b):
    return 1.0 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)


def main(n_frames: int = 12):
    z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = z["grid_R"], z["grid_Z"]
    mc = z["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)
    m = bl._model()
    rows = []
    for shot in ("203702", "203703", "203704"):
        tbl = pq.ParquetFile(str(STARTER / "parquet_data" / f"d3d_shot_{shot}.parquet")).read()
        row = {k: tbl.column(k)[0].as_py() for k in tbl.column_names}
        psi_true = np.asarray(row["efit_psirz"], dtype=np.float64)
        pred = bl.predict_row(row)["psirz"].astype(np.float64)
        T = min(len(psi_true), len(pred))
        ok = [k for k in range(T) if np.isfinite(psi_true[k]).all() and np.isfinite(pred[k]).all()]
        idx = np.asarray(ok)[np.linspace(0, len(ok) - 1, n_frames).astype(int)]
        for k in idx:
            t = psi_true[k]
            p = pred[k]
            proj = m["pca"].inverse_transform(m["pca"].transform(t.reshape(1, -1))).reshape(t.shape)
            rows.append((shot, k, r2(t, p), r2(t, proj),
                         gs_inconsistency(t, R, Z, mc, mf)[0],
                         gs_inconsistency(p, R, Z, mc, mf)[0],
                         gs_inconsistency(proj, R, Z, mc, mf)[0]))
    print(f"{'shot':>7s} {'frame':>5s} {'R2 pred':>8s} {'1-R2 PCA50':>11s} {'g(EFIT)':>8s} {'g(pred)':>8s} {'g(PCA50)':>9s}")
    for r in rows:
        print(f"{r[0]:>7s} {r[1]:5d} {r[2]:8.4f} {1-r[3]:11.2e} {r[4]:8.4f} {r[5]:8.4f} {r[6]:9.4f}")
    a = np.array([r[2:] for r in rows], dtype=float)
    a[:, 1] = 1 - a[:, 1]
    med = np.nanmedian(a, axis=0)
    print("\nмедіани: R2 pred {:.4f} | 1-R2 PCA50 {:.2e} | g(EFIT) {:.4f} | g(pred) {:.4f} | g(PCA50) {:.4f}".format(*med))
    print("LCFS не знайдено (g = nan): EFIT {}, pred {}, PCA50 {} з {}".format(
        *[int(np.isnan(a[:, j]).sum()) for j in (2, 3, 4)], len(a)))


if __name__ == "__main__":
    main()
