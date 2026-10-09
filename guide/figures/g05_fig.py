#!/usr/bin/env python3
"""Рисунок g05_critical_points: скелет критичних точок psi на істині і при 10 % шуму.

Лише викликає функції проєкту (Our try/04-novelty/topology_probe.py: inside_lcfs,
critical_points), кадр 75 розряду DIII-D 203702 серед скінченних кадрів, шум як у main()
(seed 1000 + номер кадру не відтворюється тут; seed 1075 фіксовано).
Запуск (venv starter kit):
    cd "fusion equilibrium challenge/starter"
    .venv/bin/python ../../guide/figures/g05_fig.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # десяткова кома

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(ROOT / "Our try" / "04-novelty"))
from topology_probe import inside_lcfs, critical_points  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

z = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = z["grid_R"], z["grid_Z"]
mc = z["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
a = np.asarray(pq.ParquetFile(str(STARTER / "parquet_data" / "d3d_shot_203702.parquet"))
               .read(columns=["efit_psirz"]).column("efit_psirz")[0].as_py(), dtype=np.float64)
psi_all = a[np.isfinite(a).all(axis=(1, 2))]
k = 75
psi = psi_all[k]
rng = np.random.default_rng(1075)
noisy = psi + rng.standard_normal(psi.shape) * 0.10 * np.std(psi)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
for ax, p, title in [(axs[0], psi, "EFIT (істина)"), (axs[1], noisy, "+10 % білого шуму")]:
    ins = inside_lcfs(p, R, Z, mc, mf)
    cps = critical_points(p, R, Z, ins)
    ax.contour(R, Z, psi, levels=25, colors="0.6", linewidths=0.6)
    ax.contour(R, Z, ins.astype(float), levels=[0.5], colors="tab:blue", linewidths=1.0)
    nO = nX = 0
    for c in cps:
        r = np.interp(c["ir"], np.arange(len(R)), R); zz = np.interp(c["iz"], np.arange(len(Z)), Z)
        if c["index"] > 0:
            ax.plot(r, zz, "o", color="black", ms=5); nO += 1
        else:
            ax.plot(r, zz, "x", color="tab:red", ms=6, mew=1.6); nX += 1
    ax.set_title(f"{title}: {nO} O, {nX} X", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlabel("R, м")
axs[0].set_ylabel("Z, м")
for _ax in fig.axes:
    _ax.xaxis.set_major_formatter(COMMA); _ax.yaxis.set_major_formatter(COMMA)
fig.tight_layout()
out = Path(__file__).resolve().parent / "g05_critical_points.png"
fig.savefig(out, dpi=200)
print("saved", out)
