#!/usr/bin/env python3
"""Рисунок g04_gs_residual: GS-нев'язка g на 36 кадрах трьох демо-розрядів DIII-D.

Числа НЕ перераховуються тут: скрипт читає вивід manual/code/ch11_gs_residual_baseline.py,
збережений у текстовий файл (перший аргумент). Відтворення:
    cd "fusion equilibrium challenge/starter"
    .venv/bin/python ../../manual/code/ch11_gs_residual_baseline.py > /tmp/ch11.out
    .venv/bin/python ../../guide/figures/g04_fig.py /tmp/ch11.out
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # десяткова кома

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    t = line.split()
    if len(t) == 7 and t[0].isdigit():
        rows.append([float(x) for x in t[2:]])
a = np.array(rows)            # R2pred, 1-R2PCA50, gEFIT, gpred, gPCA50
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(a))
ax.plot(x, a[:, 2], "o-", ms=3, lw=0.8, label="EFIT (істина)")
ax.plot(x, a[:, 3], "s-", ms=3, lw=0.8, label=r"прогноз PCA+Ridge (медіана $R^2=0{,}877$)")
ax.plot(x, a[:, 4], "^-", ms=3, lw=0.8, label=r"EFIT$\to$PCA-50 ($1-R^2\approx2\cdot10^{-7}$)")
for b in (11.5, 23.5):
    ax.axvline(b, color="0.7", lw=0.6)
ax.set_xlabel("кадр (розряди 203702 | 203703 | 203704, по 12 кадрів)")
ax.set_ylabel("$g$")
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
for _ax in fig.axes:
    _ax.xaxis.set_major_formatter(COMMA); _ax.yaxis.set_major_formatter(COMMA)
fig.tight_layout()
out = Path(__file__).resolve().parent / "g04_gs_residual.png"
fig.savefig(out, dpi=220)
print("saved", out)
