#!/usr/bin/env python3
"""Рисунок g07_tokamap: фазові портрети Tokamap при x_L = 0; 0,8; 2,0.

Лише викликає Our try/03-deep-dives/D3-poincare-inverse/tokamap.py (orbit). Полярні
координати (sqrt(Psi) cos 2piT, sqrt(Psi) sin 2piT), як у Balescu et al. 1998.
    .venv/bin/python ../../guide/figures/g07_fig.py     # з fusion equilibrium challenge/starter
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
sys.path.insert(0, str(ROOT / "Our try" / "03-deep-dives" / "D3-poincare-inverse"))
from tokamap import orbit  # noqa: E402

fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9), sharey=True)
psis = np.linspace(0.05, 1.6, 30)
for ax, xL in zip(axs, [0.0, 0.8, 2.0]):
    for i, p0 in enumerate(psis):
        for T0 in (0.0, 0.5):
            p, T = orbit(p0, T0, xL, 1500)
            ax.plot(T, np.sqrt(p), ",", color=plt.cm.viridis(i / len(psis)), alpha=0.8)
    ax.set_title(f"$x_L = {xL:g}$".replace(".", "{,}"), fontsize=10)
    ax.set_xlabel("$T$ (кут / $2\\pi$)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.8)
axs[0].set_ylabel(r"$\sqrt{\Psi}$")
for _ax in fig.axes:
    _ax.xaxis.set_major_formatter(COMMA); _ax.yaxis.set_major_formatter(COMMA)
fig.tight_layout()
out = Path(__file__).resolve().parent / "g07_tokamap.png"
fig.savefig(out, dpi=220)
print("saved", out)
