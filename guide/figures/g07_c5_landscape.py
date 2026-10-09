#!/usr/bin/env python3
"""Рисунок g07_c5_landscape: безшумова сума квадратів нев'язок Q уздовж фази phi_1 через істину (стенд A, екземпляр 0).

Дані -- Our try/03-deep-dives/D3-poincare-inverse/c5/diag_tokamap.json (post-hoc діагностика C5, E40).
Рівень шуму chi^2 ~ N_obs = 20 радіусів x 8 фаз = 160 (stand_tokamap.py).
    python3 guide/figures/g07_c5_landscape.py   (з кореня проєкту; далі sips -> jpg)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "Our try/03-deep-dives/D3-poincare-inverse/c5/diag_tokamap.json"
d = json.loads(SRC.read_text())
N_OBS = 160
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
for ax, key, title in ((axes[0], "phi1_fine", "крок 0,5°, ±20° навколо істини"),
                       (axes[1], "phi1_full", "крок 2,5°, повне коло")):
    g = np.rad2deg(np.array(d[key]["grid"])); c = np.array(d[key]["chi2_0"])
    pos = c > 0
    ax.semilogy(g[pos], c[pos], "-o", color="tab:blue", lw=1.2, ms=2.5)
    ax.axhline(N_OBS, color="k", lw=0.9, ls="--")
    ax.axvline(0, color="tab:red", lw=0.9, ls=":")
    ax.text(g.min() + 0.02 * np.ptp(g), N_OBS * 1.25, "рівень шуму, Q ≈ N_obs = 160", fontsize=8)
    ax.set_xlabel("зсув фази φ₁ від істини, °")
    ax.set_title(f"{title}: локальних мінімумів {d[key]['n_local_minima']}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("Q без шуму")
fig.tight_layout()
out = HERE / "g07_c5_landscape.png"
fig.savefig(out)
print(out)
