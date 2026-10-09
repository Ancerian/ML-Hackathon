#!/usr/bin/env python3
"""C2 step 5 — figure c2_arms.png from eval.json (Consistency and S per model and weighting)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = ["flat", "sk", "h1", "shuffled"]
LAB = {"flat": "flat (MSE)", "sk": "S(k)", "h1": "H$^1$", "shuffled": "shuffled"}

ev = json.loads((HERE / "eval.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
for ax, (m, title) in zip(axes, (("consistency", "Consistency"), ("S", "composite S"))):
    for j, model in enumerate(("pca", "unet")):
        for i, arm in enumerate(ARMS):
            x = i + (j - 0.5) * 0.3
            ax.plot(x, ev["point"][f"{model}_{arm}"][m], "s" if model == "pca" else "o",
                    color="tab:gray" if model == "pca" else "tab:blue", ms=7,
                    label=("PCA+Ridge" if model == "pca" else "UNet_Lite (mean of 3 seeds)") if i == 0 else None)
            if model == "unet":
                for s in ev["seeds"][f"unet_{arm}"]:
                    ax.plot(x + 0.08, s[m], ".", color="tab:blue", alpha=0.45, ms=5)
    ax.set_xticks(range(len(ARMS)), [LAB[a] for a in ARMS])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3)
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=2, frameon=False)
fig.suptitle("C2: spectral weighting of the loss, 8 held-out DIII-D shots (small dots = seeds)", fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "c2_arms.png")
print("c2_arms.png")
