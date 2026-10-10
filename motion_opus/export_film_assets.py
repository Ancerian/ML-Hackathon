#!/usr/bin/env python3
"""Export browser-safe film assets from the project data (offline only)."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data"
SRC = ROOT / "motion" / "data"

def residual_png(values, name):
    # The supplied rmap is the local GS residual map. A fixed range makes
    # comparison between truth and prediction honest.
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.imshow(values, cmap="inferno", vmin=0, vmax=1.2, origin="lower")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / name, transparent=False)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = np.load(SRC / "equilibrium_arrays.npz")
    residual_png(arrays["gt_rmap"], "residual_gt.png")
    residual_png(arrays["unet_rmap"], "residual_unet.png")
    for filename in ("contours_gt.json", "contours_unet.json", "equilibrium_meta.json"):
        (OUT / filename).write_text((SRC / filename).read_text(), encoding="utf-8")
    sweep = json.loads((SRC / "poincare_sweep.json").read_text())
    compact = {"meta": sweep["meta"], "frames": sweep["frames"]}
    (OUT / "poincare_sweep_compact.json").write_text(json.dumps(compact, ensure_ascii=False), encoding="utf-8")
    high = json.loads((SRC / "poincare_highres_chaos.json").read_text())
    (OUT / "poincare_keyframe_chaos.json").write_text(json.dumps(high, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
