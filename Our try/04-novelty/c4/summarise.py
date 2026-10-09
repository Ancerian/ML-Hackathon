#!/usr/bin/env python3
"""C4 step 4 — figure c4_gs.png from eval.json: GS_score vs R2_psi per model, and S'_GS vs w_r."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

HERE = Path(__file__).resolve().parent
ev = json.loads((HERE / "eval.json").read_text())
names = list(ev["models"])
GROUP = {"Simple MLP": ("networks (map output)", "tab:blue", "o"), "Conv Decoder": ("networks (map output)", "tab:blue", "s"),
         "UNet_Lite": ("networks (map output)", "tab:blue", "D"), "MLP (sklearn)": ("MLP (sklearn), PCA output", "tab:orange", "o")}
LIN = ("linear / RF on PCA coefficients", "tab:gray")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
seen = set()
for m in names:
    v = ev["models"][m]
    lab, c, mk = GROUP.get(m, (LIN[0], LIN[1], "o"))
    a.plot(v["r2_psi"], v["GS_score"], mk, color=c, ms=7, label=lab if lab not in seen else None)
    seen.add(lab)
a.set_xlabel(r"$R^2_\psi$"); a.set_ylabel("GS_score"); a.set_xlim(-0.02, 1.05); a.set_ylim(-0.05, 1.05)
a.grid(alpha=.3); a.set_title("GS_score vs flux-map accuracy (8 held-out shots)", fontsize=9)
a.legend(fontsize=7, loc="center")
wr = sorted(float(k) for k in ev["scan"])
seen = set()
for m in names:
    ys = [ev["scan"][f"{w:.4f}"]["S_prime"][m] for w in wr]
    lab, c, mk = GROUP.get(m, (LIN[0], LIN[1], "o"))
    b.plot(wr, ys, "-" + mk, color=c, ms=3, lw=1.2, label=lab if lab not in seen else None)
    seen.add(lab)
b.axvline(0.15 / 0.9, color="k", lw=0.8, ls="--")
b.set_xlabel("GS weight $w_r$"); b.set_ylabel(r"$S'_{GS}$")
b.grid(alpha=.3); b.set_title("S'_GS vs GS weight (dashed: design weight)", fontsize=9)
b.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "c4_gs.png")
print("c4_gs.png")
