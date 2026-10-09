#!/usr/bin/env python3
"""D2 plots: the two branches, their L2 mean, and the residual that exposes it."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from branch_mean import X, bratu_u, theta_roots, rel_residual_bratu, lam_star, bean_H  # noqa: E402

D = json.loads((HERE / "results.json").read_text())
br = [r for r in D["rows"] if r["problem"] == "bratu"]
ls, _ = lam_star()

fig = plt.figure(figsize=(13, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.25], wspace=.3)

# ---- (a) the two branches and their mean, at lam = 1
ax = fig.add_subplot(gs[0, 0])
lam = 1.0
rts = theta_roots(lam); u_lo, u_hi = bratu_u(X, rts[0]), bratu_u(X, rts[-1])
um = .5 * (u_lo + u_hi)
ax.plot(X, u_lo, lw=2, label="branch 1 (lower)")
ax.plot(X, u_hi, lw=2, label="branch 2 (upper)")
ax.plot(X, um, "k--", lw=2.2, label="L2-optimal mean")
ax.set_title(rf"(a) Bratu at $\lambda={lam}$: two exact solutions"
             "\n" "same input, two valid targets", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.legend(fontsize=8); ax.grid(alpha=.3)

# ---- (b) the residual field of the mean
ax = fig.add_subplot(gs[0, 1])
from branch_mean import d2
for u, lab, st in [(u_lo, "branch 1", "-"), (u_hi, "branch 2", "-"), (um, "L2 mean", "--")]:
    r = d2(u) + lam * np.exp(u)
    ax.plot(X, r, st, lw=2.2 if st == "--" else 1.8, color="k" if st == "--" else None, label=lab)
ax.set_title("(b) PDE residual  " r"$u'' + \lambda e^{u}$"
             "\n" "branches: zero. mean: not a solution.", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("residual"); ax.legend(fontsize=8); ax.grid(alpha=.3)

# ---- (c) residual of the mean vs lambda, with Bean as control
ax = fig.add_subplot(gs[0, 2])
lams = np.array([r["lam"] for r in br]); rmean = np.array([r["res_mean"] for r in br])
rbr = np.array([.5 * (r["res_lo"] + r["res_hi"]) for r in br])
sep = np.array([r["l2_sep"] for r in br])
ax.semilogy(lams, np.maximum(rmean, 1e-16), "o-", color="crimson", lw=2, label="Bratu: residual of the L2 MEAN")
ax.semilogy(lams, rbr, "s-", color="crimson", alpha=.45, lw=1.5, label="Bratu: residual of a BRANCH")
bn = [r for r in D["rows"] if r["problem"] == "bean"]
ax.axhline(np.mean([r["res_mean"] for r in bn]), color="steelblue", lw=2.2, ls="-",
           label="Bean (convex): residual of the L2 mean")
ax.axvline(ls, color="gray", ls=":", lw=1.4)
ax.text(ls, 2e-5, r" fold $\lambda^*$", fontsize=8, color="gray", rotation=90, va="bottom")
ax2 = ax.twinx(); ax2.plot(lams, sep, "^--", color="darkgreen", lw=1.3, ms=5)
ax2.set_ylabel("branch separation  " r"$\|u_2-u_1\|$", color="darkgreen", fontsize=9)
ax2.tick_params(axis="y", labelcolor="darkgreen")
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("relative PDE residual")
ax.set_title("(c) the gap the L2 metric cannot see\n"
             "convex control stays at machine precision", fontsize=10)
ax.legend(fontsize=7.4, loc="lower left"); ax.grid(alpha=.3, which="both")

fig.suptitle("D2: an L2-trained regressor on a multivalued target returns the mean of the branches — "
             "which is not a solution", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(HERE / "branch_mean.png", dpi=150)
print("wrote branch_mean.png")
