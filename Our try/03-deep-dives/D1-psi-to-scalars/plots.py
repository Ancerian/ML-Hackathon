#!/usr/bin/env python3
"""D1 plots + the break-even analysis, from results.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "results.json").read_text())
S = D["scalars"]; sigma = np.array(D["sigma"]); rows = D["rows"]; expo = D["exponents"]
UNITS = {"R_axis": "m", "Z_axis": "m", "kappa": "-", "tri_top": "-",
         "tri_bot": "-", "volume": "m^3", "li": "-"}
C = plt.cm.tab10(np.linspace(0, 1, 10))

def fam(f): return [r for r in rows if r["family"] == f]

# ---------------------------------------------------------------- 1. amplification
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
for ax, f, title in zip(axes, ["white", "pca-tail"],
                        ["white noise (flat spectrum)",
                         "PCA tail (what a truncated linear decoder gets wrong)"]):
    rs = fam(f); e = np.array([r["eps"] for r in rs])
    for i, n in enumerate(S):
        y = np.array([r[f"med_{n}"] for r in rs])
        ax.loglog(e, y, "o-", color=C[i], label=f"{n}  (alpha={expo[n]['alpha']:.2f})", lw=1.6, ms=4)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(e[0] * 1.05, 1.15, "1 sigma  =  R^2 of that scalar reaches 0", fontsize=8)
    ax.set_xlabel(r"relative $\psi$ error  $e=\|\delta\psi\|/\|\psi-\bar\psi\|$"
                  "\n" r"(top axis: $R^2_\psi = 1-e^2$)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3, which="both")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 1 - x**2, lambda y: np.sqrt(np.clip(1-y, 0, None))))
    sec.set_xticks([1 - v**2 for v in e])
    sec.set_xticklabels([f"{1-v**2:.5f}" for v in e], fontsize=7.5, rotation=40, ha="left")
    sec.minorticks_off()
    sec.set_xlabel(r"$R^2_\psi$", fontsize=9, labelpad=2)
axes[0].set_ylabel(r"median scalar error  $/\ \sigma$")
axes[1].legend(fontsize=7.5, loc="lower right", ncol=1)
fig.suptitle(r"D1: the map $\psi\mapsto$ derived scalars is not Lipschitz in $L^2$"
             "\n" r"every $\alpha<1$: scalar error decays SLOWER than the $\psi$ error that causes it",
             fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "amplification.png", dpi=150)
print("wrote amplification.png")

# ---------------------------------------------------------------- 2. spectral sensitivity
bands = [r for r in rows if r["family"].startswith("band-")]
def mid(r):
    _, a, b = r["family"].split("-"); return 0.5 * (int(a) + int(b))
ks = np.array([mid(r) for r in bands])
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for i, n in enumerate(S):
    ax.plot(ks, [r[f"med_{n}"] for r in bands], "o-", color=C[i], label=n, lw=1.7, ms=5)
ax.axhline(1.0, color="k", ls="--", lw=1.1)
ax.set_xlabel("radial wavenumber |k| of the injected perturbation  (grid Nyquist = 32)")
ax.set_ylabel(r"median scalar error $/\ \sigma$")
ax.set_title(r"D1: spectral sensitivity $S(k)$ at fixed $\|\delta\psi\|$  ($e=0.10$, $R^2_\psi=0.99$)"
             "\n" "which Fourier modes of the flux-map error the metric actually punishes", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(HERE / "spectral_sensitivity.png", dpi=150)
print("wrote spectral_sensitivity.png")

# ---------------------------------------------------------------- 3. break-even table
print("\n=== eps at which the median scalar error reaches 1 sigma (R^2 -> 0) ===")
print(f"{'scalar':9s} {'alpha':>7s} {'eps*':>9s} {'R2_psi*':>9s} {'err @ R2psi=0.99':>18s}")
lines = []
for i, n in enumerate(S):
    a = expo[n]["alpha"]; b = expo[n]["log_intercept"]
    eps_star = float(np.exp(-b / a))
    at99 = float(np.exp(b) * 0.1 ** a)
    abs99 = at99 * sigma[i]
    print(f"{n:9s} {a:7.3f} {eps_star:9.4f} {1-eps_star**2:9.5f} "
          f"{at99:8.2f} sigma = {abs99:.4g} {UNITS[n]}")
    lines.append((n, a, eps_star, 1 - eps_star**2, at99, abs99, UNITS[n]))
(HERE / "break_even.md").write_text(
    "| scalar | alpha | eps* (1 sigma) | R2_psi* | err @ R2psi=0.99 | absolute |\n"
    "|---|---|---|---|---|---|\n" +
    "".join(f"| `{n}` | {a:.3f} | {es:.4f} | **{r2:.5f}** | {e99:.2f} sigma | {ab:.4g} {u} |\n"
            for n, a, es, r2, e99, ab, u in lines))
print("\nwrote break_even.md")
