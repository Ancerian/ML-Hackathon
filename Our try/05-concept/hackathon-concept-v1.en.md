# The Equilibrium You Cannot See
## A student hackathon concept — v1, 2026-09-22 (amended 2026-10-06 for R11 and R12)

V. M. Horshkov AI Laboratory, Faculty of Physics and Mathematics,
Igor Sikorsky Kyiv Polytechnic Institute

> Written for an external reader — a potential partner, mentor, jury member or reviewer.
> It states what we intend to run, what is novel about it, and what we have already measured
> to justify those claims. Internal decision records and the full literature analysis sit
> alongside it in the same repository.

---

## The task

Teams reconstruct the **poloidal flux map ψ(R, Z) inside a tokamak from instruments that cannot
see it** — poloidal-field coil currents and Thomson-scattering profiles, **with no magnetic
diagnostics at all**. They train on synthetic equilibria produced by open free-boundary solvers and
are scored on **real DIII-D discharges**.

The setting is not artificial. In SPARC/ARC-class devices, neutron flux degrades magnetic sensors
through integrator drift and radiation-induced EMF, so magnetics-free equilibrium inference is
something those machines will actually have to do.

## What is different about it

Existing benchmarks in this space — the Fusion Equilibrium Challenge, TokaMark, PDEBench,
The Well — all score **how close the output looks to the target**. None of them asks whether the
output **is a solution of the governing equation**, and none asks whether the model's stated
uncertainty is **honest**. Our scoring function adds both.

```
S′ = 0.40·R²ψ + 0.10·R²_{q95,βN} + 0.10·(1 − D_LCFS) + 0.15·Consistency
   + 0.15·GS_score + 0.10·Calibration
```

The first four terms are the vendored Fusion Equilibrium Challenge composite (CC BY 4.0 data,
MIT code, attributed). The last two are ours.

### `GS_score` — is this an equilibrium at all?

A submission ships only ψ, not the flux functions `p′` and `FF′`, so a naive Grad–Shafranov
residual is not computable. The way around reuses EFIT's own structure: **the GS source is linear
in the profile coefficients once ψ is fixed**. So we score

```
min over (p′, FF′)  ‖Δ*ψ + μ₀R²p′(ψ) + FF′(ψ)‖ / ‖Δ*ψ‖
```

— *"does there exist any pair of flux functions making this ψ an equilibrium?"* — an ordinary
linear least-squares problem, numpy only, no extra submission channel. It is not defeated by mere
smoothness, because it requires `Δ*ψ` restricted to a flux surface to be an affine function of `R²`
with ψ-only coefficients: the flux-surface condition itself.

**Measured on real DIII-D frames:** ground truth scores **0.009**; the same field with 1% white
noise scores **0.650** — a **70× jump**. A uniform rescale of ψ leaves the score unchanged, as GS
linearity requires, which serves as a built-in invariance check.

We also measured what it does **not** catch — Gaussian smoothing and low-rank PCA truncation both
pass — so we present it as **complementary to** R²ψ rather than a replacement. `Δ*` is a
second-order operator: it amplifies high wavenumbers and is nearly blind to smooth error. Its
unique contribution is detecting outputs that are **not equilibria at all**.

**Tested on the baselines (2026-10-01).** As an additive term, `GS_score` does **not robustly
reorder** the eight starter baselines: no swap holds in ≥ 95 % of shot-bootstrap replicates. It also
rewards smoothness. Map-output networks with R²ψ = 0.96–0.98 have a GS-inconsistency of 0.81–0.96,
worse than ground truth with 1 % noise (0.63), while models that predict PCA coefficients sit close
to the truth only because their basis is smooth. We therefore expect to use it as a **gate** (a
multiplier or a threshold) rather than an additive term; the 0.15 weight above is provisional.

### `Calibration` — is the uncertainty honest?

Teams submit predictive bands at nominal 50/80/90/95%, scored on **empirical coverage against
nominal**. The interesting complication, and the reason this is a research question rather than a
checkbox: ψ is a **65×65 field**, so pointwise intervals give 4,225 marginal guarantees rather than
one joint one. Simultaneous or functional bands are required, and choosing among them —
max-statistic, PCA-basis projection, or conformalising the derived scalars — is a genuine
mathematical decision.

## Two measurements that motivated the design

Both were run before writing this concept; code and data are in the repository.

**The map ψ ↦ derived scalars is not Lipschitz in L².** The magnetic axis is a critical point
(∇ψ = 0); the shape scalars come from a level set through a saddle (the X-point); ℓi integrates
|∇ψ|². Perturbing ground-truth ψ at controlled L² norm and recomputing the scalars with the
scorer's own code gives amplification exponents **α ∈ [0.35, 0.70] — all below 1**, the signature
of an unbounded map. The headline: **`R_axis` reaches a 1σ error at R²ψ = 0.99993**. A flux map any
reviewer would call essentially perfect already predicts the magnetic axis no better than the
population mean; at R²ψ = 0.99 the axis error is 3.4 cm, the same order as the 4 cm that
coils-only reconstruction achieves directly. ~~The spectral sensitivity curve that falls out of this
yields a drop-in spectrally weighted loss.~~ We later tested that idea against criteria fixed in
advance, and it failed (2026-09-30): weighting the loss by the spectral sensitivity curve improved
neither PCA+Ridge nor UNet_Lite, and every 95 % confidence interval contained zero.

**An L2-trained regressor on a multivalued target returns the mean of the branches, which is not a
solution.** Ham & Farrell (2024) and Pentland et al. (2025, MAST-U with deflated continuation) show
that distinct equilibria can share identical magnetic measurements. We measured the consequence on
a closed-form surrogate — 1D Bratu/Gelfand, the family in which Bartolucci et al. place the plasma
free-boundary problem — against a **convex control** (the Bean critical state, provably unique by
Prigozhin 1996). Result: an actual branch has relative residual **6.4 × 10⁻⁸**; the L2-optimal mean
has **1.22**, a residual of the same order as the source term. The convex control stays at
**2.5 × 10⁻¹³**. The damage is invisible to L2 **by construction**, since the mean is the
L2-optimal predictor — only a residual term sees it.

## Structure

**Entry track** — prepared starter kit, leaderboard on `S′`, no plasma-physics prerequisite.

**Research track** — open sub-problems with a jury, drawn from the literature analysis so each has
a known state of the art and known limits: spectrally weighted losses (a naive weighting is
already known not to help); multivalued-target
regression; simultaneous conformal bands on a field; regularisation-parameter selection done
properly (L-curve, GCV, Morozov, Picard plots — which the fusion community does not practise); and
Deep Ritz versus PINN on Grad–Shafranov, where we found no published Deep Ritz solver for GS and
where the plasma problem's **non-convexity** means seed-variance of the branch found **is** the
result.

**Nominations** all consume the same ψ, so no nomination needs a separate engine: physics
consistency, honest uncertainty, sensor-dropout robustness, and compute-light efficiency as a
first-class scored axis rather than a self-declared badge.

## What we are careful not to claim

We have tested and **refuted** a number of our own ideas and record every one. The two below shaped
this concept most. Two more bear directly on it and are noted where they apply above: the
spectrally weighted loss, and the expectation that `GS_score` would reorder existing baselines.

A draft claim that no neural architecture hard-constrains the Green's-function boundary condition
is **false**: McClenaghan et al. (*Phys. Plasmas* 31, 082507, 2024) does exactly that,
architecturally. Only the narrower version survives — not inside a free-boundary forward solver.

A proposal to bridge Bean's critical state and the plasma free-boundary problem as "the same
mathematics" is **also false**. Bean is a convex, maximal-monotone variational inequality with a
unique solution and a KKT multiplier on an active set; the plasma problem is a non-convex nonlinear
eigenvalue problem whose free boundary is a level set of the solution, with a multiplier on an
equality constraint that *is* the eigenvalue. The two literatures have not cross-cited in fifty
years. What survives is better than the original idea: **does a non-smooth-PDE solver's failure mode
differ between a convex free-boundary problem and a non-convex one?** — with Bean as the convex
control.

We are also not competing on cross-machine transfer, which belongs to a live competition, and not
on disruption prediction, which is saturated intra-machine and whose data is not openly available.

## Open items

Dates and venue; cluster access, which the compute-light nomination requires; jury composition;
and one licensing question we cannot resolve ourselves — the CC BY 4.0 challenge corpus contains
1,206 MAST shots that may derive from the CC BY-SA 4.0 FAIR-MAST archive, so until that is
clarified we treat the MAST slice as share-alike-encumbered and do not redistribute it.

## Attribution

Data and scorer: **Fusion Equilibrium Challenge**, Sophelio and General Atomics, CC BY 4.0 (starter
code MIT), with the dataset card's full citation and the DOE acknowledgements
(DE-FC02-04ER54698; DE-SC0024426, DE-SC0024499, DE-SC0024409, DE-SC0024571).
