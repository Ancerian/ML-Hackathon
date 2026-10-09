# Physics: conventions, derivations, validation, caveats

Everything below is implemented in `src/tokviz/` and validated by
`tests/test_physics.py`. Numbers quoted are measured on DIII-D shot 203702,
frame 75 (t = 1760 ms), unless stated.

---

## 1. Conventions

SI throughout. ψ is the poloidal flux **per radian** (Wb/rad), as EFIT stores it.

```
B_R   = -(1/R) ∂ψ/∂Z
B_Z   = +(1/R) ∂ψ/∂R
B_φ   =  F(ψ) / R
B_pol =  |∇ψ| / R
```

Equivalently `B = ∇ψ × ∇φ + F ∇φ`, which is divergence-free by construction.

**Sign convention.** DIII-D stores the magnetic axis at the **minimum** of ψ,
MAST at the **maximum**. `config.AXIS_SIGN` carries `{-1.0, +1.0}` accordingly,
matching `fusion_scoring/common.py` in the challenge scorer. The O-point finder
maximises `axis_sign · ψ`, so it is machine-agnostic.

Normalised flux `ψ_N = (ψ − ψ_axis)/(ψ_bdy − ψ_axis)`: 0 on axis, 1 on the LCFS.

---

## 2. Calibrating F(ψ), which the dataset does not ship

The Fusion Equilibrium Challenge parquet contains `efit_psirz` but **no
`fpol`**, so `B_φ` cannot be derived from ψ alone. Two bad options and one good
one:

- assume a vacuum field `F = R₀B₀` — needs B₀, which is also absent;
- ignore `B_φ` — then there are no field lines at all;
- **calibrate F against the dataset's own `efit_q95`** — what we do.

The safety factor on a flux surface is

```
q = (1/2π) ∮ (B_φ / (R B_pol)) dl
  = (F/2π) ∮ dl / (R² B_pol)
  = (F/2π) ∮ dl / (R |∇ψ|)
```

q is **linear in F**, so with the ψ_N = 0.95 contour extracted and the loop
integral evaluated, F follows by a single division — no iteration, no assumed
field:

```
F = 2π q95_EFIT / ∮ dl/(R|∇ψ|)
```

**Result and independent check.** For frame 75 this gives

```
F = 3.23341 T·m   →   B_φ(R_axis = 1.678 m) = 1.927 T
```

DIII-D's toroidal field at its nominal major radius R₀ = 1.6955 m is
**1.9–2.1 T** in normal operation. We recovered the machine's real toroidal
field from q95 and geometry alone, having put in no information about B₀. That
is a genuine, independent validation of the whole ψ → B chain.

---

## 3. Straight-field-line coordinates

A perturbation `cos(mθ − nφ)` only resonates at exactly `q = m/n` if θ is the
**straight-field-line** angle θ*, defined so a field line advances
`dθ*/dφ = 1/q` uniformly. Using the geometric angle instead misplaces and
distorts the islands.

On a flux surface parametrised by poloidal arc length ℓ,

```
θ*(ℓ) = 2π · [∫₀^ℓ dl/(R|∇ψ|)] / [∮ dl/(R|∇ψ|)]
```

F cancels — θ* does not depend on the toroidal field at all.

### Why we tabulate Δ = θ* − θ_geo and not θ*

θ* has a 2π branch cut, which destroys spline derivatives. Δ does not: both
angles gain exactly 2π per poloidal circuit, so Δ is single-valued and
2π-periodic in θ_geo. It is also small and smooth, so interpolating it is far
cleaner than interpolating an oscillatory `cos(mθ*)`.

`Equilibrium.theta_star_geometry()` builds Δ on a regular (ψ_N, θ_geo) table
(200 × 512), skipping surfaces where θ_geo is not monotonic about the axis —
which happens near an X-point, where the surface stops being star-shaped.
`ThetaStarField` then gives θ* and its (R,Z) derivatives anywhere by chain rule:

```
∂θ*/∂R = (∂θ_g/∂R)(1 + ∂Δ/∂θ_g) + (∂Δ/∂ψ_N)(∂ψ_N/∂R)
```

### This mattered — a measured lesson

The first implementation rasterised `cos(mθ*)` onto the 65×65 EFIT grid and
splined that. Decomposing the result back on a flux surface showed **2.4%
spurious m = 1 content**, which produced a **fake island chain at q = 1**
(ψ_N = 0.335) from a perturbation containing only m = 2 and m = 3. Island width
goes as √amplitude, so even small spectral pollution is visible.

Evaluating analytically instead cut the spurious m = 1 to **0.4%**, and the
fake q = 1 island vanished from the Poincaré section. Residual parasitic
harmonics are ≲ 1.3% of the intended mode, i.e. parasitic islands ≲ 11% of the
intended width. That is the accuracy floor imposed by the 65×65 ψ grid.

---

## 4. The perturbation

**In strict axisymmetry the field-line system is integrable and a Poincaré
section is exactly the level sets of ψ.** Islands are therefore not something
one can extract from an EFIT reconstruction — they must be put in deliberately.

The perturbation is applied to the flux itself:

```
ψ_total(R,Z,φ) = ψ(R,Z) + δψ(R,Z,φ)
δψ = Σ_k A_k g_k(ψ_N) cos(m_k θ* − n_k φ + α_k)
```

with the field taken as `B = ∇×(ψ_total ∇φ)`, which is **divergence-free
exactly, for any δψ**, because it is a curl.

Radial envelopes:

- `vacuum` — `g = ψ_N^(m/2)`, growing toward the edge as an externally applied
  vacuum field does;
- `resonant` — a Gaussian centred on the resonant surface.

Both carry a C¹ taper `exp(−max(0, ψ_N−1)²/w²)` outside the separatrix, so
nothing is driven in the SOL. (`max(0,·)²` has zero derivative on both sides of
ψ_N = 1, so g stays continuously differentiable there.)

### Island width

**Canonical pair.** For `B = ∇ψ×∇φ + F∇φ` (ψ the poloidal flux per radian)
the field lines form a 1½-degree-of-freedom Hamiltonian system with φ as time,
the straight-field-line angle **θ\*** as the coordinate, the **toroidal flux per
radian ψ_t** as its conjugate momentum, and the **poloidal flux ψ as the
Hamiltonian**: `H = ψ(ψ_t, θ*, φ)`, `dθ*/dφ = ∂ψ/∂ψ_t = ι = 1/q`,
`dψ_t/dφ = −∂ψ/∂θ*`, and `dψ = dψ_t / q`. ψ is **not** the momentum.

**Pendulum.** With `H = ψ₀(ψ_t) + A cos(mθ* − nφ + α)` and
`ξ = mθ* − nφ + α`, near the resonance

```
dξ/dφ   = m ι′ (ψ_t − ψ_t,res),     ι′ = dι/dψ_t = −(dq/dψ)/q³
dψ_t/dφ = m A sin ξ
```

so the separatrix full width in ψ_t is `4√(A/|ι′|)`, and in ψ it is 1/q of
that:

```
W_ψ = 4 √( A q / |dq/dψ| )      ⇒      W = 4 √( ε q / |dq/dψ_N| )
```

The second form is the same expression in normalised flux, with
`ε = A/|ψ_bdy − ψ_axis|` (what `Perturbation.amplitude_at` returns) — both A and
dq/dψ rescale with the flux range, so the formula is unit-free. The
m-dependence cancels — a fact worth checking against, since it is easy to
carry a stray factor of m through this algebra.

**Correction, 2026-09-29.** Earlier versions used `W = 4√(ε q²/|dq/dψ|)`, which
is what one gets by treating ψ itself as the momentum. That overestimates W by
**√q** (41 % at q = 2, 73 % at q = 3). Checked directly: tracing single-mode
islands at amp = 10⁻³ and measuring the ψ_N extent of the librating orbits gives
0.98 (2/1) and 1.00 (3/1) of the corrected formula
(`tests/test_physics.py::test_island_width_matches_traced_separatrix`). See
`docs/CHANGES-2026-09-29-island-width.md`.

Chirikov overlap `S = (W₁+W₂)/(2|ψ₁−ψ₂|)`. **S ≥ 1 is a heuristic for the
onset of large-scale stochasticity, not a theorem**, and is reported as such.

---

## 5. Field-line tracing

With φ as the independent variable,

```
dR/dφ = R B_R/B_φ = −(R/F) ∂ψ_total/∂Z
dZ/dφ = R B_Z/B_φ = +(R/F) ∂ψ_total/∂R
```

a 1½-degree-of-freedom Hamiltonian system: φ is the time, θ* the coordinate,
the toroidal flux ψ_t its conjugate momentum, and the poloidal flux ψ the
Hamiltonian (§4, "Island width").

Integrated with `scipy.integrate.solve_ivp`, **DOP853**, `rtol` 1e-9 to 1e-10.
Seeds are integrated as one batched state vector; a line that leaves the grid
has its derivative frozen rather than being allowed to destabilise the shared
adaptive step.

Poincaré punctures are taken at `t_eval = 2πk`.

---

## 6. Validation — measured

| Check | Result |
|---|---|
| Magnetic axis vs `efit_r_axis`/`efit_z_axis` | **ΔR = 0.0002 mm, ΔZ = 0.0001 mm** (grid cell 26.6 × 50.0 mm) |
| ψ spread along the dataset's own LCFS | 8.2×10⁻⁵, i.e. **0.022%** of the flux range — confirms it really is a ψ contour |
| Calibrated B_φ(axis) | **1.927 T** vs DIII-D's real 1.9–2.1 T |
| q from contour integral vs **from tracing** | agree to **0.001 – 0.16%** across ψ_N = 0.3 … 0.95 |
| ψ drift along axisymmetric field lines, 200 toroidal turns | **1×10⁻⁷ … 1.4×10⁻⁶** relative — integrator error only |
| Unperturbed section | reproduces the ψ contours (the axisymmetry theorem, used as a unit test) |
| Spurious poloidal harmonics | ≲ 1.3% of the intended mode |

q-profile for frame 75: q₀ ≈ 0.71, q95 = 3.6995 (by construction), q(0.99) = 5.01.
Resonant surfaces: **q = 1 at ψ_N = 0.335, q = 2 at 0.752, q = 3 at 0.900**.

With m/n = 2/1 and 3/1 at amplitude 10⁻³ of the flux range:

```
2/1  ψ_N = 0.7520  ε = 7.52e-4  dq/dψ_N = 4.44   W = 0.0736
3/1  ψ_N = 0.9001  ε = 8.54e-4  dq/dψ_N = 10.70  W = 0.0619
Chirikov S = 0.458      (below 1 -- surfaces between the chains survive)
(before the 2026-09-29 √q correction: W = 0.104 / 0.107, S = 0.714)
```

The traced section shows **2 islands at q = 2 and 3 at q = 3** — an m/n chain
must show m islands in a poloidal cut — and the surfaces between them remain
intact, consistent with S < 1.

---

## 7. The scaling argument

The equilibrium is a real DIII-D discharge (R₀ = 1.678 m, a ≈ 0.67 m); the
rendered machine is reactor-class. We apply a **uniform similarity scale**
s = 3.0, giving R₀ = 5.03 m, a ≈ 2.01 m.

Because the scale is uniform, the **aspect ratio is unchanged** (2.5), and
q(ψ) — being dimensionless and dependent only on shape — is **invariant**.
So are the resonant surfaces, the island widths in ψ_N, the Chirikov parameter
and the entire Poincaré structure.

This is why the render may honestly carry a real discharge's topology at
reactor scale. It is **not** a claim to be ITER (aspect ratio 3.1) or to be
DIII-D at its true size. Say "reactor-class geometry carrying the unmodified
topology of DIII-D 203702".

---

## 8. Caveats that must appear on the image

1. **This is a prescribed vacuum-like perturbation, not a plasma response.**
   A vacuum Poincaré section **systematically overestimates stochasticity**,
   because the plasma screens resonant components. Lobes and manifolds are
   robust to the plasma response; the stochastic sea is not.
2. **Chirikov S ≥ 1 is a heuristic**, not a proved criterion.
3. **The islands are put in by hand.** They are not a property of the EFIT
   reconstruction and must never be presented as measured.
4. **The colour ramp is core-hot by convention.** A real visible-light
   photograph of a tokamak is the other way round: the core is far too hot to
   radiate in the visible and looks dark, while the cool edge glows from line
   radiation. Do not caption a core-bright render as photographic.
5. **DIII-D has no beryllium, no tungsten divertor and no cryostat** — it is a
   room-temperature copper-coil machine with graphite tiles. Those materials
   belong to the reactor-class composite, not to the machine the data came from.

---

## 9. Known limitations

- The 65×65 ψ grid is the accuracy floor: it sets the ≲1.3% spurious-harmonic
  level and limits how fine an island can be resolved.
- θ* is undefined on surfaces that are not star-shaped about the axis, so the
  table stops at ψ_N ≈ 0.985 and the last few percent of the plasma is
  extrapolated.
- The perturbation is applied to a *fixed* equilibrium; there is no
  self-consistent response, no Δ′, no island saturation physics.
- Only the ψ_N = 0.95 surface is used to pin F, so a q-profile error at other
  radii is not corrected.
- `q_from_tracing` counts poloidal turns with the geometric angle, which is
  fine over many turns but is not a per-turn measure.
