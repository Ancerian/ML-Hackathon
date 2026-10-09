# Island-width formula correction — 2026-09-29

Reason: the field-line Hamiltonian has canonical pair (θ*, ψ_t) with H = ψ
(poloidal flux per radian); dψ = dψ_t/q. Pendulum full width in ψ_t is
4√(A/|dι/dψ_t|), |dι/dψ_t| = |dq/dψ|/q³, width in ψ = width_t/q ⇒
W = 4√(A q/|dq/dψ|) = 4√(ε q/|dq/dψ_N|) in normalised flux (ε = A/|ψ_bdy−ψ_axis|,
normalisation-invariant). The old 4√(ε q²/|dq/dψ|) treated ψ as the momentum and
overestimated W by √q. Parent-register refs: E19 (correct Hamiltonian), R4 (refuted).
findings.md registers were NOT edited; reconcile from the table below.

Verification: tests/test_physics.py::test_island_width_matches_traced_separatrix
(single mode, amp 1e-3, libration of ξ = mθ*−nφ): measured/new = 0.983 (2/1), 0.996 (3/1).
Scratch run amp 5e-4, 30 seeds, 80 turns: 0.994 (2/1), 0.998 (3/1).

```yaml
formula:
  - {file: src/tokviz/perturbation.py, symbol: island_width_psin, old: "4*sqrt(eps*q**2/|dq/dpsin|)", new: "4*sqrt(eps*q/|dq/dpsin|)"}
  - {file: src/tokviz/coilfield.py, symbol: CoilPerturbation.calibrate_to_island_width, old: "inline q**2 formula", new: "calls island_width_psin", effect: "returned current scale grows by factor q (no callers in repo)"}

default_perturbation_amp_1e-3:   # frame 75, modes 2/1 + 3/1, vacuum envelope
  - {quantity: W_2/1, old: 0.1041, new: 0.0736, files: [docs/PHYSICS.md §6, docs/BLUEPRINT.md §5.6, docs/ANALYSIS.md §2.2]}
  - {quantity: W_3/1, old: 0.1072, new: 0.0619, files: [docs/PHYSICS.md §6, docs/BLUEPRINT.md §5.6, docs/ANALYSIS.md §2.2]}
  - {quantity: chirikov_S, old: 0.714, new: 0.458, files: [docs/PHYSICS.md §6, docs/BLUEPRINT.md §5.6, docs/ANALYSIS.md §2.2-2.3], register: [findings.md VE9]}

chirikov_S_scan:   # docs/ANALYSIS.md §2.3; register findings.md VE19 quotes the old S values
  - {amp: 2.5e-4, old: 0.357, new: 0.229}
  - {amp: 5.0e-4, old: 0.505, new: 0.324}
  - {amp: 1.0e-3, old: 0.714, new: 0.458}
  - {amp: 2.0e-3, old: 1.009, new: 0.647}
  - {amp: 4.0e-3, old: 1.427, new: 0.915}
  consequence: "no amplitude in the scan reaches S = 1; claim 'fraction accelerates smoothly across S≈1' withdrawn; 'chaos exists well below S=1' strengthened (from S=0.229)"

measured_over_theory:   # same measurements (pre-fix seeding, data/analysis_pre_fix_2026-09-29/), rescored; docs/ANALYSIS.md §2.2; register findings.md VE18 area
  - {amp: 5.0e-4, mode: 2/1, W_meas: 0.0492, old_ratio: 0.67, new_ratio: 0.95}
  - {amp: 1.0e-3, mode: 2/1, W_meas: 0.0752, old_ratio: 0.72, new_ratio: 1.02}
  - {amp: 1.0e-3, mode: 3/1, W_meas: 0.0628, old_ratio: 0.59, new_ratio: 1.01}
  - {amp: 2.0e-3, mode: 2/1, W_meas: 0.1044, old_ratio: 0.71, new_ratio: 1.00}
  - {amp: 4.0e-3, mode: 3/1, W_meas: 0.1578, old_ratio: 0.74, new_ratio: 1.27, note: "outside pendulum regime (edge, S~0.9, 2 orbits)"}
  interpretation_old: "deficit = regular core surviving inside a stochastic separatrix layer"
  interpretation_new: "deficit was the sqrt(q) formula error (1/sqrt2=0.71, 1/sqrt3=0.58); locked orbits fill the full separatrix; thin stochastic layers exist but do not shrink the island"

fresh_rerun_2026-09-29:   # data/analysis/analysis.json, same CLI, new seeding (refinement band follows the smaller predicted W)
  chaos_threshold_digits: {old: 4.14, new: 3.88}
  ratios: [{amp: 1.0e-3, mode: 2/1, ratio: 1.02}, {amp: 4.0e-3, mode: 2/1, ratio: 1.10}, {amp: 4.0e-3, mode: 3/1, ratio: 1.23}]
  chaos_fraction_uniform: {amps: [0, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3], pre_fix_run: [0.000, 0.075, 0.125, 0.113, 0.175, 0.263], fresh: [0.000, 0.075, 0.100, 0.125, 0.200, 0.200]}
  caveat: "classifier found no locked 2/1 band at 5e-4 and 2e-3, and flagged 2 'island' orbits at amp=0 (3/2, 5/2 plateaus, span 0.001) — flatness test sensitive to denser seeding"

parasitic_VC2:   # docs/CAMPAIGN.md §2 + English summary; src/run_parasitic.py docstring; register findings.md VE26
  - {quantity: expected width of 1.3% spurious m=1 chain, old: 0.0120, new: 0.0077, threshold: 0.004, margin_old: "3x", margin_new: "1.9x", verdict: unchanged (measured 0.0000)}
  - {quantity: driven 2/1 and 3/1 measured vs theory, measured: [0.0756, 0.0651], theory_new: [0.0736, 0.0619], ratio_new: [1.03, 1.05]}

calibration_note_docs/ANALYSIS.md_§1.2: {old_text: "min 5.29, median 6.71 -> 4.29", corrected_to_saved_json: "min 5.14, median 6.66 -> 4.14", note: "5.29 did not match the saved analysis.json; unrelated to the formula"}

not_updated:
  - "none: render/04_poincare_eevee.png re-rendered from the re-baked bake01 on 2026-09-30 (EEVEE 1920x1080, punctures regular/island/chaotic 9840/2880/3600)"

rebake_done_2026-09-29:   # data/bake01 re-baked (backup: data/bake01_pre_fix_2026-09-29); overlays regenerated via src/annotate.py
  - {quantity: W 2/1 (psiN), old: 0.1041, new: 0.0736}
  - {quantity: W 3/1 (psiN), old: 0.1072, new: 0.0619}
  - {quantity: Chirikov S, old: 0.714, new: 0.458}
  - {quantity: orbits regular/island/chaotic, old: 39/18/7, new: 41/12/15, note: "island seeds depend on W (64 -> 68 seeds)"}
  - "findings.md registers (VE9, VE18, VE19, VE26 and any quoting S=0.714/1.009/1.427 or ratios 0.59-0.74)"
```
