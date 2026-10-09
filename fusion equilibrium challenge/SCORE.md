# Official metric — my reproduced baseline

Scorer: `starter/local_score.py`, using `fusion_scoring/` (the competition's own modules, vendored
unmodified). Machine: DIII-D — the only one with released ground truth.

## Harness self-check

| Mode | Composite S | Expected |
|---|---|---|
| `--mode perfect --n-shots 2` | **1.0000** | 1.0 — PASS |

All four terms 1.0; LCFS extraction failed on 0.0 % of frames; `psi_sign = +1`.

## PCA + Ridge baseline

`my_experiments/baseline_pca_ridge.py`, trained on the first 40 streamed `diii_d_train` shots
(8,626 frames, 21 coil-current features, 50 PCA components on ψ), scored on 8 held-out shots
(`--n-shots 8 --skip 60`) it never saw.

```
COMPOSITE S = 0.1762      (8 held-out DIII-D shots)

            R2_psi    0.0640   x 0.55  =  0.0352
    R2_{q95,betaN}   -0.7362   x 0.15  =  0.0000   (clipped at 0)
        1 - D_LCFS    0.9035   x 0.10  =  0.0903
       Consistency    0.2534   x 0.20  =  0.0507

  per-derived-scalar R2:
        R_axis   0.4352      tri_bot  -0.1832
        Z_axis   0.2715       volume  -1.3664
         kappa   0.4967           li  -0.2411
       tri_top   0.5704

  LCFS extraction failed on 0.0% of frames; derivations on 0.0%
```

## What this says

**Generalization across discharges is the whole problem.** `experiments.py` reports R²ψ = 0.396 for
Ridge on its internal test split — but that split is drawn from the *same 50 shots* as training.
On 8 genuinely different discharges the same model class gets **0.064**. The internal split
flatters every model in the suite; treat its numbers as a smoke test, not as performance.

**A good boundary does not mean a good reconstruction.** `1 − D_LCFS = 0.90` alongside R²ψ = 0.06:
the LCFS contour is extracted from the *shape* of the field, which a low-rank linear fit gets
roughly right, while the flux *values* are wrong. Do not read the boundary term as validation.

**The scalar heads are worse than predicting the mean.** R²{q95,βN} = −0.74, clipped to 0 by the
scorer. Ridge on 21 coil currents cannot reach q95 and βN — they depend on the toroidal field
function and the pressure profile, which coil currents alone do not determine.

**Leaderboard context.** Top entries sit at S ≈ 0.99. This baseline is 0.18. The gap is not the
metric being generous — it is 7,041 training shots plus a nonlinear model.

## Reproduce

```bash
cd starter && source .venv/bin/activate
python local_score.py --mode perfect --n-shots 2
python my_experiments/baseline_pca_ridge.py --train --n-shots 40 --n-pca 50
python local_score.py --n-shots 8 --skip 60
```

Use a small `--skip`. The README's `--skip 7000` iterates the stream from row 0, so it pulls
through 7,000 shots before scoring anything — 32 min wall-clock for 3 min 55 s of CPU here.
