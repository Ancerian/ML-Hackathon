# Fusion Equilibrium Challenge — restored experiment

Reproduction of the baseline experiment published by the **Fusion Equilibrium Challenge**
(Sophelio, NeurIPS 2026 Competition Track) — <https://fusion-equilibrium-challenge.sophelio.io/>.

The task: reconstruct a tokamak's 2-D poloidal flux map ψ(R,Z) from **coil currents and Thomson
scattering only** — no magnetic sensors — on DIII-D, and zero-shot on MAST.

## What is here

| Path | What it is |
|---|---|
| `starter/` | Upstream starter kit, cloned from GitHub (MIT). Untouched except where noted below. |
| `starter/parquet_data/` | The 6 demo shots (3 DIII-D, 3 MAST), 67 MB — fetched without `git-lfs`. |
| `starter/.venv/` | Python 3.12 env with the pinned dependencies + torch 2.5.1 (MPS). |
| `starter/results/` | Plots from the reproduced 50-shot experiment. |
| `starter/my_experiments/` | My own PCA+Ridge baseline, wired into the scorer (gitignored upstream). |
| `downloaded_huggingface/hf_dataset/` | Demo shots re-laid-out in Hub format for `--source local`. |

Two files in `starter/` are modified: `submission_skeleton.py` (its `your_model_predict`
placeholder now delegates to `my_experiments/baseline_pca_ridge.py`) and `parquet_data/*`
(LFS pointers replaced with the real files).

## Data

Full corpus: 9,121 shots / 98 GB on Hugging Face, CC BY 4.0 —
`Sophelio/fusion-equilibrium-challenge`, configs `diii_d_train` (7,041), `diii_d_public_test`
(874), `mast_public_test` (1,206). **Nothing was bulk-downloaded**: the experiments stream
individual shots over HTTP, which works unauthenticated (rate-limited).

The 6 demo shots ship as Git LFS pointers and `git-lfs` is not installed here, so they were
pulled from GitHub's raw endpoint instead, which resolves LFS server-side:

```bash
curl -L https://github.com/Sophelio/fusion-equilibrium-challenge-starter/raw/main/parquet_data/<file>.parquet
```

## How to re-run

```bash
cd starter
source .venv/bin/activate

python experiments.py --quick --skip-pytorch          # 3 shots, ~1 min
python experiments.py --n-shots 50 --epochs 50 --device mps   # full run, ~12 min

python local_score.py --mode perfect --n-shots 2      # harness self-check, must print S = 1.0
python my_experiments/baseline_pca_ridge.py --train --n-shots 40 --n-pca 50
python local_score.py --n-shots 10 --skip 7000        # official metric on held-out shots
```

## Results reproduced

**Experiment suite** (`experiments.py`, 50 DIII-D shots, 11,050 frames → 7,766 train / 943 val /
2,341 test, split by shot; 50 PCA components on ψ; R² is on the flux map):

| Model | MSE | R²ψ | SSIM | Train time |
|---|---|---|---|---|
| Simple MLP (PyTorch) | 0.001685 | **0.9148** | 0.6555 | 16.7 s |
| MLP (sklearn) | 0.002083 | 0.8952 | 0.8169 | 1.8 s |
| Conv Decoder | 0.002178 | 0.8891 | 0.8044 | 31.9 s |
| UNet Lite | 0.002627 | 0.8670 | 0.7528 | 231.4 s |
| Random Forest | 0.002984 | 0.8536 | 0.8738 | 1.4 s |
| Ridge (CV) | 0.012506 | 0.3962 | 0.4855 | 0.0 s |
| Linear Regression | 0.012522 | 0.3954 | 0.4852 | 0.0 s |

PCA on ψ: 50 components capture ~100 % of variance, component 1 alone **94.2 %** — the site
claims ≈92 %, so that checks out.

**Scale matters more than model class.** On 3 shots (`--quick`) Ridge wins with R²ψ = 0.963 and
the nonlinear models are worse than useless (Random Forest R² = −2.49). On 50 shots the ordering
inverts: Ridge collapses to 0.396 while the neural baselines sit at 0.87–0.91. A 3-shot run is
not a small version of a 50-shot run; it is a different problem.

**Composite metric** — `S = 0.55·R²ψ + 0.15·R²{q95,βN} + 0.10·(1 − D_LCFS) + 0.20·Consistency`,
computed by `fusion_scoring/`, the competition's own scorer modules copied unmodified.

- Harness self-check: `--mode perfect` → **S = 1.0000** exactly (all four terms 1.0, LCFS
  extraction failed on 0 % of frames). The vendored scorer is intact.
- My PCA+Ridge baseline (trained on 40 shots / 8,626 frames, scored on 10 held-out shots at
  `--skip 7000`): see `SCORE.md`.

## Known gotcha reproduced

`experiments.py` applies the v1.1.0 erratum fix for DIII-D `magnetics_plasma_current_times`
(wrong on ~69 % of shots — `data_fixes.fix_d3d_ip_times`). My baseline applies it too. Without
it, Ip interpolates to the ~2 kA pre-plasma noise floor instead of ~1 MA.

The flux maps predicted by the neural baselines are pixel-noisy (see
`results/flux_Simple_MLP_PyTorch.png`): globally right, locally grainy. That is exactly the
failure the challenge warns about — a high R²ψ with a poor `Consistency` term, because the seven
ψ-derived scalars are computed from contours and gradients that noise destroys.

## Licensing

Dataset CC BY 4.0 · starter code MIT · this folder holds no redistributed challenge data beyond
the 6 public demo shots.
