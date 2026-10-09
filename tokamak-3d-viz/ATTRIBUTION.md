# Attribution and licensing

## Code

The code in this repository is released under the MIT licence (see `LICENSE`).

## Scientific data — NOT redistributed here

This repository contains **no** shot data. `.parquet` and `.npz` data files are
excluded by `.gitignore` deliberately, and `src/tokviz/config.py` reads their
location from `TOKVIZ_DATA_DIR`.

The equilibria used during development come from:

> **The Fusion Equilibrium Challenge** — Sophelio and General Atomics,
> NeurIPS 2026 Competition Track.
> Dataset: `Sophelio/fusion-equilibrium-challenge` on HuggingFace, **CC BY 4.0**.
> Nakkina et al., *The Fusion Equilibrium Challenge: Inferring Magnetic Geometry
> Without Magnetic Diagnostics*, arXiv:2609.01750 (2026).

Specifically `d3d_shot_203702.parquet` (DIII-D), frame 75, t = 1760 ms.

**Licensing caution.** The DIII-D portion is CC BY 4.0. The **MAST** portion of
the same corpus is **CC BY-SA 4.0**, and that share-alike term conflicts with
CC BY 4.0 for a combined redistribution. This is an open question in the parent
project (`Our try/00-decisions/open-questions.md`, Q13) and is the reason
nothing is vendored here. Resolve it before republishing any derived dataset.

If you publish images produced by this pipeline, attribute the dataset:

> Equilibrium data: Fusion Equilibrium Challenge (Sophelio / General Atomics),
> CC BY 4.0.

## Prior art this work builds on

- Lao et al., *Nucl. Fusion* **25**, 1611 (1985) — EFIT and the g-file conventions.
- Balescu, Vlad & Spineanu, *Phys. Rev. E* **58**, 951 (1998) — the Tokamap,
  used as the symplectic reference in the parent project's `D3-poincare-inverse`.
- Escande & Momo, *Rev. Mod. Plasma Phys.* **8**, 16 (2024) — open-access
  modern treatment of field-line Hamiltonians and Poincaré sections.
- Cary & Littlejohn, *Ann. Phys.* **151**, 1 (1983) — the action formulation.
- Nazikian et al., *Phys. Rev. Lett.* **114**, 105002 (2015) — DIII-D n = 3 RMP
  physics, the motivation for the resonant mode set.
- Sunn Pedersen et al., *Nat. Commun.* **7**, 13493 (2016) — the W7-X
  electron-beam photograph of a Poincaré section.

The `fusion_scoring` package in the challenge starter kit (MIT) defines
"correct" for O-point and LCFS extraction in the parent project; this repository
implements those independently so it can stand alone, and `tests/test_physics.py`
cross-checks against the dataset's own published values.

## Software

Blender (GPL-2.0-or-later), numpy, scipy, pyarrow, scikit-image, matplotlib,
Pillow — each under its own licence.
