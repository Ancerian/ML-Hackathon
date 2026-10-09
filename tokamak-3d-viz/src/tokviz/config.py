"""Paths and physical constants.

The DIII-D / MAST parquet corpus is third-party data (CC BY 4.0, Sophelio +
General Atomics) and is deliberately NOT vendored into this repository.  Point
``TOKVIZ_DATA_DIR`` at a directory holding the parquet files, or pass an explicit
path on the command line.  See ATTRIBUTION.md.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BAKE_DIR = Path(os.environ.get("TOKVIZ_BAKE_DIR", REPO_ROOT / "data"))
RENDER_DIR = Path(os.environ.get("TOKVIZ_RENDER_DIR", REPO_ROOT / "render"))

#: Directory containing ``*.parquet`` shot files.  Default guesses the sibling
#: challenge checkout that this project was developed against.
DATA_DIR = Path(
    os.environ.get(
        "TOKVIZ_DATA_DIR",
        REPO_ROOT.parent / "fusion equilibrium challenge" / "starter" / "parquet_data",
    )
)

#: Optional: vessel envelope masks shipped with the challenge scorer.
MASK_DIR = Path(
    os.environ.get(
        "TOKVIZ_MASK_DIR",
        REPO_ROOT.parent / "fusion equilibrium challenge" / "starter" / "fusion_scoring" / "masks",
    )
)

DEFAULT_SHOT = "d3d_shot_203702.parquet"

# ---------------------------------------------------------------------------
# Machine / sign conventions
# ---------------------------------------------------------------------------
#: EFIT sign convention per machine.  DIII-D stores the magnetic axis at the
#: MINIMUM of psi, MAST at the maximum.  Multiplying psi by this makes the axis
#: a maximum in every case, which is what the O-point finder assumes.
#: (Matches ``fusion_scoring/common.py:AXIS_SIGN`` in the challenge scorer.)
AXIS_SIGN = {"DIII-D": -1.0, "MAST": +1.0}

MU0 = 4.0e-7 * 3.141592653589793

# ---------------------------------------------------------------------------
# Reactor-class composite scaling
# ---------------------------------------------------------------------------
# The equilibrium is a real DIII-D discharge (R0 ~ 1.68 m, a ~ 0.67 m); the
# rendered machine is reactor-class.  A UNIFORM similarity scale preserves the
# aspect ratio exactly, and therefore leaves q(psi) and the entire field-line
# topology invariant -- islands, resonant surfaces and the Poincare section are
# unchanged.  This is why the render may honestly carry a real shot's topology
# at reactor scale.  See docs/PHYSICS.md section "Scaling".
GEOMETRY_SCALE = 3.0

__all__ = [
    "REPO_ROOT", "BAKE_DIR", "RENDER_DIR", "DATA_DIR", "MASK_DIR",
    "DEFAULT_SHOT", "AXIS_SIGN", "MU0", "GEOMETRY_SCALE",
]
