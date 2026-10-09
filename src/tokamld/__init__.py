"""tokamld — Differentiable PyTorch tools for Tokamak Equilibrium & ML.

Modules:
- tokamld.gs: Differentiable Grad-Shafranov operators (delta_star, residual, gate).
- tokamld.topology: Critical points (O/X), Poincare-Hopf index, canonical topology check.
- tokamld.conformal: Simultaneous conformal bands and UQ for 2D fields.
- tokamld.fieldline: Field-line tracing, Poincare punctures, Green's function, island width.
- tokamld.imas: IMAS IDS equilibrium export and import.
"""

from . import gs
from . import topology
from . import conformal
from . import fieldline
from . import imas

from .imas import EquilibriumData, to_ids, from_ids

__version__ = "0.1.0"
__all__ = ["gs", "topology", "conformal", "fieldline", "imas", "EquilibriumData", "to_ids", "from_ids"]

