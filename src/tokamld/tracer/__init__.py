"""tokamld.tracer — JAX GPU/CPU accelerated field-line tracing in toroidal geometry.

Key features:
1. Canonical symplectic integrator (Implicit Midpoint) with Newton solver (residual <= 1e-12)
   in (psi_t, theta*, phi) magnetic coordinates, verifying |det J - 1| <= 1e-10 (E19).
2. Cartesian RK4 integrator in (R, Z, phi) with alive-masking to prevent NaNs on wall strike.
3. Resonant helical perturbations cos(m*theta* - n*phi + alpha) with smooth C1 envelope
   restricted to psi_N <= 0.95 to avoid logarithmic X-point divergence.
4. Fast bicubic Catmull-Rom patch evaluator for psi(R, Z) and trigonometric angles (cos theta*, sin theta*).
5. Batched Poincare puncture mapping, 3D trajectory generation, and FTLE chaos diagnostics (E20).
6. Divertor / wall footprint detector and Web/Three.js JSON export.
"""

from .bfield import MagneticField2D
from .perturbation import HelicalPerturbation, ResonantMode
from .integrators import (
    rk4_step_cartesian,
    midpoint_step_canonical,
    IntegratorState,
)
from .tracer import (
    trace_poincare_cartesian,
    trace_poincare_canonical,
    trace_trajectories_cartesian,
    compute_ftle_cartesian,
    compute_ftle_canonical,
)
from .wall import WallBoundary
from .export_web import export_poincare_json

__all__ = [
    "MagneticField2D",
    "HelicalPerturbation",
    "ResonantMode",
    "rk4_step_cartesian",
    "midpoint_step_canonical",
    "IntegratorState",
    "trace_poincare_cartesian",
    "trace_poincare_canonical",
    "trace_trajectories_cartesian",
    "compute_ftle_cartesian",
    "compute_ftle_canonical",
    "WallBoundary",
    "export_poincare_json",
]
