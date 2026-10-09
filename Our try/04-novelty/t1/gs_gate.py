"""gs_gate.py — Grad-Shafranov equilibrium gating and penalty functions.

Provides clean APIs for evaluating whether a predicted magnetic flux map psi
satisfies the Grad-Shafranov equilibrium constraint within tolerance g_ref,
and applies hard, sigmoid, or logarithmic penalties to model competition scores.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import sys

# Ensure local dependencies can be imported
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(PROJECT / "Our try" / "04-novelty"))

from gs_residual_probe import gs_inconsistency, delta_star, plasma_mask


# Default reference threshold from ground truth + 1% noise (c4/gref.json)
DEFAULT_G_REF = 0.6340883948271788


def compute_g(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, mask_coarse: np.ndarray, mask_f: np.ndarray, n_p: int = 4, n_f: int = 4) -> float:
    """Compute relative Grad-Shafranov inconsistency g(psi).
    
    Returns 0 for perfect equilibrium, ~0.01 for EFIT ground truth,
    ~0.63 for ground truth + 1% noise, and >0.8 for unconstrained neural networks.
    """
    try:
        g, npix = gs_inconsistency(psi, R, Z, mask_coarse, mask_f, n_p=n_p, n_f=n_f)
        return float(g) if np.isfinite(g) else np.nan
    except Exception:
        return np.nan


def gate(psi: np.ndarray, score: float, g_ref: float = DEFAULT_G_REF,
         R: np.ndarray | None = None, Z: np.ndarray | None = None,
         mask_coarse: np.ndarray | None = None, mask_f: np.ndarray | None = None,
         mode: str = "hard", k: float = 10.0, beta: float = 1.0,
         precomputed_g: float | None = None) -> float:
    """Apply Grad-Shafranov gating or penalty to a competition score.

    Parameters:
    -----------
    psi : np.ndarray
        2D predicted flux map (65, 65).
    score : float
        Original competition composite score S in [0, 1].
    g_ref : float
        Reference physical threshold for acceptable GS inconsistency.
    mode : str
        'hard'    : S * 1[g < g_ref]
        'sigmoid' : S / (1 + exp(k * (g - g_ref) / g_ref))
        'log'     : S * exp(-beta * max(0, g - g_ref) / g_ref)
    k : float
        Sigmoid steepness parameter (default 10.0).
    beta : float
        Log/exponential decay rate (default 1.0).
    precomputed_g : float | None
        Optional precomputed inconsistency g(psi) to avoid recomputing.

    Returns:
    --------
    float: Gated/penalized score S' in [0, 1].
    """
    if precomputed_g is not None:
        g_val = precomputed_g
    else:
        if R is None or Z is None or mask_coarse is None or mask_f is None:
            # Load default envelope masks
            mask_path = STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz"
            z = np.load(mask_path)
            R = z["grid_R"]
            Z = z["grid_Z"]
            mask_coarse = z["mask_coarse"].astype(bool)
            mask_f = mask_coarse.astype(np.float64)
        g_val = compute_g(psi, R, Z, mask_coarse, mask_f)

    if not np.isfinite(g_val):
        return 0.0

    if mode == "hard":
        # Variant A: Hard indicator gate
        return float(score) if g_val < g_ref else 0.0

    elif mode == "sigmoid":
        # Variant B: Smooth sigmoid transition
        z_arg = k * (g_val - g_ref) / g_ref
        sig = 1.0 / (1.0 + np.exp(np.clip(z_arg, -50.0, 50.0)))
        return float(score * sig)

    elif mode == "log":
        # Variant C: Logarithmic/exponential excess penalty
        excess = max(0.0, g_val - g_ref)
        decay = np.exp(-beta * excess / g_ref)
        return float(score * decay)

    else:
        raise ValueError(f"Unknown gating mode '{mode}'. Choose 'hard', 'sigmoid', or 'log'.")
