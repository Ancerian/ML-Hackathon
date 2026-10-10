"""tokamld.gs — Differentiable Grad-Shafranov operators and residual evaluation.

Рівняння Град-Шафранова:
    Δ* ψ = -μ₀ R² p'(ψ) - F F'(ψ)
де еліптичний оператор у циліндричних координатах:
    Δ* ψ = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

MU0 = 4.0e-7 * np.pi
DEFAULT_G_REF = 0.6328


def delta_star_torch(psi: torch.Tensor, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Диференційовний оператор Δ* ψ у PyTorch.
    
    Args:
        psi: Tensor форми (..., H, W) або (H, W), де rows=Z, cols=R.
        R: 1D Tensor радіальних координат (W,).
        Z: 1D Tensor вертикальних координат (H,).
        
    Returns:
        Tensor форми (..., H, W) із Δ* ψ = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z².
    """
    orig_dim = psi.dim()
    if orig_dim == 2:
        psi = psi.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif orig_dim == 3:
        psi = psi.unsqueeze(1)               # (B, 1, H, W)

    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    # Central difference kernels
    k_dR = torch.tensor([[[-0.5, 0.0, 0.5]]], dtype=psi.dtype, device=psi.device) / dR
    k_d2R = torch.tensor([[[1.0, -2.0, 1.0]]], dtype=psi.dtype, device=psi.device) / (dR ** 2)

    k_dZ = torch.tensor([[[-0.5], [0.0], [0.5]]], dtype=psi.dtype, device=psi.device) / dZ
    k_d2Z = torch.tensor([[[1.0], [-2.0], [1.0]]], dtype=psi.dtype, device=psi.device) / (dZ ** 2)

    # 2nd order centered differences with replicate padding
    p_R = F.pad(psi, (1, 1, 0, 0), mode="replicate")
    d_dR = F.conv2d(p_R, k_dR.unsqueeze(0))
    d2_dR2 = F.conv2d(p_R, k_d2R.unsqueeze(0))

    p_Z = F.pad(psi, (0, 0, 1, 1), mode="replicate")
    d2_dZ2 = F.conv2d(p_Z, k_d2Z.unsqueeze(0))

    R_inv = (1.0 / R).view(1, 1, 1, -1).to(psi.device)
    out = d2_dR2 - R_inv * d_dR + d2_dZ2

    if orig_dim == 2:
        return out.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        return out.squeeze(1)
    return out


def delta_star_numpy(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Оператор Δ* ψ для NumPy масивів.
    
    Δ* ψ = d²ψ/dR² - (1/R) dψ/dR + d²ψ/dZ²
    """
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    d_dR = np.gradient(psi, dR, axis=-1)
    d2_dR2 = np.gradient(d_dR, dR, axis=-1)
    d2_dZ2 = np.gradient(np.gradient(psi, dZ, axis=-2), dZ, axis=-2)
    return d2_dR2 - d_dR / R[None, :] + d2_dZ2


def delta_star(psi: Union[np.ndarray, torch.Tensor],
               R: Union[np.ndarray, torch.Tensor],
               Z: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Універсальний виклик оператора Δ* ψ (автовизначення torch / numpy)."""
    if isinstance(psi, torch.Tensor):
        R_t = R if isinstance(R, torch.Tensor) else torch.as_tensor(R, dtype=psi.dtype, device=psi.device)
        Z_t = Z if isinstance(Z, torch.Tensor) else torch.as_tensor(Z, dtype=psi.dtype, device=psi.device)
        return delta_star_torch(psi, R_t, Z_t)
    return delta_star_numpy(np.asarray(psi), np.asarray(R), np.asarray(Z))


def residual(psi: Union[np.ndarray, torch.Tensor],
             R: Union[np.ndarray, torch.Tensor],
             Z: Union[np.ndarray, torch.Tensor],
             pprime: Optional[Union[np.ndarray, torch.Tensor]] = None,
             ffprime: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, torch.Tensor]:
    """Повна нев'язка Град-Шафранова:
        Res = Δ* ψ + μ₀ R² p'(ψ) + F F'(ψ)
        
    Якщо p' та FF' не задано (None), повертає Δ* ψ.
    """
    ds = delta_star(psi, R, Z)
    if pprime is None and ffprime is None:
        return ds

    if isinstance(psi, torch.Tensor):
        res = ds.clone()
        R_t = R if isinstance(R, torch.Tensor) else torch.as_tensor(R, dtype=psi.dtype, device=psi.device)
        R2 = (R_t ** 2).view(1, -1)
        if pprime is not None:
            pp = pprime if isinstance(pprime, torch.Tensor) else torch.as_tensor(pprime, dtype=psi.dtype, device=psi.device)
            res = res + MU0 * R2 * pp
        if ffprime is not None:
            ff = ffprime if isinstance(ffprime, torch.Tensor) else torch.as_tensor(ffprime, dtype=psi.dtype, device=psi.device)
            res = res + ff
        return res
    else:
        res = np.copy(ds)
        R2 = (R ** 2)[None, :]
        if pprime is not None:
            res += MU0 * R2 * pprime
        if ffprime is not None:
            res += ffprime
        return res


def gs_inconsistency(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                     mask_coarse: np.ndarray, mask_f: np.ndarray,
                     n_p: int = 4, n_f: int = 4) -> Tuple[float, int]:
    """Відносна нев'язка Град-Шафранова для заданого поля ψ без відомих профілів p' та FF'.
    
    Мінімізує нев'язку за поліномами p'(ψ_N) та FF'(ψ_N) всередині LCFS через МНК.
    """
    try:
        from .topology import plasma_mask
    except ImportError:
        from tokamld.topology import plasma_mask
    inside, psi_a, psi_b = plasma_mask(psi, R, Z, mask_coarse, mask_f)
    if inside is None or inside.sum() < 50:
        return np.nan, 0
    ds = delta_star_numpy(psi, R, Z)
    RR, _ = np.meshgrid(R, Z)
    diff = psi_b - psi_a
    if abs(diff) < 1e-12:
        return np.nan, int(inside.sum())
    psin = np.clip((psi - psi_a) / diff, 0.0, 1.0)
    m = inside
    y = ds[m]
    denom = np.linalg.norm(y)
    if denom < 1e-12:
        return 0.0, int(m.sum())
    cols = [(RR[m] ** 2) * psin[m] ** i for i in range(n_p)] + [psin[m] ** j for j in range(n_f)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, -y, rcond=None)
    r = y + A @ coef
    return float(np.linalg.norm(r) / denom), int(m.sum())


def gate(g: Optional[float] = None,
         score: float = 1.0,
         g_ref: float = DEFAULT_G_REF) -> float:
    """Неперервний фізичний шлюз Град-Шафранова G_GS (Специфікація v2.1).
    
    G_GS = 1.0 при g <= g_ref, і exp(-(g - g_ref) / g_ref) при g > g_ref.
    """
    if g is None or not np.isfinite(g):
        return 0.0
    if g <= g_ref:
        return float(score)
    factor = float(np.exp(-(g - g_ref) / g_ref))
    return float(score * factor)
