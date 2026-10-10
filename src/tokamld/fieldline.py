"""tokamld.fieldline — Field-line integration, Poincare sections, Green's function & island width.

Фізичні формули та калібрування:
1. Функція Гріна кругового витка струму (E33):
   G(R₀, Z₀; R, Z) = (μ₀ / 2π) * (√(R R₀) / k) * ((2 - k²) K(m) - 2 E(m))
   де m = k² = 4 R R₀ / ((R + R₀)² + (Z - Z₀)²).
   ВАЖЛИВО: scipy.special.ellipk та ellipe приймають параметр m = k², а не модуль k!

2. Ширина магнітного острова (R4, VR15):
   W = 4 √(ψ̃ q / |dq/dψ|)
   де ψ — полоїдальний потік, q — коефіцієнт запасу стійкості (safety factor).
   (Формула без додаткового фактора q під коренем).
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.special as special
from scipy.integrate import solve_ivp

MU0 = 4.0e-7 * np.pi


def green_function(R0: float, Z0: float, R: float, Z: float, mu: float = MU0) -> float:
    """Полоїдальний потік на радіан ψ(R, Z) [Вб/рад], створений витком одиничного струму (1 А) на (R₀, Z₀).
    
    Регресійний тест E33: scipy.special.ellipk / ellipe викликаються від параметру m = k², а не модуля k.
    """
    R0_f = float(R0)
    R_f = float(R)
    dz2 = float(Z - Z0) ** 2
    r_sum2 = (R_f + R0_f) ** 2
    denom = r_sum2 + dz2
    if denom <= 0:
        return 0.0
    k2 = 4.0 * R0_f * R_f / denom
    if k2 >= 1.0:
        k2 = 1.0 - 1e-15
    k = np.sqrt(k2)
    ellip_k = float(special.ellipk(k2))
    ellip_e = float(special.ellipe(k2))
    g = (mu * np.sqrt(R_f * R0_f) / (2.0 * np.pi * k)) * ((2.0 - k2) * ellip_k - 2.0 * ellip_e)
    return float(g)


def magnetic_island_width(psi_tilde: float, q_res: float, dq_dpsi: float) -> float:
    """Ширина магнітного острова у полоїдальному потоці (R4, VR15):
    
        W = 4 * sqrt( psi_tilde * q / |dq/dpsi| )
        
    Args:
        psi_tilde: Амплітуда резонансної збуреної моди потоку.
        q_res: Раціональне значення q на резонансній поверхні (наприклад, 2.0 або 3.0).
        dq_dpsi: Магнітний зсув (похідна профілю q за полоїдальним потоком dq/dψ).
        
    Returns:
        Повна ширина сепаратриси острова W у просторі потоку ψ.
    """
    shear = abs(float(dq_dpsi))
    if shear <= 1e-15:
        return float(np.nan)
    arg = abs(float(psi_tilde) * float(q_res)) / shear
    return float(4.0 * np.sqrt(arg))


class FieldLineIntegrator:
    """Інтегратор силових ліній магнітного поля B у тороїдальній геометрії.
    
    Система диференціальних рівнянь:
        dR/dφ = R B_R / B_φ = -(R / F) ∂ψ/∂Z
        dZ/dφ = R B_Z / B_φ = +(R / F) ∂ψ/∂R
    де тороїдальний кут φ виступає в ролі незалежної змінної часу Гамільтоніана.
    """

    def __init__(self, R_grid: np.ndarray, Z_grid: np.ndarray,
                 psi_grid: np.ndarray, F_val: float = 1.0):
        self.R = np.asarray(R_grid, dtype=np.float64)
        self.Z = np.asarray(Z_grid, dtype=np.float64)
        self.psi = np.asarray(psi_grid, dtype=np.float64)
        self.F_val = float(F_val)
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]

    def _grad_psi(self, r: float, z: float) -> Tuple[float, float]:
        """Білінійна інтерполяція компонент ∂ψ/∂R та ∂ψ/∂Z."""
        ir = int(np.clip(np.searchsorted(self.R, r) - 1, 0, len(self.R) - 2))
        iz = int(np.clip(np.searchsorted(self.Z, z) - 1, 0, len(self.Z) - 2))
        tr = (r - self.R[ir]) / self.dR
        tz = (z - self.Z[iz]) / self.dZ

        # Centered or forward differences
        # dpsi/dR:
        dpsi_dR_0 = (self.psi[iz, ir + 1] - self.psi[iz, ir]) / self.dR
        dpsi_dR_1 = (self.psi[iz + 1, ir + 1] - self.psi[iz + 1, ir]) / self.dR
        dpsi_dR = (1.0 - tz) * dpsi_dR_0 + tz * dpsi_dR_1

        # dpsi/dZ:
        dpsi_dZ_0 = (self.psi[iz + 1, ir] - self.psi[iz, ir]) / self.dZ
        dpsi_dZ_1 = (self.psi[iz + 1, ir + 1] - self.psi[iz, ir + 1]) / self.dZ
        dpsi_dZ = (1.0 - tr) * dpsi_dZ_0 + tr * dpsi_dZ_1

        return float(dpsi_dR), float(dpsi_dZ)

    def rhs(self, phi: float, state: np.ndarray) -> np.ndarray:
        r, z = state[0], state[1]
        if r <= self.R[0] or r >= self.R[-1] or z <= self.Z[0] or z >= self.Z[-1]:
            return np.zeros(2)
        dpsi_dR, dpsi_dZ = self._grad_psi(r, z)
        # dR/dphi = -(R / F) * dpsi/dZ
        # dZ/dphi = +(R / F) * dpsi/dR
        dr_dphi = -(r / self.F_val) * dpsi_dZ
        dz_dphi = +(r / self.F_val) * dpsi_dR
        return np.array([dr_dphi, dz_dphi])

    def trace(self, r0: float, z0: float, n_turns: int = 50,
              steps_per_turn: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Трасування силової лінії на n_turns обертів по тору (φ від 0 до 2π n_turns).
        
        Returns:
            (phi_eval, R_points, Z_points)
        """
        phi_max = float(2.0 * np.pi * n_turns)
        phi_eval = np.linspace(0, phi_max, n_turns * steps_per_turn + 1)
        sol = solve_ivp(self.rhs, (0.0, phi_max), [r0, z0],
                        t_eval=phi_eval, method="RK45", rtol=1e-7, atol=1e-8)
        return sol.t, sol.y[0], sol.y[1]

    def poincare_puncture(self, r0: float, z0: float, n_turns: int = 50) -> np.ndarray:
        """Повертає точки перерізу Пуанкаре на площині φ = 0 (mod 2π).
        
        Returns:
            (N, 2) масив точок [R, Z] перерізу Пуанкаре.
        """
        phi_eval = np.arange(0, n_turns + 1) * (2.0 * np.pi)
        sol = solve_ivp(self.rhs, (0.0, phi_eval[-1]), [r0, z0],
                        t_eval=phi_eval, method="RK45", rtol=1e-7, atol=1e-8)
        return np.column_stack([sol.y[0], sol.y[1]])
