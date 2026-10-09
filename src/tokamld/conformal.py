"""tokamld.conformal — Simultaneous conformal prediction bands for 2D fields.

Реалізація одночасних конформних смуг (simultaneous conformal bands, P2.5, C6):
- Калібрування шкали похибки s(x) по калібрувальній вибірці;
- Обчислення макс-статистики e_j = max_{x in Omega} |y_j(x) - hat{y}_j(x)| / s(x);
- Знаходження конформного квантиля c* рівня (1 - alpha);
- Побудова смуги [hat{y}(x) - c* s(x), hat{y}(x) + c* s(x)] із гарантією сумісного покриття.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def conformal_k(n: int, alpha: float) -> int:
    """1-based rank of the split-conformal quantile. > n means infinite band."""
    return int(np.ceil((n + 1) * alpha))


class SimultaneousConformalBands:
    """Одночасні конформні смуги для 2D карт магнітного потоку."""

    def __init__(self, alpha: float = 0.90, min_std: float = 1e-6):
        """
        Args:
            alpha: Рівень покриття (наприклад, 0.90 для 90% довіри).
            min_std: Регуляризація дисперсії проти нульового шуму на границях.
        """
        self.alpha = alpha
        self.min_std = min_std
        self.scale_map: Optional[np.ndarray] = None
        self.c_star: Optional[float] = None
        self.cal_scores: Optional[np.ndarray] = None

    def fit(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
            mask: Optional[np.ndarray] = None) -> SimultaneousConformalBands:
        """Калібрування смуг за вибіркою (N_cal, H, W).
        
        Args:
            y_true_cal: Істинні карти потоку (N, H, W).
            y_pred_cal: Прогнози моделі (N, H, W).
            mask: Бінарна маска регіону інтересу (H, W). Якщо None, використовується все поле.
        """
        residuals = np.asarray(y_true_cal - y_pred_cal, dtype=np.float64)
        n_cal = len(residuals)
        if n_cal < 2:
            raise ValueError("Need at least 2 calibration frames to fit conformal scale.")

        # 1. Поточкова шкала похибок s(x)
        s = residuals.std(axis=0)
        s_med = float(np.median(s))
        self.scale_map = np.maximum(s, self.min_std * max(s_med, 1e-6))

        # 2. Макс-статистика похибки e_j для кожного кадру
        m = np.ones(residuals.shape[1:], dtype=bool) if mask is None else mask.astype(bool)
        scores = []
        for j in range(n_cal):
            rel_err = np.abs(residuals[j])[m] / self.scale_map[m]
            scores.append(float(np.max(rel_err)))

        self.cal_scores = np.sort(np.asarray(scores))
        k = conformal_k(n_cal, self.alpha)
        if k <= n_cal:
            self.c_star = float(self.cal_scores[k - 1])
        else:
            self.c_star = float(np.inf)
        return self

    def predict_band(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Повертає нижню та верхню межі (lower, upper) форми як y_pred."""
        if self.scale_map is None or self.c_star is None:
            raise RuntimeError("Model is not calibrated. Call .fit() first.")
        y_p = np.asarray(y_pred, dtype=np.float64)
        delta = self.c_star * self.scale_map
        return y_p - delta, y_p + delta

    def evaluate_coverage(self, y_true_test: np.ndarray, y_pred_test: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Перевіряє емпіричне покриття на тестовій вибірці (N_test, H, W)."""
        lower, upper = self.predict_band(y_pred_test)
        y_t = np.asarray(y_true_test, dtype=np.float64)
        m = np.ones(y_t.shape[1:], dtype=bool) if mask is None else mask.astype(bool)

        n_test = len(y_t)
        joint_covered = []
        pix_covered = []

        for j in range(n_test):
            in_band = (y_t[j][m] >= lower[j][m]) & (y_t[j][m] <= upper[j][m])
            joint_covered.append(bool(np.all(in_band)))
            pix_covered.append(float(np.mean(in_band)))

        return {
            "joint_coverage": float(np.mean(joint_covered)),
            "mean_pixel_coverage": float(np.mean(pix_covered)),
            "c_star": self.c_star,
            "mean_band_width": float(np.mean(2.0 * self.c_star * self.scale_map[m])),
        }
