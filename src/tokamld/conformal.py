"""tokamld.conformal — Simultaneous and comparative conformal prediction bands for 2D fields.

Реалізація одночасних та порівняльних конформних смуг (simultaneous conformal bands, P2.5, C6):
- Одночасні конформні смуги (SimultaneousConformalBands, M):
  - Калібрування шкали похибки s(x) по калібрувальній вибірці;
  - Обчислення макс-статистики e_j = max_{x in Omega} |y_j(x) - hat{y}_j(x)| / s(x);
  - Знаходження конформного квантиля c* рівня (1 - alpha);
  - Побудова смуги [hat{y}(x) - c* s(x), hat{y}(x) + c* s(x)] із гарантією сумісного покриття.
- Поточкові конформні смуги (PointwiseConformalBands, P):
  - Попіксельний квантиль q_{1-alpha}(x) (колапс сумісного покриття до 0.01-0.06).
- Конформні смуги Бонферроні (BonferroniConformalBands):
  - Поправка альфа / P на множинність гіпотез (нескінченна ширина при скінченному N).
- Кластерний бутстреп за розрядами (cluster / shot-level bootstrap) для оцінки 95% CI.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def conformal_k(n: int, alpha: float) -> int:
    """1-based rank of the split-conformal quantile. > n means infinite band."""
    return int(np.ceil((n + 1) * alpha))


class SimultaneousConformalBands:
    """Одночасні конформні смуги (Simultaneous Studentized Max Bands) для 2D полів."""

    def __init__(self, alpha: float = 0.90, min_std: float = 1e-6):
        """
        Args:
            alpha: Рівень довіри сумісного покриття (наприклад, 0.90 для 90%).
            min_std: Регуляризація шкали дисперсії s(x).
        """
        self.alpha = float(alpha)
        self.min_std = float(min_std)
        self.scale_map: Optional[np.ndarray] = None
        self.c_star: Optional[float] = None
        self.cal_scores: Optional[np.ndarray] = None
        self.n_cal: int = 0

    def fit(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
            mask: Optional[np.ndarray] = None,
            scale_map: Optional[np.ndarray] = None) -> SimultaneousConformalBands:
        """Калібрування шкали s(x) та квантиля c* за вибіркою (N_cal, H, W).
        
        Args:
            y_true_cal: Істинні карти потоку (N, H, W).
            y_pred_cal: Прогнози моделі (N, H, W).
            mask: Бінарна маска регіону інтересу (H, W).
            scale_map: Заздалегідь обчислена шкала s(x) (наприклад, з частини A спліту).
        """
        residuals = np.asarray(y_true_cal - y_pred_cal, dtype=np.float64)
        n_cal = len(residuals)
        if n_cal < 2:
            raise ValueError("Need at least 2 calibration frames to fit conformal scale.")
        self.n_cal = n_cal

        m = np.ones(residuals.shape[1:], dtype=bool) if mask is None else mask.astype(bool)

        # 1. Поточкова шкала похибок s(x)
        if scale_map is not None:
            self.scale_map = np.asarray(scale_map, dtype=np.float64)
        else:
            s = residuals.std(axis=0)
            s_med = float(np.median(s))
            self.scale_map = np.maximum(s, self.min_std * max(s_med, 1e-6))

        # 2. Макс-статистика студентизованої похибки e_j для кожного кадру
        scores = []
        for j in range(n_cal):
            rel_err = np.abs(residuals[j])[m] / self.scale_map[m]
            scores.append(float(np.max(rel_err)))

        self.cal_scores = np.sort(np.asarray(scores, dtype=np.float64))
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
        if not np.isfinite(self.c_star):
            return np.full_like(y_p, -np.inf), np.full_like(y_p, np.inf)
        delta = self.c_star * self.scale_map
        return y_p - delta, y_p + delta

    def evaluate_coverage(self, y_true_test: np.ndarray, y_pred_test: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          shot_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Оцінює спільне та поточкове покриття на тестовій вибірці (N_test, H, W)."""
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

        joint_arr = np.asarray(joint_covered, dtype=bool)
        res: Dict[str, Any] = {
            "joint_coverage": float(np.mean(joint_arr)),
            "mean_pixel_coverage": float(np.mean(pix_covered)),
            "c_star": self.c_star,
            "mean_half_width": float(np.mean(self.c_star * self.scale_map[m])) if np.isfinite(self.c_star) else float(np.inf),
            "mean_full_width": float(np.mean(2.0 * self.c_star * self.scale_map[m])) if np.isfinite(self.c_star) else float(np.inf),
        }

        # Per-shot statistics if shot_ids provided
        if shot_ids is not None:
            s_ids = np.asarray(shot_ids)
            unique_shots = np.unique(s_ids)
            per_shot = {}
            for sid in unique_shots:
                mask_s = s_ids == sid
                per_shot[int(sid)] = {
                    "n_frames": int(np.sum(mask_s)),
                    "joint_coverage": float(np.mean(joint_arr[mask_s])),
                    "pixel_coverage": float(np.mean(np.asarray(pix_covered)[mask_s])),
                }
            res["per_shot_coverage"] = per_shot

        return res


class PointwiseConformalBands:
    """Поточкові конформні смуги (Pointwise Conformal Bands) для 2D полів."""

    def __init__(self, alpha: float = 0.90):
        self.alpha = float(alpha)
        self.quantile_map: Optional[np.ndarray] = None
        self.n_cal: int = 0

    def fit(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
            mask: Optional[np.ndarray] = None) -> PointwiseConformalBands:
        """Обчислює попіксельний квантиль q_{1-alpha}(x)."""
        residuals = np.abs(np.asarray(y_true_cal - y_pred_cal, dtype=np.float64))
        n_cal = len(residuals)
        if n_cal < 2:
            raise ValueError("Need at least 2 calibration frames.")
        self.n_cal = n_cal

        k = conformal_k(n_cal, self.alpha)
        if k <= n_cal:
            # Sort along axis 0
            self.quantile_map = np.partition(residuals, k - 1, axis=0)[k - 1]
        else:
            self.quantile_map = np.full(residuals.shape[1:], np.inf, dtype=np.float64)
        return self

    def predict_band(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.quantile_map is None:
            raise RuntimeError("Model is not calibrated.")
        y_p = np.asarray(y_pred, dtype=np.float64)
        return y_p - self.quantile_map, y_p + self.quantile_map

    def evaluate_coverage(self, y_true_test: np.ndarray, y_pred_test: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          shot_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
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

        joint_arr = np.asarray(joint_covered, dtype=bool)
        w = float(np.mean(self.quantile_map[m])) if self.quantile_map is not None else float(np.inf)
        return {
            "joint_coverage": float(np.mean(joint_arr)),
            "mean_pixel_coverage": float(np.mean(pix_covered)),
            "mean_half_width": w,
            "mean_full_width": 2.0 * w,
        }


class BonferroniConformalBands:
    """Конформні смуги з поправкою Бонферроні (alpha' = alpha / P)."""

    def __init__(self, alpha: float = 0.90):
        self.alpha = float(alpha)
        self.quantile_map: Optional[np.ndarray] = None
        self.is_finite: bool = False
        self.required_n: int = 0
        self.n_cal: int = 0

    def fit(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
            mask: Optional[np.ndarray] = None) -> BonferroniConformalBands:
        residuals = np.abs(np.asarray(y_true_cal - y_pred_cal, dtype=np.float64))
        n_cal = len(residuals)
        self.n_cal = n_cal

        m = np.ones(residuals.shape[1:], dtype=bool) if mask is None else mask.astype(bool)
        p_pixels = int(np.sum(m))
        alpha_prime = (1.0 - self.alpha) / max(p_pixels, 1)
        bonf_level = 1.0 - alpha_prime

        self.required_n = int(np.ceil(p_pixels / (1.0 - self.alpha))) - 1
        k = conformal_k(n_cal, bonf_level)

        if k <= n_cal:
            self.is_finite = True
            self.quantile_map = np.partition(residuals, k - 1, axis=0)[k - 1]
        else:
            self.is_finite = False
            self.quantile_map = np.full(residuals.shape[1:], np.inf, dtype=np.float64)
        return self

    def predict_band(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_p = np.asarray(y_pred, dtype=np.float64)
        if not self.is_finite or self.quantile_map is None:
            return np.full_like(y_p, -np.inf), np.full_like(y_p, np.inf)
        return y_p - self.quantile_map, y_p + self.quantile_map

    def evaluate_coverage(self, y_true_test: np.ndarray, y_pred_test: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        m = np.ones(y_true_test.shape[1:], dtype=bool) if mask is None else mask.astype(bool)
        w = float(np.mean(self.quantile_map[m])) if self.is_finite else float(np.inf)
        return {
            "is_finite": self.is_finite,
            "required_n": self.required_n,
            "available_n": self.n_cal,
            "joint_coverage": 1.0 if not self.is_finite else float(np.mean([np.all(np.abs(y_true_test[j] - y_pred_test[j])[m] <= self.quantile_map[m]) for j in range(len(y_true_test))])),
            "mean_pixel_coverage": 1.0 if not self.is_finite else float(np.mean([np.mean(np.abs(y_true_test[j] - y_pred_test[j])[m] <= self.quantile_map[m]) for j in range(len(y_true_test))])),
            "mean_half_width": w,
            "mean_full_width": 2.0 * w,
        }


def bootstrap_shot_coverage_ci(joint_covered: np.ndarray, shot_ids: np.ndarray,
                               n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Кластерний бутстреп покриття за розрядами (95% довірчий інтервал)."""
    rng = np.random.default_rng(seed)
    s_ids = np.asarray(shot_ids)
    unique_shots = np.unique(s_ids)
    n_shots = len(unique_shots)

    shot_means = np.array([float(np.mean(joint_covered[s_ids == sid])) for sid in unique_shots])

    boot_vals = []
    for _ in range(n_boot):
        sample_idx = rng.choice(n_shots, size=n_shots, replace=True)
        boot_vals.append(float(np.mean(shot_means[sample_idx])))

    return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


def compare_conformal_methods(y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
                              y_true_test: np.ndarray, y_pred_test: np.ndarray,
                              alpha: float = 0.90, mask: Optional[np.ndarray] = None,
                              test_shot_ids: Optional[np.ndarray] = None,
                              null_width: Optional[float] = None) -> Dict[str, Any]:
    """Порівнює одночасну смугу, попіксельну смугу та Бонферроні на тих самих даних."""
    sim = SimultaneousConformalBands(alpha=alpha).fit(y_true_cal, y_pred_cal, mask=mask)
    sim_res = sim.evaluate_coverage(y_true_test, y_pred_test, mask=mask, shot_ids=test_shot_ids)

    pw = PointwiseConformalBands(alpha=alpha).fit(y_true_cal, y_pred_cal, mask=mask)
    pw_res = pw.evaluate_coverage(y_true_test, y_pred_test, mask=mask, shot_ids=test_shot_ids)

    bonf = BonferroniConformalBands(alpha=alpha).fit(y_true_cal, y_pred_cal, mask=mask)
    bonf_res = bonf.evaluate_coverage(y_true_test, y_pred_test, mask=mask)

    price = sim_res["mean_half_width"] / pw_res["mean_half_width"] if pw_res["mean_half_width"] > 0 else np.nan
    kappa = sim_res["mean_half_width"] / null_width if null_width is not None and null_width > 0 else np.nan

    out: Dict[str, Any] = {
        "alpha": alpha,
        "simultaneous_M": sim_res,
        "pointwise_P": pw_res,
        "bonferroni": bonf_res,
        "price_of_simultaneity": float(price),
        "relative_width_kappa": float(kappa),
    }

    if test_shot_ids is not None:
        lower, upper = sim.predict_band(y_pred_test)
        m = np.ones(y_true_test.shape[1:], dtype=bool) if mask is None else mask.astype(bool)
        covered = np.array([bool(np.all((y_true_test[j][m] >= lower[j][m]) & (y_true_test[j][m] <= upper[j][m]))) for j in range(len(y_true_test))])
        ci_l, ci_h = bootstrap_shot_coverage_ci(covered, test_shot_ids)
        out["simultaneous_M"]["joint_coverage_ci95"] = [ci_l, ci_h]

    return out
