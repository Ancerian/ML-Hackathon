"""tokamld.topology — Critical point detection, Poincare-Hopf index and topology checking.

Канонічна топологія магнітної конфігурації всередині сепаратриси (LCFS):
- Рівно 1 еліптична O-точка (магнітна вісь, індекс +1);
- 0 гіперболічних X-точок (сідлових точок, індекс -1).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOOP = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _winding(ang: np.ndarray, iz0: int, iz1: int, ir0: int, ir1: int) -> int:
    """Winding number of grad psi around rectangle boundary [iz0..iz1] x [ir0..ir1]."""
    pts = ([(iz0, r) for r in range(ir0, ir1 + 1)]
           + [(r, ir1) for r in range(iz0 + 1, iz1 + 1)]
           + [(iz1, r) for r in range(ir1 - 1, ir0 - 1, -1)]
           + [(r, ir0) for r in range(iz1 - 1, iz0, -1)])
    a = np.array([ang[z_, r_] for z_, r_ in pts])
    d = np.diff(np.append(a, a[0]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return int(np.round(d.sum() / (2 * np.pi)))


def find_critical_points(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                         inside: np.ndarray, min_npix: int = 3) -> List[Dict]:
    """Визначає критичні точки поля ψ всередині маски `inside` із їхнім індексом Пуанкаре-Хопфа.
    
    Компоненти шуму розміром менше `min_npix` відкидаються для уникнення граничних артефактів.
    """
    from scipy.ndimage import label
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    gz, gr = np.gradient(psi, dZ, dR)
    ang = np.arctan2(gz, gr)
    nz, nr = psi.shape

    raw = np.zeros_like(psi)
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            if not inside[iz, ir]:
                continue
            a = [ang[iz + dz, ir + dr] for dz, dr in LOOP]
            a.append(a[0])
            d = np.diff(a)
            d = (d + np.pi) % (2 * np.pi) - np.pi
            raw[iz, ir] = np.round(d.sum() / (2 * np.pi))

    lab, n = label(raw != 0, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        zs, rs = np.where(lab == c)
        if len(zs) < min_npix:
            continue
        iz0, iz1 = max(zs.min() - 1, 0), min(zs.max() + 1, nz - 1)
        ir0, ir1 = max(rs.min() - 1, 0), min(rs.max() + 1, nr - 1)
        w = _winding(ang, iz0, iz1, ir0, ir1)
        if w != 0:
            out.append({
                "iz": float(zs.mean()),
                "ir": float(rs.mean()),
                "index": int(w),
                "npix": int(len(zs)),
            })
    return out


def count_critical_points(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                          inside: np.ndarray, min_npix: int = 3) -> Tuple[int, int]:
    """Повертає пару (N_O, N_X) для критичних точок всередині `inside`."""
    pts = find_critical_points(psi, R, Z, inside, min_npix=min_npix)
    n_O = sum(1 for p in pts if p["index"] > 0)
    n_X = sum(1 for p in pts if p["index"] < 0)
    return n_O, n_X


def check_canonical_topology(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                             inside: np.ndarray, min_npix: int = 3) -> Tuple[bool, int, Dict]:
    """Перевіряє умову канонічної топології: рівно 1 O-точка і 0 X-точок всередині LCFS.
    
    Returns:
        (is_canonical, N_spurious, info_dict)
    """
    n_O, n_X = count_critical_points(psi, R, Z, inside, min_npix=min_npix)
    n_spurious = abs(n_O - 1) + n_X
    is_canonical = (n_spurious == 0)
    return is_canonical, n_spurious, {"n_O": n_O, "n_X": n_X, "n_spurious": n_spurious}


def _bilin(f: np.ndarray, R: np.ndarray, Z: np.ndarray, r: float, z: float) -> float:
    ir = np.clip(np.searchsorted(R, r) - 1, 0, len(R) - 2)
    iz = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
    tr = (r - R[ir]) / (R[ir + 1] - R[ir])
    tz = (z - Z[iz]) / (Z[iz + 1] - Z[iz])
    return float(((1 - tz) * ((1 - tr) * f[iz, ir] + tr * f[iz, ir + 1])
                  + tz * ((1 - tr) * f[iz + 1, ir] + tr * f[iz + 1, ir + 1])))


def plasma_mask(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                mask_coarse: np.ndarray, mask_f: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
    """Повертає маску внутрішньої області LCFS та значення ψ на осі й межі."""
    from matplotlib.path import Path as MplPath
    # Fallback import of competition scorer functions
    try:
        from lcfs import extract_lcfs
        from derive import magnetic_axis
    except ImportError:
        import sys
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "fusion equilibrium challenge" / "starter" / "fusion_scoring"
        sys.path.insert(0, str(p))
        from lcfs import extract_lcfs
        from derive import magnetic_axis

    C = extract_lcfs(psi, R, Z, "DIII-D", mask_coarse, mask_f, n_points=256)
    if C is None:
        return None, 0.0, 0.0
    Ra, Za, iz, ir = magnetic_axis(psi, R, Z, "DIII-D", mask_coarse)
    RR, ZZ = np.meshgrid(R, Z)
    inside = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    inside[:2] = inside[-2:] = False
    inside[:, :2] = inside[:, -2:] = False
    psi_a = float(psi[iz, ir])
    psi_b = float(np.mean([_bilin(psi, R, Z, p[0], p[1]) for p in C[::8]]))
    return inside, psi_a, psi_b


def inside_lcfs(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                mask_coarse: np.ndarray, mask_f: np.ndarray, erode: int = 2) -> Optional[np.ndarray]:
    """Еродована маска всередині LCFS для безпечного обчислення шаблонів градієнта."""
    from matplotlib.path import Path as MplPath
    from scipy.ndimage import binary_erosion
    try:
        from lcfs import extract_lcfs
    except ImportError:
        import sys
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "fusion equilibrium challenge" / "starter" / "fusion_scoring"
        sys.path.insert(0, str(p))
        from lcfs import extract_lcfs

    C = extract_lcfs(psi, R, Z, "DIII-D", mask_coarse, mask_f, n_points=256)
    if C is None:
        return None
    RR, ZZ = np.meshgrid(R, Z)
    ins = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    return binary_erosion(ins, np.ones((3, 3)), iterations=erode)
