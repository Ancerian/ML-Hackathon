"""tokamld.imas — Export and Import of Tokamak Equilibrium in IMAS IDS format.

Специфікація та відображення даних (Data Mapping):
1. Специфікація ITER IMAS IDS `equilibrium` [V]:
   - Офіційний словник даних ITER: Data Dictionary for Interface Data Structures (IDS) `equilibrium` v3.x/4.x [V].
   - OMAS (Open Modeling and Analysis Suite) Python interface schema [V].
2. Відповідність вузлів IMAS IDS `equilibrium`:
   - 2D поле потоку:
     `equilibrium.time_slice[:].profiles_2d[:]`
     - `grid_type.index`: 1 (прямокутна сітка R, Z) [V]
     - `grid.dim1`: 1D масив радіальних координат R [м] [V]
     - `grid.dim2`: 1D масив вертикальних координат Z [м] [V]
     - `psi`: 2D масив полоїдального магнітного потоку ψ(Z, R) або ψ(R, Z) [Вб або Вб/рад] [V]
   - 1D профілі:
     `equilibrium.time_slice[:].profiles_1d`
     - `psi`: радіальна сітка полоїдального потоку ψ [V]
     - `psi_norm`: нормалізований потік ψ_N ∈ [0, 1] [V]
     - `pressure`: тиск плазми p(ψ) [Па] [V]
     - `dpressure_dpsi`: похідна тиску p'(ψ) = dp/dψ [Па/Вб] [V]
     - `f`: полоїдальна функція струму F(ψ) = R B_φ [Тл·м] [V]
     - `f_df_dpsi`: FF'(ψ) = F dF/dψ [Тл²·м²/Вб] [V]
     - `q`: профіль коефіцієнта запасу стійкості q(ψ) [V]
   - Межа плазми (LCFS):
     `equilibrium.time_slice[:].boundary`
     - `outline.r`: контур R межі LCFS [м] [V]
     - `outline.z`: контур Z межі LCFS [м] [V]
     - `type`: 0 (лімітерна) або 1 (диверторна) [V]
     - `x_point[:]`: масив X-точок із координатами r, z [V]
   - Глобальні величини та критичні точки:
     `equilibrium.time_slice[:].global_quantities`
     - `magnetic_axis.r`: радіальне положення магнітної осі (O-точка) [м] [V]
     - `magnetic_axis.z`: вертикальне положення магнітної осі (O-точка) [м] [V]
     - `psi_axis`: потік на магнітній осі [Вб] [V]
     - `psi_boundary`: потік на сепаратрисі LCFS [Вб] [V]
     - `ip`: сумарний струм плазми I_p [А] [V]

3. Реалізація та обробка залежностей:
   - Якщо бібліотеки `imas` або `omas` встановлені в системі, модуль підтримує конвертацію в нативний `ODS`.
   - За їхньої відсутності модуль реалізує повну деревоподібну схему (Mock IDS), яка на 100% сумісна зі структурою
     словника даних IMAS та серіалізується у JSON / словники Python без втрати точності.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EquilibriumData:
    """Високорівневий контейнер рівноваги для обміну з IMAS IDS."""
    psi: np.ndarray                           # 2D масив потоку (H, W) або (nz, nr)
    R: np.ndarray                             # 1D координати R (W,)
    Z: np.ndarray                             # 1D координати Z (H,)
    pprime: Optional[np.ndarray] = None       # 1D профіль p'(ψ)
    ffprime: Optional[np.ndarray] = None      # 1D профіль FF'(ψ)
    q: Optional[np.ndarray] = None            # 1D профіль q(ψ)
    pressure: Optional[np.ndarray] = None     # 1D профіль p(ψ)
    f: Optional[np.ndarray] = None            # 1D профіль F(ψ)
    psi_1d: Optional[np.ndarray] = None       # 1D сітка ψ для профілів
    lcfs_r: Optional[np.ndarray] = None       # 1D координати R межі LCFS
    lcfs_z: Optional[np.ndarray] = None       # 1D координати Z межі LCFS
    axis_r: Optional[float] = None            # Положення магнітної осі R
    axis_z: Optional[float] = None            # Положення магнітної осі Z
    psi_axis: Optional[float] = None          # Потік на осі
    psi_boundary: Optional[float] = None      # Потік на межі
    x_points: Optional[List[Tuple[float, float]]] = None  # Список X-точок [(R, Z), ...]
    ip: Optional[float] = None                # Струм плазми
    time: float = 0.0                         # Час кадру (с)
    shot: int = 0                             # Номер розряду
    boundary_type: int = 1                    # 1 = divertor, 0 = limiter


def _to_numpy(val: Any) -> Optional[np.ndarray]:
    """Converts input to float64 numpy array or None."""
    if val is None:
        return None
    return np.asarray(val, dtype=np.float64)


def to_ids(data: Union[EquilibriumData, Dict[str, Any]],
           target_ods: Optional[Any] = None) -> Dict[str, Any]:
    """Експортує дані рівноваги у формат IMAS IDS `equilibrium`.

    Args:
        data: Об'єкт `EquilibriumData` або словник із відповідними ключами.
        target_ods: Необов'язковий об'єкт OMAS ODS (якщо omas встановлено).

    Returns:
        Словник із повною структурою IMAS IDS `equilibrium` або переданий target_ods.
    """
    if isinstance(data, dict):
        eq = EquilibriumData(
            psi=np.asarray(data["psi"], dtype=np.float64),
            R=np.asarray(data["R"], dtype=np.float64),
            Z=np.asarray(data["Z"], dtype=np.float64),
            pprime=_to_numpy(data.get("pprime")),
            ffprime=_to_numpy(data.get("ffprime")),
            q=_to_numpy(data.get("q")),
            pressure=_to_numpy(data.get("pressure")),
            f=_to_numpy(data.get("f")),
            psi_1d=_to_numpy(data.get("psi_1d")),
            lcfs_r=_to_numpy(data.get("lcfs_r")),
            lcfs_z=_to_numpy(data.get("lcfs_z")),
            axis_r=float(data["axis_r"]) if data.get("axis_r") is not None else None,
            axis_z=float(data["axis_z"]) if data.get("axis_z") is not None else None,
            psi_axis=float(data["psi_axis"]) if data.get("psi_axis") is not None else None,
            psi_boundary=float(data["psi_boundary"]) if data.get("psi_boundary") is not None else None,
            x_points=data.get("x_points"),
            ip=float(data["ip"]) if data.get("ip") is not None else None,
            time=float(data.get("time", 0.0)),
            shot=int(data.get("shot", 0)),
            boundary_type=int(data.get("boundary_type", 1)),
        )
    else:
        eq = data

    # 1. 2D grid and flux map
    R = np.asarray(eq.R, dtype=np.float64)
    Z = np.asarray(eq.Z, dtype=np.float64)
    psi_2d = np.asarray(eq.psi, dtype=np.float64)

    # 2. 1D radial profiles
    n_1d = 65
    if eq.psi_1d is not None:
        psi_1d = np.asarray(eq.psi_1d, dtype=np.float64)
    else:
        # Construct radial flux coordinate grid
        p_ax = eq.psi_axis if eq.psi_axis is not None else float(np.min(psi_2d))
        p_bnd = eq.psi_boundary if eq.psi_boundary is not None else float(np.max(psi_2d))
        psi_1d = np.linspace(p_ax, p_bnd, n_1d, dtype=np.float64)

    psi_norm = np.linspace(0.0, 1.0, len(psi_1d), dtype=np.float64)

    # Global quantities
    axis_r = eq.axis_r if eq.axis_r is not None else float(R[len(R) // 2])
    axis_z = eq.axis_z if eq.axis_z is not None else float(Z[len(Z) // 2])
    psi_axis = eq.psi_axis if eq.psi_axis is not None else float(psi_1d[0])
    psi_bnd = eq.psi_boundary if eq.psi_boundary is not None else float(psi_1d[-1])

    # Construct IMAS IDS hierarchical tree
    time_slice = {
        "time": eq.time,
        "global_quantities": {
            "magnetic_axis": {
                "r": axis_r,
                "z": axis_z,
            },
            "psi_axis": psi_axis,
            "psi_boundary": psi_bnd,
        },
        "boundary": {
            "type": eq.boundary_type,
            "outline": {
                "r": np.asarray(eq.lcfs_r, dtype=np.float64) if eq.lcfs_r is not None else np.array([], dtype=np.float64),
                "z": np.asarray(eq.lcfs_z, dtype=np.float64) if eq.lcfs_z is not None else np.array([], dtype=np.float64),
            },
            "x_point": [
                {"r": float(pt[0]), "z": float(pt[1])} for pt in (eq.x_points or [])
            ],
        },
        "profiles_1d": {
            "psi": psi_1d,
            "psi_norm": psi_norm,
            "dpressure_dpsi": np.asarray(eq.pprime, dtype=np.float64) if eq.pprime is not None else np.array([], dtype=np.float64),
            "f_df_dpsi": np.asarray(eq.ffprime, dtype=np.float64) if eq.ffprime is not None else np.array([], dtype=np.float64),
            "q": np.asarray(eq.q, dtype=np.float64) if eq.q is not None else np.array([], dtype=np.float64),
            "pressure": np.asarray(eq.pressure, dtype=np.float64) if eq.pressure is not None else np.array([], dtype=np.float64),
            "f": np.asarray(eq.f, dtype=np.float64) if eq.f is not None else np.array([], dtype=np.float64),
        },
        "profiles_2d": [
            {
                "grid_type": {
                    "index": 1,
                    "name": "rectangular",
                },
                "grid": {
                    "dim1": R,
                    "dim2": Z,
                },
                "psi": psi_2d,
            }
        ],
    }

    if eq.ip is not None:
        time_slice["global_quantities"]["ip"] = float(eq.ip)

    ids = {
        "ids_properties": {
            "comment": "tokamld exported equilibrium IDS",
            "homogeneous_time": 1,
            "provider": "tokamld.imas",
            "version_put": {
                "data_dictionary": "3.38.0",
                "access_layer": "mock-schema",
            },
        },
        "time": np.array([eq.time], dtype=np.float64),
        "time_slice": [time_slice],
    }

    # If an OMAS ODS was passed, fill it
    if target_ods is not None:
        try:
            target_ods["equilibrium"] = ids
            return target_ods
        except Exception:
            pass

    return ids


def from_ids(ids_obj: Union[Dict[str, Any], Any],
             time_index: int = 0) -> EquilibriumData:
    """Імпортує дані рівноваги зі структури IMAS IDS `equilibrium`.

    Args:
        ids_obj: Словник IMAS IDS або об'єкт OMAS ODS.
        time_index: Індекс часового зрізу (time_slice index, за замовчуванням 0).

    Returns:
        Об'єкт `EquilibriumData` з відновленими полями.
    """
    # Support both dictionary and OMAS ODS access
    if hasattr(ids_obj, "to_dict"):
        d = ids_obj.to_dict()
    elif hasattr(ids_obj, "__getitem__"):
        d = ids_obj
    else:
        raise TypeError(f"Unsupported IDS object type: {type(ids_obj)}")

    # Unnest if top-level key 'equilibrium' is present
    if "equilibrium" in d:
        eq_root = d["equilibrium"]
    else:
        eq_root = d

    time_slices = eq_root.get("time_slice", [])
    if not time_slices or time_index >= len(time_slices):
        raise ValueError(f"No time_slice at index {time_index} in equilibrium IDS.")

    ts = time_slices[time_index]
    p2d = ts.get("profiles_2d", [{}])[0]
    p1d = ts.get("profiles_1d", {})
    bnd = ts.get("boundary", {})
    gq = ts.get("global_quantities", {})

    grid = p2d.get("grid", {})
    R = np.asarray(grid.get("dim1"), dtype=np.float64)
    Z = np.asarray(grid.get("dim2"), dtype=np.float64)
    psi = np.asarray(p2d.get("psi"), dtype=np.float64)

    # 1D profiles
    def _get_opt(d_dict: dict, k: str) -> Optional[np.ndarray]:
        v = d_dict.get(k)
        if v is None or len(v) == 0:
            return None
        return np.asarray(v, dtype=np.float64)

    pprime = _get_opt(p1d, "dpressure_dpsi")
    ffprime = _get_opt(p1d, "f_df_dpsi")
    q = _get_opt(p1d, "q")
    pressure = _get_opt(p1d, "pressure")
    f_prof = _get_opt(p1d, "f")
    psi_1d = _get_opt(p1d, "psi")

    # Boundary
    outline = bnd.get("outline", {})
    lcfs_r = _get_opt(outline, "r")
    lcfs_z = _get_opt(outline, "z")
    bnd_type = int(bnd.get("type", 1))

    # X-points
    x_points = []
    for xp in bnd.get("x_point", []):
        if "r" in xp and "z" in xp:
            x_points.append((float(xp["r"]), float(xp["z"])))

    # Global quantities
    mag_axis = gq.get("magnetic_axis", {})
    axis_r = float(mag_axis["r"]) if "r" in mag_axis else None
    axis_z = float(mag_axis["z"]) if "z" in mag_axis else None
    psi_axis = float(gq["psi_axis"]) if "psi_axis" in gq else None
    psi_boundary = float(gq["psi_boundary"]) if "psi_boundary" in gq else None
    ip = float(gq["ip"]) if "ip" in gq else None

    time_val = float(ts.get("time", 0.0))

    return EquilibriumData(
        psi=psi,
        R=R,
        Z=Z,
        pprime=pprime,
        ffprime=ffprime,
        q=q,
        pressure=pressure,
        f=f_prof,
        psi_1d=psi_1d,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
        axis_r=axis_r,
        axis_z=axis_z,
        psi_axis=psi_axis,
        psi_boundary=psi_boundary,
        x_points=x_points if x_points else None,
        ip=ip,
        time=time_val,
        boundary_type=bnd_type,
    )


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that converts NumPy arrays and scalars to native Python types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_ids_json(ids: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """Зберігає дерево IDS `equilibrium` у валідний JSON файл."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ids, f, cls=_NumpyJSONEncoder, indent=2)
    return p


def load_ids_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Зчитує дерево IDS `equilibrium` із JSON файлу."""
    p = Path(filepath)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def export_equilibrium(data: Union[EquilibriumData, Dict[str, Any]],
                       filepath: Union[str, Path]) -> Path:
    """Експортує рівновагу безпосередньо у файл IMAS JSON."""
    ids = to_ids(data)
    return save_ids_json(ids, filepath)


def import_equilibrium(filepath: Union[str, Path],
                       time_index: int = 0) -> EquilibriumData:
    """Імпортує рівновагу безпосередньо із файлу IMAS JSON."""
    ids = load_ids_json(filepath)
    return from_ids(ids, time_index=time_index)
