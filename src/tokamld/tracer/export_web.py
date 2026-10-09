"""tokamld.tracer.export_web — JSON export for 3D/Poincare web viewer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def export_poincare_json(
    output_path: str | Path,
    machine: str,
    shot_id: int | str,
    q95: float,
    f_pol: float,
    r_axis: float,
    z_axis: float,
    surfaces: List[Dict[str, Any]],
    wall_polygon: Optional[Dict[str, Any]] = None,
    footprints: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Exports field-line punctures and magnetic surface data to a structured JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "tokamld-poincare-v1",
        "machine": str(machine),
        "shot": str(shot_id),
        "q95": float(q95),
        "f_pol": float(f_pol),
        "axis": [float(r_axis), float(z_axis)],
        "surfaces": surfaces,
        "wall": wall_polygon or {},
        "footprints": footprints or [],
        "metadata": metadata or {},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path
