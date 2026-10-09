"""Write the bake that Blender consumes.

Blender 5.2 ships numpy (2.3.4) but no scipy, so every heavy computation
happens here and Blender only ever loads ``.npz``/``.json``.  One directory per
bake; ``manifest.json`` records the provenance and every number the annotation
layer prints on the image.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class Bake:
    """A bake directory under construction."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "objects": {},
        }

    # -- writing ------------------------------------------------------------
    def save(self, name: str, **arrays) -> str:
        """Save one ``.npz`` and register it in the manifest."""
        path = self.dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        self.manifest["objects"][name] = {
            "file": path.name,
            "arrays": {k: list(np.shape(v)) for k, v in arrays.items()},
        }
        return str(path)

    def save_meshes(self, name: str, meshes: dict) -> str:
        """Save a ``{key: (verts, faces)}`` collection into one archive."""
        arrays = {}
        keys = []
        for i, (k, (v, f)) in enumerate(sorted(meshes.items())):
            arrays[f"v_{i:03d}"] = np.asarray(v, dtype=np.float32)
            arrays[f"f_{i:03d}"] = np.asarray(f, dtype=np.int32)
            keys.append(float(k) if isinstance(k, (int, float)) else str(k))
        arrays["keys"] = np.asarray(keys)
        return self.save(name, **arrays)

    def save_mesh_list(self, name: str, meshes: list, labels=None) -> str:
        arrays = {}
        for i, (v, f) in enumerate(meshes):
            arrays[f"v_{i:03d}"] = np.asarray(v, dtype=np.float32)
            arrays[f"f_{i:03d}"] = np.asarray(f, dtype=np.int32)
        arrays["count"] = np.asarray([len(meshes)])
        if labels is not None:
            arrays["labels"] = np.asarray([str(x) for x in labels])
        return self.save(name, **arrays)

    def set(self, key: str, value) -> None:
        self.manifest[key] = _jsonable(value)

    def finish(self) -> Path:
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))
        return path


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# reader used from inside Blender
# ---------------------------------------------------------------------------
def load_meshes(npz_path):
    """Yield ``(key, verts, faces)`` from an archive written by
    :meth:`Bake.save_meshes` or :meth:`Bake.save_mesh_list`."""
    z = np.load(npz_path, allow_pickle=False)
    if "keys" in z:
        keys = z["keys"]
        n = len(keys)
    else:
        n = int(z["count"][0])
        keys = z["labels"] if "labels" in z else np.arange(n)
    for i in range(n):
        yield keys[i], z[f"v_{i:03d}"], z[f"f_{i:03d}"]
