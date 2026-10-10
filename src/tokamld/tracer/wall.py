"""tokamld.tracer.wall — Wall boundary, strike detection, and divertor footprint geometry."""
from __future__ import annotations

import dataclasses
from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class WallBoundary:
    """Tokamak first wall / divertor polygon."""

    r_wall: np.ndarray      # shape (Nw,) metres
    z_wall: np.ndarray      # shape (Nw,) metres
    s_wall: np.ndarray      # cumulative arc length along wall (Nw,) metres

    @classmethod
    def from_polygon(cls, r: np.ndarray, z: np.ndarray) -> "WallBoundary":
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        # Ensure closed polygon
        if r[0] != r[-1] or z[0] != z[-1]:
            r = np.append(r, r[0])
            z = np.append(z, z[0])
        dr = np.diff(r)
        dz = np.diff(z)
        seg_lens = np.hypot(dr, dz)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        return cls(r_wall=r, z_wall=z, s_wall=s)

    @classmethod
    def default_d3d_limiter(cls) -> "WallBoundary":
        """Approximated outer/inner vessel limiter polygon for DIII-D."""
        r = np.array([1.01, 1.01, 1.15, 1.37, 1.68, 2.37, 2.37, 1.68, 1.37, 1.15, 1.01])
        z = np.array([-1.25, 1.25, 1.38, 1.38, 1.35, 0.00, -1.00, -1.35, -1.38, -1.38, -1.25])
        return cls.from_polygon(r, z)

    @property
    def total_length(self) -> float:
        return float(self.s_wall[-1])

    def strike_coordinate(self, r: float, z: float) -> float:
        """Finds closest point on the wall polygon and returns arc length coordinate s_wall."""
        r_f = float(r)
        z_f = float(z)
        dists_sq = (self.r_wall - r_f)**2 + (self.z_wall - z_f)**2
        idx = int(np.argmin(dists_sq))
        return float(self.s_wall[idx])
