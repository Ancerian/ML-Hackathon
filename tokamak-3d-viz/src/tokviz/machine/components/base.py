"""Result containers shared by all component builders."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geom import Mesh, rot_z
from ..schema import MachineDB, Tracker


class RecordOnly(Exception):
    """Raised by a builder for a database entry that carries facts but no
    geometry (it is listed in the manifest as ``record_only``)."""


@dataclass
class Part:
    """One unique mesh plus the transforms of all its instances."""
    name: str
    mesh: Mesh
    material: str
    transforms: np.ndarray                       # (K, 4, 4)
    labels: list = field(default_factory=list)   # one per instance
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        T = np.asarray(self.transforms, float)
        if T.ndim == 2:
            T = T[None]
        self.transforms = T
        if not self.labels:
            self.labels = [f"{self.name}_{i:02d}" for i in range(len(T))]

    @property
    def n_instances(self) -> int:
        return len(self.transforms)


@dataclass
class ComponentResult:
    name: str
    type: str
    parts: list
    provenance: list
    params: dict
    spec_provenance: dict
    info: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    consumed: dict = field(default_factory=dict)     # other specs whose params were used

    def part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)


def finish(tr: Tracker, parts: list, info: dict | None = None) -> ComponentResult:
    prov = tr.provenance()
    return ComponentResult(
        name=tr.spec.name, type=tr.spec.type, parts=parts, provenance=prov,
        params=dict(tr.params_used), spec_provenance=dict(tr.spec.provenance),
        info=info or {},
        assumptions=[p["path"] for p in prov if p.get("confidence") == "assumed"],
        consumed=dict(getattr(tr, "consumed", {})))


# ---------------------------------------------------------------------------
# machine layout (azimuths) -- shared by several builders
# ---------------------------------------------------------------------------
class Layout:
    """Toroidal layout: TF coils, rigid sectors, octants, port sectors."""

    def __init__(self, tr: Tracker):
        self.n_tf = int(tr.v("tf_coils.n_coils"))
        self.pitch = 2 * np.pi / self.n_tf
        self.phi0 = np.deg2rad(tr.k("tf_phi0_deg"))
        self.n_pieces = int(tr.v("vessel.n_rigid_sectors"))
        self.n_oct = int(tr.v("vessel.n_octants"))
        # geometric sector pitches = TF pitches.  n_pieces == n_tf: 1975 layout
        # (octant welds in the middle of sector 4j); n_pieces == n_tf + n_oct:
        # the weld sectors are two separate half-width end pieces (1983 reading).
        if self.n_pieces == self.n_tf:
            self.split_end = False
        elif self.n_pieces == self.n_tf + self.n_oct:
            self.split_end = True
        else:
            raise ValueError(f"{self.n_pieces} rigid pieces for {self.n_tf} TF pitches "
                             f"and {self.n_oct} octants: layout not supported")
        self.n_sec = self.n_tf
        self.per_oct = self.n_sec // self.n_oct
        self.hport_k = int(tr.k("hport_sector_in_octant"))
        self.vlarge_k = int(tr.k("vport_large_sector_in_octant"))
        self.vsmall_k = int(tr.k("vport_small_sector_in_octant"))

    def tf_phi(self, i):
        return self.phi0 + np.asarray(i) * self.pitch

    def sector_phi(self, i):
        """Rigid sector i sits between TF coils i and i+1."""
        return self.phi0 + (np.asarray(i) + 0.5) * self.pitch

    def octant_of(self, i: int) -> int:
        return i // self.per_oct

    def sectors_with(self, k_in_octant: int) -> list:
        return [j * self.per_oct + k_in_octant for j in range(self.n_oct)]

    def sector_kind(self, i: int) -> str:
        k = i % self.per_oct
        if k == self.hport_k:
            return "hport"
        if k == self.vlarge_k:
            return "vport_large"
        if k == self.vsmall_k:
            return "vport_small"
        return "plain"

    def rot_sectors(self, idx) -> np.ndarray:
        return np.stack([rot_z(self.sector_phi(i)) for i in idx])

    def rot_tf(self, idx) -> np.ndarray:
        return np.stack([rot_z(self.tf_phi(i)) for i in idx])

    def hport_sector(self, octant_1based: int) -> int:
        """Sector index of the main horizontal port of octant k (1-based, as in
        the 1983 deltas; kernel labels use the 0-based index k-1)."""
        k = int(octant_1based)
        if not 1 <= k <= self.n_oct:
            raise ValueError(f"octant {k} outside 1..{self.n_oct}")
        return (k - 1) * self.per_oct + self.hport_k
