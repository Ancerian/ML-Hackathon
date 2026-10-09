"""Component builders.  ``REGISTRY[type](tracker, spec) -> ComponentResult``."""
from __future__ import annotations

from .ex_vessel import (build_gas_inlet, build_nbi_adaptor, build_port_entry,
                        build_pump_chamber, build_restraint_ring, build_rotary_valve)
from .generic import build_limiter_module, build_rect_coil_set, build_revolved_profile
from .human import build_human
from .iron_core import build_iron
from .limiters import build_limiters
from .nbi import build_nbi
from .pf_coils import build_pf
from .plasma import build_plasma
from .ports import build_ports
from .structure import build_structure
from .tf_coil import build_tf
from .vessel import build_bellows, build_vessel

REGISTRY = {
    "plasma": build_plasma,
    "vessel": build_vessel,
    "bellows": build_bellows,
    "ports": build_ports,
    "tf_coils": build_tf,
    "pf_coils": build_pf,
    "iron_core": build_iron,
    "structure": build_structure,
    "limiters": build_limiters,
    "nbi": build_nbi,
    "human": build_human,
    # generic, for components added by delta variants
    "rect_coil_set": build_rect_coil_set,
    "revolved_profile": build_revolved_profile,
    "limiter_module": build_limiter_module,
    # JET 1983 (machines/jet1983/deltas.yaml components_add)
    "vessel_restraint_ring": build_restraint_ring,
    "pump_chamber": build_pump_chamber,
    "nbi_adaptor": build_nbi_adaptor,
    "rotary_valve": build_rotary_valve,
    "gas_inlet": build_gas_inlet,
    "port": build_port_entry,
}

#: types that carry facts only (no geometry); listed in the manifest as
#: ``record_only``.  ``vessel_sector`` / ``vessel_bellows`` / ``pf_coil_stack``
#: have no builder of their own: their params are consumed by the vessel,
#: bellows and pf_coils builders (manifest state ``realized``).
RECORD_TYPES = {"operating_parameter", "first_wall", "pf_coil_record",
                "iron_core_record", "structure_record"}

#: (name, type, default params) -- the component list used when a base
#: database does not declare ``components:`` itself.
DEFAULT_COMPONENTS = [
    ("plasma", "plasma", {"n_pol": 256, "n_tor": 256}),
    ("vessel", "vessel", {"n_pol": 280, "n_tor_sector": 16, "openings": True}),
    ("bellows", "bellows", {"n_pol": 280, "samples_per_convolution": 8}),
    ("ports", "ports", {}),
    ("tf_coils", "tf_coils", {"n_contour": 320}),
    ("pf_coils", "pf_coils", {"n_tor": 192}),
    ("iron_core", "iron_core", {"n_tor": 128}),
    ("structure", "structure", {"n_tor": 192, "n_tor_block": 8}),
    ("limiters", "limiters", {}),
    ("nbi", "nbi", {}),
    ("human", "human", {}),
]
