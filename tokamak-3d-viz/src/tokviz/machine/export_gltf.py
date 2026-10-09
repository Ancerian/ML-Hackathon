"""Web export: JET machine models as self-contained glTF 2.0 binaries (GLB).

Target: an interactive three.js viewer on a host that blocks runtime decoder
fetches, so NO Draco / meshopt / KTX2.  The size budget is met by geometry
level-of-detail instead:

* the machine is rebuilt through the geometry kernel from the same sourced
  database as the bake (``machines/<name>``) with web resolution parameters
  (``WEB_LOD``) and two resolution caps on hard-coded kernel samplings
  (``_resolution_caps``); the component / part / instance structure is checked
  against the bake manifest (``data/bake_<name>/manifest.json``);
* bellows convolutions are replaced by one smooth band per ply;
* the plasma is the physics bake (Solov'ev LCFS + 3 inner flux surfaces,
  ``data/bake_jet1975/physics``), revolved coarsely; field lines are
  LINE_STRIP primitives.

glTF conventions: metres, +Y up (kernel Z -> glTF Y, kernel Y -> glTF -Z),
origin on the machine axis, midplane at y = 0.  Repeated parts share one mesh
referenced by several nodes (plain node -> mesh reuse, no instancing
extension).  Materials are core metallic-roughness (+ KHR_materials_unlit on
field lines, optional: loaders without it fall back to a lit material).

Node tree (``node.extras`` -> three.js ``object.userData``)::

    JET_1975 (root)              extras.kind = "machine"
      TF | Vessel | ... (group)  extras.kind = "group"
        <component>__<part>      extras.kind = "part"  (tooltip data)
          <component>__<part>__<label>   extras.kind = "instance", mesh

Pure numpy + json; nothing here imports bpy.
"""
from __future__ import annotations

import contextlib
import json
import math
import re
import struct
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .build import _status_of, _variant_of, build_machine
from .schema import load_machine

# ---------------------------------------------------------------------------
# budget / LOD
# ---------------------------------------------------------------------------
MAX_FILE_BYTES = 12 * 1024 * 1024
TRI_BUDGET = 250_000                   # rendered (instanced) triangles per file

#: kernel builder parameters for the web level of detail (per component TYPE)
WEB_LOD = {
    "plasma": {"n_pol": 64, "n_tor": 16},          # provenance only (mesh from physics)
    "vessel": {"n_pol": 96, "n_tor_sector": 4},    # D-profile 96 rows (>= 96)
    "bellows": {"n_pol": 96, "samples_per_convolution": 1},
    "tf_coils": {"n_contour": 96},
    "pf_coils": {"n_tor": 48},
    "iron_core": {"n_tor": 48},
    "structure": {"n_tor": 48, "n_tor_block": 2},
    "vessel_restraint_ring": {"n_tor": 4},
    "rect_coil_set": {"n_tor": 48},
    "revolved_profile": {"n_tor": 48},
}

#: hard-coded kernel samplings capped for the web (module -> {n_in: n_out})
_RESAMPLE_CAPS = {200: 72, 720: 240}      # structure: ring collar 200, shell plates 720
_RAIL_NTOR = 3                            # limiters: rail plates revolved with n_tor=9

#: plasma / field lines resolution
LCFS_RES = (80, 48)            # (poloidal, toroidal) -> 7 680 triangles
INNER_RES = (48, 32)           # -> 3 072 triangles each
INNER_PSI = (0.3, 0.6, 0.9)
FIELDLINE_POINTS = 420         # per line (7 lines -> ~2 900 points)

CREASE_DEG = 35.0

# kernel (x, y, z; z up) -> glTF (x, z, -y; y up)
C_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


@contextlib.contextmanager
def _resolution_caps():
    """Temporarily cap two hard-coded samplings in kernel builders (web LOD)."""
    from .components import limiters as LIM
    from .components import structure as STR
    orig_rc, orig_sr = STR.resample_closed, LIM.solid_of_revolution

    def rc(p, n):
        return orig_rc(p, _RESAMPLE_CAPS.get(int(n), n))

    def sr(poly, n_tor=128, **kw):
        return orig_sr(poly, n_tor=min(int(n_tor), _RAIL_NTOR) if n_tor == 9 else n_tor, **kw)

    STR.resample_closed, LIM.solid_of_revolution = rc, sr
    try:
        yield
    finally:
        STR.resample_closed, LIM.solid_of_revolution = orig_rc, orig_sr


def build_web_machine(machine_dir, log=None):
    """Load the database and build it at web resolution."""
    db = load_machine(machine_dir)
    for spec in db.components:
        if spec.type in WEB_LOD:
            spec.params = {**spec.params, **WEB_LOD[spec.type]}
    with _resolution_caps():
        mb = build_machine(db, log=log)
    return db, mb


def check_against_bake(mb, bake_dir) -> list:
    """Compare component / part / instance-label structure with the bake
    manifest.  Returns a list of mismatch strings (empty = identical)."""
    man = json.loads((Path(bake_dir) / "manifest.json").read_text())
    comps = man["components"]
    out = []
    if set(comps) != set(mb.components):
        out.append(f"components differ: bake-only {sorted(set(comps) - set(mb.components))}, "
                   f"web-only {sorted(set(mb.components) - set(comps))}")
    for name, res in mb.components.items():
        if name not in comps:
            continue
        bp = comps[name]["parts"]
        for p in res.parts:
            if p.name not in bp:
                out.append(f"{name}.{p.name}: not in bake")
            elif list(bp[p.name]["labels"]) != list(p.labels):
                out.append(f"{name}.{p.name}: instance labels differ")
        if set(bp) != {p.name for p in res.parts}:
            out.append(f"{name}: part sets differ")
    return out


# ---------------------------------------------------------------------------
# subsystems, labels, materials
# ---------------------------------------------------------------------------
GROUPS = OrderedDict([
    ("TF", "Котушки тороїдального поля (TF)"),
    ("Vessel", "Вакуумна камера"),
    ("Bellows", "Сильфони камери"),
    ("Ports", "Порти камери"),
    ("PF", "Котушки полоїдального поля (PF)"),
    ("IronCore", "Залізний магнітопровід"),
    ("Structure", "Механічна структура"),
    ("Limiters", "Лімітери"),
    ("NBI", "Інжекція нейтральних пучків (NBI)"),
    ("Plasma", "Плазма"),
    ("FieldLines", "Силові лінії поля"),
    ("Human", "Людина (масштаб)"),
    ("Extras1983", "Нове у 1983 (зовнішні вузли)"),
])

#: component TYPE -> group
TYPE_GROUP = {
    "tf_coils": "TF", "vessel": "Vessel", "bellows": "Bellows", "ports": "Ports",
    "port": "Ports", "pf_coils": "PF", "rect_coil_set": "PF", "iron_core": "IronCore",
    "structure": "Structure", "limiters": "Limiters", "limiter_module": "Limiters",
    "nbi": "NBI", "nbi_adaptor": "NBI", "plasma": "Plasma", "human": "Human",
    "vessel_restraint_ring": "Extras1983", "pump_chamber": "Extras1983",
    "rotary_valve": "Extras1983", "gas_inlet": "Extras1983",
    "revolved_profile": "Extras1983",
}

_PART_UK = {
    ("tf_coils", "casing"): "Котушка TF — сталевий корпус",
    ("tf_coils", "support_pads"): "Опорні подушки котушки TF",
    ("vessel", "rigid_sector_plain"): "Жорсткий сектор камери (без портів)",
    ("vessel", "rigid_sector_hport"): "Жорсткий сектор камери з горизонтальним портом",
    ("vessel", "rigid_sector_vport_large"): "Жорсткий сектор камери з великим вертикальним портом",
    ("vessel", "rigid_sector_vport_small"): "Жорсткий сектор камери з малим вертикальним портом",
    ("vessel", "rigid_sector_end_a"): "Торцевий напівсектор октанта (бік −φ)",
    ("vessel", "rigid_sector_end_b"): "Торцевий напівсектор октанта (бік +φ)",
    ("vessel", "rigid_sector_end_half"): "Торцевий напівсектор октанта",
    ("bellows", "bellows_outer_ply"): "Сильфон — зовнішній шар (гофри спрощено до смуги)",
    ("bellows", "bellows_inner_ply"): "Сильфон — внутрішній шар (гофри спрощено до смуги)",
    ("bellows", "bellows_ply"): "Сильфон (гофри спрощено до смуги)",
    ("pf_coils", "coil1"): "Котушка PF 1 (центральний соленоїд, блок)",
    ("pf_coils", "coil2"): "Котушка PF 2",
    ("pf_coils", "coil3"): "Котушка PF 3",
    ("pf_coils", "coil4"): "Котушка PF 4",
    ("pf_coils", "coil1_support_cylinder"): "Опорний циліндр котушки PF 1",
    ("iron_core", "upper_radial_arm"): "Верхнє радіальне плече магнітопроводу",
    ("iron_core", "lower_radial_arm"): "Нижнє радіальне плече магнітопроводу",
    ("iron_core", "outer_limb"): "Зовнішня вертикальна гілка магнітопроводу",
    ("iron_core", "foot_plate"): "Опорна плита (ніжка)",
    ("iron_core", "central_pillar"): "Центральний стовп магнітопроводу",
    ("iron_core", "pillar_top_hub"): "Верхня маточина центрального стовпа",
    ("iron_core", "centre_piece_upper"): "Верхня центральна частина магнітопроводу",
    ("iron_core", "centre_piece_lower"): "Нижня центральна частина магнітопроводу",
    ("structure", "upper_ring_collar"): "Верхнє кільце — внутрішній комір",
    ("structure", "lower_ring_collar"): "Нижнє кільце — внутрішній комір",
    ("structure", "upper_ring_block_plain"): "Верхнє кільце — клиновий блок",
    ("structure", "upper_ring_block_vport_large"): "Верхнє кільце — блок з вирізом під великий вертикальний порт",
    ("structure", "upper_ring_block_vport_small"): "Верхнє кільце — блок з вирізом під малий вертикальний порт",
    ("structure", "lower_ring_block_plain"): "Нижнє кільце — клиновий блок",
    ("structure", "lower_ring_block_vport_large"): "Нижнє кільце — блок з вирізом під великий вертикальний порт",
    ("structure", "lower_ring_block_vport_small"): "Нижнє кільце — блок з вирізом під малий вертикальний порт",
    ("structure", "shell_plates"): "Плити зовнішньої оболонки між котушками",
    ("structure", "inner_cylinder"): "Внутрішній циліндр",
    ("limiters", "outer_rail_plate"): "Зовнішній рейковий лімітер (пластина Mo)",
    ("limiters", "upper_rail_plate"): "Верхній рейковий лімітер (пластина Mo)",
    ("limiters", "lower_rail_plate"): "Нижній рейковий лімітер (пластина Mo)",
    ("nbi", "beam_duct"): "Тракт пучка NBI",
    ("nbi", "injector_tank"): "Бак інжектора NBI",
    ("human", "figure_1p80m"): "Людина 1,80 м (для масштабу)",
    ("vessel_restraint_rings", "upper_ring_skin"): "Кільце жорсткості камери, верхнє — обшивка",
    ("vessel_restraint_rings", "lower_ring_skin"): "Кільце жорсткості камери, нижнє — обшивка",
    ("vessel_restraint_rings", "upper_ring_bridge"): "Кільце жорсткості, верхнє — ізольований місток через сильфон",
    ("vessel_restraint_rings", "lower_ring_bridge"): "Кільце жорсткості, нижнє — ізольований місток через сильфон",
}

_PORT_UK = {"hport": "Горизонтальний порт", "vport_large_top": "Великий вертикальний порт (верх)",
            "vport_large_bottom": "Великий вертикальний порт (низ)",
            "vport_small_top": "Малий вертикальний порт (верх)",
            "vport_small_bottom": "Малий вертикальний порт (низ)"}
_TYPE_UK = {"pump_chamber": "Насосна камера", "nbi_adaptor": "Адаптер NBI (середній порт)",
            "rotary_valve": "Поворотний клапан", "gas_inlet": "Модуль напуску газу",
            "port": "Лімітерний порт"}
_SUB_UK = {"chamber": "корпус", "spool": "перехідний патрубок", "door": "двері Ø1,2 м",
           "turbo_pumps": "турбомолекулярні насоси", "adaptor": "корпус",
           "valve_body": "корпус", "gas_module": "модуль", "duct": "патрубок",
           "flange": "фланець"}


def part_label_uk(comp: str, ctype: str, part: str) -> str:
    if (comp, part) in _PART_UK:
        return _PART_UK[(comp, part)]
    if comp == "ports":
        base, _, sub = part.rpartition("_")
        return f"{_PORT_UK.get(base, base)} — {_SUB_UK.get(sub, sub)}"
    if ctype == "limiter_module":
        k = comp.split("_", 1)[1]
        return (f"Графітовий лімітер {k}" if k.startswith("C")
                else f"Лімітер {k}: CuCr з Ni-плакуванням 1,5 мм")
    if ctype in _TYPE_UK:
        m = re.search(r"oct(\d+)", comp)
        where = f", октант {m.group(1)}" if m else ""
        return f"{_TYPE_UK[ctype]}{where} — {_SUB_UK.get(part, part)}"
    return f"{comp} / {part}"


#: material KIND -> (label_uk, baseColor linear RGB, metallic, roughness)
#: (G3 palette: values follow src/blender/tokblend/jet_materials.py)
MATERIALS = {
    "iron": ("залізо магнітопроводу (фарбоване)", (0.20, 0.235, 0.225), 0.0, 0.56),
    "magnetic_steel": ("магнітна сталь (фарбована)", (0.16, 0.18, 0.19), 0.0, 0.50),
    "cast_iron": ("аустенітний високоміцний чавун", (0.19, 0.20, 0.20), 0.2, 0.62),
    "tf_case": ("сталевий корпус котушки TF", (0.46, 0.47, 0.49), 0.85, 0.40),
    "copper": ("мідь з епоксидною ізоляцією", (0.80, 0.45, 0.27), 1.0, 0.33),
    "inconel": ("Inconel 600 / Nicrofer 7216 (сатин)", (0.60, 0.58, 0.54), 1.0, 0.30),
    "inconel625": ("Inconel 625", (0.64, 0.62, 0.58), 1.0, 0.26),
    "moly": ("молібден", (0.56, 0.57, 0.58), 1.0, 0.34),
    "al_cast": ("литий алюмінієвий сплав", (0.80, 0.80, 0.78), 1.0, 0.55),
    "al_plate": ("алюмінієвий лист", (0.86, 0.86, 0.85), 1.0, 0.38),
    "stainless": ("нержавна сталь", (0.55, 0.55, 0.56), 1.0, 0.36),
    "steel": ("конструкційна сталь", (0.42, 0.43, 0.44), 1.0, 0.45),
    "graphite": ("графіт", (0.035, 0.035, 0.038), 0.0, 0.80),
    "nickel": ("Ni-плакування на CuCr", (0.66, 0.61, 0.53), 1.0, 0.24),
    "human": ("масштабна фігура", (0.55, 0.52, 0.48), 0.0, 0.70),
    "neutral": ("—", (0.5, 0.5, 0.5), 0.0, 0.5),
}

_TAG_RULES = (
    (r"copper_epoxy_steel_case|tf_case", "tf_case"),
    (r"ni.?clad|nickel", "nickel"),
    (r"copper|cucr", "copper"),
    (r"iron_laminated|iron_solid|^iron", "iron"),
    (r"nodular|cast_iron", "cast_iron"),
    (r"magnetic_steel", "magnetic_steel"),
    (r"inconel625|inconel_625", "inconel625"),
    (r"inconel|nicrofer", "inconel"),
    (r"molybd|^mo$", "moly"),
    (r"al_alloy_cast|al_cast", "al_cast"),
    (r"al_alloy|alumin", "al_plate"),
    (r"graphite|carbon", "graphite"),
    (r"stainless|304|316", "stainless"),
    (r"steel", "steel"),
    (r"human", "human"),
)


def material_kind(tag: str, variant_machine=None) -> str:
    t = str(tag).lower()
    for pat, kind in _TAG_RULES:
        if re.search(pat, t):
            return kind
    return "neutral"


def linear_to_hex(rgb) -> str:
    def s(c):
        c = max(0.0, min(1.0, float(c)))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return int(round(c * 255))
    return "#{:02x}{:02x}{:02x}".format(*(s(c) for c in rgb))


#: group colour for jet_meta (linear RGB of the dominant material)
GROUP_KIND = {"TF": "tf_case", "Vessel": "inconel", "Bellows": "inconel625",
              "Ports": "stainless", "PF": "copper", "IronCore": "iron",
              "Structure": "al_cast", "Limiters": "moly", "NBI": "stainless",
              "Human": "human", "Extras1983": "stainless"}
PLASMA_RGB = (1.0, 0.50, 0.30)
FIELDLINE_RGB = (1.0, 0.85, 0.35)

# ---------------------------------------------------------------------------
# provenance -> sources / confidence / key dimensions
# ---------------------------------------------------------------------------
DOCS = {
    "JET1975": "EUR 5516e",
    "JET_1975_Design_Proposal_EUR5516e.pdf": "EUR 5516e",
    "JET_JU_1983_Progress_Report.pdf": "JET JU Progress Report 1983 (EUR 9472)",
    "JET_JU_Brochure.pdf": "JET JU Brochure 1982 (EUR 8306)",
    "JET_JU_1979_Annual_Report.pdf": "JET JU Annual Report 1979",
    "JET_Wesson_1999_Science_of_JET.pdf": "Wesson, The Science of JET (1999)",
    "KERNEL": "припущення ядра моделі",
}
CONF_RANK = {"drawing": 5, "table": 5, "text": 4, "digitized": 3, "assumed": 1}


def doc_label(src) -> str:
    return DOCS.get(str(src), str(src))


def cite(src, page) -> str:
    return f"{doc_label(src)}, с. {page}" if page not in (None, "") else doc_label(src)


def _sources(prov_entries) -> list:
    seen, out = set(), []
    for e in prov_entries:
        src, page = e.get("source"), e.get("page")
        if not src or src == "KERNEL" or page in (None, ""):
            continue
        key = (doc_label(src), page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"doc": key[0], "page": page, "cite": cite(src, page)})
    return sorted(out, key=lambda d: (d["doc"], d["page"] if isinstance(d["page"], int) else 0))


def _confidence(prov_entries, spec_prov=None):
    counts = Counter(e.get("confidence") for e in prov_entries if e.get("confidence"))
    sourced = Counter({k: v for k, v in counts.items() if k != "assumed"})
    if spec_prov and spec_prov.get("confidence") and spec_prov.get("source") != "KERNEL":
        level = spec_prov["confidence"]
    elif sourced:
        level = sourced.most_common(1)[0][0]
    else:
        level = "assumed"
    return level, dict(counts)


#: component TYPE -> key database paths (value, unit, source, page, confidence)
KEY_DIMS = {
    "tf_coils": ["tf_coils.n_coils", "tf_coils.overall_height", "tf_coils.overall_width",
                 "tf_coils.bore_R_inner", "tf_coils.bore_R_outer",
                 "tf_coils.outer_leg_toroidal_width", "tf_coils.turns_per_coil",
                 "tf_coils.weight", "tf_coils.total_TF_current_basic"],
    "vessel": ["vessel.n_rigid_sectors", "vessel.n_octants", "vessel.rs_R_inner_wall_inboard",
               "vessel.rs_R_inner_wall_outboard", "vessel.rs_Z_inner_wall_top",
               "vessel.wall_thickness_Ri_Ra", "vessel.wall_thickness_max",
               "vessel.total_weight", "vessel.volume", "vessel.material_rigid_sectors"],
    "bellows": ["vessel.bellows_per_section", "vessel.bellows_wall_thickness",
                "vessel.material_bellows"],
    "ports": ["ports.horizontal_port_width", "ports.horizontal_port_height",
              "ports.horizontal_port_count", "ports.vertical_port_large",
              "ports.vertical_port_small"],
    "pf_coils": ["pf_coils.n_pf_coils_total"],
    "iron_core": ["iron_core.n_limbs", "iron_core.mass_total", "iron_core.limb_R_outer",
                  "iron_core.upper_arm_Z_top", "iron_core.lower_arm_Z_bottom",
                  "iron_core.B_iron_max", "overall.overall_diameter", "overall.overall_height"],
    "structure": ["structure.ring_outer_diameter", "structure.ring_height",
                  "structure.inner_cylinder_R", "structure.inner_cylinder_thickness",
                  "structure.shell_description"],
    "limiters": ["limiters.outer_limiter_plates", "limiters.plate_size",
                 "limiters.plate_material", "limiters.outer_limiter_R",
                 "limiters.bellows_shield_Re"],
    "nbi": ["nbi.injectors_per_port", "nbi.beam_energy_H", "nbi.neutral_power_per_injector",
            "nbi.total_power_goal", "nbi.injection_angle_to_axis"],
    "plasma": ["plasma.R0", "plasma.a", "plasma.b", "plasma.elongation_b_over_a",
               "plasma.Ip_D_basic", "plasma.Btor_R0_basic", "plasma.TF_ripple"],
}
_PART_DIMS = {
    ("iron_core", "upper_radial_arm"): ["iron_core.mass_upper_radial_arm", "iron_core.arm_toroidal_width"],
    ("iron_core", "lower_radial_arm"): ["iron_core.mass_lower_radial_arm", "iron_core.arm_toroidal_width"],
    ("iron_core", "outer_limb"): ["iron_core.mass_vertical_limb", "iron_core.limb_R_inner",
                                  "iron_core.limb_toroidal_width"],
    ("iron_core", "foot_plate"): ["iron_core.mass_feet_structural"],
    ("iron_core", "central_pillar"): ["iron_core.mass_central_pillar", "iron_core.central_pillar_R"],
    ("iron_core", "centre_piece_upper"): ["iron_core.mass_centre_piece", "iron_core.centre_piece_R_outer"],
    ("iron_core", "centre_piece_lower"): ["iron_core.mass_centre_piece", "iron_core.centre_piece_R_outer"],
}
for _k in (1, 2, 3, 4):
    _PART_DIMS[("pf_coils", f"coil{_k}")] = [f"pf_coils.coil{_k}_{s}" for s in
                                             ("count", "mean_diameter", "Z", "turns", "cu_mass")]
_PART_DIMS[("pf_coils", "coil1_support_cylinder")] = ["pf_coils.coil1_support_cylinder_bore",
                                                      "pf_coils.coil1_support_cylinder_radial"]


def _dim_entry(d) -> dict:
    v = d.value
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str) and re.fullmatch(r"[-+]?\d+(\.\d*)?[eE][-+]?\d+", v.strip()):
        v = float(v)                     # PyYAML reads '2.8e6' as a string
    e = {"value": v, "unit": d.unit, "confidence": d.confidence,
         "source": doc_label(d.source), "page": d.page, "cite": cite(d.source, d.page)}
    if d.ref:
        e["ref"] = d.ref
    return e


def _key_dims(db, paths) -> dict:
    out = {}
    for p in paths:
        try:
            d = db.dim(p)
        except KeyError:
            continue
        out[d.id] = _dim_entry(d)
    return out


def _param_unit(k: str, v) -> str:
    if k.endswith("_deg"):
        return "deg"
    if k.endswith("_C"):
        return "°C"
    if k.endswith("_W_m2"):
        return "W/m^2"
    if k == "area":
        return "m^2"
    if isinstance(v, int) or k in ("octant", "count_total", "per_octant", "per_octant_count"):
        return "count"
    return "m"


def _param_dims(spec) -> dict:
    """Numeric params of a variant-added component (1983) as key dims."""
    if spec is None:
        return {}
    assumed = set(spec.params.get("assumed_params", []) or [])
    prov = spec.provenance or {}
    out = {}
    for k, v in spec.params.items():
        if k in WEB_LOD.get(spec.type, {}) or k == "assumed_params":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                pass
            else:
                continue
        out[k] = {"value": v, "unit": _param_unit(k, v),
                  "confidence": "assumed" if k in assumed else prov.get("confidence"),
                  "source": doc_label(prov.get("source")), "page": prov.get("page"),
                  "cite": cite(prov.get("source"), prov.get("page"))}
    return out


# ---------------------------------------------------------------------------
# mesh utilities
# ---------------------------------------------------------------------------
def to_yup(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) @ C_ZUP_TO_YUP.T


def matrix_to_yup(X: np.ndarray) -> np.ndarray:
    C4 = np.eye(4)
    C4[:3, :3] = C_ZUP_TO_YUP
    return C4 @ np.asarray(X, float) @ C4.T


def crease_normals(V: np.ndarray, F: np.ndarray, crease_deg: float = CREASE_DEG):
    """Split vertices along creases sharper than ``crease_deg`` and return
    (V', N', F') with area-weighted smooth normals inside each smooth region."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area2 = np.linalg.norm(fn, axis=1)
    keep = area2 > 1e-14 * max(1.0, float(np.max(area2)) if len(area2) else 1.0)
    F, fn, area2 = F[keep], fn[keep], area2[keep]
    fu = fn / area2[:, None]
    nF = len(F)
    vid = F.ravel()
    fid = np.repeat(np.arange(nF), 3)
    order = np.argsort(vid, kind="stable")
    sv, sf = vid[order], fid[order]
    starts = np.searchsorted(sv, np.arange(len(V)))
    deg = np.bincount(vid, minlength=len(V))
    D = int(deg.max()) if len(deg) else 0
    rank = np.arange(len(sv)) - starts[sv]
    A = np.full((len(V), max(D, 1)), -1, np.int64)
    A[sv, rank] = sf
    nb = A[vid]                                          # (3nF, D) neighbour faces
    valid = nb >= 0
    nbc = np.where(valid, nb, 0)
    dots = np.einsum("kj,kdj->kd", fu[fid], fu[nbc])
    w = valid & (dots > math.cos(math.radians(crease_deg)))
    cn = np.einsum("kd,kdj->kj", w.astype(float), fn[nbc])
    cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-30)
    q = np.round(cn * 1000).astype(np.int64)
    key = np.c_[vid, q]
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Nsum = np.zeros((len(uniq), 3))
    np.add.at(Nsum, inv, cn)
    Nout = Nsum / np.maximum(np.linalg.norm(Nsum, axis=1, keepdims=True), 1e-30)
    Vout = V[uniq[:, 0]]
    return Vout, Nout, inv.reshape(-1, 3)


def mesh_triangles(m) -> np.ndarray:
    return m.tris()


def bellows_band(m, rows_keep=None):
    """Kernel bellows ply (structured (nt, npol) grid, built with one sample
    per convolution so every row lies on the skin) -> one smooth band between
    the two flange rows."""
    V = m.v
    # the grid is nt rows of npol vertices; first quad = [0, 1, npol + 1, npol]
    q0 = m.q[0]
    npol = int(q0[3] - q0[0])
    if len(V) % npol or q0[1] != q0[0] + 1:
        raise ValueError("bellows_band: unexpected bellows grid layout")
    nt = len(V) // npol
    G = V.reshape(nt, npol, 3)
    rows = rows_keep or [0, nt - 1]
    G2 = G[rows]
    ia = np.arange(len(rows) - 1)[:, None]
    ib = np.arange(npol)[None, :]
    b1 = (ib + 1) % npol
    Q = np.stack([(ia * npol + ib).ravel(), (ia * npol + b1).ravel(),
                  ((ia + 1) * npol + b1).ravel(), ((ia + 1) * npol + ib).ravel()], -1)
    from .geom import Mesh
    return Mesh(G2.reshape(-1, 3), Q)


def revolve_profile(prof_rz: np.ndarray, n_pol: int, n_tor: int):
    """Closed (R, Z) profile -> torus-like surface (Z up), resampled by arc length."""
    p = np.asarray(prof_rz, float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], n_pol, endpoint=False)
    pc = np.vstack([p, p[:1]])
    R = np.interp(t, s, pc[:, 0])
    Z = np.interp(t, s, pc[:, 1])
    ph = np.linspace(0, 2 * np.pi, n_tor, endpoint=False)
    V = np.stack([R[None, :] * np.cos(ph)[:, None], R[None, :] * np.sin(ph)[:, None],
                  np.broadcast_to(Z, (n_tor, n_pol))], -1).reshape(-1, 3)
    it = np.arange(n_tor)[:, None]
    ip = np.arange(n_pol)[None, :]
    a = it * n_pol + ip
    b = it * n_pol + (ip + 1) % n_pol
    c = ((it + 1) % n_tor) * n_pol + (ip + 1) % n_pol
    d = ((it + 1) % n_tor) * n_pol + ip
    # orientation: outward normals for a CCW (R, Z) profile
    ccw = np.sum((np.roll(R, -1) - R) * (np.roll(Z, -1) + Z)) < 0
    F = np.r_[np.stack([a, d, c], -1).reshape(-1, 3), np.stack([a, c, b], -1).reshape(-1, 3)]
    if not ccw:
        F = F[:, ::-1]
    return V, F


# ---------------------------------------------------------------------------
# GLB writer
# ---------------------------------------------------------------------------
class GLB:
    FLOAT, USHORT, UINT = 5126, 5123, 5125

    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "tokviz.machine.export_gltf"},
                  "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
                  "materials": [], "accessors": [], "bufferViews": [], "buffers": []}
        self._mat = {}
        self.extensions = set()

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin.extend(data)
        self.j["bufferViews"].append(bv)
        return len(self.j["bufferViews"]) - 1

    def _acc(self, arr, ctype, typ, target, minmax=False) -> int:
        bv = self._view(arr.tobytes(), target)
        a = {"bufferView": bv, "componentType": ctype, "count": int(len(arr)), "type": typ}
        if minmax:
            a["min"] = [float(x) for x in arr.min(0)]
            a["max"] = [float(x) for x in arr.max(0)]
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    def material(self, key, name, rgb, metallic, rough, alpha=1.0, emissive=None,
                 double=False, unlit=False, extras=None) -> int:
        if key in self._mat:
            return self._mat[key]
        m = {"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": [float(c) for c in rgb] + [float(alpha)],
            "metallicFactor": float(metallic), "roughnessFactor": float(rough)}}
        if alpha < 1.0:
            m["alphaMode"] = "BLEND"
        if emissive is not None:
            m["emissiveFactor"] = [float(c) for c in emissive]
        if double:
            m["doubleSided"] = True
        if unlit:
            m["extensions"] = {"KHR_materials_unlit": {}}
            self.extensions.add("KHR_materials_unlit")
        if extras:
            m["extras"] = extras
        self.j["materials"].append(m)
        self._mat[key] = len(self.j["materials"]) - 1
        return self._mat[key]

    def mesh(self, name, V, N, F, mat, extras=None) -> int:
        V = np.ascontiguousarray(V, np.float32)
        N = np.ascontiguousarray(N, np.float32)
        idx_t = np.uint16 if len(V) < 65536 else np.uint32
        I = np.ascontiguousarray(np.asarray(F).ravel(), idx_t)
        prim = {"attributes": {"POSITION": self._acc(V, self.FLOAT, "VEC3", 34962, True),
                               "NORMAL": self._acc(N, self.FLOAT, "VEC3", 34962)},
                "indices": self._acc(I, self.USHORT if idx_t == np.uint16 else self.UINT,
                                     "SCALAR", 34963, False),
                "material": mat, "mode": 4}
        self.j["accessors"][prim["indices"]]["min"] = [int(I.min())]
        self.j["accessors"][prim["indices"]]["max"] = [int(I.max())]
        m = {"name": name, "primitives": [prim]}
        if extras:
            m["extras"] = extras
        self.j["meshes"].append(m)
        return len(self.j["meshes"]) - 1

    def line_mesh(self, name, P, mat, extras=None) -> int:
        P = np.ascontiguousarray(P, np.float32)
        prim = {"attributes": {"POSITION": self._acc(P, self.FLOAT, "VEC3", 34962, True)},
                "material": mat, "mode": 3}
        m = {"name": name, "primitives": [prim]}
        if extras:
            m["extras"] = extras
        self.j["meshes"].append(m)
        return len(self.j["meshes"]) - 1

    def node(self, name, parent=None, mesh=None, matrix=None, extras=None) -> int:
        n = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        if matrix is not None and not np.allclose(matrix, np.eye(4), atol=1e-12):
            M = np.where(np.abs(matrix) < 1e-12, 0.0, matrix)
            n["matrix"] = [float(x) for x in M.T.ravel()]      # column-major
        if extras is not None:
            n["extras"] = extras
        self.j["nodes"].append(n)
        i = len(self.j["nodes"]) - 1
        if parent is None:
            self.j["scenes"][0]["nodes"].append(i)
        else:
            self.j["nodes"][parent].setdefault("children", []).append(i)
        return i

    def write(self, path) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        if self.extensions:
            self.j["extensionsUsed"] = sorted(self.extensions)
        js = json.dumps(self.j, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))
        return total


# ---------------------------------------------------------------------------
# GLB reader (tests / stats; independent of the writer)
# ---------------------------------------------------------------------------
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


def read_glb(path):
    """Parse a GLB; validates the container.  Returns (json_dict, bin_bytes)."""
    b = Path(path).read_bytes()
    magic, ver, total = struct.unpack_from("<III", b, 0)
    if magic != 0x46546C67 or ver != 2 or total != len(b):
        raise ValueError("bad GLB header")
    l0, t0 = struct.unpack_from("<II", b, 12)
    if t0 != 0x4E4F534A or l0 % 4:
        raise ValueError("first chunk must be 4-byte aligned JSON")
    js = json.loads(b[20:20 + l0].decode("utf-8"))
    off = 20 + l0
    l1, t1 = struct.unpack_from("<II", b, off)
    if t1 != 0x004E4942 or l1 % 4 or off % 4:
        raise ValueError("second chunk must be 4-byte aligned BIN")
    binb = b[off + 8: off + 8 + l1]
    if off + 8 + l1 != len(b):
        raise ValueError("trailing bytes after BIN chunk")
    return js, binb


def accessor_array(js, binb, i) -> np.ndarray:
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    dt = np.dtype(_DT[a["componentType"]])
    n = _NCOMP[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    if off % dt.itemsize:
        raise ValueError(f"accessor {i} misaligned")
    if off + a["count"] * n * dt.itemsize > bv.get("byteOffset", 0) + bv["byteLength"]:
        raise ValueError(f"accessor {i} overruns its bufferView")
    arr = np.frombuffer(binb, dt, a["count"] * n, off)
    return arr.reshape(a["count"], n) if n > 1 else arr


def node_world_matrices(js) -> dict:
    out = {}

    def local(n):
        if "matrix" in n:
            return np.asarray(n["matrix"], float).reshape(4, 4).T
        M = np.eye(4)
        if "translation" in n:
            M[:3, 3] = n["translation"]
        return M

    def walk(i, P):
        W = P @ local(js["nodes"][i])
        out[i] = W
        for c in js["nodes"][i].get("children", []):
            walk(c, W)
    for r in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    return out


def glb_stats(path) -> dict:
    """Triangles / points per group (instanced), mesh reuse, world bbox."""
    js, binb = read_glb(path)
    W = node_world_matrices(js)
    parent = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []):
            parent[c] = i

    def group_of(i):
        while i in parent:
            if (js["nodes"][i].get("extras") or {}).get("kind") == "group":
                return js["nodes"][i]["name"]
            i = parent[i]
        return None
    tri_mesh, pts_mesh = {}, {}
    for mi, m in enumerate(js["meshes"]):
        t = p = 0
        for pr in m["primitives"]:
            if pr.get("mode", 4) == 4:
                t += js["accessors"][pr["indices"]]["count"] // 3
            else:
                p += js["accessors"][pr["attributes"]["POSITION"]]["count"]
        tri_mesh[mi], pts_mesh[mi] = t, p
    groups, reuse = {}, Counter()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    glo = {}
    for i, n in enumerate(js["nodes"]):
        if "mesh" not in n:
            continue
        g = group_of(i)
        reuse[n["mesh"]] += 1
        s = groups.setdefault(g, {"triangles": 0, "line_points": 0, "nodes": 0})
        s["triangles"] += tri_mesh[n["mesh"]]
        s["line_points"] += pts_mesh[n["mesh"]]
        s["nodes"] += 1
        for pr in js["meshes"][n["mesh"]]["primitives"]:
            a = js["accessors"][pr["attributes"]["POSITION"]]
            c = np.array([[x, y, z] for x in (a["min"][0], a["max"][0])
                          for y in (a["min"][1], a["max"][1]) for z in (a["min"][2], a["max"][2])])
            wc = c @ W[i][:3, :3].T + W[i][:3, 3]
            lo, hi = np.minimum(lo, wc.min(0)), np.maximum(hi, wc.max(0))
            gl = glo.setdefault(g, [np.full(3, np.inf), np.full(3, -np.inf)])
            gl[0], gl[1] = np.minimum(gl[0], wc.min(0)), np.maximum(gl[1], wc.max(0))
    return {"bytes": Path(path).stat().st_size,
            "triangles_rendered": int(sum(s["triangles"] for s in groups.values())),
            "triangles_unique": int(sum(tri_mesh.values())),
            "line_points": int(sum(pts_mesh.values())),
            "groups": groups, "mesh_reuse": dict(reuse),
            "bbox": [lo.tolist(), hi.tolist()],
            "group_bbox": {g: [a.tolist(), b.tolist()] for g, (a, b) in glo.items()},
            "n_nodes": len(js["nodes"]), "n_meshes": len(js["meshes"]),
            "n_materials": len(js["materials"])}


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _safe(s: str) -> str:
    """three.js PropertyBinding.sanitizeNodeName-proof names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s)).strip("_")


def _variant_label(v):
    return {"jet1975": "проєкт 1975", "jet1983": "як побудовано, 1983"}.get(v, v)


def _machine_status(res, spec):
    st = _status_of(spec.params if spec is not None else {}, res)
    out = {"status": st["status"]}
    if st.get("status_note"):
        out["status_note"] = st["status_note"]
    return out


def _load_physics(physics_dir):
    pd = Path(physics_dir)
    lc = np.load(pd / "lcfs.npz")
    fs = np.load(pd / "flux_surfaces.npz")
    fl = np.load(pd / "fieldlines.npz")
    man = json.loads((pd / "manifest.json").read_text())
    levels = [float(x) for x in fs["keys"]]
    n_pol = len(lc["profile"])
    surf = {}
    for i, lev in enumerate(levels):
        v = fs[f"v_{i:03d}"]
        surf[round(lev, 3)] = v[:n_pol][:, [0, 2]].astype(float)   # ring at phi = 0
    return {"lcfs": lc["profile"].astype(float), "surfaces": surf,
            "fieldlines": {k: fl[k] for k in fl.files}, "manifest": man}


def export_machine(machine_dir, bake_dir, physics_dir, out_path, log=print) -> dict:
    db, mb = build_web_machine(machine_dir)
    mism = check_against_bake(mb, bake_dir)
    for m in mism:
        log(f"  WARNING bake mismatch: {m}")
    phys = _load_physics(physics_dir)
    physics_json = Path(physics_dir).parent / "physics.json"
    pj = json.loads(physics_json.read_text()) if physics_json.exists() else {}

    g = GLB()
    variant = db.name
    root = g.node(f"JET_{variant.replace('jet', '')}", extras={
        "kind": "machine", "machine": db.name, "title": db.meta.get("variant", db.meta.get("machine")),
        "lineage": db.lineage, "units": "m", "up_axis": "+Y",
        "frame": "glTF: x = kernel x, y = kernel Z (machine axis), z = -kernel y; "
                 "origin on the machine axis, y = 0 midplane",
        "page_convention": "page = номер сторінки PDF (як у базі), друкований номер може відрізнятися",
        "lod": {"kernel_params": WEB_LOD, "resample_caps": _RESAMPLE_CAPS,
                "rail_plate_n_tor": _RAIL_NTOR, "lcfs": LCFS_RES, "inner_surfaces": INNER_RES,
                "fieldline_points": FIELDLINE_POINTS, "crease_deg": CREASE_DEG},
        "bake_structure_check": "identical" if not mism else mism})
    gnode = {}
    for gid, uk in GROUPS.items():
        gnode[gid] = g.node(gid, parent=root, extras={
            "kind": "group", "subsystem": gid, "label_uk": uk})

    stats = {"parts": {}, "groups": Counter()}
    specs = {s.name: s for s in db.components}
    for cname, res in mb.components.items():
        spec = specs.get(cname)
        group = TYPE_GROUP.get(res.type, "Extras1983")
        if group == "Plasma":
            continue                     # plasma from the physics bake (below)
        comp_variant = _variant_of(db, spec.origin if spec is not None else "")
        conf, counts = _confidence(res.provenance, spec.provenance if spec else None)
        srcs = _sources(res.provenance + ([spec.provenance] if spec and spec.provenance else []))
        base_dims = _key_dims(db, KEY_DIMS.get(res.type, []))
        base_dims.update(_param_dims(spec) if comp_variant != "jet1975" else {})
        status = _machine_status(res, spec)
        for p in res.parts:
            kind = material_kind(p.material)
            if group == "Structure" and variant != "jet1975" and kind in ("al_cast", "al_plate") \
                    and "nodular" in str(db.value("structure.shell_description")):
                kind = "cast_iron"               # 1983: rings + shell in nodular cast iron
            label, rgb, met, rough = MATERIALS[kind]
            dbl = res.type == "bellows"
            mat = g.material(kind + ("_2s" if dbl else ""), f"JET_{kind}", rgb, met, rough,
                             double=dbl, extras={"label_uk": label, "kind": kind})
            mesh = p.mesh
            if res.type == "bellows":
                mesh = bellows_band(mesh)
            V, N, F = crease_normals(to_yup(mesh.v), mesh_triangles(mesh))
            part_id = _safe(f"{cname}__{p.name}")
            mi = g.mesh(part_id, V, N, F, mat)
            dims = dict(base_dims)
            dims.update(_key_dims(db, _PART_DIMS.get((res.type, p.name), [])))
            extras = {
                "kind": "part", "id": part_id, "component": cname, "part": p.name,
                "subsystem": group, "label_uk": part_label_uk(cname, res.type, p.name),
                "material": p.material, "material_uk": label,
                "variant": comp_variant, "variant_uk": _variant_label(comp_variant),
                "confidence": conf, "confidence_counts": counts,
                "sources": srcs, **status, "key_dims": dims,
                "n_instances": p.n_instances,
                "assumed_inputs": res.assumptions,
                "simplified": ("гофри сильфона замінено гладкою смугою" if res.type == "bellows"
                               else None),
            }
            if extras["simplified"] is None:
                del extras["simplified"]
            pn = g.node(part_id, parent=gnode[group], extras=extras)
            for k, (X, lab) in enumerate(zip(p.transforms, p.labels)):
                phi = math.degrees(math.atan2(X[1, 0], X[0, 0]))
                g.node(_safe(f"{part_id}__{lab}"), parent=pn, mesh=mi,
                       matrix=matrix_to_yup(X),
                       extras={"kind": "instance", "label": lab, "part_id": part_id,
                               "index": k, "phi_deg": round(phi, 4)})
            ntri = len(F)
            stats["parts"][part_id] = {"group": group, "tris_unique": ntri,
                                       "instances": p.n_instances,
                                       "tris_rendered": ntri * p.n_instances}
            stats["groups"][group] += ntri * p.n_instances

    # ---- plasma (physics bake) -------------------------------------------
    pres = mb.components.get("plasma")
    pconf, pcounts = _confidence(pres.provenance) if pres else ("table", {})
    psrc = _sources(pres.provenance) if pres else []
    pdims = _key_dims(db, KEY_DIMS["plasma"])
    eq = phys["manifest"].get("equilibrium", pj.get("equilibrium", {}))
    phys_note = phys["manifest"].get("provenance", {})
    for k, val, unit in (("q95_model", eq.get("q95"), "-"), ("q0_model", eq.get("q0"), "-"),
                         ("kappa_model", eq.get("kappa"), "-"), ("delta_model", eq.get("delta"), "-"),
                         ("volume_model", eq.get("volume_m3"), "m^3")):
        if val is not None:
            pdims[k] = {"value": round(float(val), 4), "unit": unit, "confidence": "derived",
                        "source": "модель Solov'ev (Cerfon-Freidberg)", "page": None,
                        "cite": "розрахунок: рівновага Solov'ev за EUR 5516e, с. 83, 332"}
    surfs = [("lcfs", phys["lcfs"], LCFS_RES, 1.0, "Остання замкнена магнітна поверхня (LCFS)",
              PLASMA_RGB, 0.30, (0.85, 0.36, 0.20))]
    warm = {0.3: (1.0, 0.85, 0.45), 0.6: (1.0, 0.62, 0.30), 0.9: (0.95, 0.42, 0.25)}
    for psi in INNER_PSI:
        surfs.append((f"flux_psi{int(round(psi * 100)):02d}", phys["surfaces"][round(psi, 3)],
                      INNER_RES, psi, f"Магнітна поверхня ψN = {psi:.1f}".replace(".", ","),
                      warm[psi], 0.18, tuple(0.6 * c for c in warm[psi])))
    for name, prof, (npol, ntor), psi, uk, rgb, alpha, emis in surfs:
        V, F = revolve_profile(prof, npol, ntor)
        V2, N2, F2 = crease_normals(to_yup(V), F, crease_deg=60.0)
        mat = g.material(f"plasma_{name}", f"JET_plasma_{name}", rgb, 0.0, 0.9, alpha=alpha,
                         emissive=emis, double=True,
                         extras={"label_uk": "плазма (напівпрозора, світна)", "kind": "plasma"})
        pid = f"plasma__{name}"
        mi = g.mesh(pid, V2, N2, F2, mat)
        R = prof[:, 0]
        pn = g.node(pid, parent=gnode["Plasma"], extras={
            "kind": "part", "id": pid, "component": "plasma", "part": name,
            "subsystem": "Plasma", "label_uk": uk, "material": "plasma",
            "material_uk": "плазма", "variant": "jet1975", "variant_uk": _variant_label("jet1975"),
            "confidence": pconf, "confidence_counts": pcounts, "sources": psrc,
            "status": "installed", "psi_n": psi,
            "key_dims": {**pdims, "R_min": {"value": round(float(R.min()), 4), "unit": "m",
                                            "confidence": "derived", "source": "модель",
                                            "page": None, "cite": "модель"},
                         "R_max": {"value": round(float(R.max()), 4), "unit": "m",
                                   "confidence": "derived", "source": "модель", "page": None,
                                   "cite": "модель"}},
            "physics": {"model": eq.get("model"), "note": phys_note.get("sources")},
            "n_instances": 1})
        g.node(f"{pid}__0", parent=pn, mesh=mi,
               extras={"kind": "instance", "label": name, "part_id": pid, "index": 0,
                       "phi_deg": 0.0})
        stats["parts"][pid] = {"group": "Plasma", "tris_unique": len(F2), "instances": 1,
                               "tris_rendered": len(F2)}
        stats["groups"]["Plasma"] += len(F2)

    # ---- field lines ------------------------------------------------------
    fl = phys["fieldlines"]
    npts_total = 0
    for i in range(len(fl["xyz"])):
        xyz = fl["xyz"][i].astype(float)
        sel = np.unique(np.r_[np.round(np.linspace(0, len(xyz) - 1, FIELDLINE_POINTS)).astype(int)])
        P = to_yup(xyz[sel])
        q, psi, kind = float(fl["q"][i]), float(fl["psi_n0"][i]), str(fl["kind"][i])
        rgb = (1.0, 0.85, 0.35) if kind == "rational" else (0.45, 0.80, 1.0)
        mat = g.material(f"fl_{kind}", f"JET_fieldline_{kind}", rgb, 0.0, 1.0,
                         unlit=True, extras={"label_uk": "силова лінія", "kind": "fieldline"})
        fid = _safe(f"fieldline_{i}_q{q:.3f}")
        mi = g.line_mesh(fid, P, mat)
        uk = (f"Силова лінія, q = {q:.2f}".replace(".", ",")
              + (" (раціональна)" if kind == "rational" else " (ірраціональна)"))
        pn = g.node(fid, parent=gnode["FieldLines"], extras={
            "kind": "part", "id": fid, "component": "fieldlines", "part": f"line_{i}",
            "subsystem": "FieldLines", "label_uk": uk, "q": round(q, 4),
            "psi_n0": round(psi, 4), "line_kind": kind, "variant": "jet1975",
            "confidence": "derived", "status": "model",
            "sources": psrc, "key_dims": {"q": {"value": round(q, 4), "unit": "-",
                                                 "confidence": "derived", "source": "модель",
                                                 "page": None, "cite": "модель Solov'ev"}},
            "n_instances": 1, "n_points": int(len(P))})
        g.node(f"{fid}__0", parent=pn, mesh=mi,
               extras={"kind": "instance", "label": fid, "part_id": fid, "index": 0,
                       "phi_deg": 0.0})
        npts_total += len(P)
    stats["line_points"] = npts_total

    total = g.write(out_path)
    stats["bytes"] = total
    stats["triangles_rendered"] = int(sum(stats["groups"].values()))
    stats["groups"] = dict(stats["groups"])
    stats["bake_mismatches"] = mism
    return {"db": db, "mb": mb, "stats": stats, "physics": phys, "physics_json": pj}


# ---------------------------------------------------------------------------
# jet_meta.json
# ---------------------------------------------------------------------------
CHANGES_1983 = [
    "Вакуумна камера: 40 жорстких деталей замість 32 (октант = 5 жорстких секторів + 4 сильфони; "
    "торцеві сектори октанта — окремі половинки); 32-кратна симетрія збережена "
    "(JET JU Progress Report 1983, с. 24).",
    "Камера важча: 108 т замість 68 т; тороїдальний опір 0,6 мОм замість 0,52 мОм; "
    "матеріал Nicrofer 7216 LC (≈ Inconel 600).",
    "Сильфони: лист 2 мм (було 1,7 мм), гофри 120 мм, паралельні фланці; "
    "з'явилися два кільця жорсткості на Z = ±1 м.",
    "Лімітери: замість трьох рейкових Mo-лімітерів — 12 дискретних модулів 0,40 × 0,80 м на "
    "зовнішньому екваторі (4 графітові + 8 CuCr з Ni-плакуванням); у 1983 працювали лише графітові.",
    "Нові зовнішні вузли: насосні камери на октантах 1 і 5, адаптери NBI на октантах 4 і 8, "
    "модулі напуску газу на октантах 2 і 6, лімітерні порти; поворотний клапан встановлено "
    "лише в січні 1984.",
    "Нагріву в 1983 не було: розділ NBI (6 інжекторів на порт, концепція 1975) прибрано; "
    "перший бокс NBI — вересень 1984, ICRH — з 1985.",
    "PF: котушка 1 — 8 блоків замість 10 (усього 14 PF-котушок замість 16), струм 40 кА.",
    "Залізний магнітопровід: 2800 т (проєкт 1975: 2263 т), зовнішній контур без змін "
    "(габарит 11,5 × 14,8 м).",
    "Механічна оболонка: аустенітний високоміцний чавун замість литого Al-сплаву, внутрішній циліндр — "
    "SS 304 з 32 пазами (≈ 470 т).",
    "Котушки TF — без змін: 32 котушки, 24 витки, 5,68 × 3,86 м.",
]


def _fmt(v, nd=2):
    if isinstance(v, float):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    return str(v)


def build_meta(results: dict, out_dir) -> dict:
    """results: machine name -> export_machine() result."""
    meta = {"schema": "tokviz.jet_meta/1", "units": "SI (м, кг, А, Т)",
            "page_convention": "«с. N» = номер сторінки PDF документа (як у базі даних проєкту)",
            "files": {}, "machines": {}}
    for name, r in results.items():
        db, mb, st = r["db"], r["mb"], r["stats"]
        meta["files"][name] = f"{name}.glb"
        subs = OrderedDict()
        for gid, uk in GROUPS.items():
            kind = GROUP_KIND.get(gid)
            if gid == "Structure" and name != "jet1975":
                kind = "cast_iron"
            rgb = (PLASMA_RGB if gid == "Plasma" else FIELDLINE_RGB if gid == "FieldLines"
                   else MATERIALS[kind][1])
            subs[gid] = {"label_uk": uk, "colour": linear_to_hex(rgb), "components": [],
                         "key_dims": {}, "confidence_counts": Counter(),
                         "triangles": int(st["groups"].get(gid, 0))}
        for cname, res in mb.components.items():
            gid = TYPE_GROUP.get(res.type, "Extras1983")
            s = subs[gid]
            s["components"].append(cname)
            for e in res.provenance:
                if e.get("confidence"):
                    s["confidence_counts"][e["confidence"]] += 1
            for k, d in _key_dims(db, KEY_DIMS.get(res.type, [])).items():
                s["key_dims"].setdefault(k, d)
        for s in subs.values():
            s["confidence_counts"] = dict(s["confidence_counts"])
        subs["FieldLines"]["components"] = ["fieldlines (physics bake)"]
        subs["FieldLines"]["key_dims"] = {
            "n_lines": {"value": int(len(r["physics"]["fieldlines"]["q"])), "unit": "count"},
            "q_values": {"value": [round(float(q), 3) for q in r["physics"]["fieldlines"]["q"]],
                         "unit": "-"}}
        meta["machines"][name] = {
            "title": db.meta.get("variant", db.meta.get("machine")),
            "lineage": db.lineage, "file": f"{name}.glb", "bytes": st["bytes"],
            "triangles_rendered": st["triangles_rendered"], "subsystems": subs,
            "facts": _machine_facts(db)}
    # global facts (design / model)
    r0 = next(iter(results.values()))
    pj, db0 = r0["physics_json"], r0["db"]
    eq = pj.get("equilibrium", {})
    rip = pj.get("ripple", {})
    eps = rip.get("at_reference_radius", {}).get("1_filament", {}).get("eps_src_pct")
    eps_src = rip.get("source_value_pct", 3.6)

    def dd(path):
        try:
            return _dim_entry(db0.dim(path))
        except KeyError:
            return None
    meta["global"] = {
        "R0": dd("plasma.R0"), "a": dd("plasma.a"), "b": dd("plasma.b"),
        "kappa": dd("plasma.elongation_b_over_a"),
        "B0": dd("plasma.Btor_R0_basic"), "B0_extended": dd("plasma.Btor_R0_extended"),
        "I_p": dd("plasma.Ip_D_basic"), "I_p_extended": dd("plasma.Ip_D_extended"),
        "q95": {"value": round(float(eq.get("q95", float("nan"))), 3), "unit": "-",
                "source": "модель Solov'ev (Cerfon-Freidberg), I_p = 3,8 МА",
                "note": f"проєктне q(a) = 6 (EUR 5516e, с. 83); модельне q95 = "
                        f"{_fmt(float(eq.get('q95', 0)), 2)}"},
        "ripple": {"model_pct": round(float(eps), 2) if eps else None, "source_pct": eps_src,
                   "R_m": rip.get("reference_radius_m", 4.21),
                   "text_uk": (f"Гофрування TF на R = 4,21 м: {_fmt(float(eps), 2)} % (модель: 32 котушки, "
                               f"1 нитка на котушку) проти {_fmt(float(eps_src), 1)} % (EUR 5516e, с. 83)")
                   if eps else None,
                   "convention": rip.get("conventions", {}).get("eps_src")},
        "overall_diameter": dd("overall.overall_diameter"),
        "overall_height": dd("overall.overall_height"),
        "masses": {},
    }
    for name, r in results.items():
        db = r["db"]
        m = {}
        for key, path in (("machine", "overall.machine_weight"), ("iron_core", "iron_core.mass_total"),
                          ("vessel", "vessel.total_weight"), ("tf_coil_each", "tf_coils.weight")):
            try:
                m[key] = _dim_entry(db.dim(path))
            except KeyError:
                pass
        meta["global"]["masses"][name] = m
    meta["changes_1983"] = CHANGES_1983
    meta["changes_1983_source"] = "machines/jet1983/README.md"
    p = Path(out_dir) / "jet_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default))
    return meta


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _machine_facts(db) -> dict:
    out = {}
    for path in ("tf_coils.n_coils", "vessel.n_rigid_sectors", "vessel.n_octants",
                 "pf_coils.n_pf_coils_total", "pf_coils.coil1_count", "iron_core.n_limbs",
                 "vessel.total_weight", "iron_core.mass_total", "overall.machine_weight"):
        try:
            d = db.dim(path)
        except KeyError:
            continue
        out[d.path] = _dim_entry(d)
    return out
