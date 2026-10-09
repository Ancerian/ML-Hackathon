"""Materials for the engineering (JET) machine scene.

Bake material TAGS (``materials`` array of each component npz) are mapped to
shaders here.  Metals reuse :func:`tokblend.materials.metal_material`; the JET
presets are ADDED to ``METAL_PRESETS`` in the same style (base colour = linear
normal-incidence reflectance, roughness, anisotropy, brush scale, scratch,
heat), so the DIII-D presets are untouched.

Palette intent (G3 style gate may change this): neutral, "1970s engineering
hall" -- painted iron in a muted grey-green, satin Inconel, copper with an
amber epoxy coat, dull cast aluminium, dark graphite.  No invented colours
coded by subsystem; the section-fill colours (visible only on cut faces) ARE a
code, like hatching on a drawing.
"""
from __future__ import annotations

import re

import bpy

from . import materials as M
from .bl import NodeBuilder, set_in, sout, sin
from . import jet_cut as JC

#: Added to materials.METAL_PRESETS (same schema).  Heat tint kept low: the
#: vessel bakes at 300-500 C but the model is not claiming an oxide state.
JET_METAL_PRESETS = {
    # Inconel 600 / Nicrofer 7216 rigid sectors: satin nickel alloy
    "inconel": dict(base=(0.60, 0.58, 0.54), rough=0.30, aniso=0.30,
                    brush_scale=(40.0, 3.0, 3.0), scratch=0.05, heat=0.06),
    # Inconel 625 bellows: slightly brighter, finer finish
    "inconel625": dict(base=(0.64, 0.62, 0.58), rough=0.24, aniso=0.45,
                       brush_scale=(60.0, 2.5, 2.5), scratch=0.04, heat=0.04),
    # molybdenum limiter plates: grey, slightly blue-neutral
    "molybdenum": dict(base=(0.56, 0.57, 0.58), rough=0.34, aniso=0.20,
                       brush_scale=(30.0, 4.0, 4.0), scratch=0.06, heat=0.15),
    # cast aluminium alloy (1975 wedge blocks): dull, coarse
    "al_cast": dict(base=(0.80, 0.80, 0.78), rough=0.55, aniso=0.05,
                    brush_scale=(8.0, 8.0, 8.0), scratch=0.15, heat=0.0),
    # rolled aluminium plate (1975 outer shell)
    "al_plate": dict(base=(0.86, 0.86, 0.85), rough=0.38, aniso=0.35,
                     brush_scale=(50.0, 3.0, 3.0), scratch=0.15, heat=0.0),
    # plain structural steel (pads, foot plates, support cylinder)
    "steel_dark": dict(base=(0.42, 0.43, 0.44), rough=0.42, aniso=0.20,
                       brush_scale=(30.0, 3.0, 3.0), scratch=0.25, heat=0.0),
    # TF coil steel case (painted/satin stainless skin over the winding)
    "tf_case_steel": dict(base=(0.50, 0.51, 0.52), rough=0.36, aniso=0.25,
                          brush_scale=(30.0, 3.0, 3.0), scratch=0.10, heat=0.0),
    # nickel cladding (1983 Ni-clad CuCr limiters)
    "nickel": dict(base=(0.66, 0.61, 0.53), rough=0.22, aniso=0.25,
                   brush_scale=(30.0, 4.0, 4.0), scratch=0.05, heat=0.20),
}
for _k, _v in JET_METAL_PRESETS.items():
    M.METAL_PRESETS.setdefault(_k, _v)

#: Section-fill colours (cut faces only), drawing-style code.
SECTION = {
    "copper": (0.75, 0.33, 0.12),
    "iron": (0.28, 0.36, 0.34),
    "inconel": (0.36, 0.45, 0.62),
    "steel": (0.55, 0.55, 0.58),
    "aluminium": (0.70, 0.70, 0.62),
    "moly": (0.45, 0.47, 0.55),
    "graphite": (0.12, 0.12, 0.13),
    "nickel": (0.62, 0.55, 0.40),
    "neutral": (0.5, 0.5, 0.5),
}

_CACHE = {}


def _principled(name, base, rough, metallic=0.0, spec=0.5, coat=0.0,
                coat_tint=(1, 1, 1), coat_rough=0.1, noise_amt=0.0,
                noise_scale=3.0, bands=0.0, bands_scale=40.0):
    """Non-metal / coated shader with optional subtle paint variation."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)
    bsdf = nb.new("ShaderNodeBsdfPrincipled", col=6, row=0, label="Principled")
    set_in(bsdf, "Metallic", float(metallic))
    set_in(bsdf, "Roughness", float(rough))
    set_in(bsdf, "Specular IOR Level", float(spec))
    if coat > 0:
        set_in(bsdf, "Coat Weight", float(coat))
        set_in(bsdf, "Coat Roughness", float(coat_rough))
        set_in(bsdf, "Coat Tint", (*coat_tint, 1.0))
    if noise_amt > 0 or bands > 0:
        tc = nb.new("ShaderNodeTexCoord", col=0, row=0, label="TexCoord")
        noise = nb.new("ShaderNodeTexNoise", col=1, row=0, label="PaintNoise")
        noise.noise_dimensions = "3D"
        noise.normalize = True
        set_in(noise, "Scale", float(noise_scale))
        set_in(noise, "Detail", 4.0)
        nb.link(tc, "Object", noise, "Vector")
        ramp = nb.new("ShaderNodeValToRGB", col=3, row=0, label="PaintVar")
        cr = ramp.color_ramp
        lo = tuple(max(0.0, c * (1 - noise_amt)) for c in base)
        hi = tuple(min(1.0, c * (1 + noise_amt)) for c in base)
        cr.elements[0].position = 0.3
        cr.elements[0].color = (*lo, 1.0)
        cr.elements[1].position = 0.7
        cr.elements[1].color = (*hi, 1.0)
        nb.link(noise, "Factor", ramp, "Factor")
        nb.link(ramp, "Color", bsdf, "Base Color")
        if bands > 0:
            # laminations: fine horizontal bands as a bump (iron yoke plates)
            wave = nb.new("ShaderNodeTexWave", col=1, row=2, label="Laminations")
            wave.wave_type = "BANDS"
            wave.bands_direction = "Z"
            set_in(wave, "Scale", float(bands_scale))
            set_in(wave, "Distortion", 0.0)
            nb.link(tc, "Object", wave, "Vector")
            bump = nb.new("ShaderNodeBump", col=4, row=2, label="LamBump")
            set_in(bump, "Strength", float(bands))
            set_in(bump, "Distance", 0.002)
            nb.link(wave, "Factor", bump, "Height")
            nb.link(bump, "Normal", bsdf, "Normal")
    else:
        set_in(bsdf, "Base Color", (*base, 1.0))
    out = nb.new("ShaderNodeOutputMaterial", col=8, row=0, label="Output")
    nb.link(bsdf, "BSDF", out, "Surface")
    return mat


def _copper_epoxy(name, coat=0.75):
    """Copper conductor under an amber epoxy-glass coat (the insulation hint)."""
    mat = M.metal_material(name, "copper", heat_amount=0.0)
    bsdf = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    set_in(bsdf, "Coat Weight", float(coat))
    set_in(bsdf, "Coat Roughness", 0.18)
    set_in(bsdf, "Coat Tint", (0.92, 0.72, 0.42, 1.0))
    set_in(bsdf, "Coat IOR", 1.55)
    return mat


def _soften(mat, bump=0.035, scratch_scale=0.25):
    """Tone down the DIII-D metal micro-relief for metre-scale JET parts.

    The shared metal shader was tuned for 10-cm tiles; on a 3-m vessel wall
    its FBM bump + Voronoi scratches read as a hammered / crazed surface
    (phase F5 review of C3).  Keep the brushed anisotropy and roughness
    variation, drop most of the relief."""
    for nd in mat.node_tree.nodes:
        if nd.bl_idname == "ShaderNodeBump" and nd.label == "Bump":
            nd.inputs["Strength"].default_value = float(bump)
        elif nd.bl_idname == "ShaderNodeMath" and nd.label == "ScratchAmt":
            nd.inputs[1].default_value *= float(scratch_scale)
    return mat


def _build(kind):
    """kind -> (material, section key)."""
    if kind == "copper_epoxy":
        return _copper_epoxy("JET_CopperEpoxy"), "copper"
    if kind == "tf_case":
        # tag "copper_epoxy_steel_case".  G3 decision: the STEEL CASE is what
        # the outer surfaces show; the copper winding appears only on cut
        # faces (section code "copper").
        return _soften(M.metal_material("JET_TF_SteelCase", "tf_case_steel",
                                        heat_amount=0.0, bevel_radius=0.008)), "copper"
    if kind == "iron_laminated":
        return _principled("JET_IronPainted_Laminated", (0.20, 0.235, 0.225), 0.58,
                           spec=0.35, noise_amt=0.10, noise_scale=2.0,
                           bands=0.08, bands_scale=60.0), "iron"
    if kind == "iron_solid":
        return _principled("JET_IronPainted", (0.20, 0.235, 0.225), 0.55,
                           spec=0.35, noise_amt=0.10, noise_scale=2.0), "iron"
    if kind == "magnetic_steel":
        return _principled("JET_MagSteelPainted", (0.16, 0.18, 0.19), 0.50,
                           spec=0.4, noise_amt=0.08), "steel"
    if kind == "cast_iron":
        return _principled("JET_NodularCastIron", (0.19, 0.20, 0.20), 0.62,
                           spec=0.35, noise_amt=0.12, noise_scale=6.0), "iron"
    if kind == "graphite":
        return _principled("JET_Graphite", (0.035, 0.035, 0.038), 0.78,
                           spec=0.35, noise_amt=0.25, noise_scale=40.0), "graphite"
    if kind == "human":
        return _principled("JET_HumanScale", (0.55, 0.40, 0.18), 0.7,
                           spec=0.3), "neutral"
    if kind == "floor":
        return _principled("JET_Floor", (0.16, 0.16, 0.155), 0.85, spec=0.25,
                           noise_amt=0.12, noise_scale=0.6), None
    presets = {"inconel": ("JET_Inconel600", "inconel", "inconel"),
               "inconel625": ("JET_Inconel625", "inconel625", "inconel"),
               "moly": ("JET_Molybdenum", "molybdenum", "moly"),
               "al_cast": ("JET_AlCast", "al_cast", "aluminium"),
               "al_plate": ("JET_AlPlate", "al_plate", "aluminium"),
               "stainless": ("JET_Stainless", "steel", "steel"),
               "steel": ("JET_SteelDark", "steel_dark", "steel"),
               "nickel": ("JET_NickelClad", "nickel", "nickel")}
    if kind in presets:
        nm, preset, sec = presets[kind]
        return _soften(M.metal_material(nm, preset,
                                        heat_amount=M.METAL_PRESETS[preset]["heat"],
                                        bevel_radius=0.006)), sec
    return _principled(f"JET_Unmapped_{kind}", (0.5, 0.5, 0.5), 0.5), "neutral"


#: tag regex -> kind.  First match wins; unknown tags fall back to neutral.
TAG_RULES = (
    (r"copper_epoxy_steel_case|tf_case", "tf_case"),
    (r"ni.?clad|nickel", "nickel"),          # before copper: "Ni-clad ... CuCr"
    (r"copper|cucr", "copper_epoxy"),
    (r"iron_laminated|laminat", "iron_laminated"),
    (r"nodular|cast_iron", "cast_iron"),
    (r"iron", "iron_solid"),
    (r"magnetic_steel", "magnetic_steel"),
    (r"inconel625|inconel_625", "inconel625"),
    (r"inconel|nicrofer", "inconel"),
    (r"molybd|^mo$", "moly"),
    (r"al_alloy_cast|al_cast|alumin.*cast", "al_cast"),
    (r"al_alloy|alumin", "al_plate"),
    (r"graphite|carbon", "graphite"),
    (r"stainless|304|316|ss_", "stainless"),
    (r"steel", "steel"),
    (r"human", "human"),
)


def kind_for_tag(tag):
    t = str(tag).lower()
    for pat, kind in TAG_RULES:
        if re.search(pat, t):
            return kind
    return "unmapped"


def material_for_tag(tag, comp=None, part=None, plasma_mat=None, cut=True):
    """Resolver passed to :class:`machine_table.MachineTable`."""
    if str(tag) == "plasma":
        return plasma_mat
    kind = kind_for_tag(tag)
    if kind not in _CACHE:
        mat, sec = _build(kind if kind != "unmapped" else f"x_{tag}")
        if cut and sec is not None:
            JC.wrap_material(mat, SECTION.get(sec, SECTION["neutral"]))
        mat["tok_kind"] = kind
        _CACHE[kind] = mat
    return _CACHE[kind]


def floor_material():
    mat, _ = _build("floor")
    return mat


# ---------------------------------------------------------------------------
# plasma
# ---------------------------------------------------------------------------
def plasma_shell(name="JET_PlasmaShell", color=(1.0, 0.55, 0.85), strength=2.2,
                 alpha=0.55):
    """Soft emissive D-shaped shell (fallback when no psi_N data exist):
    fresnel-weighted emission, transparent face-on."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)
    lw = nb.new("ShaderNodeLayerWeight", col=0, row=1, label="Rim")
    set_in(lw, "Blend", 0.35)
    em = nb.new("ShaderNodeEmission", col=2, row=0, label="Emission")
    set_in(em, "Color", (*color, 1.0))
    set_in(em, "Strength", float(strength))
    tr = nb.new("ShaderNodeBsdfTransparent", col=2, row=2, label="Transparent")
    mul = nb.new("ShaderNodeMath", col=2, row=1, operation="MULTIPLY", label="Alpha")
    nb.link(lw, "Facing", mul, "Value", b_idx=0)
    set_in(mul, "Value", float(alpha), idx=1)
    mix = nb.new("ShaderNodeMixShader", col=4, row=1, label="Mix")
    nb.link(mul, "Value", mix, "Factor")
    nt.links.new(sout(tr, "BSDF"), mix.inputs[1])
    nt.links.new(sout(em, "Emission"), mix.inputs[2])
    out = nb.new("ShaderNodeOutputMaterial", col=6, row=1, label="Output")
    nb.link(mix, "Shader", out, "Surface")
    from .bl import set_transparency
    set_transparency(mat, "BLENDED")
    mat.use_backface_culling = False
    return mat


def plasma_section_material(name="JET_PlasmaSection", strength=1.1):
    """Flat section through the plasma, coloured by the per-vertex ``psi_n``
    attribute with the same warm-core / cool-edge ramp as the flux surfaces."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)
    attr = nb.new("ShaderNodeAttribute", col=0, row=0, label="psi_n")
    attr.attribute_name = "psi_n"
    ramp = nb.new("ShaderNodeValToRGB", col=2, row=0, label="Palette")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (1.0, 0.85, 0.45, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.15, 0.35, 0.95, 1.0)
    cr.elements.new(0.55).color = (0.95, 0.35, 0.25, 1.0)
    nb.link(attr, "Factor", ramp, "Factor")
    em = nb.new("ShaderNodeEmission", col=4, row=0, label="Emission")
    nb.link(ramp, "Color", em, "Color")
    set_in(em, "Strength", float(strength))
    out = nb.new("ShaderNodeOutputMaterial", col=6, row=0, label="Output")
    nb.link(em, "Emission", out, "Surface")
    mat.use_backface_culling = False
    return mat
