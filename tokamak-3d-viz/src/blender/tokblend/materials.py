"""Shader library.

Everything is authored so one node tree renders in BOTH engines.  Where a node
is Cycles-only its branch is mixed in behind a factor that EEVEE evaluates to
zero, so nothing silently changes meaning between look-dev and finals:

* ``ShaderNodeBevel`` and ``ShaderNodeAmbientOcclusion`` are Cycles-only.  In
  EEVEE, Bevel passes the unmodified normal through and AO returns white.
* ``Geometry -> Pointiness`` is likewise Cycles-only, so curvature wear is
  built from procedural textures instead of relying on it.
* EEVEE volume EMISSION does not light surrounding surfaces.  The plasma
  therefore also carries a proxy light rig (see :mod:`tokblend.cameras`);
  Cycles needs none.

Node names below are the Blender 5.2 spelling.  4.2 LTS differences are noted
inline; the ones that need code live in :mod:`tokblend.bl`.
"""
from __future__ import annotations

import bpy

from .bl import NodeBuilder, sin, sout, set_in, set_transparency


# ---------------------------------------------------------------------------
# psi_n as a float image (exact for an axisymmetric equilibrium)
# ---------------------------------------------------------------------------
def psi_n_image(name, psi_n_grid):
    """Pack psi_n(Z, R) into a non-colour float image.

    psi_n is written into R, G and B alike, so feeding the image's Color output
    into a float socket (which averages the channels) returns psi_n exactly.
    """
    import numpy as np

    g = np.asarray(psi_n_grid, dtype=np.float32)
    nz, nr = g.shape
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, width=nr, height=nz, float_buffer=True,
                                  is_data=True)
    img.colorspace_settings.name = "Non-Color"
    px = np.empty((nz, nr, 4), dtype=np.float32)
    px[..., 0] = g
    px[..., 1] = g
    px[..., 2] = g
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.update()
    return img


def _rz_lookup(nb, img, R_range, Z_range, col0=0):
    """Nodes computing psi_n at the shading point from object coordinates.

    Object space -> (R, Z) -> normalised UV -> image lookup.  Returns the
    Image Texture node whose Color output carries psi_n.
    """
    tc = nb.new("ShaderNodeTexCoord", col=col0, row=0, label="TexCoord")
    sep = nb.new("ShaderNodeSeparateXYZ", col=col0 + 1, row=0, label="SepXYZ")
    nb.link(tc, "Object", sep, "Vector")

    # R = hypot(x, y)
    xx = nb.new("ShaderNodeMath", col=col0 + 2, row=0, operation="MULTIPLY", label="x2")
    nb.link(sep, "X", xx, "Value", b_idx=0)
    nb.link(sep, "X", xx, "Value", b_idx=1)
    yy = nb.new("ShaderNodeMath", col=col0 + 2, row=1, operation="MULTIPLY", label="y2")
    nb.link(sep, "Y", yy, "Value", b_idx=0)
    nb.link(sep, "Y", yy, "Value", b_idx=1)
    add = nb.new("ShaderNodeMath", col=col0 + 3, row=0, operation="ADD", label="x2+y2")
    nb.link(xx, "Value", add, "Value", b_idx=0)
    nb.link(yy, "Value", add, "Value", b_idx=1)
    rad = nb.new("ShaderNodeMath", col=col0 + 4, row=0, operation="SQRT", label="R")
    nb.link(add, "Value", rad, "Value")

    # normalise R and Z onto [0, 1]
    mr = nb.new("ShaderNodeMapRange", col=col0 + 5, row=0, label="R->u")
    set_in(mr, "From Min", float(R_range[0]))
    set_in(mr, "From Max", float(R_range[1]))
    set_in(mr, "To Min", 0.0)
    set_in(mr, "To Max", 1.0)
    nb.link(rad, "Value", mr, "Value")

    mz = nb.new("ShaderNodeMapRange", col=col0 + 5, row=1, label="Z->v")
    set_in(mz, "From Min", float(Z_range[0]))
    set_in(mz, "From Max", float(Z_range[1]))
    set_in(mz, "To Min", 0.0)
    set_in(mz, "To Max", 1.0)
    nb.link(sep, "Z", mz, "Value")

    uv = nb.new("ShaderNodeCombineXYZ", col=col0 + 6, row=0, label="UV")
    nb.link(mr, "Result", uv, "X")
    nb.link(mz, "Result", uv, "Y")

    tex = nb.new("ShaderNodeTexImage", col=col0 + 7, row=0, label="psi_n")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    nb.link(uv, "Vector", tex, "Vector")
    return tex


# ---------------------------------------------------------------------------
# plasma volume
# ---------------------------------------------------------------------------
def plasma_volume(name, psi_n_grid, R_range, Z_range,
                  density=6.0, emission=14.0, turbulence=0.35,
                  t_core=11000.0, t_edge=1800.0):
    """Emissive, layered plasma volume driven by the real psi_n(R, Z).

    Graph
    -----
    ::

        TexCoord.Object -> SeparateXYZ -> (R = sqrt(x^2+y^2), Z)
                        -> MapRange x2 -> CombineXYZ -> ImageTexture(psi_n)
        psi_n -> ColorRamp(radial profile) --+--> Math(*) -> Density
                                             |
        NoiseTexture(4D) ------------------- +
        psi_n -> MapRange(inverted) -> Blackbody -> Emission Color
        psi_n -> ColorRamp(emission profile) -> Math(*) -> Emission Strength
        -> Principled Volume -> Material Output.Volume

    Physical honesty: the colour ramp runs core-hot / edge-cool because that is
    what the brief asks and what reads as "plasma".  A real VISIBLE-light
    photograph of a tokamak is the other way round -- the core is far too hot to
    radiate in the visible and appears dark, while the cool edge glows from line
    radiation.  Say so in any caption that claims photographic realism.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)

    img = psi_n_image(f"{name}_psi_n", psi_n_grid)
    tex = _rz_lookup(nb, img, R_range, Z_range)

    # --- radial density profile -------------------------------------------
    ramp_d = nb.new("ShaderNodeValToRGB", col=9, row=0, label="DensityProfile")
    cr = ramp_d.color_ramp
    cr.interpolation = "B_SPLINE"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    e = cr.elements.new(0.72)
    e.color = (0.55, 0.55, 0.55, 1.0)
    nb.link(tex, "Color", ramp_d, "Factor")

    # --- turbulence --------------------------------------------------------
    noise = nb.new("ShaderNodeTexNoise", col=9, row=2, label="Turbulence")
    noise.noise_dimensions = "4D"          # W drives time; static for a still
    noise.noise_type = "FBM"               # Musgrave was folded in here at 4.1
    noise.normalize = True
    set_in(noise, "Scale", 5.5)
    set_in(noise, "Detail", 7.0)
    set_in(noise, "Roughness", 0.55)
    set_in(noise, "W", 0.0)

    n_mix = nb.new("ShaderNodeMapRange", col=10, row=2, label="TurbGain")
    set_in(n_mix, "From Min", 0.0)
    set_in(n_mix, "From Max", 1.0)
    set_in(n_mix, "To Min", 1.0 - float(turbulence))
    set_in(n_mix, "To Max", 1.0 + float(turbulence))
    nb.link(noise, "Factor", n_mix, "Value")

    d_mul = nb.new("ShaderNodeMath", col=11, row=0, operation="MULTIPLY", label="DensMul")
    nb.link(ramp_d, "Color", d_mul, "Value", b_idx=0)
    nb.link(n_mix, "Result", d_mul, "Value", b_idx=1)
    d_scale = nb.new("ShaderNodeMath", col=12, row=0, operation="MULTIPLY", label="DensScale")
    nb.link(d_mul, "Value", d_scale, "Value", b_idx=0)
    set_in(d_scale, "Value", float(density), idx=1)

    # --- emission profile and colour --------------------------------------
    ramp_e = nb.new("ShaderNodeValToRGB", col=9, row=4, label="EmissionProfile")
    ce = ramp_e.color_ramp
    ce.interpolation = "B_SPLINE"
    ce.elements[0].position = 0.0
    ce.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    ce.elements[1].position = 0.92
    ce.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    ce.elements.new(0.55).color = (0.35, 0.35, 0.35, 1.0)
    nb.link(tex, "Color", ramp_e, "Factor")

    e_scale = nb.new("ShaderNodeMath", col=11, row=4, operation="MULTIPLY", label="EmitScale")
    nb.link(ramp_e, "Color", e_scale, "Value", b_idx=0)
    set_in(e_scale, "Value", float(emission), idx=1)

    temp = nb.new("ShaderNodeMapRange", col=9, row=6, label="psi_n->T")
    set_in(temp, "From Min", 0.0)
    set_in(temp, "From Max", 1.0)
    set_in(temp, "To Min", float(t_core))
    set_in(temp, "To Max", float(t_edge))
    temp.clamp = True
    nb.link(tex, "Color", temp, "Value")

    bb = nb.new("ShaderNodeBlackbody", col=10, row=6, label="Blackbody")
    nb.link(temp, "Result", bb, "Temperature")

    # --- output ------------------------------------------------------------
    vol = nb.new("ShaderNodeVolumePrincipled", col=13, row=2, label="PrincipledVolume")
    set_in(vol, "Anisotropy", 0.25)
    nb.link(d_scale, "Value", vol, "Density")
    nb.link(e_scale, "Value", vol, "Emission Strength")
    nb.link(bb, "Color", vol, "Emission Color")
    set_in(vol, "Color", (0.12, 0.28, 0.75, 1.0))

    out = nb.new("ShaderNodeOutputMaterial", col=15, row=2, label="Output")
    nb.link(vol, "Volume", out, "Volume")
    return mat


# ---------------------------------------------------------------------------
# flux-surface / field-line shading
# ---------------------------------------------------------------------------
def flux_surface_material(name, alpha=0.22, emission=2.5):
    """Translucent shell tinted by the per-vertex ``psi_n`` attribute."""
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
    cr.elements[0].color = (1.0, 0.85, 0.45, 1.0)     # core, warm
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.15, 0.35, 0.95, 1.0)    # edge, cool
    cr.elements.new(0.55).color = (0.95, 0.35, 0.25, 1.0)
    nb.link(attr, "Factor", ramp, "Factor")

    # Fresnel rim so nested shells stay readable when stacked
    lw = nb.new("ShaderNodeLayerWeight", col=2, row=2, label="Rim")
    set_in(lw, "Blend", 0.28)

    emis = nb.new("ShaderNodeEmission", col=4, row=0, label="Emission")
    nb.link(ramp, "Color", emis, "Color")
    set_in(emis, "Strength", float(emission))

    transp = nb.new("ShaderNodeBsdfTransparent", col=4, row=2, label="Transparent")

    a_mul = nb.new("ShaderNodeMath", col=4, row=4, operation="MULTIPLY", label="Alpha")
    nb.link(lw, "Facing", a_mul, "Value", b_idx=0)
    set_in(a_mul, "Value", float(alpha), idx=1)

    mix = nb.new("ShaderNodeMixShader", col=6, row=1, label="MixShader")
    nb.link(a_mul, "Value", mix, "Factor")
    nt.links.new(sout(transp, "BSDF"), mix.inputs[1])
    nt.links.new(sout(emis, "Emission"), mix.inputs[2])

    out = nb.new("ShaderNodeOutputMaterial", col=8, row=1, label="Output")
    nb.link(mix, "Shader", out, "Surface")

    set_transparency(mat, "BLENDED")
    mat.use_backface_culling = False
    return mat


def emissive_material(name, color=(0.4, 0.8, 1.0, 1.0), strength=6.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)
    e = nb.new("ShaderNodeEmission", col=0, row=0, label="Emission")
    set_in(e, "Color", color)
    set_in(e, "Strength", float(strength))
    out = nb.new("ShaderNodeOutputMaterial", col=2, row=0, label="Output")
    nb.link(e, "Emission", out, "Surface")
    return mat


def matte_material(name, color=(0.02, 0.022, 0.03, 1.0), roughness=0.95):
    """Flat, almost-black backdrop -- e.g. the phi = 0 section card, which must
    sit behind the punctures without competing with them."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)
    b = nb.new("ShaderNodeBsdfPrincipled", col=0, row=0, label="Principled")
    set_in(b, "Base Color", color)
    set_in(b, "Roughness", float(roughness))
    set_in(b, "Metallic", 0.0)
    set_in(b, "Specular IOR Level", 0.15)
    out = nb.new("ShaderNodeOutputMaterial", col=2, row=0, label="Output")
    nb.link(b, "BSDF", out, "Surface")
    return mat


# ---------------------------------------------------------------------------
# metals
# ---------------------------------------------------------------------------
#: Base colours are measured normal-incidence reflectances, linear sRGB.
METAL_PRESETS = {
    # tungsten: grey, slightly warm; divertor monoblock material
    "tungsten": dict(base=(0.54, 0.52, 0.49), rough=0.34, aniso=0.65,
                     brush_scale=(70.0, 2.5, 2.5), scratch=0.55, heat=0.75),
    # beryllium: light, low reflectance, matte (pre-2024 ITER first wall)
    "beryllium": dict(base=(0.66, 0.65, 0.62), rough=0.48, aniso=0.15,
                      brush_scale=(18.0, 6.0, 6.0), scratch=0.35, heat=0.25),
    # 316LN stainless: vessel / ports / structure
    "steel": dict(base=(0.56, 0.57, 0.58), rough=0.26, aniso=0.35,
                  brush_scale=(48.0, 3.0, 3.0), scratch=0.30, heat=0.10),
    # copper: DIII-D-style resistive coils
    "copper": dict(base=(0.72, 0.36, 0.20), rough=0.30, aniso=0.30,
                   brush_scale=(30.0, 4.0, 4.0), scratch=0.25, heat=0.15),
}


def metal_material(name, preset="tungsten", heat_amount=None, use_bevel=True,
                   bevel_radius=0.012):
    """Procedural brushed metal with scratches, bump and heat tinting.

    Graph
    -----
    ::

        TexCoord.Object -> Mapping(anisotropic scale) -> Noise(FBM) --> brush streaks
                                                      -> Voronoi(F1) --> scratches
        brush + scratch -> Math -> Bump.Height -> Principled.Normal
        brush          -> MapRange -> Principled.Roughness
        heat mask      -> ColorRamp(temper colours) -> Mix -> Principled.Base Color
        Bevel.Normal   -> Bump.Normal          (Cycles only; pass-through in EEVEE)

    The heat mask is the vertical object coordinate by default, so tiles nearer
    the plasma temper more strongly; drive it from an attribute instead when a
    real heat-flux map is available.
    """
    cfg = dict(METAL_PRESETS[preset])
    if heat_amount is not None:
        cfg["heat"] = float(heat_amount)

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nb = NodeBuilder(nt)

    tc = nb.new("ShaderNodeTexCoord", col=0, row=0, label="TexCoord")
    mapn = nb.new("ShaderNodeMapping", col=1, row=0, label="BrushMapping")
    set_in(mapn, "Scale", tuple(cfg["brush_scale"]))
    nb.link(tc, "Object", mapn, "Vector")

    # --- brushed streaks ---------------------------------------------------
    brush = nb.new("ShaderNodeTexNoise", col=3, row=0, label="BrushNoise")
    brush.noise_dimensions = "3D"
    brush.noise_type = "FBM"
    brush.normalize = True
    set_in(brush, "Scale", 9.0)
    set_in(brush, "Detail", 6.0)
    set_in(brush, "Roughness", 0.62)
    nb.link(mapn, "Vector", brush, "Vector")

    # --- scratches ---------------------------------------------------------
    smap = nb.new("ShaderNodeMapping", col=1, row=3, label="ScratchMapping")
    set_in(smap, "Scale", (cfg["brush_scale"][0] * 0.35, 1.2, 1.2))
    nb.link(tc, "Object", smap, "Vector")

    scr = nb.new("ShaderNodeTexVoronoi", col=3, row=3, label="Scratches")
    scr.feature = "DISTANCE_TO_EDGE"
    scr.voronoi_dimensions = "3D"
    scr.normalize = True
    set_in(scr, "Scale", 26.0)
    set_in(scr, "Randomness", 0.9)
    nb.link(smap, "Vector", scr, "Vector")

    scr_ramp = nb.new("ShaderNodeValToRGB", col=5, row=3, label="ScratchMask")
    sc = scr_ramp.color_ramp
    sc.elements[0].position = 0.0
    sc.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    sc.elements[1].position = 0.06
    sc.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    nb.link(scr, "Distance", scr_ramp, "Factor")

    scr_amt = nb.new("ShaderNodeMath", col=7, row=3, operation="MULTIPLY", label="ScratchAmt")
    nb.link(scr_ramp, "Color", scr_amt, "Value", b_idx=0)
    set_in(scr_amt, "Value", float(cfg["scratch"]), idx=1)

    # --- combined height ---------------------------------------------------
    height = nb.new("ShaderNodeMath", col=9, row=1, operation="ADD", label="Height")
    nb.link(brush, "Factor", height, "Value", b_idx=0)
    nb.link(scr_amt, "Value", height, "Value", b_idx=1)

    bump = nb.new("ShaderNodeBump", col=11, row=1, label="Bump")
    set_in(bump, "Strength", 0.22)
    set_in(bump, "Distance", 0.004)
    nb.link(height, "Value", bump, "Height")

    # Cycles-only contact bevel; in EEVEE this node passes the normal through
    if use_bevel:
        bev = nb.new("ShaderNodeBevel", col=9, row=-1, label="Bevel_CyclesOnly")
        set_in(bev, "Radius", float(bevel_radius))
        nb.link(bev, "Normal", bump, "Normal")

    # --- roughness ---------------------------------------------------------
    rough = nb.new("ShaderNodeMapRange", col=6, row=0, label="Roughness")
    set_in(rough, "From Min", 0.0)
    set_in(rough, "From Max", 1.0)
    set_in(rough, "To Min", max(cfg["rough"] - 0.10, 0.02))
    set_in(rough, "To Max", min(cfg["rough"] + 0.14, 0.95))
    nb.link(brush, "Factor", rough, "Value")

    # --- heat tinting ------------------------------------------------------
    hsep = nb.new("ShaderNodeSeparateXYZ", col=1, row=6, label="HeatCoord")
    nb.link(tc, "Object", hsep, "Vector")
    hmap = nb.new("ShaderNodeMapRange", col=3, row=6, label="HeatMask")
    set_in(hmap, "From Min", -3.0)
    set_in(hmap, "From Max", 3.0)
    set_in(hmap, "To Min", 1.0)
    set_in(hmap, "To Max", 0.0)
    hmap.clamp = True
    nb.link(hsep, "Z", hmap, "Value")

    hmul = nb.new("ShaderNodeMath", col=5, row=6, operation="MULTIPLY", label="HeatAmt")
    nb.link(hmap, "Result", hmul, "Value", b_idx=0)
    set_in(hmul, "Value", float(cfg["heat"]), idx=1)

    temper = nb.new("ShaderNodeValToRGB", col=7, row=6, label="TemperColours")
    tr = temper.color_ramp
    tr.elements[0].position = 0.0
    tr.elements[0].color = (*cfg["base"], 1.0)
    tr.elements[1].position = 1.0
    tr.elements[1].color = (0.20, 0.26, 0.52, 1.0)      # deep blue, hottest
    for pos, col in ((0.30, (0.62, 0.50, 0.24, 1.0)),   # straw
                     (0.52, (0.48, 0.24, 0.14, 1.0)),   # brown
                     (0.72, (0.34, 0.18, 0.40, 1.0))):  # purple
        tr.elements.new(pos).color = col
    nb.link(hmul, "Value", temper, "Factor")

    # --- output ------------------------------------------------------------
    bsdf = nb.new("ShaderNodeBsdfPrincipled", col=13, row=1, label="Principled")
    set_in(bsdf, "Metallic", 1.0)
    set_in(bsdf, "Anisotropic", float(cfg["aniso"]))
    set_in(bsdf, "Specular IOR Level", 0.5)   # 4.0 renamed this from 'Specular'
    nb.link(temper, "Color", bsdf, "Base Color")
    nb.link(rough, "Result", bsdf, "Roughness")
    nb.link(bump, "Normal", bsdf, "Normal")

    tang = nt.nodes.new("ShaderNodeTangent")
    tang.direction_type = "RADIAL"
    tang.axis = "Z"
    tang.location = (bsdf.location[0] - 220, bsdf.location[1] - 520)
    nt.links.new(sout(tang, "Tangent"), sin(bsdf, "Tangent"))

    out = nb.new("ShaderNodeOutputMaterial", col=15, row=1, label="Output")
    nb.link(bsdf, "BSDF", out, "Surface")
    return mat
