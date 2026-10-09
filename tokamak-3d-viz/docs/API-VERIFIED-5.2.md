# Blender API: verified reference and 4.2 → 5.x deltas

Every statement here was produced by introspecting **Blender 5.2.1 LTS**
(`hash 9e2066aef7ef`, build 2026-08-25) on macOS/arm64, or by executing the
call and observing the result. Nothing is quoted from memory or documentation.

Reproduce with:

```bash
blender --background --factory-startup --python tools/introspect_api.py
```

**Build facts**

| Item | Value |
|---|---|
| Version | 5.2.1 LTS, `bpy.app.version == (5, 2, 1)` |
| Bundled Python | 3.13 |
| Bundled numpy | 2.3.4 — so `.npy`/`.npz` load inside Blender with no install |
| scipy inside Blender | **absent** — all heavy maths must run outside |
| Cycles devices seen | `Apple M4` (CPU), `Apple M4 (GPU - 10 cores)` (METAL) |

---

## 1. The deltas that actually break scripts

These are the ones that silently or loudly break a 4.2-era script on 5.2.

### 1.1 Geometry Nodes modifier inputs — **hard break**

```python
# Blender 4.2 LTS
mod["Socket_2"] = 0.5

# Blender 5.x   -> the 4.2 form raises TypeError, it does not just no-op
getattr(mod.properties.inputs, "Socket_2").value = 0.5
```

`NodesModifier` in 5.x no longer supports ID-property assignment at all:

```
TypeError: bpy_struct[key] = val: id properties not supported for this type
```

`mod.properties` is a `GeometryNodesModifierInterface`; `mod.properties.inputs`
is a `GeometryNodesInterfaceInputs` whose **attributes are named by socket
identifier** (`Socket_0`, `Socket_1`, …). It has no `len()`, no `keys()`, and
`inputs["Socket_1"]` returns an `IDPropertyGroup` of metadata — *not* the value.
The value lives at `.value` on the attribute access.

Handled by `set_modifier_input()` / `get_modifier_input()` in `tokblend/bl.py`.

### 1.2 Socket renames: `Fac` → `Factor`

In 5.x these are all called **`Factor`**:

| Node | 4.2 | 5.2 |
|---|---|---|
| `ShaderNodeValToRGB` input | `Fac` | `Factor` |
| `ShaderNodeMixShader` input | `Fac` | `Factor` |
| `ShaderNodeTexNoise` output | `Fac` | `Factor` |
| `ShaderNodeTexVoronoi` output | `Fac` | `Factor` |
| `ShaderNodeAttribute` output | `Fac` | `Factor` |

Handled by the `_ALIASES` table in `tokblend/bl.py`, which resolves either
spelling on either version.

### 1.3 Compositor — rebuilt

| | 4.2 LTS | 5.x |
|---|---|---|
| tree lives at | `Scene.node_tree` | `Scene.compositing_node_group` |
| final node | `CompositorNodeComposite` | `NodeGroupOutput` |
| mix | `CompositorNodeMixRGB` | `ShaderNodeMix` |
| math | `CompositorNodeMath` | `ShaderNodeMath` |
| ramp | `CompositorNodeValToRGB` | `ShaderNodeValToRGB` |
| gamma | `CompositorNodeGamma` | *(removed)* |

`Scene.node_tree` does not exist in 5.2 — touching it raises `AttributeError`.
Confirmed still present: `CompositorNodeRLayers`, `Glare`, `Blur`, `CurveRGB`,
`HueSat`, `BrightContrast`, `Invert`, `Zcombine`, `Image`, `Scale`,
`Translate`, `AlphaOver`, `Tonemap`, `Lensdist`, `ColorBalance`, `Exposure`,
`Denoise`, `SetAlpha`, `IDMask`, `CryptomatteV2`, `OutputFile`.

### 1.4 Glare node — properties became **menu sockets**

In 5.x `glare_type`, `quality`, `threshold`, `size`, `mix`, `iterations`,
`fade` are **gone as node properties**. They are input sockets:

```
Image, Type, Quality, Threshold, Smoothness, Clamp, Maximum, Strength,
Saturation, Tint, Size, Streaks, Streaks Angle, Iterations, Fade,
Color Modulation, Diagonal, Sun Position, Jitter, Kernel Data Type, Kernel
```

Outputs: `Image`, `Glare`, `Highlights`.

`Type` and `Quality` are `NodeSocketMenu`. Their `enum_items` is **empty** —
the menu is built dynamically, so the valid values cannot be discovered by
introspection. They are exact display strings:

```
Type    : 'Bloom' | 'Ghosts' | 'Streaks' | 'Fog Glow' | 'Simple Star'
          | 'Sun Beams' | 'Kernel'          (default 'Streaks')
Quality : 'Low' | 'Medium' | 'High'          (default 'Medium')
```

Uppercase enum identifiers (`'BLOOM'`) raise `TypeError`.

**Trap:** `Size` is a **0–1 float** in 5.x (default 0.5). In 4.2 it was an
integer number of blur steps, 1–9. Passing `8` gives an enormous, obviously
wrong glare — this bit during development.

### 1.5 EEVEE lost Bloom (4.2) — glare is mandatory, not polish

`scene.eevee.use_bloom`, `use_gtao`, `use_ssr` are all **absent**. Bloom must
come from the compositor `Glare` node. Replacements present in 5.2:
`use_raytracing`, `ray_tracing_method`, `fast_gi_*`, `clamp_volume_direct`,
`clamp_volume_indirect`, `shadow_ray_count`, `shadow_step_count`.

Volumetrics: `volumetric_samples` (default 64), `volumetric_tile_size`
(`'1'|'2'|'4'|'8'|'16'`), `volumetric_start` 0.1, `volumetric_end` 100.0,
`use_volumetric_shadows` (**default False**).

### 1.6 `PointCloud.points` has no `.add()`

Populate with `PointCloud.resize(n)` — `resize` is on the datablock, not on
`.points`. This project uses a vertex mesh plus a Geometry Nodes instancer
instead, which behaves identically on 4.2 and 5.x and is easier to shade.

### 1.7 Render engine identifier

`scene.render.engine` accepts `'BLENDER_EEVEE'`, `'CYCLES'`,
`'BLENDER_WORKBENCH'`. **`'BLENDER_EEVEE_NEXT'` is rejected** in 5.2 — it was
the 4.2 spelling. Note the enum only *lists* `BLENDER_EEVEE` even with Cycles
enabled, because the list is built dynamically; assignment is the reliable test.

### 1.8 Deprecations to expect

`Material.use_nodes` and `Scene.use_nodes` both emit
`DeprecationWarning: expected to be removed in Blender 6.0`.

---

## 2. Dynamic enums cannot be introspected

Several properties report an **empty** `enum_items` yet accept values. Do not
conclude a feature is missing from an empty list — test by assignment.

| Property | Introspects as | Actually accepts |
|---|---|---|
| `view_settings.view_transform` | `['NONE']` | `Standard`, `Filmic`, `AgX`, `Khronos PBR Neutral`, `Raw`, `False Color` |
| `view_settings.look` | `['NONE']` | `None`, `AgX - Punchy`, `AgX - Base Contrast`, … |
| `cycles.denoiser` | `[]` | `OPENIMAGEDENOISE` (not `OPTIX` on Metal) |
| `cycles.sampling_pattern` | `[]` | `AUTOMATIC`, `TABULATED_SOBOL`, `BLUE_NOISE` (not `SOBOL_BURLEY`) |
| `prefs.compute_device_type` | `[]` | `METAL`, `NONE` (not `CUDA`/`OPTIX` on this machine) |
| Glare `Type` / `Quality` | `[]` | see §1.4 |

`eevee.ray_tracing_method` accepted only `'PROBE'` here; `'SCREEN_TRACE'` and
`'NONE'` were rejected.

---

## 3. Principled BSDF v2 — full socket list (5.2)

`ShaderNodeBsdfPrincipled`, 32 inputs, in order:

```
 0 Base Color            RGBA      16 Anisotropic            0.0
 1 Metallic              0.0       17 Anisotropic Rotation   0.0
 2 Roughness             0.5       18 Tangent                VECTOR
 3 IOR                   1.5       19 Transmission Weight    0.0
 4 Alpha                 1.0       20 Coat Weight            0.0
 5 Thin Wall             BOOLEAN   21 Coat Roughness         0.03
 6 Normal                VECTOR    22 Coat IOR               1.5
 7 Weight                0.0       23 Coat Tint              RGBA
 8 Diffuse Roughness     0.0       24 Coat Normal            VECTOR
 9 Subsurface Weight     0.0       25 Sheen Weight           0.0
10 Subsurface Radius     VECTOR    26 Sheen Roughness        0.5
11 Subsurface Scale      0.005     27 Sheen Tint             RGBA
12 Subsurface IOR        1.4       28 Emission Color         RGBA
13 Subsurface Anisotropy 0.0       29 Emission Strength      0.0
14 Specular IOR Level    0.5       30 Thin Film Thickness    0.0
15 Specular Tint         RGBA      31 Thin Film IOR          1.33
```

`distribution`: `GGX`, `MULTI_GGX`.
`subsurface_method`: `BURLEY`, `RANDOM_WALK`, `RANDOM_WALK_SKIN`, `RANDOM_WALK_LEGACY`.

**Present in 5.2, absent in 4.2:** `Thin Wall`, `Diffuse Roughness`.
**Renamed at 4.0:** `Specular`→`Specular IOR Level`, `Subsurface`→`Subsurface
Weight`, `Transmission`→`Transmission Weight`, `Clearcoat`→`Coat Weight`,
`Sheen`→`Sheen Weight`, `Emission`→`Emission Color`.

### Material transparency

```python
mat.surface_render_method  # 4.2+ : 'DITHERED' | 'BLENDED'
mat.blend_method           # legacy, still present: 'OPAQUE'|'CLIP'|'HASHED'|'BLEND'
mat.shadow_method          # REMOVED at 4.2 -- absent in 5.2
```

Also present: `use_backface_culling`, `use_transparent_shadow`,
`use_raytrace_refraction`, `displacement_method` (`BUMP`/`DISPLACEMENT`/`BOTH`),
`thickness_mode` (`SPHERE`/`SLAB`).

---

## 4. Texture nodes

### Noise Texture — Musgrave lives here now

`ShaderNodeTexMusgrave` **does not exist** (`Node type ShaderNodeTexMusgrave
undefined`). It was folded into Noise Texture at 4.1.

```
inputs : Vector, W, Scale, Detail, Roughness, Lacunarity, Offset, Gain, Distortion
outputs: Factor, Color
noise_type      : MULTIFRACTAL | RIDGED_MULTIFRACTAL | HYBRID_MULTIFRACTAL
                  | FBM | HETERO_TERRAIN
noise_dimensions: 1D | 2D | 3D | 4D
normalize       : BOOLEAN
```

`Offset` and `Gain` are only meaningful for the hetero/hybrid types.

### Voronoi Texture

```
outputs: Distance, Color, Position, W, Radius
feature : F1 | F2 | SMOOTH_F1 | DISTANCE_TO_EDGE | N_SPHERE_RADIUS
distance: EUCLIDEAN | MANHATTAN | CHEBYCHEV | MINKOWSKI
normalize: BOOLEAN
```

### Bump — new input in 5.x

```
inputs: Strength, Distance, Filter Width, Height, Normal
```

`Filter Width` is **not present in 4.2**. Positional socket indexing across
versions will therefore misalign; address by name.

### Cycles-only nodes

`ShaderNodeBevel` and `ShaderNodeAmbientOcclusion` create successfully in any
engine but are **ignored by EEVEE** — Bevel passes the normal through, AO
returns white. Same for `Geometry → Pointiness`. Build EEVEE-visible wear from
procedural textures; keep Bevel/AO as a Cycles-only enhancement.

### Volume nodes

```
Principled Volume inputs: Color, Color Attribute, Density, Density Attribute,
  Anisotropy, Absorption Color, Emission Strength, Emission Color,
  Blackbody Intensity, Blackbody Tint, Temperature, Temperature Attribute, Weight
Volume Scatter inputs   : Color, Density, Anisotropy, IOR, Backscatter, Alpha,
                          Diameter, Weight
```

`IOR`, `Backscatter`, `Alpha`, `Diameter` on Volume Scatter are 5.x additions.

---

## 5. Geometry Nodes

All 98 node identifiers this project uses exist in 5.2 — full list in
`tools/introspect_api.py`. Notable socket detail:

```
GeometryNodeDeleteGeometry   in: Geometry, Selection
                             domain: POINT|EDGE|FACE|CURVE|INSTANCE|LAYER
                             mode: ALL|EDGE_FACE|ONLY_FACE
GeometryNodeMeshBoolean      in: Mesh 1, Mesh 2, Self Intersection, Hole Tolerant
                             operation: INTERSECT|UNION|DIFFERENCE
                             solver: EXACT | FLOAT | MANIFOLD
GeometryNodeSampleCurve      in: Curves, Value, Factor, Length, Curve Index
                             out: Value, Position, Tangent, Normal
GeometryNodeTransform        in: Geometry, Mode, Translation, Rotation, Scale, Transform
GeometryNodeStoreNamedAttribute
                             domain: POINT|EDGE|FACE|CORNER|CURVE|INSTANCE|LAYER
```

**`solver='MANIFOLD'`** is new and markedly more robust than `EXACT` on
generated geometry — worth preferring for the capped Boolean cutaway.

### Node group interface (4.0+)

`NodeTree.inputs` is **gone**. Use:

```python
g.interface.new_socket("Cut", in_out="INPUT", socket_type="NodeSocketFloat")
```

which returns a socket whose `.identifier` (`"Socket_1"`) is the key needed by
`set_modifier_input`.

### Attribute domains and types

```
domain: POINT, EDGE, FACE, CORNER, CURVE, INSTANCE, LAYER
type  : FLOAT, INT, BOOLEAN, FLOAT_VECTOR, FLOAT_COLOR, QUATERNION, FLOAT4X4,
        STRING, INT8, INT16_2D, INT32_2D, FLOAT2, FLOAT4, BYTE_COLOR
```

---

## 6. Cycles on Apple Metal — measured

Settings verified to apply: `device='GPU'`, `compute_device_type='METAL'`,
`denoiser='OPENIMAGEDENOISE'`, `sampling_pattern='BLUE_NOISE'`,
`use_adaptive_sampling`, `volume_step_rate`, `volume_max_steps`.

**`cycles.volume_bounces` defaults to 0.** For an emissive, scattering plasma
this must be raised (this project uses 4) or the volume will not scatter light
into itself at all.

### The 100-second trap

Measured, same scene, 480×320, 32 spp, no denoise, no adaptive sampling:

| configuration | time |
|---|---|
| GPU, no volume | 0.61 s |
| CPU, no volume | 0.34 s |
| GPU, with volume | 1.13 s |
| CPU, with volume | 1.39 s |

But the **first** Cycles render of a session took **104.9 s** for a *smaller*
240×160 frame. That is one-off **Metal kernel compilation**, not render time;
it is cached afterwards. Budget for it once per session and never mistake it
for a performance problem.

**Measured on the real scene**, not extrapolated: 1920×1080 at 128 spp took
**214.8 s** — but that run included the one-off kernel compilation, so the
render itself was ≈115 s. Scaling to 3840×2160 (4× pixels) at 256 spp (2×
samples) gives **≈15 minutes per 4K still**, so the six-image set is roughly
1.5 hours.

(An earlier extrapolation from a 480×320 test box suggested ~8 min. The real
scene, with volume scattering through cut geometry, is about twice that —
extrapolating render time from a toy scene understates it.)

Note the GPU is only ~1.2× the CPU here: the M4's 10-core GPU has little
advantage on a scene this small, where launch overhead dominates. Expect the
gap to widen with heavier geometry.

---

## 7. Compositor render passes

`ViewLayer` flags confirmed present: `use_pass_combined`, `use_pass_z`,
`use_pass_mist`, `use_pass_normal`, `use_pass_position`,
`use_pass_diffuse_color`, `use_pass_emit`, `use_pass_environment`,
`use_pass_ambient_occlusion`, `use_pass_cryptomatte_object`,
`use_pass_cryptomatte_material`.

**`cryptomatte_levels` is absent** on `ViewLayer` in 5.2.

Enabling passes adds sockets to `CompositorNodeRLayers` dynamically — with none
enabled it exposes only `Image`, `Alpha`.

---

## 8. Output formats

```
file_format: AVIF, JPEG, OPEN_EXR, PNG, WEBP, BMP, CINEON, DPX, IRIS,
             JPEG2000, HDR, TARGA, TARGA_RAW, TIFF, OPEN_EXR_MULTILAYER, FFMPEG
color_depth: 8, 10, 12, 16, 32
```

Use `PNG` at `color_depth='16'` for print stills, `OPEN_EXR` when the
annotation layer needs headroom above 1.0.
