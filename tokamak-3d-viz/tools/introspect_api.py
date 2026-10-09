"""Headless Blender API introspection.

Dumps every node id, socket name and enum this project relies on, so the
reference in docs/API-VERIFIED-5.2.md can be regenerated against whatever
Blender build you actually have.

    blender --background --factory-startup --python tools/introspect_api.py

Writes api_dump.txt next to this script.  Read-only with respect to the
project: it creates datablocks in memory and never saves a .blend.

Note: with --factory-startup the Cycles add-on is NOT enabled, so the engine
enum and the cycles.* enums come back empty.  tools/introspect_cycles.py
enables it first.
"""
import bpy, sys

import os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_dump.txt")
_fh = open(LOG, "w")
OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    _fh.write(s + "\n")
    _fh.flush()

p("=== VERSION ===")
p("bpy.app.version_string:", bpy.app.version_string)
p("bpy.app.version:", bpy.app.version)
import numpy
p("numpy in blender:", numpy.__version__)

p("\n=== RENDER ENGINE ENUM ===")
for it in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items:
    p("  engine:", repr(it.identifier), "|", it.name)

p("\n=== CYCLES DEVICE ===")
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for it in prefs.bl_rna.properties['compute_device_type'].enum_items:
        p("  compute_device_type:", repr(it.identifier))
    cs = bpy.types.Scene.bl_rna.properties.get('cycles')
except Exception as e:
    p("  cycles prefs err:", e)
try:
    import _cycles
    p("  cycles addon present: True")
except Exception as e:
    p("  _cycles:", e)

p("\n=== PRINCIPLED BSDF v2 SOCKETS ===")
mat = bpy.data.materials.new("PROBE")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
p("  bl_idname:", bsdf.bl_idname)
for i, s in enumerate(bsdf.inputs):
    dv = ""
    try:
        dv = str(round(s.default_value, 4)) if isinstance(s.default_value, float) else ""
    except Exception:
        pass
    p(f"  [{i:2d}] {s.name!r:28s} type={s.type:8s} {dv}")
p("  OUTPUTS:", [s.name for s in bsdf.outputs])
for prop in ('distribution', 'subsurface_method'):
    pr = bsdf.bl_rna.properties.get(prop)
    if pr:
        p(f"  {prop}:", [e.identifier for e in pr.enum_items])

p("\n=== MATERIAL SETTINGS (EEVEE) ===")
for prop in ('surface_render_method', 'blend_method', 'shadow_method',
             'use_backface_culling', 'use_transparent_shadow',
             'use_raytrace_refraction', 'displacement_method', 'thickness_mode'):
    pr = mat.bl_rna.properties.get(prop)
    if pr is None:
        p(f"  {prop}: ABSENT")
    elif pr.type == 'ENUM':
        p(f"  {prop}: ENUM", [e.identifier for e in pr.enum_items])
    else:
        p(f"  {prop}: {pr.type} default={getattr(mat, prop, None)!r}")

p("\n=== NOISE / VORONOI / MUSGRAVE ===")
def probe(idname, label):
    try:
        n = nt.nodes.new(idname)
    except Exception as e:
        p(f"  {label} ({idname}): MISSING -> {e}")
        return None
    p(f"  {label} ({idname}): OK")
    p(f"    inputs : {[s.name for s in n.inputs]}")
    p(f"    outputs: {[s.name for s in n.outputs]}")
    for prop in ('noise_type', 'noise_dimensions', 'normalize', 'feature',
                 'distance', 'voronoi_dimensions', 'musgrave_type',
                 'color_ramp', 'operation', 'data_type', 'clamp_type',
                 'interpolation', 'gradient_type', 'wave_type', 'bands_direction',
                 'wave_profile', 'component', 'mode', 'domain', 'blend_type'):
        pr = n.bl_rna.properties.get(prop)
        if pr is None:
            continue
        if pr.type == 'ENUM':
            p(f"    {prop}: {[e.identifier for e in pr.enum_items]}")
        else:
            p(f"    {prop}: {pr.type}")
    return n

probe('ShaderNodeTexNoise', 'Noise Texture')
probe('ShaderNodeTexMusgrave', 'Musgrave (should be GONE in 4.1+)')
probe('ShaderNodeTexVoronoi', 'Voronoi Texture')
probe('ShaderNodeTexWave', 'Wave Texture')
probe('ShaderNodeValToRGB', 'Color Ramp')
probe('ShaderNodeMapRange', 'Map Range')
probe('ShaderNodeMath', 'Math')
probe('ShaderNodeVectorMath', 'Vector Math')
probe('ShaderNodeMixRGB', 'MixRGB (legacy)')
probe('ShaderNodeMix', 'Mix (new)')
probe('ShaderNodeBump', 'Bump')
probe('ShaderNodeNewGeometry', 'Geometry (shader input)')
probe('ShaderNodeTexCoord', 'Texture Coordinate')
probe('ShaderNodeMapping', 'Mapping')
probe('ShaderNodeEmission', 'Emission')
probe('ShaderNodeVolumePrincipled', 'Principled Volume')
probe('ShaderNodeVolumeScatter', 'Volume Scatter')
probe('ShaderNodeVolumeAbsorption', 'Volume Absorption')
probe('ShaderNodeBsdfTransparent', 'Transparent BSDF')
probe('ShaderNodeMixShader', 'Mix Shader')
probe('ShaderNodeAttribute', 'Attribute')
probe('ShaderNodeBevel', 'Bevel (Cycles-only)')
probe('ShaderNodeAmbientOcclusion', 'AO (Cycles-only)')
probe('ShaderNodeLayerWeight', 'Layer Weight')
probe('ShaderNodeBlackbody', 'Blackbody')
probe('ShaderNodeObjectInfo', 'Object Info')
probe('ShaderNodeSeparateXYZ', 'Separate XYZ')
probe('ShaderNodeCombineXYZ', 'Combine XYZ')
probe('ShaderNodeClamp', 'Clamp')
probe('ShaderNodeFloatCurve', 'Float Curve')
probe('ShaderNodeRGBCurve', 'RGB Curves')

p("\n=== VOLUME PRINCIPLED DETAIL ===")
vp = nt.nodes.get('Principled Volume')
if vp:
    for s in vp.inputs:
        p(f"  in  {s.name!r:24s} {s.type}")
    p("  out:", [s.name for s in vp.outputs])

p("\n=== SCENE EEVEE PROPS ===")
ee = bpy.context.scene.eevee
names = sorted(pr.identifier for pr in ee.bl_rna.properties if pr.identifier != 'rna_type')
p("  " + ", ".join(names))

p("\n=== GEOMETRY NODES: node id probe ===")
gn = bpy.data.node_groups.new("PROBE_GN", 'GeometryNodeTree')
GN_IDS = [
 'NodeGroupInput','NodeGroupOutput',
 'GeometryNodeMeshCircle','GeometryNodeMeshLine','GeometryNodeMeshGrid',
 'GeometryNodeCurvePrimitiveCircle','GeometryNodeCurvePrimitiveLine',
 'GeometryNodeCurveToMesh','GeometryNodeCurveToPoints','GeometryNodeMeshToCurve',
 'GeometryNodeMeshToPoints','GeometryNodePointsToVertices',
 'GeometryNodeFillCurve','GeometryNodeResampleCurve','GeometryNodeTrimCurve',
 'GeometryNodeSetCurveRadius','GeometryNodeSetCurveTilt','GeometryNodeCurveSpiral',
 'GeometryNodeInstanceOnPoints','GeometryNodeRealizeInstances',
 'GeometryNodeRotateInstances','GeometryNodeScaleInstances','GeometryNodeTranslateInstances',
 'GeometryNodeTransform','GeometryNodeSetPosition','GeometryNodeSetMaterial',
 'GeometryNodeDeleteGeometry','GeometryNodeSeparateGeometry','GeometryNodeJoinGeometry',
 'GeometryNodeMeshBoolean','GeometryNodeDualMesh','GeometryNodeSubdivideMesh',
 'GeometryNodeSubdivisionSurface','GeometryNodeExtrudeMesh','GeometryNodeFlipFaces',
 'GeometryNodeStoreNamedAttribute','GeometryNodeInputNamedAttribute',
 'GeometryNodeCaptureAttribute','GeometryNodeAttributeStatistic',
 'GeometryNodeInputPosition','GeometryNodeInputNormal','GeometryNodeInputIndex',
 'GeometryNodeInputID','GeometryNodeInputRadius','GeometryNodeInputMeshEdgeAngle',
 'GeometryNodeSampleCurve','GeometryNodeSampleIndex','GeometryNodeSampleNearest',
 'GeometryNodeSampleNearestSurface','GeometryNodeProximity',
 'GeometryNodeSimulationInput','GeometryNodeSimulationOutput',
 'GeometryNodeRepeatInput','GeometryNodeRepeatOutput',
 'GeometryNodeForeachGeometryElementInput','GeometryNodeForeachGeometryElementOutput',
 'GeometryNodeDistributePointsOnFaces','GeometryNodeDistributePointsInVolume',
 'GeometryNodePointsToVolume','GeometryNodeVolumeToMesh','GeometryNodeVolumeCube',
 'GeometryNodeSetShadeSmooth','GeometryNodeSetID',
 'GeometryNodeObjectInfo','GeometryNodeCollectionInfo','GeometryNodeSelfObject',
 'GeometryNodeSwitch','GeometryNodeIndexSwitch','GeometryNodeMenuSwitch',
 'GeometryNodeBoundBox','GeometryNodeConvexHull',
 'GeometryNodeMergeByDistance','GeometryNodeScaleElements',
 'ShaderNodeMath','ShaderNodeVectorMath','ShaderNodeMapRange','ShaderNodeValToRGB',
 'ShaderNodeMix','ShaderNodeSeparateXYZ','ShaderNodeCombineXYZ','ShaderNodeClamp',
 'ShaderNodeTexNoise','ShaderNodeTexWhiteNoise','ShaderNodeFloatCurve',
 'FunctionNodeRandomValue','FunctionNodeAlignRotationToVector',
 'FunctionNodeAlignEulerToVector','FunctionNodeRotationToAxisAngle',
 'FunctionNodeAxisAngleToRotation','FunctionNodeRotateVector',
 'FunctionNodeCompare','FunctionNodeBooleanMath','FunctionNodeInputVector',
 'FunctionNodeInputInt','FunctionNodeInputBool','FunctionNodeQuaternionToRotation',
 'GeometryNodeInputSceneTime','GeometryNodeSetPointRadius',
]
missing, present = [], []
for nid in GN_IDS:
    try:
        n = gn.nodes.new(nid)
        present.append(nid)
    except Exception:
        missing.append(nid)
p("  PRESENT (%d): %s" % (len(present), ", ".join(present)))
p("  MISSING (%d): %s" % (len(missing), ", ".join(missing) or "none"))

p("\n=== KEY GN NODE SOCKET DETAIL ===")
for nid in ['GeometryNodeDeleteGeometry','GeometryNodeSeparateGeometry','GeometryNodeMeshBoolean',
            'GeometryNodeInstanceOnPoints','GeometryNodeCurveToMesh','GeometryNodeSampleCurve',
            'GeometryNodeStoreNamedAttribute','GeometryNodeCaptureAttribute',
            'GeometryNodeSimulationInput','GeometryNodeSimulationOutput',
            'GeometryNodeDistributePointsInVolume','GeometryNodeMeshToPoints',
            'FunctionNodeRandomValue','GeometryNodeSetMaterial','GeometryNodeTransform',
            'GeometryNodeCurveSpiral','GeometryNodeSetCurveRadius']:
    n = next((x for x in gn.nodes if x.bl_idname == nid), None)
    if n is None:
        p(f"  {nid}: ABSENT"); continue
    p(f"  {nid}")
    p(f"    in : {[s.name for s in n.inputs]}")
    p(f"    out: {[s.name for s in n.outputs]}")
    for prop in ('domain','mode','operation','data_type','component','solver',
                 'rotation_type','input_type','data_type_item'):
        pr = n.bl_rna.properties.get(prop)
        if pr is not None and pr.type == 'ENUM':
            p(f"    {prop}: {[e.identifier for e in pr.enum_items]}")

p("\n=== ATTRIBUTE ENUMS ===")
me = bpy.data.meshes.new("PROBEMESH")
a = me.attributes
p("  domain:", [e.identifier for e in a.bl_rna.functions['new'].parameters['domain'].enum_items])
p("  type  :", [e.identifier for e in a.bl_rna.functions['new'].parameters['type'].enum_items])

p("\n=== MODIFIER TYPES ===")
p("  " + ", ".join(e.identifier for e in bpy.types.Modifier.bl_rna.properties['type'].enum_items))

p("\n=== COLOR MANAGEMENT ===")
vs = bpy.context.scene.view_settings
p("  view_transform:", [e.identifier for e in vs.bl_rna.properties['view_transform'].enum_items])
p("  look:", [e.identifier for e in vs.bl_rna.properties['look'].enum_items][:14], "...")

p("\n=== COMPOSITOR ACCESS (5.x moved this) ===")
sc = bpy.context.scene
for cand in ('node_tree', 'compositing_node_group', 'use_nodes'):
    pr = sc.bl_rna.properties.get(cand)
    p(f"  Scene.{cand}: {'ABSENT' if pr is None else pr.type}")
cnt = None
try:
    cnt = bpy.data.node_groups.new("PROBE_COMP", 'CompositorNodeTree')
    if sc.bl_rna.properties.get('compositing_node_group'):
        sc.compositing_node_group = cnt
    p("  created CompositorNodeTree via bpy.data.node_groups: OK")
except Exception as e:
    p("  CompositorNodeTree creation failed:", e)

if cnt is not None:
    for nid in ['CompositorNodeRLayers','CompositorNodeComposite','CompositorNodeViewer',
                'CompositorNodeGlare','CompositorNodeMixRGB','CompositorNodeCryptomatteV2',
                'CompositorNodeAlphaOver','CompositorNodeTonemap','CompositorNodeLensdist',
                'CompositorNodeColorBalance','CompositorNodeExposure','CompositorNodeDenoise',
                'CompositorNodeSetAlpha','CompositorNodeIDMask','CompositorNodeOutputFile',
                'NodeGroupOutput','NodeGroupInput']:
        try:
            n = cnt.nodes.new(nid)
            extra = ""
            for pn in ('glare_type','quality','tonemap_type','filter_type'):
                pr = n.bl_rna.properties.get(pn)
                if pr is not None and pr.type == 'ENUM':
                    extra += f" {pn}={[e.identifier for e in pr.enum_items]}"
            p(f"  {nid}: OK{extra}")
        except Exception as e:
            p(f"  {nid}: MISSING ({e})")

p("\n=== VIEW LAYER PASSES ===")
vl = bpy.context.view_layer
for pn in ('use_pass_combined','use_pass_z','use_pass_mist','use_pass_normal',
           'use_pass_position','use_pass_diffuse_color','use_pass_emit',
           'use_pass_environment','use_pass_ambient_occlusion',
           'use_pass_cryptomatte_object','use_pass_cryptomatte_material',
           'cryptomatte_levels'):
    pr = vl.bl_rna.properties.get(pn)
    p(f"  ViewLayer.{pn}: {'ABSENT' if pr is None else pr.type}")

p("\n=== LIGHT / WORLD ===")
lt = bpy.data.lights.new("PROBELIGHT", type='AREA')
p("  light types:", [e.identifier for e in bpy.types.Light.bl_rna.properties['type'].enum_items])
p("  area shapes:", [e.identifier for e in bpy.types.AreaLight.bl_rna.properties['shape'].enum_items])

p("\n=== CYCLES SCENE PROPS (subset) ===")
try:
    cy = bpy.context.scene.cycles
    for pn in ('device','feature_set','samples','preview_samples','use_adaptive_sampling',
               'adaptive_threshold','use_denoising','denoiser','denoising_input_passes',
               'volume_step_rate','volume_max_steps','volume_bounces','max_bounces',
               'transmission_bounces','transparent_max_bounces','caustics_reflective'):
        pr = cy.bl_rna.properties.get(pn)
        if pr is None:
            p(f"  cycles.{pn}: ABSENT")
        elif pr.type == 'ENUM':
            p(f"  cycles.{pn}: {[e.identifier for e in pr.enum_items]}")
        else:
            p(f"  cycles.{pn}: {pr.type} default={getattr(cy, pn, None)!r}")
except Exception as e:
    p("  cycles scene props err:", e)

_fh.close()
print("\n".join(OUT))
