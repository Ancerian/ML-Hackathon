"""Plasma geometry: flux surfaces, field lines, Poincare punctures, volume."""
from __future__ import annotations

import bpy
import numpy as np

from .bl import mesh_object, curve_object, points_object, instance_points, store_attribute
from .machine import load_meshes


def build_flux_surfaces(bake_dir, coll, mat, keep=None):
    """Nested flux surfaces, each carrying its psi_n as a vertex attribute."""
    out = []
    for key, v, f in load_meshes(f"{bake_dir}/flux_surfaces.npz"):
        s = float(key)
        if keep is not None and not keep(s):
            continue
        ob = mesh_object(f"FluxSurface_{s:.3f}", v, f, coll)
        store_attribute(ob.data, "psi_n", np.full(len(v), s, dtype=np.float32))
        ob.data.materials.append(mat)
        out.append(ob)
    return out


def build_plasma_volume(bake_dir, coll, mat):
    """The LCFS shell, used as the bounding mesh for the volume shader.

    A volume material needs a closed mesh to bound it.  The surface itself is
    left un-shaded (no Surface output in the plasma material), so only the
    volume contributes.
    """
    z = np.load(f"{bake_dir}/lcfs.npz")
    ob = mesh_object("PlasmaVolume", z["verts"], z["faces"], coll)
    ob.data.materials.append(mat)
    return ob


def build_fieldlines(bake_dir, coll, mat, bevel=0.012, stride=1):
    z = np.load(f"{bake_dir}/fieldlines.npz")
    xyz = z["xyz"]
    lines = [xyz[i] for i in range(0, xyz.shape[0], stride)]
    ob = curve_object("FieldLines", lines, coll, bevel=bevel)
    ob.data.materials.append(mat)
    return ob


def build_punctures(bake_dir, coll, mat, radius=0.018, plane_phi=0.0):
    """Poincare punctures as one point object on the phi = const plane."""
    z = np.load(f"{bake_dir}/punctures.npz")
    ob = points_object("PoincarePunctures", z["points"], coll,
                       attributes={"psi_n": z["psi_n"].astype(float)})
    instance_points(ob, radius=radius)
    ob.data.materials.append(mat)
    return ob


def build_punctures_classified(bake_dir, coll, materials, radius=0.018):
    """One puncture object per orbit class, so each can be coloured and toggled.

    ``materials`` maps a class name to a material.  Falls back to a single
    object if the bake predates classification.
    """
    z = np.load(f"{bake_dir}/punctures.npz")
    if "label_id" not in z:
        return {"all": build_punctures(bake_dir, coll,
                                       list(materials.values())[0], radius)}
    pts = z["points"]
    lab = z["label_id"]
    names = [str(x) for x in z["label_names"]] if "label_names" in z else         ["regular", "island", "chaotic", "escaped"]
    out = {}
    for i, nm in enumerate(names):
        sel = lab == i
        if not sel.any():
            continue
        ob = points_object(f"Poincare_{nm}", pts[sel], coll,
                           attributes={"psi_n": z["psi_n"][sel].astype(float)})
        mat = materials.get(nm, list(materials.values())[0])
        ob.data.materials.append(mat)
        instance_points(ob, radius=radius, name=f"TokPointInstancer_{nm}",
                        material=mat)
        out[nm] = ob
    return out


def build_poincare_plane(bake_dir, coll, mat=None, back=0.05):
    """Dark card marking the phi = 0 section, sitting just BEHIND the punctures.

    Without a material this renders as Blender's default grey and fills the
    frame, hiding the very points it is meant to back.
    """
    z = np.load(f"{bake_dir}/psi_n_grid.npz")
    r0, r1 = float(z["R_scaled"][0]), float(z["R_scaled"][-1])
    z0, z1 = float(z["Z_scaled"][0]), float(z["Z_scaled"][-1])
    y = float(back)
    verts = np.array([[r0, y, z0], [r1, y, z0], [r1, y, z1], [r0, y, z1]])
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    ob = mesh_object("PoincarePlane", verts, faces, coll, smooth=False)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob
