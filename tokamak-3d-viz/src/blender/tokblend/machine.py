"""Procedural reactor-class machine, built from the bake.

Nothing is imported from CAD: Blender has no native STEP/IGES reader in any 4.x
or 5.x release, and this project has no CAD anyway.  Every part is generated
from the measured vessel profile and the real 19-coil table in the shot file,
so the geometry is reproducible and re-drivable from a different discharge.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from .bl import collection, mesh_object
from . import materials as M


def _revolve(profile, n_tor, phi0=0.0, phi1=2 * math.pi, closed=True):
    prof = np.asarray(profile, float)
    n_pol = prof.shape[0]
    phis = (np.linspace(phi0, phi1, n_tor, endpoint=False) if closed
            else np.linspace(phi0, phi1, n_tor))
    c, s = np.cos(phis), np.sin(phis)
    X = prof[:, 0][None, :] * c[:, None]
    Y = prof[:, 0][None, :] * s[:, None]
    Z = np.repeat(prof[:, 1][None, :], phis.size, axis=0)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    n_seg = n_tor if closed else n_tor - 1
    i = np.arange(n_seg)[:, None]
    j = np.arange(n_pol)[None, :]
    faces = np.stack([(i * n_pol + j).ravel(),
                      (((i + 1) % n_tor) * n_pol + j).ravel(),
                      (((i + 1) % n_tor) * n_pol + (j + 1) % n_pol).ravel(),
                      (i * n_pol + (j + 1) % n_pol).ravel()], axis=-1)
    return verts, faces


def build_vessel(bake_dir, coll, mat):
    z = np.load(f"{bake_dir}/vessel.npz")
    ob = mesh_object("VacuumVessel", z["verts"], z["faces"], coll)
    ob.data.materials.append(mat)
    return ob, z["profile"]


def load_meshes(npz_path):
    """Yield ``(key, verts, faces)`` from a mesh archive written by the bake."""
    z = np.load(npz_path, allow_pickle=False)
    if "keys" in z:
        keys = z["keys"]
        n = len(keys)
    else:
        n = int(z["count"][0])
        keys = z["labels"] if "labels" in z else np.arange(n)
    for i in range(n):
        yield keys[i], z[f"v_{i:03d}"], z[f"f_{i:03d}"]


def build_pf_coils(bake_dir, coll, mat):
    out = []
    for key, v, f in load_meshes(f"{bake_dir}/pf_coils.npz"):
        ob = mesh_object(f"PF_{key}", v, f, coll, smooth=False)
        ob.data.materials.append(mat)
        out.append(ob)
    return out


def build_tf_coils(bake_dir, coll, mat, n_coils=18, width=0.55, depth=0.32):
    """Discrete TF coils on a D cross-section, evenly spaced in phi."""
    prof = np.load(f"{bake_dir}/tf_profile.npz")["profile"]
    out = []
    for i in range(n_coils):
        phi_c = 2 * math.pi * i / n_coils
        half = 0.5 * width / max(float(np.mean(prof[:, 0])), 1e-6)
        v, f = _revolve(prof, 4, phi0=phi_c - half, phi1=phi_c + half, closed=False)
        ob = mesh_object(f"TF_{i:02d}", v, f, coll, smooth=False)
        sol = ob.modifiers.new("Solidify", "SOLIDIFY")
        sol.thickness = depth
        sol.offset = 0.0
        bev = ob.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.04
        bev.segments = 2
        bev.limit_method = "ANGLE"
        ob.data.materials.append(mat)
        out.append(ob)
    return out


def build_cryostat(profile, coll, mat, clearance=2.1, n_tor=96):
    """Outer cryostat: a smoothed, expanded envelope of the machine."""
    prof = np.asarray(profile, float)
    r0 = float(prof[:, 0].mean())
    z0 = float(prof[:, 1].mean())
    d = np.stack([prof[:, 0] - r0, prof[:, 1] - z0], axis=-1)
    rad = np.hypot(d[:, 0], d[:, 1])
    rmax = float(rad.max()) + clearance
    t = np.linspace(0, 2 * math.pi, 96, endpoint=False)
    # rounded-rectangle cross-section, capped in R at the machine bore
    rr = r0 + rmax * np.cos(t) * 1.02
    zz = z0 + rmax * np.sin(t) * 1.12
    rr = np.clip(rr, 0.25 * r0, None)
    v, f = _revolve(np.stack([rr, zz], axis=-1), n_tor)
    ob = mesh_object("Cryostat", v, f, coll)
    ob.data.materials.append(mat)
    return ob


def build_divertor(profile, coll, mat, n_tor=96, n_cassette=54):
    """Divertor cassette ring at the bottom of the vessel."""
    prof = np.asarray(profile, float)
    k = int(np.argmin(prof[:, 1]))
    lo = prof[(k - 14) % len(prof):(k + 15) % len(prof)] if k > 14 else prof[:29]
    if lo.shape[0] < 4:
        lo = prof[:29]
    inner = lo.copy()
    outer = lo.copy()
    outer[:, 1] -= 0.30
    ring = np.vstack([inner, outer[::-1]])
    v, f = _revolve(ring, n_tor)
    ob = mesh_object("Divertor", v, f, coll, smooth=False)
    ob.data.materials.append(mat)
    return ob


def build_tiles(profile, coll, mat, n_tor=140, n_pol_group=3, gap=0.012):
    """First-wall armour tiles: the wall profile split into discrete panels.

    Modelled as a separate shell just inside the vessel so the tile grid reads
    at close range without disturbing the vessel mesh itself.
    """
    prof = np.asarray(profile, float)
    n_pol = prof.shape[0]
    inner = prof * 0.995
    verts = []
    faces = []
    phis = np.linspace(0, 2 * math.pi, n_tor, endpoint=False)
    dphi = (phis[1] - phis[0]) * (1.0 - gap)
    vi = 0
    for i, ph in enumerate(phis):
        for j in range(0, n_pol - n_pol_group, n_pol_group):
            a, b = j, min(j + n_pol_group, n_pol - 1)
            quad_rz = inner[a:b + 1]
            for s_off in (0.0, dphi):
                c, s = math.cos(ph + s_off), math.sin(ph + s_off)
                for rz in quad_rz:
                    verts.append((rz[0] * c, rz[0] * s, rz[1]))
            m = len(quad_rz)
            for t in range(m - 1):
                faces.append((vi + t, vi + t + 1, vi + m + t + 1, vi + m + t))
            vi += 2 * m
    ob = mesh_object("ArmourTiles", np.array(verts), np.array(faces, dtype=np.int32),
                     coll, smooth=False)
    ob.data.materials.append(mat)
    return ob


def build_ports(profile, coll, mat, n_ports=12, r_port=0.75, length=3.0):
    """Equatorial diagnostic ports as cylinders through the vessel wall."""
    prof = np.asarray(profile, float)
    r_out = float(prof[:, 0].max())
    z_mid = float(prof[:, 1].mean())
    out = []
    for i in range(n_ports):
        ph = 2 * math.pi * (i + 0.5) / n_ports
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=r_port, depth=length,
            location=((r_out + length * 0.35) * math.cos(ph),
                      (r_out + length * 0.35) * math.sin(ph), z_mid),
            rotation=(0.0, math.pi / 2, ph))
        ob = bpy.context.active_object
        ob.name = f"Port_{i:02d}"
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        coll.objects.link(ob)
        ob.data.materials.append(mat)
        out.append(ob)
    return out
