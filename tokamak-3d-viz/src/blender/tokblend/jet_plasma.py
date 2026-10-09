"""Plasma for the engineering scene: volume, section cards, physics extras.

Two sources, in order of preference:

1. a PHYSICS bake (``src/run_bake_jet_physics.py``) in ``<bake>/physics/`` or
   in the bake directory itself -- same file names as the DIII-D bake:
   ``psi_n_grid.npz`` (2-D ``psi_n`` + 1-D R, Z), ``lcfs.npz`` (closed mesh +
   ``profile``), ``flux_surfaces.npz`` (mesh archive, keys = psi_N),
   ``fieldlines.npz`` (``xyz`` (n, m, 3)), ``manifest.json`` (equilibrium).
   Detected, never required.
2. otherwise a SCHEMATIC psi_N proxy built from the bake's own LCFS
   (``components.plasma.info.profile_rz``, the Miller D fitted to Table I.3-1):
   psi_N = rho^2 with rho the self-similar scaling of the boundary about a
   point 0.05 a outboard of R0.  It is labelled as such (``tok_psi_source``).

The volume shader is :func:`tokblend.materials.plasma_volume` (the DIII-D one)
fed with whichever psi_N grid exists, so the look is shared.
"""
from __future__ import annotations

import glob
import math
import os

import bpy
import numpy as np

from .bl import mesh_object, store_attribute, curve_object
from . import materials as M
from .machine import load_meshes

_R_KEYS = ("R", "R_grid", "r", "R_scaled", "rgrid", "R_1d")
_Z_KEYS = ("Z", "Z_grid", "z", "Z_scaled", "zgrid", "Z_1d")


# ---------------------------------------------------------------------------
# psi_N sources
# ---------------------------------------------------------------------------
def find_physics_grid(bake_dir, component_files=()):
    """Return (psi_n (nz, nr), R (nr,), Z (nz,), path) or None."""
    for path in sorted(glob.glob(os.path.join(bake_dir, "*.npz"))):
        if os.path.basename(path) in component_files:
            continue
        try:
            z = np.load(path, allow_pickle=False)
        except Exception:
            continue
        if "psi_n" not in z.files or z["psi_n"].ndim != 2:
            continue
        R = next((z[k] for k in _R_KEYS if k in z.files and z[k].ndim == 1), None)
        Z = next((z[k] for k in _Z_KEYS if k in z.files and z[k].ndim == 1), None)
        if R is None or Z is None:
            continue
        g = np.asarray(z["psi_n"], float)
        if g.shape == (len(R), len(Z)):          # stored (nr, nz) -> transpose
            g = g.T
        if g.shape != (len(Z), len(R)):
            continue
        return g, np.asarray(R, float), np.asarray(Z, float), path
    return None


class ProxyPsi:
    """psi_N = rho^2, rho = self-similar scaling of the LCFS about an axis."""

    def __init__(self, profile_rz, R0, a, shift=0.05):
        p = np.asarray(profile_rz, float)
        self.ax = np.array([R0 + shift * a, 0.0])
        d = p - self.ax
        th = np.arctan2(d[:, 1], d[:, 0])
        rb = np.hypot(d[:, 0], d[:, 1])
        o = np.argsort(th)
        th, rb = th[o], rb[o]
        self.th = np.concatenate([th[-1:] - 2 * np.pi, th, th[:1] + 2 * np.pi])
        self.rb = np.concatenate([rb[-1:], rb, rb[:1]])
        self.profile = p

    def boundary_r(self, theta):
        return np.interp(np.mod(theta + np.pi, 2 * np.pi) - np.pi, self.th, self.rb)

    def rho(self, R, Z):
        dR, dZ = R - self.ax[0], Z - self.ax[1]
        return np.hypot(dR, dZ) / self.boundary_r(np.arctan2(dZ, dR))

    def grid(self, nr=160, nz=240, pad=0.15):
        p = self.profile
        R = np.linspace(p[:, 0].min() - pad, p[:, 0].max() + pad, nr)
        Z = np.linspace(p[:, 1].min() - pad, p[:, 1].max() + pad, nz)
        RR, ZZ = np.meshgrid(R, Z)
        return np.clip(self.rho(RR, ZZ) ** 2, 0.0, 1.6), R, Z


def _bilinear(grid, R, Z, r, z):
    fi = np.clip((r - R[0]) / (R[-1] - R[0]) * (len(R) - 1), 0, len(R) - 1.001)
    fj = np.clip((z - Z[0]) / (Z[-1] - Z[0]) * (len(Z) - 1), 0, len(Z) - 1.001)
    i, j = fi.astype(int), fj.astype(int)
    u, v = fi - i, fj - j
    return ((1 - u) * (1 - v) * grid[j, i] + u * (1 - v) * grid[j, i + 1]
            + (1 - u) * v * grid[j + 1, i] + u * v * grid[j + 1, i + 1])


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
class JetPlasma:
    def __init__(self, table, log=print, physics_fallback=None):
        """``physics_fallback``: another bake directory whose ``physics/``
        is used when this bake has none (the 1983 variant reuses the 1975
        equilibrium -- same plasma design; the -5.625 deg TF rotation only
        changes the ripple, not the axisymmetric equilibrium)."""
        self.table = table
        self.borrowed = None
        info = table.plasma_info or {}
        self.R0 = float(info.get("R0", 2.96))
        self.a = float(info.get("a", 1.25))
        self.kappa = float(info.get("kappa", 1.68))
        prof = info.get("profile_rz")
        if prof is None:                         # Miller D from the table values
            t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
            d = float(info.get("delta", 0.3))
            prof = np.stack([self.R0 + self.a * np.cos(t + np.arcsin(d) * np.sin(t)),
                             self.kappa * self.a * np.sin(t)], -1)
        self.proxy = ProxyPsi(prof, self.R0, self.a)
        self.surfaces, self.lines, self.volume_objs = [], [], []
        files = {c.get("file") for c in table.manifest.get("components", {}).values()}
        self.phys_dir = None
        self.phys_manifest = {}
        phys = None
        cands = [os.path.join(table.bake, "physics"), table.bake]
        if physics_fallback:
            cands.append(os.path.join(physics_fallback, "physics"))
        for d in cands:
            if os.path.isdir(d):
                phys = find_physics_grid(d, component_files=files)
                if phys is not None:
                    self.phys_dir = d
                    if not os.path.abspath(d).startswith(os.path.abspath(table.bake)):
                        self.borrowed = d
                    break
        if phys is not None:
            self.grid, self.R, self.Z, src = phys
            rel = os.path.relpath(src, table.bake)
            self.source = f"physics:{rel}" + (" (borrowed)" if self.borrowed else "")
            mp = os.path.join(self.phys_dir, "manifest.json")
            if os.path.exists(mp):
                import json
                self.phys_manifest = json.load(open(mp))
            lp = os.path.join(self.phys_dir, "lcfs.npz")
            if os.path.exists(lp):
                zl = np.load(lp)
                if "profile" in zl.files:           # section cards follow the real LCFS
                    eq = self.phys_manifest.get("equilibrium", {})
                    self.proxy = ProxyPsi(zl["profile"], self.R0, self.a,
                                          shift=(float(eq["r_axis"]) - self.R0) / self.a
                                          if "r_axis" in eq else 0.05)
        else:
            self.grid, self.R, self.Z = self.proxy.grid()
            self.source = "schematic_rho2_proxy"
        log(f"[plasma] psi_N source: {self.source} grid {self.grid.shape}")
        self.log = log

    def psi_at(self, r, z):
        if self.source.startswith("physics"):
            return _bilinear(self.grid, self.R, self.Z, r, z)
        return np.clip(self.proxy.rho(r, z) ** 2, 0, 1.6)

    def volume_material(self, density=0.3, emission=0.4, t_core=7000.0, t_edge=1600.0):
        mat = M.plasma_volume(f"JET_PlasmaVolume_{self.table.label}", self.grid,
                              (float(self.R[0]), float(self.R[-1])),
                              (float(self.Z[0]), float(self.Z[-1])),
                              density=density, emission=emission, turbulence=0.25,
                              t_core=t_core, t_edge=t_edge)
        mat["tok_psi_source"] = self.source
        return mat

    def physics_volume_object(self, coll, mat):
        """Closed physics LCFS mesh as the volume's bounding mesh (or None)."""
        if not self.phys_dir:
            return None
        lp = os.path.join(self.phys_dir, "lcfs.npz")
        if not os.path.exists(lp):
            return None
        z = np.load(lp)
        ob = mesh_object("PlasmaVolume_physics", z["verts"], z["faces"], coll)
        ob.data.materials.append(mat)
        ob["tok_psi_source"] = self.source
        ob["tok_subsystem"] = "Plasma"
        ob["tok_nocut"] = 1.0
        return ob

    def caption_note(self):
        if not self.source.startswith("physics"):
            return "плазма: схематичний psi_N = rho^2 від межі D (не розрахунок рівноваги)"
        eq = self.phys_manifest.get("equilibrium", {})
        bits = []
        if "Ip_A" in eq:
            bits.append(f"I_p = {eq['Ip_A'] / 1e6:.1f} МА".replace(".", ","))
        if "B0_vacuum_T" in eq:
            bits.append(f"B0 = {eq['B0_vacuum_T']:.2f} Тл".replace(".", ","))
        if "q95" in eq:
            bits.append(f"q95 = {eq['q95']:.2f}".replace(".", ","))
        note = ("плазма: аналітична рівновага Солов'єва (" + ", ".join(bits) +
                "), physics/manifest.json")
        if self.borrowed:
            note += " (та сама рівновага, що й для проекту 1975 р.)"
        return note

    # -- section cards ------------------------------------------------------------
    def _d_card(self, n_rho=28, n_th=192):
        th = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
        rb = self.proxy.boundary_r(th)
        ax = self.proxy.ax
        rho = np.linspace(0.0, 1.0, n_rho + 1)[1:]
        R = ax[0] + rho[:, None] * rb[None, :] * np.cos(th)[None, :]
        Z = ax[1] + rho[:, None] * rb[None, :] * np.sin(th)[None, :]
        RZ = np.concatenate([[ax], np.stack([R.ravel(), Z.ravel()], -1)])
        faces = []
        for j in range(n_th):                      # centre fan
            faces.append((0, 1 + j, 1 + (j + 1) % n_th))
        for i in range(n_rho - 1):
            b0, b1 = 1 + i * n_th, 1 + (i + 1) * n_th
            for j in range(n_th):
                jn = (j + 1) % n_th
                faces.append((b0 + j, b1 + j, b1 + jn, b0 + jn))
        return RZ, faces

    def elevation_cards(self, coll, mat, phi, eps=0.004):
        """Poloidal D-sections on the plane through the axis at azimuth ``phi``
        (both halves: phi and phi + 180 deg), nudged ``eps`` behind the plane."""
        RZ, faces = self._d_card()
        psi = self.psi_at(RZ[:, 0], RZ[:, 1])
        out = []
        n = np.array([-math.sin(phi), math.cos(phi), 0.0])
        for k, ph in enumerate((phi, phi + math.pi)):
            v = np.stack([RZ[:, 0] * math.cos(ph), RZ[:, 0] * math.sin(ph), RZ[:, 1]], -1)
            v = v - eps * n
            ob = mesh_object(f"PlasmaSection_RZ_{k}", v, faces, coll, smooth=True)
            store_attribute(ob.data, "psi_n", np.clip(psi, 0, 1).astype(np.float32))
            ob.data.materials.append(mat)
            ob["tok_psi_source"] = self.source
            out.append(ob)
        return out

    def plan_card(self, coll, mat, z=0.0, n_r=40, n_phi=256):
        """Midplane annulus of the plasma, psi_N-coloured."""
        rb_out = self.proxy.ax[0] + self.proxy.boundary_r(np.array([0.0]))[0]
        rb_in = self.proxy.ax[0] - self.proxy.boundary_r(np.array([np.pi]))[0]
        r = np.linspace(rb_in, rb_out, n_r)
        ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        v = np.stack([(r[None, :] * np.cos(ph)[:, None]).ravel(),
                      (r[None, :] * np.sin(ph)[:, None]).ravel(),
                      np.full(n_r * n_phi, z)], -1)
        faces = []
        for i in range(n_phi):
            i2 = (i + 1) % n_phi
            for j in range(n_r - 1):
                faces.append((i * n_r + j, i * n_r + j + 1, i2 * n_r + j + 1, i2 * n_r + j))
        psi = np.tile(self.psi_at(r, np.zeros_like(r)), n_phi)
        ob = mesh_object("PlasmaSection_Plan", v, faces, coll, smooth=True)
        store_attribute(ob.data, "psi_n", np.clip(psi, 0, 1).astype(np.float32))
        ob.data.materials.append(mat)
        ob["tok_psi_source"] = self.source
        return ob

    # -- physics extras -----------------------------------------------------------
    def physics_extras(self, coll_plasma, coll_lines, flux_mat, line_mat, bevel=0.012):
        """Flux surfaces / field lines, only if the physics bake wrote them."""
        made = {"flux_surfaces": 0, "field_lines": 0}
        self.surfaces, self.lines = [], []
        base = self.phys_dir or self.table.bake
        fs = os.path.join(base, "flux_surfaces.npz")
        if os.path.exists(fs):
            try:
                for key, v, f in load_meshes(fs):
                    try:
                        s = float(key)
                    except ValueError:
                        s = 0.5
                    if s >= 0.999:                 # the LCFS itself: volume covers it
                        continue
                    ob = mesh_object(f"FluxSurface_{s:.3f}", v, f, coll_plasma)
                    ob["tok_subsystem"] = "Plasma"
                    ob["tok_role"] = "flux_surface"
                    self.surfaces.append(ob)
                    store_attribute(ob.data, "psi_n", np.full(len(v), s, np.float32))
                    ob.data.materials.append(flux_mat)
                    ob["tok_psi_source"] = self.source
                    made["flux_surfaces"] += 1
            except Exception as exc:                        # format drift
                self.log(f"[plasma] flux_surfaces.npz not understood: {exc}")
        fl = os.path.join(base, "fieldlines.npz")
        if os.path.exists(fl):
            try:
                z = np.load(fl, allow_pickle=False)
                if "xyz" in z.files:
                    xyz = z["xyz"]
                    lines = [xyz[i] for i in range(xyz.shape[0])]
                else:
                    lines = [z[k] for k in sorted(z.files)
                             if z[k].ndim == 2 and z[k].shape[1] == 3]
                if lines:
                    ob = curve_object("FieldLines", lines, coll_lines, bevel=bevel)
                    ob["tok_nocut"] = 1.0
                    self.lines.append(ob)
                    ob.data.materials.append(line_mat)
                    made["field_lines"] = len(lines)
            except Exception as exc:
                self.log(f"[plasma] fieldlines.npz not understood: {exc}")
        self.log(f"[plasma] physics extras: {made}")
        return made
