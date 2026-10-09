"""Generic builders for components added by delta variants (components_add).

* ``rect_coil_set`` -- axisymmetric rectangular coils.  params: ``coils``: list
  of ``{name?, R, Z, dR|width, dZ|height}`` (m), optional ``material``.
* ``revolved_profile`` -- a closed (R, Z) polygon revolved 360 deg.  params:
  ``points: [[R, Z], ...]`` or ``profile: <csv name>`` (+ ``part``), optional
  ``material``.

Values come from the component's own params; its provenance is the
``components_add`` entry (source/page/confidence), recorded in the manifest.
"""
from __future__ import annotations

import numpy as np

from ..geom import solid_of_revolution
from .base import Part, finish


def build_rect_coil_set(tr, spec):
    coils = tr.p("coils", [])
    mat = tr.p("material", "copper_epoxy")
    n_tor = int(tr.p("n_tor", 192))
    parts = []
    for i, c in enumerate(coils):
        R, Z = float(c["R"]), float(c["Z"])
        dR = float(c.get("dR", c.get("width")))
        dZ = float(c.get("dZ", c.get("height")))
        poly = np.array([[R - dR / 2, Z - dZ / 2], [R + dR / 2, Z - dZ / 2],
                         [R + dR / 2, Z + dZ / 2], [R - dR / 2, Z + dZ / 2]])
        nm = str(c.get("name", f"coil_{i}"))
        parts.append(Part(nm, solid_of_revolution(poly, n_tor=n_tor), mat, np.eye(4)[None],
                          [nm], meta={"closed_solid": True, "R": R, "Z": Z, "dR": dR, "dZ": dZ}))
    return finish(tr, parts, {"n_coils": len(parts)})


def build_revolved_profile(tr, spec):
    pts = tr.p("points")
    if pts is None:
        prof = tr.profile(tr.p("profile"), tr.p("part"))
        pts = np.asarray(prof, float)
    mat = tr.p("material", "generic")
    m = solid_of_revolution(np.asarray(pts, float), n_tor=int(tr.p("n_tor", 192)))
    return finish(tr, [Part(spec.name, m, mat, np.eye(4)[None], [spec.name],
                            meta={"closed_solid": True, "axisymmetric": True})])


def build_limiter_module(tr, spec):
    """Discrete limiter module (e.g. the 1983 outer-midplane modules): front face
    at R_front, toroidally curved (a sector of revolution), width x height from
    params, radial depth from params['thickness'] or the kernel default."""
    from ..geom import rot_z
    R = float(tr.p("R_front"))
    w = float(tr.p("width_toroidal"))
    h = float(tr.p("height_poloidal"))
    zc = float(tr.p("Z_center", 0.0))
    phi = np.deg2rad(float(tr.p("phi_deg", 0.0)))
    t = tr.p("thickness")
    t = float(t) if t is not None else float(tr.k("limiter_module_thickness"))
    mat = str(tr.p("material", "limiter"))
    poly = np.array([[R, zc - h / 2], [R + t, zc - h / 2], [R + t, zc + h / 2], [R, zc + h / 2]])
    dphi = w / R
    m = solid_of_revolution(poly, n_tor=9, phi0=-dphi / 2, phi1=dphi / 2)
    return finish(tr, [Part(spec.name, m, mat, rot_z(phi)[None], [spec.name],
                            meta={"closed_solid": True, "R_front": R, "phi_deg": float(np.rad2deg(phi))})],
                  {"assumed_params": tr.spec.params.get("assumed_params", [])})
