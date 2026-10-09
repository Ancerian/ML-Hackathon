"""Poloidal-field coils: coil 1 (stack of 10) + coils 2-4 (up/down pairs).

Rectangular winding packs revolved about the machine axis; one mesh per coil
type, positions as instance translations (coil 1 x10, others x2).
"""
from __future__ import annotations

import numpy as np

from ..geom import solid_of_revolution, translate, merge
from .base import Part, finish


def _ring(r_in, r_out, z0, z1, n_tor):
    return solid_of_revolution(np.array([[r_in, z0], [r_out, z0], [r_out, z1], [r_in, z1]]),
                               n_tor=n_tor)


def pf_packs(tr) -> dict:
    """Winding-pack rectangles {name: dict(R_in, R_out, dz_blocks, Z_list)}."""
    tr.v("pf_coils.n_pf_coils_total")
    # coil 1: drawing Fig. IV.4-13 (outer d2170, radial build 327, pitch 497)
    r1_out = tr.v("pf_coils.coil1_outer_diameter") / 2
    r1_in = r1_out - tr.v("pf_coils.coil1_winding_radial")
    pitch = tr.v("pf_coils.coil1_pitch_height")
    n1 = int(tr.v("pf_coils.coil1_count"))
    h1 = pitch - float(tr.k("pf_coil1_interpack_gap"))
    tr.v("pf_coils.coil1_mean_diameter"), tr.v("pf_coils.coil1_section_area")
    z1 = (np.arange(n1) - (n1 - 1) / 2) * pitch
    # coil 2: table R and Z, section assumed square
    r2 = tr.v("pf_coils.coil2_mean_diameter") / 2
    s2 = tr.v("pf_coils.coil2_section_assumed")
    z2 = tr.v("pf_coils.coil2_Z")
    # coil 3: drawing R3970, 388 x 448
    r3 = tr.v("pf_coils.coil3_R")
    w3, h3 = tr.v("pf_coils.coil3_width"), tr.v("pf_coils.coil3_height")
    z3 = tr.v("pf_coils.coil3_Z")
    tr.v("pf_coils.coil3_mean_diameter")
    # coil 4: drawing R5242, width 264, two 430 blocks over 1080 overall
    r4 = tr.v("pf_coils.coil4_R")
    w4 = tr.v("pf_coils.coil4_width")
    hb4, H4 = tr.v("pf_coils.coil4_block_height"), tr.v("pf_coils.coil4_overall_height")
    z4 = tr.v("pf_coils.coil4_Z")
    tr.v("pf_coils.coil4_mean_diameter")
    for k in (2, 3, 4):
        tr.v(f"pf_coils.coil{k}_count")
    return {
        "coil1": dict(R=(r1_in, r1_out), blocks=[(-h1 / 2, h1 / 2)], Z=list(z1)),
        "coil2": dict(R=(r2 - s2[0] / 2, r2 + s2[0] / 2), blocks=[(-s2[1] / 2, s2[1] / 2)],
                      Z=[+z2, -z2]),
        "coil3": dict(R=(r3 - w3 / 2, r3 + w3 / 2), blocks=[(-h3 / 2, h3 / 2)], Z=[+z3, -z3]),
        "coil4": dict(R=(r4 - w4 / 2, r4 + w4 / 2),
                      blocks=[(-H4 / 2, -H4 / 2 + hb4), (H4 / 2 - hb4, H4 / 2)],
                      Z=[+z4, -z4]),
    }


def build_pf(tr, spec):
    n_tor = int(tr.p("n_tor", 192))
    packs = pf_packs(tr)
    parts = []
    for name, pk in packs.items():
        m = merge([_ring(pk["R"][0], pk["R"][1], b0, b1, n_tor) for b0, b1 in pk["blocks"]])
        T = np.stack([translate(z=z) for z in pk["Z"]])
        parts.append(Part(name, m, "copper_epoxy", T,
                          [f"{name}_{i}" for i in range(len(pk["Z"]))],
                          meta={"closed_solid": True, "R_in": pk["R"][0], "R_out": pk["R"][1],
                                "Z": [float(z) for z in pk["Z"]], "blocks": pk["blocks"]}))
    # coil-1 steel support cylinder (bore d1040, 200 mm radial), over the stack
    r_in = tr.v("pf_coils.coil1_support_cylinder_bore") / 2
    t = tr.v("pf_coils.coil1_support_cylinder_radial")
    zmax = max(packs["coil1"]["Z"]) + (packs["coil1"]["blocks"][0][1])
    cyl = _ring(r_in, r_in + t, -zmax, zmax, n_tor)
    parts.append(Part("coil1_support_cylinder", cyl, "steel", np.eye(4)[None],
                      ["coil1_support_cylinder"], meta={"closed_solid": True}))
    info = {k: {"R_in": v["R"][0], "R_out": v["R"][1], "Z": [float(z) for z in v["Z"]]}
            for k, v in packs.items()}
    # a variant's coil-1 stack entry (jet1983 pf_coil1_stack_1983) is realised by the
    # overrides coil1_count / coil1_pitch_height; check that it agrees with them
    stack = tr.component("pf_coil_stack")
    if stack is not None:
        sp = stack.params
        z_db = sorted(float(z) for z in sp.get("Z_centres", []))
        z_mod = sorted(packs["coil1"]["Z"])
        ok_n = int(sp.get("n_coils", len(z_mod))) == len(z_mod)
        ok_p = abs(float(sp.get("pitch", tr.v("pf_coils.coil1_pitch_height")))
                   - tr.v("pf_coils.coil1_pitch_height")) < 1e-9
        dz = max((abs(a - b) for a, b in zip(z_db, z_mod)), default=0.0) if z_db else 0.0
        if not (ok_n and ok_p and (not z_db or (len(z_db) == len(z_mod) and dz < 1e-6))):
            raise ValueError(f"{stack.name}: coil-1 stack disagrees with the pf_coils overrides "
                             f"(n {sp.get('n_coils')} vs {len(z_mod)}, max dZ {dz:.4f} m)")
        span = max(z_mod) - min(z_mod) + tr.v("pf_coils.coil1_pitch_height")
        info["coil1_stack_check"] = {"entry": stack.name, "n_coils": len(z_mod),
                                     "max_dZ_m": dz, "stack_height_model_m": span,
                                     "stack_height_entry_m": sp.get("stack_height")}
    return finish(tr, parts, info)
