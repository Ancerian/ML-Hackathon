"""Neutral-beam injection, System A layout (Fig. IV.5-4): six injectors on one
horizontal port.  Beam lines converge on the port mouth.  "Injection at 45 deg
to the magnetic axis" (p.301) is read as: the central beam crosses the circle
R = R0 at 45 deg (tangency radius R0 cos 45 = 2.09 m).  25 deg TYP between
injector columns in plan, 27 deg TYP between rows in the side view; the
column/row split (2 x 3) is a kernel assumption.
Each injector = a beam duct (d 0.30 m, the d300 beam duct of Fig. IV.1-1) of
the digitized duct length + a tank of the digitized length/height.  The beam
path inside the TF/PF region is not meshed (stand-off radius in the kernel).
"""
from __future__ import annotations

import numpy as np

from ..geom import box, cylinder, frame, rot_z
from .base import Layout, Part, finish


def build_nbi(tr, spec):
    L = Layout(tr)
    n_inj = int(tr.v("nbi.injectors_per_port"))
    plan = np.deg2rad(tr.v("nbi.plan_angle_A"))
    vert = np.deg2rad(tr.v("nbi.vertical_angle_A"))
    to_axis = np.deg2rad(tr.v("nbi.injection_angle_to_axis"))
    L_duct, L_tank = tr.v("nbi.duct_length_A"), tr.v("nbi.tank_length_A")
    H_tank = tr.v("nbi.tank_height_A")
    W_tank = H_tank * float(tr.k("nbi_tank_width_over_height"))
    d_duct = tr.v("ports.horizontal_port_beam_d300")
    tr.v("nbi.gate_valve_bore"), tr.v("nbi.neutral_power_per_injector")
    r_stand = float(tr.k("nbi_standoff_R"))
    oct_j = int(tr.k("nbi_port_octant"))
    side = float(tr.k("nbi_plan_side"))
    i_sec = oct_j * L.per_oct + L.hport_k
    phi_p = float(L.sector_phi(i_sec))
    R_mouth = tr.v("vessel.rs_R_outer_wall_outboard")
    P = np.array([R_mouth * np.cos(phi_p), R_mouth * np.sin(phi_p), 0.0])
    e_R = np.array([np.cos(phi_p), np.sin(phi_p), 0.0])
    e_phi = np.array([-np.sin(phi_p), np.cos(phi_p), 0.0])
    e_Z = np.array([0.0, 0.0, 1.0])

    # "injection at 45 deg to the magnetic axis": the beam crosses the magnetic-axis
    # circle R = R0 at 45 deg -> tangency radius p = R0 cos(45); at the port mouth
    # the beam then makes asin(p / R_mouth) with the outward radial.
    R0 = tr.v("plasma.R0")
    p_tan = R0 * np.cos(to_axis)
    psi_c = np.arcsin(p_tan / R_mouth)
    n_cols = int(tr.k("nbi_plan_columns"))
    n_rows = n_inj // n_cols
    psis = psi_c + plan * (np.arange(n_cols) - (n_cols - 1) / 2)
    thetas = vert * (np.arange(n_rows) - (n_rows - 1) / 2)

    duct = cylinder(d_duct / 2, 0.0, L_duct, n=24)
    # cylinder is along local Z; rotate so that it runs along local +X
    swap = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], float)
    duct = duct.transformed(swap)
    tank = box((0.0, -W_tank / 2, -H_tank / 2), (L_tank, W_tank / 2, H_tank / 2))
    Td, Tt, lab, lines = [], [], [], []
    for th in thetas:
        for psi in psis:
            d = np.cos(th) * (np.cos(psi) * e_R + side * np.sin(psi) * e_phi) + np.sin(th) * e_Z
            # stand-off: first s with horizontal radius >= r_stand
            a = d[0] ** 2 + d[1] ** 2
            b = 2 * (P[0] * d[0] + P[1] * d[1])
            c = P[0] ** 2 + P[1] ** 2 - r_stand ** 2
            s0 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
            o = P + s0 * d
            Td.append(frame(o, d))
            Tt.append(frame(o + L_duct * d, d))
            lab.append(f"inj_psi{np.rad2deg(psi):+.0f}_th{np.rad2deg(th):+.1f}")
            lines.append({"start": o.tolist(), "dir": d.tolist(), "s0": float(s0)})
    parts = [Part("beam_duct", duct, "stainless_steel", np.stack(Td), lab,
                  meta={"closed_solid": True}),
             Part("injector_tank", tank, "stainless_steel", np.stack(Tt), lab,
                  meta={"closed_solid": True})]
    return finish(tr, parts, {"port_sector": i_sec, "port_phi_deg": float(np.rad2deg(phi_p)),
                              "pivot": P.tolist(), "beam_lines": lines,
                              "tangency_radius_m": float(p_tan),
                              "plan_angles_from_radial_deg": [float(np.rad2deg(p)) for p in psis],
                              "row_angles_deg": [float(np.rad2deg(t)) for t in thetas]})

