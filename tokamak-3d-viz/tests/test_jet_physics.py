#!/usr/bin/env python3
"""JET 1975 phase F4 physics checks (Solov'ev equilibrium + TF ripple).

    pytest tests/test_jet_physics.py
    python tests/test_jet_physics.py        # standalone

No external data needed: everything comes from machines/jet1975.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokviz.machine import equilibrium_analytic as EA          # noqa: E402
from tokviz.machine import ripple as RP                        # noqa: E402

_cache: dict = {}


def eqm():
    if "eq" not in _cache:
        eq, sol, info = EA.build_equilibrium()
        x, q = eq.q_profile(np.r_[np.linspace(0.02, 0.99, 60), 1.0])
        _cache.update(eq=eq, sol=sol, info=info, x=x, q=q)
    return _cache


def rip():
    if "loops" not in _cache:
        c = RP.winding_centreline()
        _cache["loops"] = RP.tf_filaments(c, n_coils=32, current_total=4.1e7)
    return _cache["loops"]


def _R_of_psin(eq, s):
    lo, hi = eq.r_axis, 4.3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eq.psi_n(mid, 0.0) < s else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- equilibrium
def test_homogeneous_basis_is_exact():
    for h in EA.CF_HOMOGENEOUS + EA.homogeneous_basis(8):
        assert all(abs(c) < 1e-9 for c in h.gs().t.values())


def test_boundary_fit_residual_below_1cm():
    info = eqm()["info"]
    assert info["boundary_residual_max_m"] < 0.01
    # and the psi = 0 contour of the GRID equilibrium sits on the D as well
    eq, sol = eqm()["eq"], eqm()["sol"]
    lcfs = eq.surface(1.0)
    b = sol.boundary(2000)
    d = np.min(np.hypot(lcfs[:, None, 0] - b[None, :, 0], lcfs[:, None, 1] - b[None, :, 1]), axis=1)
    assert d.max() < 0.01


def test_plasma_current_within_1pct():
    info = eqm()["info"]
    assert abs(info["Ip_A"] / 3.8e6 - 1) < 0.01
    assert abs(info["Ip_area_A"] / 3.8e6 - 1) < 0.01      # area integral of j_phi


def test_B0_within_1pct():
    sol = eqm()["sol"]
    assert abs(sol.F0 / 2.96 - 2.77) / 2.77 < 1e-12      # vacuum, by construction
    Bphi = float(sol.F_of_psi(sol.psi(2.96, 0.0)) / 2.96)
    assert abs(Bphi / 2.77 - 1) < 0.01                   # with the plasma's FF'
    # the Biot-Savart TF set gives the same field on axis
    _, _, bmx, bmn = RP.ripple(rip(), np.array([2.96]))
    assert abs(0.5 * (bmx[0] + bmn[0]) / 2.77 - 1) < 0.01


def test_beta_p_matches_target_and_pressure_positive():
    info = eqm()["info"]
    assert abs(info["beta_p"] - 0.9) < 0.005
    assert info["p_axis_Pa"] > 0
    assert 0.9 < info["F_axis_over_F0"] < 1.1


def test_q_profile_monotone():
    x, q = eqm()["x"], eqm()["q"]
    assert x[-1] == 1.0
    assert np.all(np.diff(q) > 0)
    q0 = eqm()["eq"].q_axis()
    assert q0 < q[0] and abs(q0 / q[0] - 1) < 0.05


def test_traced_q_matches_contour_q():
    eq = eqm()["eq"]
    fl = EA.FieldLinesFpsi(eq, None)
    for s in (0.3, 0.6, 0.9, 0.95):
        r = _R_of_psin(eq, s)
        qc = eq.q_at(s)
        qt = EA.q_traced(fl, r, 0.0, n_turns=150)
        assert abs(qt / qc - 1) < 0.005, (s, qc, qt)


def test_psi_conserved_along_lines():
    eq = eqm()["eq"]
    fl = EA.FieldLinesFpsi(eq, None)
    r0 = np.array([_R_of_psin(eq, s) for s in (0.2, 0.6, 0.95)])
    phi = np.linspace(0, 2 * np.pi * 60, 60 * 40 + 1)
    R, Z, alive = fl.trace(r0, np.zeros(3), phi, rtol=1e-10, atol=1e-12)
    assert alive.all()
    psi = eq.psi_at(R.ravel(), Z.ravel()).reshape(R.shape)
    drift = np.abs(psi - psi[:, :1]).max() / abs(eq.psi_bdy - eq.psi_axis)
    assert drift < 1e-5, drift


# -------------------------------------------------------------------- ripple
def test_ripple_axis_much_smaller_than_edge():
    d, _, _, _ = RP.ripple(rip(), np.array([2.96, 4.21]))
    assert d[0] < 1e-3 * d[1]


def test_ripple_at_source_reference_radius():
    """Table I.3-1 (p.83): 3.6 % in the SOURCE convention eps = 2(Bmax-Bmin)/
    (Bmax+Bmin) (Fig. IV.2-4 caption, p.330), at the plasma edge Re = 4.21 m."""
    _, eps, _, _ = RP.ripple(rip(), np.array([4.21]))
    assert abs(eps[0] / 0.036 - 1) < 0.30, eps[0]


def test_ripple_along_table_IV24_boundary():
    tab = RP.load_csv("plasma_boundary", cols=3)
    _, eps, _, _ = RP.ripple(rip(), tab[:, 0], tab[:, 1])
    ratio = eps * 100 / tab[:, 2]
    assert np.all(np.abs(ratio - 1) < 0.30)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
