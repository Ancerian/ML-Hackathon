#!/usr/bin/env python3
"""C5 stand A: multi-mode Tokamap with the CORRECT generating function (E35), vectorised.

    V(T) = sum_k (a_k/2pi) sin(2pi m_k T + phi_k)
    h(T) = -sum_k a_k/(4pi^2 m_k) cos(2pi m_k T + phi_k)       (h' = V; E35)
    P = Psi - 1 - V(T);  Psi' = (P + sqrt(P^2 + 4 Psi))/2;  T' = T + 1/q(Psi') + h(T)/(1+Psi')^2

Observable: PHASE-RESOLVED Psi-excursion (ptp over N_IT iterations) on a fixed grid of
(radius, seed phase) -- the analogue of the phase-resolved observable of tokamak-3d-viz.
Seeds are deterministic (no random offset), so the forward model is a deterministic function
of (a, phi).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokamap import q_of                                                 # noqa: E402

MODES = np.array([2.0, 3.0, 4.0])
RADII = np.linspace(0.35, 1.25, 20)
N_PHASE = 8
N_IT = 400
PSI0, T0 = np.meshgrid(RADII, np.arange(N_PHASE) / N_PHASE, indexing="ij")
PSI0, T0 = PSI0.ravel(), T0.ravel()


def step(psi, T, a, phi):
    arg = 2 * np.pi * MODES[None, :] * T[:, None] + phi[None, :]
    V = (a / (2 * np.pi) * np.sin(arg)).sum(1)
    h = -(a / (4 * np.pi ** 2 * MODES) * np.cos(arg)).sum(1)
    P = psi - 1.0 - V
    psi_n = 0.5 * (P + np.sqrt(P * P + 4.0 * psi))
    return psi_n, np.mod(T + 1.0 / q_of(psi_n) + h / (1.0 + psi_n) ** 2, 1.0)


def observable(a, phi, n_it=N_IT):
    a = np.asarray(a, float); phi = np.asarray(phi, float)
    psi, T = PSI0.copy(), T0.copy()
    lo, hi = psi.copy(), psi.copy()
    for _ in range(n_it):
        psi, T = step(psi, T, a, phi)
        psi = np.clip(psi, 0.0, 25.0)
        lo = np.minimum(lo, psi); hi = np.maximum(hi, psi)
    return hi - lo


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, 3)
    for amp in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        t = time.time(); y = observable(np.full(3, amp), phi); dt = time.time() - t
        print(f"a={amp:4.2f}: median exc {np.median(y):.4f}, frac exc>0.3 {np.mean(y > 0.3):.3f}, "
              f"max {y.max():.3f}, {dt*1e3:.0f} ms")
    # smoothness of the observable in a parameter (relevant for finite-difference fitting)
    base = observable([0.1, 0.1, 0.1], phi)
    for d in (1e-4, 1e-3, 1e-2):
        y = observable([0.1 * (1 + d), 0.1, 0.1], phi)
        print(f"rel step {d:g}: |dy|/|y| = {np.linalg.norm(y - base) / np.linalg.norm(base):.2e}")
