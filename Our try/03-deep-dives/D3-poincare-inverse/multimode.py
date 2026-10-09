#!/usr/bin/env python3
"""Is the Level-3 inverse problem WELL-POSED, or degenerate?

Single-amplitude recovery from a Tokamap is trivial: the chaos observable is monotone in x_L.
The real task is recovering a SPECTRUM -- several (m,n) amplitudes at once. The decisive
question before committing weeks: are the modes separately identifiable, or do different
spectra produce indistinguishable Poincare structure?

Multi-mode tokamap. Keep the generating-function structure so symplecticity survives:
    V(T)    = sum_k (a_k/2pi) sin(2pi m_k T + phi_k)
    P       = Psi - 1 - V(T)
    Psi'    = 1/2 (P + sqrt(P^2 + 4 Psi))
    T'      = T + 1/q(Psi') - (1/4pi^2) V'(T) / (1+Psi')^2
with V'(T) = sum_k a_k m_k cos(2pi m_k T + phi_k).  Symplecticity is CHECKED, not assumed.

Each mode m_k resonates where q(Psi) = m_k, so different modes act at different radii --
that is the physical reason to hope they are separable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokamap import q_of                                        # noqa: E402

MODES = np.array([2.0, 3.0, 4.0])          # m of each perturbing mode


def _V(T, a, phi):
    return np.sum(a / (2 * np.pi) * np.sin(2 * np.pi * MODES * T + phi))


def _dV(T, a, phi):
    return np.sum(a * MODES * np.cos(2 * np.pi * MODES * T + phi))


def step(psi, T, a, phi):
    P = psi - 1.0 - _V(T, a, phi)
    psi_n = 0.5 * (P + np.sqrt(P * P + 4.0 * psi))
    T_n = T + 1.0 / q_of(psi_n) - _dV(T, a, phi) / (4 * np.pi ** 2) / (1.0 + psi_n) ** 2
    return psi_n, np.mod(T_n, 1.0)


def det_J(psi, T, a, phi, h=1e-7):
    p0, t0 = step(psi, T, a, phi)
    p1, t1 = step(psi + h, T, a, phi)
    p2, t2 = step(psi, T + h, a, phi)
    return ((p1 - p0) / h) * ((t2 - t0) / h) - ((p2 - p0) / h) * ((t1 - t0) / h)


def radial_profile(a, phi, n_psi=40, n_it=1500, burn=150, seed=0):
    """The OBSERVABLE: Psi-excursion as a function of starting radius.

    This is the map-model stand-in for a divertor footprint: a 1-D signal, measured from
    outside, that encodes where each mode acted.
    """
    rng = np.random.default_rng(seed)
    psis = np.linspace(0.08, 2.2, n_psi)
    out = np.zeros(n_psi)
    for i, p0 in enumerate(psis):
        p, t = float(p0), float(rng.uniform(0, 1))
        tr = []
        for k in range(n_it):
            tr.append(p)
            p, t = step(p, t, a, phi)
            if not np.isfinite(p) or p < 0 or p > 25:
                break
        tr = np.array(tr)
        out[i] = np.ptp(tr[burn:]) if len(tr) > burn + 50 else np.nan
    return psis, out


def main():
    rng = np.random.default_rng(1)
    phi0 = np.array([0.0, 0.7, 1.9])

    print("=== 1. symplecticity survives the multi-mode generalisation? ===")
    for amp in [0.0, 0.2, 0.5]:
        a = np.full(3, amp)
        d = np.array([det_J(rng.uniform(0.1, 2.0), rng.uniform(0, 1), a, phi0) for _ in range(400)])
        d = d[np.isfinite(d)]
        print(f"  a={amp:4.2f}  det J mean={d.mean():.9f}  max|det-1|={np.abs(d - 1).max():.2e}")

    print("\n=== 2. does each mode act at its OWN radius?  (q(Psi)=m resonances) ===")
    for m in MODES:
        r = np.sqrt(max((m - 1.0) / 3.0, 0.0))     # q = 1 + 3*Psi^2 = m
        print(f"  mode m={m:.0f}  resonant at Psi = {r:.3f}   (q={q_of(r):.2f})")

    print("\n=== 3. IDENTIFIABILITY: do different spectra give different observables? ===")
    base = np.array([0.30, 0.30, 0.30])
    psis, ref = radial_profile(base, phi0)
    ok = np.isfinite(ref)
    print(f"  reference spectrum a={base}, usable radii {ok.sum()}/{len(ok)}")

    print(f"\n  {'perturbed spectrum':>28s} {'||d obs||/||obs||':>18s}  verdict")
    tests = {
        "a2 +30%      ": np.array([0.39, 0.30, 0.30]),
        "a3 +30%      ": np.array([0.30, 0.39, 0.30]),
        "a4 +30%      ": np.array([0.30, 0.30, 0.39]),
        "all +10%     ": base * 1.10,
        "swap a2<->a4 ": np.array([0.30, 0.30, 0.30])[::-1] * np.array([1.4, 1.0, 0.6]),
        "same |a|, new phase": base,
    }
    for name, a in tests.items():
        ph = phi0 + (np.array([0.4, -0.3, 0.9]) if "phase" in name else 0.0)
        _, obs = radial_profile(a, ph)
        m = ok & np.isfinite(obs)
        rel = np.linalg.norm(obs[m] - ref[m]) / np.linalg.norm(ref[m])
        verdict = "distinguishable" if rel > 0.05 else "DEGENERATE"
        print(f"  {name:>28s} {rel:18.4f}  {verdict}")

    print("\n=== 4. noise floor: how much does the observable move on RESEED alone? ===")
    rels = []
    for s in range(1, 6):
        _, o = radial_profile(base, phi0, seed=s)
        m = ok & np.isfinite(o)
        rels.append(np.linalg.norm(o[m] - ref[m]) / np.linalg.norm(ref[m]))
    print(f"  reseed-only relative change: median={np.median(rels):.4f}  max={np.max(rels):.4f}")
    print("  --> any signal BELOW this floor is not recoverable, however clever the method.")


if __name__ == "__main__":
    main()
