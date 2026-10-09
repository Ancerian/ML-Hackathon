#!/usr/bin/env python3
"""Fix the observable, then ask whether the inverse problem is answerable AT ALL.

Two failures in the first pilot (both recorded, neither hidden):
  (1) the multi-mode generalisation BROKE symplecticity (max|detJ-1| = 1.7) -- so that map is
      not a field-line map and nothing measured on it means anything physically;
  (2) the observable used ONE random initial phase per radius, so its reseed noise floor (0.32)
      was as large as every signal being tested (0.38-0.43). The test tested nothing.

Here we fix (2) by averaging the excursion over many initial phases per radius, and we stay on
the SINGLE-mode tokamap, which was verified symplectic to 1e-8. The question becomes the honest
prerequisite one: with a clean observable on a valid map, is a single amplitude x_L recoverable,
and how far above the noise floor does the signal sit?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokamap import step                                       # noqa: E402

N_PHASE = 24        # initial phases averaged per radius -- this is the fix
N_IT, BURN = 1200, 120
PSIS = np.linspace(0.08, 2.2, 32)


def profile(xL, seed=0):
    """Mean Psi-excursion vs radius, averaged over N_PHASE initial phases."""
    rng = np.random.default_rng(seed)
    out = np.full(len(PSIS), np.nan)
    for i, p0 in enumerate(PSIS):
        vals = []
        for T0 in (np.arange(N_PHASE) + rng.uniform()) / N_PHASE:
            p, t = float(p0), float(T0)
            tr = []
            for _ in range(N_IT):
                tr.append(p)
                p, t = step(p, t, xL)
                if not np.isfinite(p) or p < 0 or p > 25:
                    break
            if len(tr) > BURN + 50:
                vals.append(np.ptp(np.array(tr)[BURN:]))
        if vals:
            out[i] = float(np.mean(vals))
    return out


def rel(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.linalg.norm(a[m] - b[m]) / np.linalg.norm(b[m]))


def main():
    XL = 0.60
    ref = profile(XL, seed=0)
    print(f"reference x_L = {XL},  usable radii {np.isfinite(ref).sum()}/{len(PSIS)}")

    print("\n=== noise floor: reseed only, same x_L ===")
    fl = [rel(profile(XL, seed=s), ref) for s in range(1, 6)]
    floor = float(np.median(fl))
    print(f"  median={floor:.4f}   max={max(fl):.4f}   (was 0.3244 with 1 phase/radius)")

    print("\n=== signal: change x_L, measure the observable shift ===")
    print(f"{'x_L':>7s} {'rel change':>12s} {'SNR vs floor':>14s}  verdict")
    for x in [0.60, 0.62, 0.66, 0.72, 0.80, 0.90, 1.10]:
        r = rel(profile(x, seed=0), ref)
        snr = r / floor if floor > 0 else np.inf
        v = "recoverable" if snr > 3 else ("marginal" if snr > 1.5 else "BELOW FLOOR")
        d = 100 * (x - XL) / XL
        print(f"{x:7.2f} {r:12.4f} {snr:14.2f}  {v}   (x_L {d:+.0f}%)")

    print("\n=== what this decides ===")
    print("  If a 10% change in x_L sits at SNR > 3, a single amplitude is comfortably")
    print("  recoverable and the SPECTRUM version is the task worth building.")
    print("  If 10% sits below the floor, the observable is still wrong and no amount of")
    print("  ML will fix it -- the information is not in the measurement.")


if __name__ == "__main__":
    main()
