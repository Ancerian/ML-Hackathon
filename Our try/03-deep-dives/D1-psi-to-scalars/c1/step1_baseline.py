#!/usr/bin/env python3
"""C1 step 1 — D1's own 60 frames (3 demo DIII-D shots), wider eps grid, three families.

Checks that the engine reproduces D1's global alpha, then reports local slopes and the
pre-saturation ("regime") alpha. Output: step1_baseline.npz, step1_baseline.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc                                                     # noqa: E402
import alpha_tools as at                                                    # noqa: E402

if __name__ == "__main__":
    psi = sc.d1.load_psi(60)
    sw = sc.run_sweep(psi, "DIII-D", seeds=3)
    np.savez_compressed(HERE / "step1_baseline.npz", **sw)
    names = list(sw["scalars"])
    res = {}
    for fam in sw["families"]:
        r = at.analyse(sw, fam, d1_eps=sc.D1_EPS)
        res[fam] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        print(f"\n=== {fam}")
        print("  alpha_global :", " ".join(f"{n}={a:.3f}" for n, a in zip(names, r["alpha_global"])))
        print("  alpha_regime :", " ".join(f"{n}={a:.3f}" for n, a in zip(names, r["alpha_regime"])))
        for key in ("alpha_global", "alpha_regime"):
            p = r[key + "_part"]
            print(f"  {key}: best k={p['k']} sil={p['silhouette']:.2f}  {p['partition']}"
                  f"   | k=3: {p['k3'][0]} (sil {p['k3'][1]:.2f})")
        print("  local slopes (rows = eps interval):")
        for j in range(len(sw["eps"]) - 1):
            print(f"   {sw['eps'][j]:.1e}-{sw['eps'][j+1]:.1e} "
                  + " ".join(f"{s:6.2f}" for s in r["slopes"][j]))
    (HERE / "step1_baseline.json").write_text(json.dumps(
        {"scalars": names, "eps": sw["eps"].tolist(), "results": res,
         "C1_partition": at.C1_PARTITION, "Halt_partition": at.HALT_PARTITION}, indent=1, default=str))
