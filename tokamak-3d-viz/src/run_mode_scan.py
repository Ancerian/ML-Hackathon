#!/usr/bin/env python3
"""VC7: does the inverse problem's conditioning degrade past three modes?

Runs the C5 conditioning test for mode sets of increasing size, packing the
resonances more and more tightly, and reports how the condition number and the
weakest singular direction move.

    python src/run_mode_scan.py --out data/mode_scan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETS = [
    ("K2", "2/1,3/1"),
    ("K3", "2/1,5/2,3/1"),
    ("K4", "3/2,2/1,5/2,3/1"),
    ("K5", "3/2,2/1,5/2,3/1,7/2"),
    ("K6", "1/1,3/2,2/1,5/2,3/1,7/2"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mode_scan")
    ap.add_argument("--n-levels", type=int, default=16)
    ap.add_argument("--n-phase", type=int, default=5)
    ap.add_argument("--n-turns", type=int, default=24)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    for tag, modes in SETS:
        d = out / tag
        cmd = [args.python, str(HERE / "run_inverse.py"),
               "--out", str(d), "--modes", modes, "--fit-phases",
               "--n-levels", str(args.n_levels), "--n-phase", str(args.n_phase),
               "--n-turns", str(args.n_turns)]
        print(f"\n=== {tag}: {modes} ===", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
            continue
        payload = json.loads((d / "inverse.json").read_text())
        c = payload["conditioning"]
        K = len(payload["modes"])
        row = dict(tag=tag, modes=modes, K=K,
                   n_params=2 * K,
                   cond_avg=c["averaged"]["condition_number"],
                   cond_res=c["phase-resolved"]["condition_number"],
                   cond_res_sub=c["phase-resolved-subsampled"]["condition_number"],
                   weakest_avg=c["averaged"]["relative_spectrum"][-1],
                   weakest_res=c["phase-resolved"]["relative_spectrum"][-1],
                   err10_avg=[x for x in payload["inversion"]["averaged"]
                              if abs(x["noise"] - 0.10) < 1e-9][0],
                   err10_res=[x for x in payload["inversion"]["phase-resolved"]
                              if abs(x["noise"] - 0.10) < 1e-9][0])
        rows.append(row)
        print(f"  K={K}  cond avg={row['cond_avg']:.1f}  res={row['cond_res']:.1f}  "
              f"weakest(res)={row['weakest_res']:.3f}  "
              f"phase err@10%={row['err10_res']['median_phase_error']*100:.1f}%",
              flush=True)

    (out / "mode_scan.json").write_text(json.dumps(
        dict(sets=rows, total_seconds=time.time() - t0), indent=2))
    print(f"\n{'tag':5} {'K':>2} {'cond_avg':>9} {'cond_res':>9} {'weakest_res':>12} "
          f"{'amp_err10':>10} {'phase_err10':>12}")
    for r in rows:
        print(f"{r['tag']:5} {r['K']:>2} {r['cond_avg']:>9.1f} {r['cond_res']:>9.1f} "
              f"{r['weakest_res']:>12.3f} "
              f"{r['err10_res']['median_amp_error']*100:>9.1f}% "
              f"{r['err10_res']['median_phase_error']*100:>11.1f}%")
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}/mode_scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
