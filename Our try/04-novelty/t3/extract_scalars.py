#!/usr/bin/env python3
"""extract_scalars.py — Precompute ground-truth consistency scalars for train, val, test shots."""
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))

from common import AXIS_SIGN, CONS_SCALARS
from derive import derive_frame
from lcfs import extract_lcfs
import train as c2_train

N_POINTS = 512


def extract_for_shots(shots, R, Z, mc, mf, sign, machine="DIII-D"):
    all_scalars = []
    all_masks = []
    t0 = time.time()
    n_total = sum(len(s["psi"]) for s in shots)
    print(f"Extracting 7 consistency scalars for {len(shots)} shots ({n_total} frames)...")

    for s_idx, s in enumerate(shots):
        psi = s["psi"]
        T = len(psi)
        sc_arr = np.full((T, len(CONS_SCALARS)), np.nan, dtype=np.float32)
        for t in range(T):
            c = extract_lcfs(psi[t], R, Z, machine, mc, mf, n_points=N_POINTS, axis_sign=sign)
            v = derive_frame(psi[t], R, Z, machine, mc, mf, contour=c, axis_sign=sign)
            for j, k in enumerate(CONS_SCALARS):
                val = v.get(k)
                if val is not None and np.isfinite(val):
                    sc_arr[t, j] = float(val)
        all_scalars.append(sc_arr)
        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == len(shots):
            print(f"  Processed {s_idx + 1}/{len(shots)} shots in {time.time() - t0:.1f} s")

    return np.concatenate(all_scalars, axis=0)


def main():
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mc = mask["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)
    sign = AXIS_SIGN["DIII-D"]

    cache_file = HERE / "scalars_cache.npz"
    if cache_file.exists():
        print(f"Scalars cache already exists at {cache_file}.")
        return

    tr_shots = c2_train.load(c2_train.TRAIN)
    va_shots = c2_train.load(c2_train.VAL)
    te_shots = c2_train.load(c2_train.TEST)

    Y_tr_scal = extract_for_shots(tr_shots, R, Z, mc, mf, sign)
    Y_va_scal = extract_for_shots(va_shots, R, Z, mc, mf, sign)
    Y_te_scal = extract_for_shots(te_shots, R, Z, mc, mf, sign)

    # Compute mean and std on train
    mean = np.nanmean(Y_tr_scal, axis=0)
    std = np.nanstd(Y_tr_scal, axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    np.savez_compressed(
        cache_file,
        Y_tr=Y_tr_scal,
        Y_va=Y_va_scal,
        Y_te=Y_te_scal,
        mean=mean,
        std=std,
        columns=CONS_SCALARS
    )
    print(f"\nSaved consistency scalars cache to {cache_file}.")
    for j, name in enumerate(CONS_SCALARS):
        print(f"  {name:10s}: mean = {mean[j]:10.4f}, std = {std[j]:10.4f}")


if __name__ == "__main__":
    main()
