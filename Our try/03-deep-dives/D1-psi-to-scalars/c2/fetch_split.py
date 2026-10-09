#!/usr/bin/env python3
"""C2 step 1 — cache the SCORE.md split locally: train = streamed shots 0..39, test = 60..67.

Same split as SCORE.md / my_experiments/baseline_pca_ridge.py (streaming order of the HF
split diii_d_train). Per shot we keep INPUTS (21 coil-current features, via the starter's own
features_for_row, which never reads efit_* columns) and TARGETS (psi, q95, betaN).

Data: Sophelio/fusion-equilibrium-challenge, CC BY 4.0.
    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/c2/fetch_split.py"
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "my_experiments"))
from baseline_pca_ridge import features_for_row, REPO_ID                    # noqa: E402

CACHE = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
TRAIN = range(0, 40)
TEST = range(60, 68)


def main():
    from datasets import load_dataset
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    want = set(TRAIN) | set(TEST)
    index = {}
    t0 = time.time()
    for i, row in enumerate(ds):
        if i > max(want):
            break
        if i not in want:
            continue
        X = features_for_row(row)
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim != 3:
            psi = np.stack([np.asarray(f, dtype=np.float32) for f in row["efit_psirz"]])
        q95 = np.asarray(row["efit_q95"], dtype=np.float64).ravel()
        bn = np.asarray(row["efit_beta_n"], dtype=np.float64).ravel()
        split = "train" if i in TRAIN else "test"
        np.savez_compressed(CACHE / f"shot_{i:03d}.npz", X=X, psi=psi, q95=q95, betaN=bn)
        index[i] = {"split": split, "T": int(psi.shape[0]), "features": int(X.shape[1])}
        print(f"  stream #{i:3d} [{split}] T={psi.shape[0]}  ({time.time() - t0:.0f}s)", flush=True)
    (CACHE / "index.json").write_text(json.dumps(
        {"repo": REPO_ID, "license": "CC BY 4.0", "config": "diii_d_train",
         "train": list(TRAIN), "test": list(TEST), "shots": index}, indent=1))


if __name__ == "__main__":
    main()
