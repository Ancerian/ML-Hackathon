#!/usr/bin/env python3
"""C6 step 1 — cache the conformal CALIBRATION shots in the c2_cache format (THEORY.md §2).

Calibration = the 25 evenly spaced DIII-D shots already downloaded for C1
(03-deep-dives/D1-psi-to-scalars/c1/hf_shots.json) minus index 0, which is stream shot #0 of the
C2 training set (checked by first-frame psi; the other 24 match no train/val/test shot).
Per shot: X (21 coil-current features via the starter's features_for_row), psi, q95, betaN —
exactly what c2/fetch_split.py stores. No model is touched here.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/04-novelty/c6/fetch_cal.py"
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
HF = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset"
C2CACHE = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
OUT = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c6_cal_cache"
sys.path.insert(0, str(STARTER)); sys.path.insert(0, str(STARTER / "my_experiments"))
from baseline_pca_ridge import features_for_row                              # noqa: E402


def main():
    shots = json.loads((PROJECT / "Our try/03-deep-dives/D1-psi-to-scalars/c1/hf_shots.json").read_text())
    first = {Path(f).stem: np.load(f)["psi"][0] for f in sorted(glob.glob(str(C2CACHE / "shot_*.npz")))}
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for k, (i, f) in enumerate(zip(shots["indices"], shots["files"])):
        row = pq.read_table(HF / f).to_pylist()[0]
        psi = np.stack([np.asarray(p, dtype=np.float32) for p in row["efit_psirz"]])
        clash = [n for n, p in first.items() if p.shape == psi[0].shape and np.allclose(p, psi[0])]
        if clash:
            print(f"  skip #{i} ({Path(f).stem}): same as {clash}")
            continue
        X = features_for_row(row)
        q95 = np.asarray(row["efit_q95"], dtype=np.float64).ravel()
        bn = np.asarray(row["efit_beta_n"], dtype=np.float64).ravel()
        np.savez_compressed(OUT / f"cal_{k:02d}.npz", X=X, psi=psi, q95=q95, betaN=bn)
        index[k] = {"file": f, "hf_index": i, "T": int(psi.shape[0]), "T_X": int(len(X)),
                    "nonfinite_X_rows": int((~np.isfinite(X).all(1)).sum()),
                    "nonfinite_psi_frames": int((~np.isfinite(psi.reshape(len(psi), -1)).all(1)).sum())}
        print(f"  cal_{k:02d} #{i}: T={psi.shape[0]} X={X.shape}", flush=True)
    (OUT / "index.json").write_text(json.dumps({"source": "c1/hf_shots.json minus stream #0", "shots": index}, indent=1))
    print("calibration shots:", len(index), " frames:", sum(v["T"] for v in index.values()))


if __name__ == "__main__":
    main()
