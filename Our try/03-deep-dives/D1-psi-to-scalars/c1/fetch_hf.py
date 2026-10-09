#!/usr/bin/env python3
"""Download an evenly spaced, reproducible subset of DIII-D training shots (C1, step 4).

Source: Hugging Face dataset Sophelio/fusion-equilibrium-challenge (CC BY 4.0,
Sophelio / General Atomics). Shot files are named by hash, so we sort the full
list of data/diii_d_train/*.parquet and take N evenly spaced indices.

    cd "fusion equilibrium challenge/starter"
    .venv/bin/python "../../Our try/03-deep-dives/D1-psi-to-scalars/c1/fetch_hf.py" --n 25
"""
import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "Sophelio/fusion-equilibrium-challenge"
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DEST = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "hf_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("data/diii_d_train/") and f.endswith(".parquet"))
    idx = np.linspace(0, len(files) - 1, args.n).round().astype(int)
    chosen = [files[i] for i in idx]
    for i, f in enumerate(chosen):
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(DEST))
        print(f"[{i + 1}/{len(chosen)}] {p}", flush=True)
    (HERE / "hf_shots.json").write_text(json.dumps(
        {"repo": REPO, "license": "CC BY 4.0", "n_total_train": len(files),
         "indices": idx.tolist(), "files": chosen}, indent=2))


if __name__ == "__main__":
    main()
