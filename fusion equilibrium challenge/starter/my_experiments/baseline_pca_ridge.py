#!/usr/bin/env python3
"""Reproduction of the challenge's documented baseline: PCA-on-psi + Ridge.

Trains on DIII-D `diii_d_train` shots streamed from Hugging Face, then exposes a
`predict_row()` that `submission_skeleton.your_model_predict` delegates to, so the
competition's own scorer (`local_score.py`) can grade it end to end.

Reads INPUTS ONLY at predict time (coil currents + efit_times) -- the efit_* columns are
never touched, so the local score stays meaningful and the same code works on test splits.

    python my_experiments/baseline_pca_ridge.py --train --n-shots 60 --n-pca 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data_fixes import fix_d3d_ip_times                                    # noqa: E402
from experiments import (D3D_MAGNETICS_SIGNALS, EFIT_GRID_SIZE,            # noqa: E402
                         interpolate_magnetics_to_efit)

ARTIFACT = HERE / "artifacts" / "baseline_pca_ridge.joblib"
REPO_ID = "Sophelio/fusion-equilibrium-challenge"


def build_inputs_only(row: dict) -> dict:
    """Shot dict with inputs only -- deliberately never reads any efit_* target column."""
    shot = {"source": row.get("source", "DIII-D"),
            "efit_times": np.asarray(row["efit_times"], dtype=np.float64)}
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = fix_d3d_ip_times(row)          # v1.1.0 erratum #6
    mag = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            continue
        mag[sig] = {"values": np.asarray(row[col], dtype=np.float32),
                    "times": ip_times if sig == "plasma_current" else shared}
    shot["magnetics"] = mag
    return shot


def features_for_row(row: dict) -> np.ndarray:
    return interpolate_magnetics_to_efit(build_inputs_only(row))


def train(n_shots: int, n_pca: int, seed: int = 0) -> dict:
    from datasets import load_dataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import joblib

    print(f"Streaming {n_shots} DIII-D train shots from {REPO_ID} ...")
    ds = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    X_parts, Y_parts, q_parts, b_parts = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(ds):
        if len(X_parts) >= n_shots:
            break
        X = features_for_row(row)
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim != 3:
            psi = np.stack([np.asarray(f, dtype=np.float32) for f in row["efit_psirz"]])
        q95 = np.asarray(row["efit_q95"], dtype=np.float64).ravel()
        bn = np.asarray(row["efit_beta_n"], dtype=np.float64).ravel()
        T = min(len(X), len(psi), len(q95), len(bn))
        X_parts.append(X[:T]); Y_parts.append(psi[:T])
        q_parts.append(q95[:T]); b_parts.append(bn[:T])
        print(f"  shot {i}: T={T}  ({time.time()-t0:.0f}s)")

    X = np.concatenate(X_parts); Y = np.concatenate(Y_parts)
    q = np.concatenate(q_parts); b = np.concatenate(b_parts)
    ok = (np.isfinite(X).all(1) & np.isfinite(Y.reshape(len(Y), -1)).all(1)
          & np.isfinite(q) & np.isfinite(b))
    X, Y, q, b = X[ok], Y[ok], q[ok], b[ok]
    print(f"Frames: {len(X)}  features: {X.shape[1]}  (dropped {int((~ok).sum())} non-finite)")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(n_pca, len(X))).fit(Y.reshape(len(Y), -1).astype(np.float64))
    C = pca.transform(Y.reshape(len(Y), -1).astype(np.float64))
    print(f"PCA {pca.n_components_} comps -> {100*pca.explained_variance_ratio_.sum():.3f}% variance"
          f" (comp 1 alone {100*pca.explained_variance_ratio_[0]:.1f}%)")

    ridge_psi = Ridge(alpha=1.0).fit(Xs, C)
    ridge_q95 = Ridge(alpha=1.0).fit(Xs, q)
    ridge_bn = Ridge(alpha=1.0).fit(Xs, b)
    print(f"In-sample R2: psi-coeffs {ridge_psi.score(Xs, C):.4f} | "
          f"q95 {ridge_q95.score(Xs, q):.4f} | betaN {ridge_bn.score(Xs, b):.4f}")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    model = {"scaler": scaler, "pca": pca, "ridge_psi": ridge_psi,
             "ridge_q95": ridge_q95, "ridge_betaN": ridge_bn,
             "n_train_shots": n_shots, "n_frames": int(len(X))}
    joblib.dump(model, ARTIFACT)
    print(f"Saved -> {ARTIFACT}")
    return model


_CACHE = None


def _model():
    global _CACHE
    if _CACHE is None:
        import joblib
        if not ARTIFACT.exists():
            raise FileNotFoundError(f"Train first: python {__file__} --train")
        _CACHE = joblib.load(ARTIFACT)
    return _CACHE


def predict_row(row: dict, source: str = "DIII-D") -> dict:
    m = _model()
    X = features_for_row(row)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = m["scaler"].transform(X)
    psi = m["pca"].inverse_transform(m["ridge_psi"].predict(Xs))
    psi = psi.reshape(-1, EFIT_GRID_SIZE, EFIT_GRID_SIZE).astype(np.float32)
    return {"psirz": psi,
            "q95": m["ridge_q95"].predict(Xs).astype(np.float32),
            "betaN": m["ridge_betaN"].predict(Xs).astype(np.float32)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-shots", type=int, default=60)
    ap.add_argument("--n-pca", type=int, default=50)
    a = ap.parse_args()
    if a.train:
        train(a.n_shots, a.n_pca)
    else:
        ap.print_help()
