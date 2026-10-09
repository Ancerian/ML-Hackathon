"""bench.base — Unified Model interface and benchmark execution harness.

Any new model is added with <= 50 lines of code by subclassing `BenchmarkModel`.
"""
from __future__ import annotations
import abc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"

import sys
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C2))

import train as c2_train
from eval_submission import evaluate_submission


def get_split_hash() -> str:
    """Computes SHA-256 hash of the held-out test split files to guarantee immutability."""
    cache_dir = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
    hasher = hashlib.sha256()
    for shot_idx in range(60, 68):
        f = cache_dir / f"shot_{shot_idx:03d}.npz"
        if f.exists():
            hasher.update(f.read_bytes())
        else:
            hasher.update(f"missing_{shot_idx}".encode())
    return hasher.hexdigest()[:16]


class BenchmarkModel(abc.ABC):
    """Abstract base class for all TokaBench-GS models."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.code_version = "tokamld-0.1.0"

    @abc.abstractmethod
    def fit(self, train_shots: List[Dict[str, Any]], val_shots: Optional[List[Dict[str, Any]]] = None) -> BenchmarkModel:
        """Fit model on training shots."""
        pass

    @abc.abstractmethod
    def predict(self, test_shots: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """Predict psirz (T, 65, 65), q95 (T,), betaN (T,) for each test shot."""
        pass

    def evaluate(self, n_boot: int = 1000, save_result: bool = True) -> Dict[str, Any]:
        """Fit, predict, and evaluate model against held-out C2 test shots using tokamld."""
        t0 = time.time()
        print(f"Loading C2 data for {self.name}...")
        tr_shots = c2_train.load(c2_train.TRAIN)
        va_shots = c2_train.load(c2_train.VAL)
        te_shots = c2_train.load(c2_train.TEST)

        print(f"Training {self.name} (seed={self.seed})...")
        self.fit(tr_shots, va_shots)

        print(f"Predicting on {len(te_shots)} test shots...")
        preds = self.predict(te_shots)

        # Temporary submission file
        sub_dict = {}
        for i, p in enumerate(preds):
            sub_dict[f"shot_{i:04d}_psirz"] = np.asarray(p["psirz"], dtype=np.float64)
            sub_dict[f"shot_{i:04d}_q95"] = np.asarray(p["q95"], dtype=np.float64)
            sub_dict[f"shot_{i:04d}_betaN"] = np.asarray(p["betaN"], dtype=np.float64)

        tmp_path = HERE / "models" / f"tmp_{self.name.lower().replace(' ', '_')}.npz"
        np.savez_compressed(tmp_path, **sub_dict)

        print(f"Scoring {self.name} with tokamld harness...")
        rep = evaluate_submission(sub_path=tmp_path, mode="file", n_boot=n_boot, seed=self.seed)
        elapsed = time.time() - t0

        split_hash = get_split_hash()
        res = {
            "model_name": self.name,
            "seed": self.seed,
            "code_version": self.code_version,
            "split_hash": split_hash,
            "train_shots": "0-35",
            "test_shots": "60-67",
            "runtime_sec": round(elapsed, 2),
            "official": rep["official"],
            "extended_s_prime": rep["extended_s_prime"],
            "per_shot": rep["per_shot"],
        }

        if save_result:
            out_file = HERE / "results" / f"{self.name.lower().replace(' ', '_')}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved benchmark result to: {out_file}")

        return res
