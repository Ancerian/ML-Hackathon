#!/usr/bin/env python3
"""C6 step 2 — retrain the eight C4 baselines and predict BOTH calibration and test shots (THEORY.md §2).

Same code and hyper-parameters as c4/train.py (sklearn four, Simple MLP, Conv Decoder) and
c2/train.py (UNet_Lite flat, PCA+Ridge flat); seeds 0-2 for the networks. A band has to be
calibrated on the very model it is applied to, so the C4 test predictions are not reused.
Also writes the null model N0 (training-mean psi map).

Outputs
  preds/test/<run>.npz                    local_score format (shot_XXXX_psirz / q95 / betaN), 8 test shots
  <data>/c6_preds_cal/<run>.npz           cal_XX -> psi prediction [T, 65, 65] float32, 24 calibration shots
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
C4 = HERE.parent / "c4"
DATA = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface"
CAL = DATA / "c6_cal_cache"
OUT_TEST = HERE / "preds" / "test"
OUT_CAL = DATA / "c6_preds_cal"
sys.path.insert(0, str(C2))
import train as c2                                                          # noqa: E402
sys.path.insert(0, str(c2.STARTER))
from experiments import get_sklearn_models                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("c4train", C4 / "train.py")
c4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c4)
SEEDS = [0, 1, 2]


def load_cal():
    idx = json.loads((CAL / "index.json").read_text())["shots"]
    return [dict(np.load(CAL / f"cal_{int(k):02d}.npz")) | {"key": f"cal_{int(k):02d}"} for k in sorted(idx, key=int)]


def save(name, psis, heads, cal_keys, n_test):
    d = {}
    for i in range(n_test):
        d[f"shot_{i:04d}_psirz"] = psis[i].astype(np.float32)
        d[f"shot_{i:04d}_q95"] = heads[i][0].astype(np.float32)
        d[f"shot_{i:04d}_betaN"] = heads[i][1].astype(np.float32)
    np.savez_compressed(OUT_TEST / f"{name}.npz", **d)
    np.savez_compressed(OUT_CAL / f"{name}.npz", **{k: p.astype(np.float32) for k, p in zip(cal_keys, psis[n_test:])})


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    OUT_TEST.mkdir(parents=True, exist_ok=True); OUT_CAL.mkdir(parents=True, exist_ok=True)
    tr, va, te = c2.load(c2.TRAIN), c2.load(c2.VAL), c2.load(c2.TEST)
    cal = load_cal()
    X, Y, q, b = c2.frames(tr)
    Xv, Yv, _, _ = c2.frames(va)
    sc = StandardScaler().fit(X)
    Xs, Xvs = sc.transform(X), sc.transform(Xv)
    n_test = len(te)
    # test shots first, then calibration shots — identical processing (c2.test_inputs)
    Xt_list = [sc.transform(c2.test_inputs(s)) for s in te] + [sc.transform(c2.test_inputs(s)) for s in cal]
    cal_keys = [s["key"] for s in cal]
    print(f"train frames {len(X)}, val {len(Xv)}, test shots {n_test} "
          f"({sum(len(s['psi']) for s in te)} fr), cal shots {len(cal)} ({sum(len(s['psi']) for s in cal)} fr)", flush=True)

    fin = np.isfinite(q) & np.isfinite(b)
    rq, rb = Ridge(alpha=1.0).fit(Xs[fin], q[fin]), Ridge(alpha=1.0).fit(Xs[fin], b[fin])
    heads = [(rq.predict(x), rb.predict(x)) for x in Xt_list[:n_test]]

    # N0: training-mean psi map
    mean_map = Y.mean(axis=0)
    save("null_mean", [np.repeat(mean_map[None], len(x), 0) for x in Xt_list], heads, cal_keys, n_test)
    print("  null_mean", flush=True)

    # PCA+Ridge (SCORE.md), flat weight == c2 pca_flat
    t0 = time.time()
    save("pca_ridge", c2.pca_ridge(Xs, Y, np.ones((c2.N, c2.N)), Xt_list), heads, cal_keys, n_test)
    print(f"  pca_ridge: {time.time() - t0:.0f}s", flush=True)

    # the four sklearn models of experiments.py (predict 50 PCA coefficients)
    pca = PCA(n_components=50, random_state=0).fit(Y.reshape(len(Y), -1))
    C = pca.transform(Y.reshape(len(Y), -1))
    for sm in get_sklearn_models():
        t0 = time.time()
        sm.fit(Xs, C)
        psis = [pca.inverse_transform(sm.predict(x)).reshape(-1, c2.N, c2.N) for x in Xt_list]
        name = sm.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save(name, psis, heads, cal_keys, n_test)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)

    # networks
    from experiments_torch import ConvDecoder, SimpleMLP
    mu, sd = Y.mean(), Y.std()
    for cls, name in ((SimpleMLP, "simple_mlp"), (ConvDecoder, "conv_decoder")):
        for seed in SEEDS:
            t0 = time.time()
            preds = c4.train_torch(cls, Xs, (Y - mu) / sd, Xvs, (Yv - mu) / sd, seed, Xt_list)
            save(f"{name}_s{seed}", [p * sd + mu for p in preds], heads, cal_keys, n_test)
            print(f"  {name}_s{seed}: {time.time() - t0:.0f}s", flush=True)
    for seed in SEEDS:
        t0 = time.time()
        preds, _ = c2.unet(Xs, (Y - mu) / sd, Xvs, (Yv - mu) / sd, np.ones((c2.N, c2.N)), seed, Xt_list)
        save(f"unet_lite_s{seed}", [p * sd + mu for p in preds], heads, cal_keys, n_test)
        print(f"  unet_lite_s{seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
