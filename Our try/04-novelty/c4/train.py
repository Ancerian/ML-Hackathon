#!/usr/bin/env python3
"""C4 step 2 — train the seven experiments.py baselines on the C2 split (THEORY.md §2).

sklearn (predict 50 PCA coefficients of psi, as experiments.py): Linear Regression, Ridge (CV),
Random Forest, MLP (sklearn). PyTorch (experiments_torch.py classes, plain MSE, 50 epochs, Adam 1e-3,
batch 256, ReduceLROnPlateau on validation, last epoch, seeds 0-2): Simple MLP, Conv Decoder.
UNet_Lite and PCA+Ridge are taken from the C2 run (same procedure; flat weight == MSE).
Predictions -> preds/<name>.npz in the local_score format (shot_XXXX_psirz / q95 / betaN).
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
C2 = PROJECT / "Our try" / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
sys.path.insert(0, str(C2))
import train as c2                                                          # noqa: E402
sys.path.insert(0, str(c2.STARTER))
from experiments import get_sklearn_models                                   # noqa: E402

SEEDS = [0, 1, 2]


def save(name, psis, heads):
    d = {}
    for i, (p, (q, b)) in enumerate(zip(psis, heads)):
        d[f"shot_{i:04d}_psirz"] = p.astype(np.float32)
        d[f"shot_{i:04d}_q95"] = q.astype(np.float32)
        d[f"shot_{i:04d}_betaN"] = b.astype(np.float32)
    (HERE / "preds").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "preds" / f"{name}.npz", **d)


def train_torch(cls, Xs, Yn, Xv, Yvn, seed, Xt_list):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = cls(Xs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    lossf = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    Xt_ = torch.tensor(Xs, dtype=torch.float32); Yt_ = torch.tensor(Yn, dtype=torch.float32)
    Xv_ = torch.tensor(Xv, dtype=torch.float32, device=dev); Yv_ = torch.tensor(Yvn, dtype=torch.float32, device=dev)
    for ep in range(50):
        model.train()
        perm = torch.randperm(len(Xt_), generator=g)
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]
            if len(j) < 2:          # BatchNorm needs >1 sample
                continue
            loss = lossf(model(Xt_[j].to(dev)), Yt_[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sched.step(float(lossf(model(Xv_), Yv_)))
    with torch.no_grad():
        return [model(torch.tensor(x, dtype=torch.float32, device=dev)).cpu().numpy().astype(np.float64)
                for x in Xt_list]


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    tr, va, te = c2.load(c2.TRAIN), c2.load(c2.VAL), c2.load(c2.TEST)
    X, Y, q, b = c2.frames(tr)
    Xv, Yv, _, _ = c2.frames(va)
    sc = StandardScaler().fit(X)
    Xs, Xvs = sc.transform(X), sc.transform(Xv)
    Xt_list = [sc.transform(c2.test_inputs(s)) for s in te]
    fin = np.isfinite(q) & np.isfinite(b)
    rq, rb = Ridge(alpha=1.0).fit(Xs[fin], q[fin]), Ridge(alpha=1.0).fit(Xs[fin], b[fin])
    heads = [(rq.predict(x), rb.predict(x)) for x in Xt_list]
    pca = PCA(n_components=50, random_state=0).fit(Y.reshape(len(Y), -1))
    C = pca.transform(Y.reshape(len(Y), -1))
    for sm in get_sklearn_models():
        t0 = time.time()
        sm.fit(Xs, C)
        psis = [pca.inverse_transform(sm.predict(x)).reshape(-1, c2.N, c2.N) for x in Xt_list]
        name = sm.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save(name, psis, heads)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    from experiments_torch import ConvDecoder, SimpleMLP
    mu, sd = Y.mean(), Y.std()
    for cls, name in ((SimpleMLP, "simple_mlp"), (ConvDecoder, "conv_decoder")):
        for seed in SEEDS:
            t0 = time.time()
            preds = train_torch(cls, Xs, (Y - mu) / sd, Xvs, (Yv - mu) / sd, seed, Xt_list)
            save(f"{name}_s{seed}", [p * sd + mu for p in preds], heads)
            print(f"  {name}_s{seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
