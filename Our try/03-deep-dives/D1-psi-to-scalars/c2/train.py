#!/usr/bin/env python3
"""C2 step 3 — train PCA+Ridge and UNet_Lite with four spectral weightings (THEORY.md §2-4).

  flat      w = 1                         (plain MSE / plain PCA)
  sk        w_b from S(k) re-measured on 28 shots at the operating point e_op
  h1        1 + (k/k_c)^2, same max/min over band centres as sk
  shuffled  the eight sk band weights in a random order (rng seed 0, never the identity)

e_op = sqrt(1 - R2_psi) of flat PCA+Ridge on the validation shots (stream #36-39).
Writes weights.json and preds/<model>_<arm>[_s<seed>].npz (test-shot psi predictions + q95, betaN).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
CACHE = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
sys.path.insert(0, str(STARTER))
N = 65
BANDS = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 13), (13, 18), (18, 24), (24, 32)]
TRAIN, VAL, TEST = range(0, 36), range(36, 40), range(60, 68)
ARMS = ["flat", "sk", "h1", "shuffled"]
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------- data
def load(ids):
    out = []
    for i in ids:
        z = np.load(CACHE / f"shot_{i:03d}.npz")
        out.append({k: z[k] for k in z.files} | {"id": i})
    return out


def frames(shots):
    X, Y, q, b = [], [], [], []
    for s in shots:
        T = min(len(s["X"]), len(s["psi"]), len(s["q95"]), len(s["betaN"]))
        X.append(s["X"][:T]); Y.append(s["psi"][:T]); q.append(s["q95"][:T]); b.append(s["betaN"][:T])
    X, Y, q, b = map(np.concatenate, (X, Y, q, b))
    ok = np.isfinite(X).all(1) & np.isfinite(Y.reshape(len(Y), -1)).all(1)
    return X[ok], Y[ok].astype(np.float64), q[ok], b[ok]


def test_inputs(s):
    """Features for EVERY frame of a test shot, aligned to its psi (the scorer needs T frames)."""
    T = len(s["psi"])
    X = np.nan_to_num(s["X"], nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < T:
        X = np.vstack([X, np.repeat(X[-1:], T - len(X), 0)])
    return X[:T]


# ---------------------------------------------------------------------------- weights
def kgrid():
    k = np.fft.fftfreq(N) * N
    return np.hypot(k[:, None], k[None, :])


def band_index(K):
    idx = np.full(K.shape, len(BANDS) - 1)
    for b in range(len(BANDS) - 1, -1, -1):
        idx[K < BANDS[b][1]] = b
    idx[K < 1] = 0
    return idx


def weight_maps(e_op, sk_file):
    sk = json.loads(sk_file.read_text())
    eps = min(sk["eps"], key=lambda e: abs(np.log(e) - np.log(e_op)))
    S = np.array(sk["median_sigma"][str(eps)])                  # [B, 7]
    wb = (S / S.max(axis=0)).mean(axis=1)                       # mean over the 7 scalars
    rng = np.random.default_rng(0)
    perm = np.arange(len(wb))
    while (perm == np.arange(len(wb))).all():
        perm = rng.permutation(len(wb))
    centres = np.array([(lo + hi) / 2 for lo, hi in BANDS])
    ratio = wb.max() / wb.min()
    kc = centres.max() / np.sqrt(ratio - 1) if ratio > 1 else np.inf  # 1+(kmax/kc)^2 = ratio
    K = kgrid()
    bi = band_index(K)
    maps = {"flat": np.ones((N, N)), "sk": wb[bi], "shuffled": wb[perm][bi],
            "h1": 1 + (np.minimum(K, centres.max()) / kc) ** 2}
    maps = {a: m / m.mean() for a, m in maps.items()}
    meta = {"e_op": e_op, "eps_used": eps, "band_weights_sk": wb.tolist(), "perm": perm.tolist(),
            "band_weights_shuffled": wb[perm].tolist(), "h1_kc": kc, "ratio_max_min": ratio,
            "band_centres": centres.tolist(),
            "h1_at_centres": (1 + (centres / kc) ** 2).tolist()}
    return maps, meta


# ---------------------------------------------------------------------------- PCA+Ridge
def spectral(Y, w_sqrt, inverse=False):
    F = np.fft.fft2(Y.reshape(-1, N, N))
    F = F / w_sqrt if inverse else F * w_sqrt
    return np.real(np.fft.ifft2(F)).reshape(len(Y), -1)


def pca_ridge(Xs, Y, w, Xt_list):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    ws = np.sqrt(w)
    Yt = spectral(Y, ws)
    pca = PCA(n_components=50, random_state=0).fit(Yt)
    ridge = Ridge(alpha=1.0).fit(Xs, pca.transform(Yt))
    return [spectral(pca.inverse_transform(ridge.predict(x)), ws, inverse=True).reshape(-1, N, N)
            for x in Xt_list]


# ---------------------------------------------------------------------------- UNet_Lite
def unet(Xs, Yn, Xv, Yvn, w, seed, Xt_list):
    import torch
    from experiments_torch import UNetLite
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # rfft2 layout: columns 1..32 stand for two conjugate halves -> multiplicity 2
    wr = torch.tensor(w[:, : N // 2 + 1] * np.r_[1.0, 2.0 * np.ones(N // 2)][None, :],
                      dtype=torch.float32, device=dev) / (N * N) ** 2

    def loss_fn(p, y):
        F = torch.fft.rfft2(p - y)
        return ((F.real ** 2 + F.imag ** 2) * wr).sum(dim=(1, 2)).mean()

    model = UNetLite(Xs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    g = torch.Generator().manual_seed(seed)
    Xt_ = torch.tensor(Xs, dtype=torch.float32); Yt_ = torch.tensor(Yn, dtype=torch.float32)
    Xv_ = torch.tensor(Xv, dtype=torch.float32, device=dev); Yv_ = torch.tensor(Yvn, dtype=torch.float32, device=dev)
    hist = []
    for ep in range(50):
        model.train()
        perm = torch.randperm(len(Xt_), generator=g)
        tl = 0.0
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]
            xb, yb = Xt_[j].to(dev), Yt_[j].to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xv_), Yv_))
        sched.step(vl)
        hist.append([tl / len(perm), vl])
    with torch.no_grad():
        preds = [model(torch.tensor(x, dtype=torch.float32, device=dev)).cpu().numpy().astype(np.float64)
                 for x in Xt_list]
    return preds, hist


# ---------------------------------------------------------------------------- main
def main():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    tr, va, te = load(TRAIN), load(VAL), load(TEST)
    X, Y, q, b = frames(tr)
    Xv, Yv, _, _ = frames(va)
    sc = StandardScaler().fit(X)
    Xs, Xvs = sc.transform(X), sc.transform(Xv)
    Xt_list = [sc.transform(test_inputs(s)) for s in te]
    print(f"train frames {len(X)}, val frames {len(Xv)}, test shots {len(te)}")

    # operating point from flat PCA+Ridge on validation
    pv = pca_ridge(Xs, Y, np.ones((N, N)), [Xvs])[0]
    r2v = 1 - ((pv - Yv) ** 2).sum() / ((Yv - Yv.mean()) ** 2).sum()
    e_op = float(np.sqrt(max(1e-12, 1 - r2v)))
    maps, meta = weight_maps(e_op, HERE / "sk_remeasure.json")
    meta["val_r2_psi_flat_pca"] = float(r2v)
    print(f"flat PCA+Ridge val R2_psi = {r2v:.4f}  ->  e_op = {e_op:.3f}  ->  S(k) at e = {meta['eps_used']}")
    print("  band weights sk      :", np.round(meta["band_weights_sk"], 3).tolist())
    print("  band weights shuffled:", np.round(meta["band_weights_shuffled"], 3).tolist())
    print("  h1 at band centres   :", np.round(meta["h1_at_centres"], 3).tolist())
    (HERE / "weights.json").write_text(json.dumps(meta, indent=1))

    # scalar heads (identical in every arm)
    fin = np.isfinite(q) & np.isfinite(b)
    rq, rb = Ridge(alpha=1.0).fit(Xs[fin], q[fin]), Ridge(alpha=1.0).fit(Xs[fin], b[fin])
    heads = [(rq.predict(x), rb.predict(x)) for x in Xt_list]

    out = HERE / "preds"
    out.mkdir(exist_ok=True)

    def save(name, psis):
        d = {}
        for i, (p, (qq, bb)) in enumerate(zip(psis, heads)):
            d[f"shot_{i:04d}_psirz"] = p.astype(np.float32)
            d[f"shot_{i:04d}_q95"] = qq.astype(np.float32)
            d[f"shot_{i:04d}_betaN"] = bb.astype(np.float32)
        np.savez_compressed(out / f"{name}.npz", **d)

    for arm in ARMS:
        t0 = time.time()
        save(f"pca_{arm}", pca_ridge(Xs, Y, maps[arm], Xt_list))
        print(f"  pca_{arm}: {time.time() - t0:.0f}s", flush=True)

    mu, sd = Y.mean(), Y.std()
    hists = {}
    for arm in ARMS:
        for seed in SEEDS:
            t0 = time.time()
            preds, hist = unet(Xs, (Y - mu) / sd, Xvs, (Yv - mu) / sd, maps[arm], seed, Xt_list)
            save(f"unet_{arm}_s{seed}", [p * sd + mu for p in preds])
            hists[f"{arm}_s{seed}"] = hist
            print(f"  unet_{arm}_s{seed}: {time.time() - t0:.0f}s  final train/val loss "
                  f"{hist[-1][0]:.4g} / {hist[-1][1]:.4g}", flush=True)
    (HERE / "train_history.json").write_text(json.dumps(hists))


if __name__ == "__main__":
    main()
