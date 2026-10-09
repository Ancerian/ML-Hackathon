#!/usr/bin/env python3
"""C4 step 3 — S vs S'_GS rankings of the baselines, bootstrap, verdict (THEORY.md §1, §3).

Official terms come from the scorer's own functions (via ../../03-deep-dives/.../c2/evaluate.py);
GS_score per frame = clip(1 - (g_pred - g_true) / (g_ref - g_true), 0, 1), 0 where g_pred is
undefined. Bootstrap: 1000x over test shots; networks averaged over their 3 seeds in every replicate.
Output: eval.json.
"""
import json
import multiprocessing as mp
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOV = HERE.parent
C2 = NOV.parent / "03-deep-dives" / "D1-psi-to-scalars" / "c2"
sys.path.insert(0, str(NOV)); sys.path.insert(0, str(C2))
import evaluate as c2e                                                      # noqa: E402
from gs_residual_probe import gs_inconsistency                              # noqa: E402
from metrics import finalize_machine, Accum                                 # noqa: E402

MODELS = {  # display name -> list of prediction files (seeds)
    "Linear Regression": [HERE / "preds/linear_regression.npz"],
    "Ridge (CV)": [HERE / "preds/ridge_cv.npz"],
    "Random Forest": [HERE / "preds/random_forest.npz"],
    "MLP (sklearn)": [HERE / "preds/mlp_sklearn.npz"],
    "Simple MLP": [HERE / f"preds/simple_mlp_s{s}.npz" for s in range(3)],
    "Conv Decoder": [HERE / f"preds/conv_decoder_s{s}.npz" for s in range(3)],
    "UNet_Lite": [C2 / f"preds/unet_flat_s{s}.npz" for s in range(3)],
    "PCA+Ridge (SCORE.md)": [C2 / "preds/pca_flat.npz"],
}
W_DESIGN = np.array([0.40, 0.10, 0.10, 0.15]) / 0.75      # psi, q95/betaN, LCFS, Consistency
WR_MAIN = 0.15 / 0.90
WR_SCAN = [0.0, 0.05, 0.10, 0.15, WR_MAIN, 0.20, 0.25, 0.30]
N_BOOT = 1000
_W = {}


def _init(frames, grid):
    _W.update(frames=frames, grid=grid)


def _g(i):
    R, Z, mc, mf = _W["grid"]
    try:
        return gs_inconsistency(_W["frames"][i], R, Z, mc, mf)[0]
    except Exception:
        return np.nan


def g_all(frames, grid):
    with mp.get_context("fork").Pool(None, initializer=_init, initargs=(frames, grid)) as pool:
        return np.array(pool.map(_g, range(len(frames)), chunksize=8))


def load(path, n):
    z = np.load(path)
    return [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64), "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
             "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)} for i in range(n)]


def terms(parts, stats, idx, sign):
    acc = Accum(); acc.psi_sign = sign
    for i in idx:
        acc.add(parts[i])
    r = finalize_machine(acc, c2e.sum_stats(stats, idx))
    return np.array([max(0.0, r["r2_psi"]), max(0.0, r["r2_qb"]), 1.0 - min(1.0, r["dlcfs"]), r["consistency"]]), r["S"]


def main():
    gref = json.loads((HERE / "gref.json").read_text())["g_ref"]
    mask = np.load(c2e.STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))
    shots = c2e.load_test()
    n = len(shots)
    refs, stats = c2e.references(shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n)))
    T = [len(s["psi"]) for s in shots]
    off = np.r_[0, np.cumsum(T)]
    g_true = g_all(np.concatenate([s["psi"].astype(np.float64) for s in shots]), grid)
    print(f"test frames {off[-1]}; g_true median {np.nanmedian(g_true):.4f}; g_ref {gref:.4f}", flush=True)

    runs = {}
    for name, files in MODELS.items():
        runs[name] = []
        for f in files:
            preds = load(f, n)
            parts, sign = c2e.score_parts(shots, refs, preds, grid, means)
            g = g_all(np.concatenate([p["psirz"] for p in preds]), grid)
            den = np.maximum(gref - g_true, 1e-12)
            gs = np.where(np.isfinite(g) & np.isfinite(g_true), np.clip(1 - (g - g_true) / den, 0, 1), 0.0)
            runs[name].append({"parts": parts, "sign": sign, "gs": gs, "g": g})
        print(f"  {name}: median g_pred {np.nanmedian(np.concatenate([r['g'] for r in runs[name]])):.3f}", flush=True)

    def evaluate(idx):
        """-> {model: (terms[4], S, GS)} averaged over seeds, for shot indices idx."""
        frames = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx])
        out = {}
        for name, rs in runs.items():
            tt, ss, gg = [], [], []
            for r in rs:
                t, s = terms(r["parts"], stats, idx, r["sign"])
                tt.append(t); ss.append(s); gg.append(float(np.mean(r["gs"][frames])))
            out[name] = (np.mean(tt, 0), float(np.mean(ss)), float(np.mean(gg)))
        return out

    def s_prime(t, gs, wr):
        return (1 - wr) * float(W_DESIGN @ t) + wr * gs

    names = list(MODELS)
    full = evaluate(list(range(n)))
    rng = np.random.default_rng(0)
    boots = [evaluate(list(rng.integers(0, n, n))) for _ in range(N_BOOT)]

    res = {"g_ref": gref, "g_true_median_test": float(np.nanmedian(g_true)), "models": {}, "scan": {}}
    for m in names:
        t, s, gs = full[m]
        res["models"][m] = {"S": s, "r2_psi": t[0], "r2_qb": t[1], "lcfs": t[2], "consistency": t[3],
                            "GS_score": gs, "S_prime_main": s_prime(t, gs, WR_MAIN),
                            "g_pred_median": float(np.nanmedian(np.concatenate([r["g"] for r in runs[m]])))}

    def rank(vals):
        return [names[i] for i in np.argsort(-np.array(vals))]

    def kendall(a, b):
        pa = {m: i for i, m in enumerate(a)}; pb = {m: i for i, m in enumerate(b)}
        c = d = 0
        for x, y in combinations(names, 2):
            s1 = np.sign(pa[x] - pa[y]); s2 = np.sign(pb[x] - pb[y])
            c += s1 == s2; d += s1 != s2
        return (c - d) / (c + d)

    S_rank = rank([full[m][1] for m in names])
    res["rank_S"] = S_rank
    for wr in WR_SCAN:
        sp = {m: s_prime(full[m][0], full[m][2], wr) for m in names}
        rk = rank([sp[m] for m in names])
        swaps = []
        for a, b in combinations(names, 2):
            for i, j in ((a, b), (b, a)):
                frac = np.mean([(bt[i][1] > bt[j][1]) and (s_prime(bt[j][0], bt[j][2], wr) > s_prime(bt[i][0], bt[i][2], wr))
                                for bt in boots])
                if frac > 0.5:
                    swaps.append({"S_above": i, "Sprime_above": j, "frac": float(frac)})
        res["scan"][f"{wr:.4f}"] = {"rank": rk, "tau": kendall(S_rank, rk), "S_prime": sp,
                                    "swaps_majority": swaps,
                                    "robust_swaps": [s for s in swaps if s["frac"] >= 0.95]}
    main_key = f"{WR_MAIN:.4f}"
    robust = res["scan"][main_key]["robust_swaps"]
    tau = res["scan"][main_key]["tau"]
    res["verdict"] = {"C4_GS": "CONFIRMED" if robust else "REFUTED", "tau_main": tau,
                      "n_robust_swaps": len(robust),
                      "first_wr_with_robust_swap": next((k for k, v in res["scan"].items() if v["robust_swaps"]), None)}

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    gsv = [full[m][2] for m in names]
    res["spearman_GS_vs_consistency"] = spearman(gsv, [full[m][0][3] for m in names])
    res["spearman_GS_vs_r2psi"] = spearman(gsv, [full[m][0][0] for m in names])
    (HERE / "eval.json").write_text(json.dumps(res, indent=1, default=float))

    print("\n=== models (UNet / MLP / ConvDecoder: mean of 3 seeds)")
    print(f"  {'model':22s} {'S':>7s} {'R2psi':>7s} {'Cons':>7s} {'GS':>7s} {'g_med':>7s} {'S_GS':>7s}")
    for m in names:
        v = res["models"][m]
        print(f"  {m:22s} {v['S']:7.4f} {v['r2_psi']:7.4f} {v['consistency']:7.4f} {v['GS_score']:7.4f} {v['g_pred_median']:7.3f} {v['S_prime_main']:7.4f}")
    print("\n  rank by S :", " > ".join(S_rank))
    for k, v in res["scan"].items():
        print(f"  w_r={float(k):.3f}: tau={v['tau']:+.3f}  robust swaps={len(v['robust_swaps'])}  rank: {' > '.join(v['rank'])}")
        for s in v["robust_swaps"]:
            print(f"        S: {s['S_above']} > {s['Sprime_above']}   S': reversed   ({s['frac']:.1%})")
    print("\n  Spearman(GS, Consistency) =", round(res["spearman_GS_vs_consistency"], 3),
          "  Spearman(GS, R2psi) =", round(res["spearman_GS_vs_r2psi"], 3))
    print("\nVERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
