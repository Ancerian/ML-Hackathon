#!/usr/bin/env python3
"""C4-Cal — S vs S'_Cal rankings, bootstrap, verdict (THEORY.md §6). Output: c4cal_eval.json.

Official S terms come from the scorer's functions exactly as in c4/evaluate.py (imported from there);
Calibration per run = Cov x Sharp from the form-M bands of bands/<run>.npz, calibrated once on all of
half B; only the test side is resampled. Bootstrap: 1000x over test shots with rng 0 — the same
replicates as C4. Networks: 3 seeds averaged inside every replicate.
"""
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C4 = HERE.parent / "c4"
_spec = importlib.util.spec_from_file_location("c4eval", C4 / "evaluate.py")
c4e = importlib.util.module_from_spec(_spec); sys.modules["c4eval"] = c4e; _spec.loader.exec_module(c4e)
c2e = c4e.c2e
from analyse_c6 import LEVELS, MODELS, NULL                                  # noqa: E402

W_DESIGN = c4e.W_DESIGN                                    # 0.40 : 0.10 : 0.10 : 0.15, normalised
W_FULL = np.array([0.40, 0.10, 0.10, 0.15])                # + 0.15 GS + 0.10 Cal = 1
WU_MAIN = 0.10 / 0.85
WU_SCAN = [0.0, 0.05, 0.10, WU_MAIN, 0.15, 0.20, 0.25, 0.30]
N_BOOT = 1000
LV = np.array(LEVELS)


def band_fixed(run):
    """Q_a and W_a of form M calibrated on all of half B; test scores grouped by shot."""
    z = np.load(HERE / "bands" / f"{run}.npz")
    eB = np.sort(z["eB_full"]); n = len(eB)
    Q = np.array([eB[int(np.ceil((n + 1) * a)) - 1] for a in LEVELS])
    return {"Q": Q, "W": Q * float(z["smean_full"]), "eT": z["eT_full"], "sid": z["sid_T"]}


def main():
    gref = json.loads((C4 / "gref.json").read_text())["g_ref"]
    mask = np.load(c2e.STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))
    shots = c2e.load_test()
    n = len(shots)
    refs, stats = c2e.references(shots, grid)
    means = c2e.means_of(c2e.sum_stats(stats, range(n)))
    T = [len(s["psi"]) for s in shots]
    off = np.r_[0, np.cumsum(T)]
    g_true = c4e.g_all(np.concatenate([s["psi"].astype(np.float64) for s in shots]), grid)
    nullb = band_fixed(NULL)
    print(f"test frames {off[-1]}; g_true median {np.nanmedian(g_true):.4f}; g_ref {gref:.4f}", flush=True)

    runs = {}
    for name, files in MODELS.items():
        runs[name] = []
        for f in files:
            preds = c4e.load(HERE / "preds" / "test" / f"{f}.npz", n)
            parts, sign = c2e.score_parts(shots, refs, preds, grid, means)
            g = c4e.g_all(np.concatenate([p["psirz"] for p in preds]), grid)
            den = np.maximum(gref - g_true, 1e-12)
            gs = np.where(np.isfinite(g) & np.isfinite(g_true), np.clip(1 - (g - g_true) / den, 0, 1), 0.0)
            b = band_fixed(f)
            sharp = float(np.mean(np.clip(1 - b["W"] / nullb["W"], 0, 1)))
            runs[name].append({"parts": parts, "sign": sign, "gs": gs, "g": g, "band": b, "sharp": sharp})
        print(f"  {name}: sign {[r['sign'] for r in runs[name]]}  sharp {np.round([r['sharp'] for r in runs[name]], 3).tolist()}", flush=True)

    def cov(b, idx):
        e = np.concatenate([b["eT"][b["sid"] == i] for i in idx])
        emp = np.array([np.mean(e <= q) for q in b["Q"]])
        return 1 - float(np.mean(np.abs(emp - LV) / np.maximum(LV, 1 - LV))), emp

    def evaluate(idx):
        frames = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx])
        out = {}
        for name, rs in runs.items():
            tt, ss, gg, cc, cv = [], [], [], [], []
            for r in rs:
                t, s = c4e.terms(r["parts"], stats, idx, r["sign"])
                c, _ = cov(r["band"], idx)
                tt.append(t); ss.append(s); gg.append(float(np.mean(r["gs"][frames])))
                cc.append(c * r["sharp"]); cv.append(c)
            out[name] = {"t": np.mean(tt, 0), "S": float(np.mean(ss)), "GS": float(np.mean(gg)),
                         "Cal": float(np.mean(cc)), "Cov": float(np.mean(cv))}
        return out

    def sp_cal(o, wu, key="Cal"):
        return (1 - wu) * float(W_DESIGN @ o["t"]) + wu * o[key]

    def sp_full(o):
        return float(W_FULL @ o["t"]) + 0.15 * o["GS"] + 0.10 * o["Cal"]

    names = list(MODELS)
    full = evaluate(list(range(n)))
    rng = np.random.default_rng(0)
    boots = [evaluate(list(rng.integers(0, n, n))) for _ in range(N_BOOT)]

    def rank(f):
        v = [f(full[m]) for m in names]
        return [names[i] for i in np.argsort(-np.array(v))]

    def kendall(a, b):
        pa = {m: i for i, m in enumerate(a)}; pb = {m: i for i, m in enumerate(b)}
        c = d = 0
        for x, y in combinations(names, 2):
            s1, s2 = np.sign(pa[x] - pa[y]), np.sign(pb[x] - pb[y])
            c += s1 == s2; d += s1 != s2
        return (c - d) / (c + d)

    def swaps(f):
        out = []
        for a, b in combinations(names, 2):
            for i, j in ((a, b), (b, a)):
                frac = np.mean([(bt[i]["S"] > bt[j]["S"]) and (f(bt[j]) > f(bt[i])) for bt in boots])
                if frac > 0.5:
                    out.append({"S_above": i, "Sprime_above": j, "frac": float(frac)})
        return out

    S_rank = rank(lambda o: o["S"])
    c4ref = json.loads((C4 / "eval.json").read_text())["models"]
    res = {"rank_S": S_rank, "models": {}, "scan": {}, "scan_cov_only": {}}
    for m in names:
        o = full[m]
        res["models"][m] = {"S": o["S"], "r2_psi": o["t"][0], "r2_qb": o["t"][1], "lcfs": o["t"][2], "consistency": o["t"][3],
                            "GS_score": o["GS"], "Cal": o["Cal"], "Cov": o["Cov"],
                            "Sharp": float(np.mean([r["sharp"] for r in runs[m]])),
                            "emp_M_test": np.mean([cov(r["band"], range(n))[1] for r in runs[m]], 0).tolist(),
                            "S_prime_cal_main": sp_cal(o, WU_MAIN), "S_prime_full": sp_full(o),
                            "C4_S": c4ref[m]["S"], "C4_r2_psi": c4ref[m]["r2_psi"], "C4_GS_score": c4ref[m]["GS_score"]}
    for wu in WU_SCAN:
        for key, store in (("Cal", res["scan"]), ("Cov", res["scan_cov_only"])):
            f = lambda o, wu=wu, key=key: sp_cal(o, wu, key)      # noqa: E731
            rk = rank(f); sw = swaps(f)
            store[f"{wu:.4f}"] = {"rank": rk, "tau": kendall(S_rank, rk), "swaps_majority": sw,
                                  "robust_swaps": [s for s in sw if s["frac"] >= 0.95]}
    rk = rank(sp_full); sw = swaps(sp_full)
    res["full_S_prime"] = {"rank": rk, "tau": kendall(S_rank, rk), "swaps_majority": sw,
                           "robust_swaps": [s for s in sw if s["frac"] >= 0.95]}
    mk = f"{WU_MAIN:.4f}"
    robust = res["scan"][mk]["robust_swaps"]
    res["verdict"] = {"C4_Cal": "CONFIRMED" if robust else "REFUTED", "tau_main": res["scan"][mk]["tau"],
                      "n_robust_swaps": len(robust),
                      "first_wu_with_robust_swap": next((k for k, v in res["scan"].items() if v["robust_swaps"]), None)}

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    cal = [full[m]["Cal"] for m in names]
    res["spearman_Cal_vs_r2psi"] = spearman(cal, [full[m]["t"][0] for m in names])
    res["spearman_Cal_vs_consistency"] = spearman(cal, [full[m]["t"][3] for m in names])
    (HERE / "c4cal_eval.json").write_text(json.dumps(res, indent=1, default=float))

    print(f"\n  {'model':22s} {'S':>7s} {'C4 S':>7s} {'R2psi':>7s} {'C4 R2':>7s} {'Cons':>6s} {'GS':>6s} {'Cov':>6s} {'Sharp':>6s} {'Cal':>6s} {'S_Cal':>7s} {'S_full':>7s}")
    for m in names:
        v = res["models"][m]
        print(f"  {m:22s} {v['S']:7.4f} {v['C4_S']:7.4f} {v['r2_psi']:7.4f} {v['C4_r2_psi']:7.4f} {v['consistency']:6.3f} "
              f"{v['GS_score']:6.3f} {v['Cov']:6.3f} {v['Sharp']:6.3f} {v['Cal']:6.3f} {v['S_prime_cal_main']:7.4f} {v['S_prime_full']:7.4f}")
    print("\n  rank by S :", " > ".join(S_rank))
    for label, store in (("Cal", res["scan"]), ("Cov only", res["scan_cov_only"])):
        print(f"\n  S'_{label}:")
        for k, v in store.items():
            print(f"  w_u={float(k):.3f}: tau={v['tau']:+.3f} robust={len(v['robust_swaps'])} "
                  f"majority={[(s['S_above'], s['Sprime_above'], round(s['frac'], 3)) for s in v['swaps_majority']]}")
    v = res["full_S_prime"]
    print(f"\n  full S': tau={v['tau']:+.3f} robust={len(v['robust_swaps'])} rank: {' > '.join(v['rank'])}")
    print(f"    majority swaps: {[(s['S_above'], s['Sprime_above'], round(s['frac'], 3)) for s in v['swaps_majority']]}")
    print("\n  Spearman(Cal, R2psi) =", round(res["spearman_Cal_vs_r2psi"], 3),
          "  Spearman(Cal, Consistency) =", round(res["spearman_Cal_vs_consistency"], 3))
    print("\nVERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
