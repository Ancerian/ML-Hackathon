#!/usr/bin/env python3
"""C2 step 4 — score every arm with the official scorer's functions, bootstrap, verdict.

Uses local_score.build_reference / score_shot and metrics.finalize_machine exactly as
local_score.main does (metric v3.1): references and means from the 8 test shots, one global flux
sign per prediction set. Bootstrap: resample TEST SHOTS (1000x); per-shot scorer outputs and
per-shot reference sums are re-accumulated, so no prediction is re-scored (the flux sign and the
fallback means stay those of the full test set — a small, stated approximation).
UNet arms: the 3 seeds are averaged inside every bootstrap replicate.

Output: eval.json (point estimates, per-scalar R2, paired deltas with 95 % CI, verdict).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
STARTER = PROJECT / "fusion equilibrium challenge" / "starter"
CACHE = PROJECT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
sys.path.insert(0, str(STARTER)); sys.path.insert(0, str(STARTER / "fusion_scoring"))
import local_score as ls                                                    # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS      # noqa: E402
from metrics import Accum, finalize_machine                                 # noqa: E402

TEST = list(range(60, 68))
ARMS = ["flat", "sk", "h1", "shuffled"]
SEEDS = [0, 1, 2]
N_BOOT = 1000


def load_test():
    shots = []
    for i in TEST:
        z = np.load(CACHE / f"shot_{i:03d}.npz")
        shots.append({"psi": z["psi"], "q95": z["q95"], "betaN": z["betaN"]})
    return shots


def references(shots, grid):
    R, Z, mc, mf = grid
    refs, stats = [], []
    for s in shots:
        ref = ls.build_reference(s["psi"], R, Z, mc, mf)
        refs.append(ref)
        g = s["psi"].astype(np.float64); fin = np.isfinite(g)
        st = {"psi_sum": float(g[fin].sum()), "psi_sumsq": float((g[fin] ** 2).sum()), "psi_n": int(fin.sum()),
              "scal_sum": np.zeros(N_SCALARS), "scal_sumsq": np.zeros(N_SCALARS), "scal_n": np.zeros(N_SCALARS),
              "cons_sum": np.zeros(N_CONS), "cons_sumsq": np.zeros(N_CONS), "cons_n": np.zeros(N_CONS)}
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64); v = v[np.isfinite(v)]
            st["scal_sum"][j] += v.sum(); st["scal_sumsq"][j] += (v ** 2).sum(); st["scal_n"][j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            st["cons_sum"][j] += v.sum(); st["cons_sumsq"][j] += (v ** 2).sum(); st["cons_n"][j] += v.size
        stats.append(st)
    return refs, stats


def sum_stats(stats, idx):
    out = {}
    for k in stats[0]:
        out[k] = sum(stats[i][k] for i in idx) if not isinstance(stats[0][k], np.ndarray) \
            else np.sum([stats[i][k] for i in idx], axis=0)
    return out


def means_of(st):
    mp = st["psi_sum"] / st["psi_n"]
    ms = np.where(st["scal_n"] > 0, st["scal_sum"] / np.where(st["scal_n"] > 0, st["scal_n"], 1), 0.0)
    mc = np.where(st["cons_n"] > 0, st["cons_sum"] / np.where(st["cons_n"] > 0, st["cons_n"], 1), 0.0)
    return mp, ms, mc


def score_parts(shots, refs, preds, grid, means):
    R, Z, mc, mf = grid
    tot = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = ls.psi_residuals(s_["psi"], p["psirz"], means[0])
        for sg in PSI_SIGNS:
            tot[sg] += rr[sg]
    sign = min(PSI_SIGNS, key=lambda sg: tot[sg])
    return [ls.score_shot(s_, r, p, R, Z, mc, mf, sign, means) for s_, r, p in zip(shots, refs, preds)], sign


def finalize(parts, stats, idx, sign):
    acc = Accum(); acc.psi_sign = sign
    for i in idx:
        acc.add(parts[i])
    res = finalize_machine(acc, sum_stats(stats, idx))
    return {"S": res["S"], "consistency": res["consistency"], "r2_psi": res["r2_psi"],
            "dlcfs": res["dlcfs"], "r2_cons_each": res["r2_cons_each"]}


def load_pred(name, n):
    z = np.load(HERE / "preds" / f"{name}.npz")
    return [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64), "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
             "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)} for i in range(n)]


def main():
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = (mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool), mask["mask_coarse"].astype(np.float64))
    shots = load_test()
    refs, stats = references(shots, grid)
    means = means_of(sum_stats(stats, range(len(shots))))
    runs = {}
    for arm in ARMS:
        runs[("pca", arm)] = [score_parts(shots, refs, load_pred(f"pca_{arm}", len(shots)), grid, means)]
        runs[("unet", arm)] = [score_parts(shots, refs, load_pred(f"unet_{arm}_s{s}", len(shots)), grid, means)
                               for s in SEEDS]
        print(f"  scored {arm}", flush=True)
    n = len(shots)
    full = list(range(n))
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    metrics = ["consistency", "S", "r2_psi"]

    def value(model, arm, idx):
        vals = [finalize(p, stats, idx, sg) for p, sg in runs[(model, arm)]]
        return {m: float(np.mean([v[m] for v in vals])) for m in metrics}, vals

    out = {"point": {}, "delta": {}, "per_scalar": {}, "seeds": {}}
    for model in ("pca", "unet"):
        for arm in ARMS:
            pt, vals = value(model, arm, full)
            out["point"][f"{model}_{arm}"] = pt
            out["per_scalar"][f"{model}_{arm}"] = {k: float(np.mean([v["r2_cons_each"][k] if v["r2_cons_each"][k] is not None else np.nan for v in vals]))
                                                   for k in CONS_SCALARS}
            if model == "unet":
                out["seeds"][f"{model}_{arm}"] = [{m: v[m] for m in metrics} for v in vals]
        bs = {arm: [value(model, arm, list(ix))[0] for ix in boots] for arm in ARMS}
        for arm in ARMS[1:]:
            for ref in ["flat"] + [a for a in ARMS[1:] if a != arm]:
                d = {m: np.array([bs[arm][r][m] - bs[ref][r][m] for r in range(N_BOOT)]) for m in metrics}
                out["delta"][f"{model}: {arm} - {ref}"] = {
                    m: {"point": out["point"][f"{model}_{arm}"][m] - out["point"][f"{model}_{ref}"][m],
                        "ci95": np.percentile(d[m], [2.5, 97.5]).tolist()} for m in metrics}
    # ---- verdict (THEORY.md §6)
    def ci_pos(key, m="consistency"):
        return out["delta"][key][m]["ci95"][0] > 0

    def ci_has0_or_neg(key, m="consistency"):
        return out["delta"][key][m]["ci95"][0] <= 0

    both_improve = all(ci_pos(f"{m}: sk - flat") for m in ("pca", "unet"))
    s_ok = all(out["delta"][f"{m}: sk - flat"]["S"]["point"] >= -0.01 for m in ("pca", "unet"))
    beats_ctrl = all(out["point"][f"{m}_sk"]["consistency"] > max(out["point"][f"{m}_h1"]["consistency"],
                                                                     out["point"][f"{m}_shuffled"]["consistency"])
                     for m in ("pca", "unet"))
    ctrl_ci = any(ci_pos(f"{m}: sk - {'h1' if out['point'][f'{m}_h1']['consistency'] >= out['point'][f'{m}_shuffled']['consistency'] else 'shuffled'}")
                  for m in ("pca", "unet"))
    refuted = all(ci_has0_or_neg(f"{m}: sk - flat") for m in ("pca", "unet"))
    if both_improve and s_ok and beats_ctrl and ctrl_ci:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "REFINED/PARTIAL"
    out["verdict"] = {"C2": verdict, "both_improve": both_improve, "S_not_worse": s_ok,
                      "beats_controls_point": beats_ctrl, "beats_best_control_ci_any": ctrl_ci}
    (HERE / "eval.json").write_text(json.dumps(out, indent=1))
    print("\n=== point estimates (UNet: mean of 3 seeds)")
    for k, v in out["point"].items():
        print(f"  {k:14s} Consistency {v['consistency']:.4f}   S {v['S']:.4f}   R2psi {v['r2_psi']:.4f}")
    print("\n=== paired deltas (95 % CI, bootstrap over test shots)")
    for k, v in out["delta"].items():
        print(f"  {k:24s} dCons {v['consistency']['point']:+.4f} [{v['consistency']['ci95'][0]:+.4f}, {v['consistency']['ci95'][1]:+.4f}]"
              f"   dS {v['S']['point']:+.4f} [{v['S']['ci95'][0]:+.4f}, {v['S']['ci95'][1]:+.4f}]")
    print("\n=== per-scalar R2")
    for k, v in out["per_scalar"].items():
        print(f"  {k:14s} " + " ".join(f"{n}={x:+.3f}" for n, x in v.items()))
    print("\nVERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
