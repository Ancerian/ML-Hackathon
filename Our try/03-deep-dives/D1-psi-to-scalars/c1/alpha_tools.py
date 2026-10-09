#!/usr/bin/env python3
"""C1 — amplification exponents, clustering and bootstrap (definitions: THEORY.md §4).

Given a sweep from sweep_core.run_sweep:
  * med_curve      median |d f| / sigma_f over frames and seeds, per eps  (D1's statistic)
  * local_slopes   alpha between neighbouring eps points
  * alpha_global   one log-log line through D1's eps points 0.003 ... 0.3  (reproduces D1)
  * alpha_regime   line through the points whose median RAW error is below 0.25 x the sample
                   range of that scalar (pre-saturation, sigma-independent)
  * partition      exact 1-D k-means (contiguous groups of sorted alphas) for k = 2, 3, 4,
                   with the silhouette of each; "clusters exist" iff best silhouette > 0.5
  * bootstrap      resample SHOTS with replacement; recompute alphas and the partition
"""
from __future__ import annotations

import itertools

import numpy as np

REGIME_FRAC = 0.25


def sample_sigma_range(base, frames):
    b = base[frames]
    return np.nanstd(b, axis=0), np.nanmax(b, axis=0) - np.nanmin(b, axis=0)


def med_curve(err_f, frames, norm):
    """err_f[E, N, S, 7] -> median over (frames, seeds) of err / norm  ->  [E, 7]."""
    e = err_f[:, frames] / norm
    e = e.reshape(e.shape[0], -1, e.shape[-1])
    return np.nanmedian(e, axis=1)


def local_slopes(eps, med):
    le = np.log(eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(med)
    return (lm[1:] - lm[:-1]) / (le[1:, None] - le[:-1, None])


def _fit(eps, y, use):
    m = use & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(eps[m]), np.log(y[m]), 1)[0])


def alpha_global(eps, med, d1_eps):
    use = np.isin(np.round(eps, 12), np.round(d1_eps, 12))
    return np.array([_fit(eps, med[:, k], use) for k in range(med.shape[1])])


def alpha_regime(eps, med_raw, rng_k):
    """med_raw: median RAW |d f| [E, 7]; pre-saturation = below REGIME_FRAC x sample range."""
    out = []
    for k in range(med_raw.shape[1]):
        use = med_raw[:, k] < REGIME_FRAC * rng_k[k]
        out.append(_fit(eps, med_raw[:, k], use))
    return np.array(out)


# ---------------------------------------------------------------------------- clustering
def _silhouette(x, labels):
    n = len(x)
    if len(set(labels)) < 2:
        return np.nan
    s = []
    for i in range(n):
        same = [abs(x[i] - x[j]) for j in range(n) if j != i and labels[j] == labels[i]]
        if not same:
            s.append(0.0)
            continue
        a = np.mean(same)
        b = min(np.mean([abs(x[i] - x[j]) for j in range(n) if labels[j] == c])
                for c in set(labels) if c != labels[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


def kmeans_1d(x, k):
    """Exact 1-D k-means: best split of the sorted values into k contiguous groups."""
    order = np.argsort(x)
    xs = x[order]
    best = None
    for cuts in itertools.combinations(range(1, len(xs)), k - 1):
        groups = np.split(xs, cuts)
        sse = sum(((g - g.mean()) ** 2).sum() for g in groups)
        if best is None or sse < best[0]:
            best = (sse, cuts)
    labels = np.empty(len(x), int)
    for gi, g in enumerate(np.split(order, best[1])):
        labels[g] = gi
    return labels


def partition(alpha, names, ks=(2, 3, 4)):
    """Best k by silhouette -> (k, silhouette, canonical partition string, all silhouettes)."""
    ok = np.isfinite(alpha)
    x, nm = alpha[ok], [n for n, o in zip(names, ok) if o]
    sils = {}
    labs = {}
    for k in ks:
        if k >= len(x):
            continue
        labs[k] = kmeans_1d(x, k)
        sils[k] = _silhouette(x, labs[k])
    k = max(sils, key=lambda kk: sils[kk])
    return k, sils[k], canon(labs[k], nm), sils


def canon(labels, names):
    groups = {}
    for lab, n in zip(labels, names):
        groups.setdefault(lab, []).append(n)
    return " | ".join(sorted(",".join(sorted(g)) for g in groups.values()))


def fixed_k_partition(alpha, names, k):
    ok = np.isfinite(alpha)
    x, nm = alpha[ok], [n for n, o in zip(names, ok) if o]
    lab = kmeans_1d(x, k)
    return canon(lab, nm), _silhouette(x, lab)


# ---------------------------------------------------------------------------- analysis
def analyse(sw, fam, frames=None, d1_eps=None):
    """All exponents for one family on a frame subset."""
    names = list(sw["scalars"])
    fi = list(sw["families"]).index(fam)
    base, err, eps = sw["base"], sw["err"][fi], sw["eps"]
    if frames is None:
        frames = np.where(np.isfinite(base).all(axis=1))[0]
    sig, rng_k = sample_sigma_range(base, frames)
    med = med_curve(err, frames, sig)
    med_raw = med_curve(err, frames, np.ones_like(sig))
    out = dict(eps=eps, med=med, slopes=local_slopes(eps, med),
               alpha_regime=alpha_regime(eps, med_raw, rng_k), sigma=sig, range=rng_k)
    if d1_eps is not None:
        out["alpha_global"] = alpha_global(eps, med, d1_eps)
    for key in ("alpha_global", "alpha_regime"):
        if key in out:
            k, s, p, sils = partition(out[key], names)
            out[key + "_part"] = dict(k=k, silhouette=s, partition=p, silhouettes=sils,
                                      k3=fixed_k_partition(out[key], names, 3))
    return out


def bootstrap(sw, fam, shot_ids, n_boot=1000, d1_eps=None, seed=0):
    """Resample shots with replacement -> alpha samples and partition frequencies."""
    names = list(sw["scalars"])
    good = np.isfinite(sw["base"]).all(axis=1)
    shots = np.unique(shot_ids[good])
    rng = np.random.default_rng(seed)
    A = {"alpha_global": [], "alpha_regime": []}
    parts = {"alpha_global": {}, "alpha_regime": {}}
    parts3 = {"alpha_global": {}, "alpha_regime": {}}
    for _ in range(n_boot):
        pick = rng.choice(shots, len(shots), replace=True)
        frames = np.concatenate([np.where(good & (shot_ids == s))[0] for s in pick])
        r = analyse(sw, fam, frames, d1_eps)
        for key in A:
            if key not in r:
                continue
            A[key].append(r[key])
            p = r[key + "_part"]["partition"]
            parts[key][p] = parts[key].get(p, 0) + 1
            p3 = r[key + "_part"]["k3"][0]
            parts3[key][p3] = parts3[key].get(p3, 0) + 1
    out = {}
    for key in A:
        if not A[key]:
            continue
        a = np.array(A[key])
        out[key] = dict(
            median=np.nanmedian(a, 0).tolist(),
            ci95=np.nanpercentile(a, [2.5, 97.5], axis=0).T.tolist(),
            best_partition_freq={p: c / n_boot for p, c in sorted(parts[key].items(), key=lambda t: -t[1])},
            k3_partition_freq={p: c / n_boot for p, c in sorted(parts3[key].items(), key=lambda t: -t[1])},
            names=names)
    return out


C1_PARTITION = canon([0, 0, 1, 1, 1, 2, 2],
                     ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"])
HALT_PARTITION = canon([0, 0, 1, 0, 0, 1, 1],
                       ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"])
