"""C5 shared fitting core -- implements THEORY.md §2-§4 exactly; do not tune after launch."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares

SIGMA = 0.03
H_LNA, H_PHI = 0.05, 0.10            # forward-difference steps (THEORY §3)
TOL_AMP, TOL_PHI = 0.10, np.deg2rad(20.0)
SPURIOUS_FACTOR = 1.5


def draw(rng, lo, hi, k=3):
    """Log-uniform amplitudes, uniform phases."""
    a = np.exp(rng.uniform(np.log(lo), np.log(hi), k))
    return a, rng.uniform(0.0, 2 * np.pi, k)


def noisy(y, seed, sigma=SIGMA):
    rng = np.random.default_rng(seed + 1000)
    return y * (1.0 + sigma * rng.standard_normal(y.shape))


def make_residual(forward, y_obs, sigma=SIGMA):
    scale = sigma * np.maximum(y_obs, 0.1 * np.median(y_obs))
    k = None

    def res(x):
        nonlocal k
        k = len(x) // 2
        return (forward(np.exp(x[:k]), x[k:]) - y_obs) / scale
    return res


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def within(a_hat, p_hat, a, p):
    da = np.abs(a_hat - a) / a
    dp = np.abs(wrap(p_hat - p))
    return bool(np.all(da <= TOL_AMP) and np.all(dp <= TOL_PHI)), da, dp


def fit(res, x0, lna_bounds, max_nfev):
    cache = {}

    def f(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear(); cache[key] = res(x)
        return cache[key]

    def jac(x):
        f0 = f(x)
        steps = np.r_[np.full(len(x) // 2, H_LNA), np.full(len(x) // 2, H_PHI)]
        J = np.empty((f0.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += steps[j]
            if j < len(x) // 2 and xp[j] > lna_bounds[1]:
                xp[j] = x[j] - steps[j]; J[:, j] = (f0 - res(xp)) / steps[j]
            else:
                J[:, j] = (res(xp) - f0) / steps[j]
        return J

    k = len(x0) // 2
    lb = np.r_[np.full(k, lna_bounds[0]), np.full(k, -np.inf)]
    ub = np.r_[np.full(k, lna_bounds[1]), np.full(k, np.inf)]
    t = time.time()
    sol = least_squares(f, x0, jac=jac, bounds=(lb, ub), method="trf", max_nfev=max_nfev)
    return sol, time.time() - t


def run_one(forward, a_true, p_true, seed, start_seed, prior, bounds, max_nfev, sigma=SIGMA):
    """One (instance, start) fit; returns a JSON-able record."""
    y_obs = noisy(forward(a_true, p_true), seed, sigma)
    res = make_residual(forward, y_obs, sigma)
    chi2_true = float(np.sum(res(np.r_[np.log(a_true), p_true]) ** 2))
    a0, p0 = draw(np.random.default_rng(start_seed), *prior)
    x0 = np.r_[np.log(np.clip(a0, bounds[0] * 1.001, bounds[1] * 0.999)), p0]
    sol, dt = fit(res, x0, (np.log(bounds[0]), np.log(bounds[1])), max_nfev)
    k = len(a_true)
    a_hat, p_hat = np.exp(sol.x[:k]), sol.x[k:]
    ok, da, dp = within(a_hat, p_hat, a_true, p_true)
    chi2 = float(2 * sol.cost)
    return dict(seed=seed, start=start_seed, a_true=a_true.tolist(), p_true=p_true.tolist(),
                a_hat=a_hat.tolist(), p_hat=wrap(p_hat).tolist(), rel_amp_err=da.tolist(),
                phase_err_deg=np.rad2deg(dp).tolist(), within=ok, chi2=chi2,
                chi2_true=chi2_true, n_obs=int(y_obs.size), nfev=int(sol.nfev),
                status=int(sol.status), seconds=dt,
                spurious=bool((not ok) and chi2 <= SPURIOUS_FACTOR * chi2_true),
                tech_fail=bool(sol.status == 0 and chi2 > 2 * chi2_true))


def summarise(records):
    by = {}
    for r in records:
        by.setdefault(r["seed"], []).append(r)
    inst = []
    for s, rs in sorted(by.items()):
        best = min(rs, key=lambda r: r["chi2"])
        inst.append(dict(seed=s, success=best["within"], spurious=any(r["spurious"] for r in rs),
                         best=best, n_starts=len(rs),
                         tech_fail=sum(r["tech_fail"] for r in rs),
                         cpu_seconds=sum(r["seconds"] for r in rs)))
    n = len(inst)
    return dict(n_instances=n,
                success_rate=sum(i["success"] for i in inst) / n,
                spurious_rate=sum(i["spurious"] for i in inst) / n,
                tech_fail_rate=sum(i["tech_fail"] for i in inst) / max(1, len(records)),
                median_amp_err=float(np.median([max(i["best"]["rel_amp_err"]) for i in inst])),
                median_phase_err_deg=float(np.median([max(i["best"]["phase_err_deg"]) for i in inst])),
                median_cpu_hours_per_instance=float(np.median([i["cpu_seconds"] for i in inst]) / 3600),
                instances=inst)
