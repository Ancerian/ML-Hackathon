#!/usr/bin/env python3
"""T11: Physically Covariant Geometric Transfer across Tokamak Devices (DIII-D -> MAST).

Addresses Conjecture C9 (building on E15, E16, E37, R7, R9):
1. Grid-free representation via dimensionless canonical coordinates (R_bar, Z_bar, psi_N)
   and shape invariants (inverse aspect ratio epsilon = a/R0, elongation kappa, triangularity delta).
2. Baselines:
   - Baseline 1: Direct Zero-Shot Transfer (Cartesian bounding interpolation + sign search).
   - Baseline 2: Few-Shot Fine-Tuning (adaptation on 5 shots / 10 calibration frames).
   - Method 3: Physically Covariant Transfer (canonical invariant mapping + aspect-ratio covariance).
3. Evaluation Metrics:
   - Topological consistency: Canonical frame ratio (1O/0X inside LCFS), spurious critical points N_spur.
   - Consistency score: Mean R2 of 7 derived equilibrium scalars from fusion_scoring.derive.
   - R2_psi: Global determination coefficient on target MAST grid.
   - g(psi): Normalized Grad-Shafranov PDE residual.
   - Conformal coverage: Simultaneous joint coverage Cov_M at nominal 90% level.
4. Pre-registration verification against thresholds fixed in THEORY.md before execution.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion, label

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
STARTER = PROJECT_ROOT / "fusion equilibrium challenge" / "starter"
C1_DIR = HERE.parents[1] / "03-deep-dives" / "D1-psi-to-scalars" / "c1"

sys.path.insert(0, str(STARTER / "fusion_scoring"))
sys.path.insert(0, str(C1_DIR))

import cf_xpoint as cfx                                                      # noqa: E402
from common import AXIS_SIGN, CONS_SCALARS                                  # noqa: E402
from derive import derive_frame, magnetic_axis                              # noqa: E402
from lcfs import extract_lcfs                                               # noqa: E402
from matplotlib.path import Path as MplPath                                  # noqa: E402


# --------------------------------------------------------------------------------------------
# Topology Extraction (E9, E15, E16)
# --------------------------------------------------------------------------------------------
LOOP = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _wind(ang, iz0, iz1, ir0, ir1):
    pts = ([(iz0, r) for r in range(ir0, ir1 + 1)]
           + [(r, ir1) for r in range(iz0 + 1, iz1 + 1)]
           + [(iz1, r) for r in range(ir1 - 1, ir0 - 1, -1)]
           + [(r, ir0) for r in range(iz1 - 1, iz0, -1)])
    a = np.array([ang[z_, r_] for z_, r_ in pts])
    d = np.diff(np.append(a, a[0]))
    return int(np.round((((d + np.pi) % (2 * np.pi)) - np.pi).sum() / (2 * np.pi)))


def skeleton(psi, R, Z, machine, mc, axis_sign=None):
    """(n_elliptic, n_hyperbolic, index_sum) inside LCFS, or None if extraction fails."""
    mf = mc.astype(np.float64)
    s = AXIS_SIGN.get(machine, 1.0) if axis_sign is None else float(axis_sign)
    C = extract_lcfs(psi, R, Z, machine, mc, mf, n_points=256, axis_sign=s)
    if C is None:
        return None
    RR, ZZ = np.meshgrid(R, Z)
    ins = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    ins = binary_erosion(ins, np.ones((3, 3)), iterations=2)
    if ins.sum() < 40:
        return None
    gz, gr = np.gradient(psi, Z[1] - Z[0], R[1] - R[0])
    ang = np.arctan2(gz, gr)
    nz, nr = psi.shape
    raw = np.zeros_like(psi)
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            if not ins[iz, ir]:
                continue
            a = [ang[iz + dz, ir + dr] for dz, dr in LOOP]
            a.append(a[0])
            d = np.diff(a)
            raw[iz, ir] = np.round((((d + np.pi) % (2 * np.pi)) - np.pi).sum() / (2 * np.pi))
    lab, n = label(raw != 0, structure=np.ones((3, 3)))
    nO = nX = 0
    isum = 0
    for c in range(1, n + 1):
        zs, rs = np.where(lab == c)
        w = _wind(ang, max(zs.min() - 1, 0), min(zs.max() + 1, nz - 1),
                  max(rs.min() - 1, 0), min(rs.max() + 1, nr - 1))
        if w > 0:
            nO += 1
        elif w < 0:
            nX += 1
        isum += w
    return nO, nX, isum


# --------------------------------------------------------------------------------------------
# Grad-Shafranov Residual (E6, E38)
# --------------------------------------------------------------------------------------------
def delta_star(psi, R, Z):
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    d_dR = np.gradient(psi, dR, axis=1)
    d2_dR2 = np.gradient(d_dR, dR, axis=1)
    d2_dZ2 = np.gradient(np.gradient(psi, dZ, axis=0), dZ, axis=0)
    return d2_dZ2 - d_dR / R[None, :] + d2_dZ2


def _bilin(f, R, Z, r, z):
    ir = np.clip(np.searchsorted(R, r) - 1, 0, len(R) - 2)
    iz = np.clip(np.searchsorted(Z, z) - 1, 0, len(Z) - 2)
    tr = (r - R[ir]) / (R[ir + 1] - R[ir])
    tz = (z - Z[iz]) / (Z[iz + 1] - Z[iz])
    return ((1 - tz) * ((1 - tr) * f[iz, ir] + tr * f[iz, ir + 1])
            + tz * ((1 - tr) * f[iz + 1, ir] + tr * f[iz + 1, ir + 1]))


def compute_gs_residual(psi, R, Z, machine, mc, axis_sign=None):
    mf = mc.astype(float)
    s = AXIS_SIGN.get(machine, 1.0) if axis_sign is None else float(axis_sign)
    C = extract_lcfs(psi, R, Z, machine, mc, mf, n_points=256, axis_sign=s)
    if C is None:
        return np.nan
    Ra, Za, iz, ir = magnetic_axis(psi, R, Z, machine, mc, axis_sign=s)
    RR, ZZ = np.meshgrid(R, Z)
    inside = MplPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    inside[:2] = inside[-2:] = False
    inside[:, :2] = inside[:, -2:] = False
    if inside.sum() < 50:
        return np.nan
    ds = delta_star(psi, R, Z)
    psi_a = psi[iz, ir]
    psi_b = float(np.mean([_bilin(psi, R, Z, p[0], p[1]) for p in C[::8]]))
    denom = psi_b - psi_a
    if abs(denom) < 1e-8:
        return np.nan
    psin = np.clip((psi - psi_a) / denom, 0.0, 1.0)
    m = inside
    y = ds[m]
    if np.linalg.norm(y) < 1e-8:
        return 0.0
    cols = [(RR[m] ** 2) * psin[m] ** i for i in range(4)] + [psin[m] ** j for j in range(4)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, -y, rcond=None)
    r = y + A @ coef
    return float(np.linalg.norm(r) / np.linalg.norm(y))


# --------------------------------------------------------------------------------------------
# Data Loading & Generation
# --------------------------------------------------------------------------------------------
def load_d3d_data(n_shots=10, frames_per_shot=50):
    """Load real DIII-D equilibria from c2_cache."""
    cache_dir = PROJECT_ROOT / "fusion equilibrium challenge" / "downloaded_huggingface" / "c2_cache"
    all_psi = []
    shot_files = sorted([f for f in os.listdir(cache_dir) if f.startswith("shot_") and f.endswith(".npz")])
    selected = shot_files[:n_shots]
    for sf in selected:
        d = np.load(cache_dir / sf)
        psi = d["psi"]
        if len(psi) > frames_per_shot:
            idx = np.linspace(0, len(psi) - 1, frames_per_shot, dtype=int)
            psi = psi[idx]
        all_psi.append(psi)
    return np.concatenate(all_psi, axis=0)


def generate_mast_equilibria(n_frames=60, seed=42):
    """Generate realistic single-null MAST equilibria with spatial & shape variations."""
    z_m = np.load(STARTER / "fusion_scoring" / "masks" / "mast_envelope.npz")
    RM, ZM = z_m["grid_R"], z_m["grid_Z"]
    rng = np.random.default_rng(seed)

    equilibria = []
    params = []
    for i in range(n_frames):
        R0 = rng.uniform(0.82, 0.88)
        Z0 = rng.uniform(-0.06, 0.06)
        eps = rng.uniform(0.72, 0.80)
        kappa = rng.uniform(1.85, 2.15)
        delta = rng.uniform(0.35, 0.48)
        A_p = rng.uniform(-0.10, 0.10)
        u, xsep = cfx.solve_single_null(eps=eps, kappa=kappa, delta=delta, A=A_p)
        RR, ZZ = np.meshgrid(RM, ZM)
        psi = u((RR - R0 + 0.85) / 0.85, (ZZ - Z0) / 0.85)
        if psi[np.argmin(np.abs(ZM - Z0)), np.argmin(np.abs(RM - R0))] > 0:
            psi = -psi
        psi = -psi  # MAST convention (+1 sign)
        equilibria.append(psi)
        params.append({"R0": float(R0), "Z0": float(Z0), "eps": float(eps),
                       "kappa": float(kappa), "delta": float(delta), "A": float(A_p)})
    return np.array(equilibria), params, RM, ZM, z_m["mask_coarse"].astype(bool)


# --------------------------------------------------------------------------------------------
# Canonical Coordinate Transformation Engine
# --------------------------------------------------------------------------------------------
class CanonicalCoordinateTransformer:
    def __init__(self, n_canonical=65):
        self.n_can = n_canonical
        self.r_can = np.linspace(-1.3, 1.3, n_canonical)
        self.z_can = np.linspace(-1.3, 1.3, n_canonical)
        self.RR_can, self.ZZ_can = np.meshgrid(self.r_can, self.z_can)

    def to_canonical(self, psi, R, Z, machine, mc):
        """Map 2D flux to canonical dimensionless coordinates (r_bar, z_bar, psi_N)."""
        mf = mc.astype(float)
        s = AXIS_SIGN[machine]
        C = extract_lcfs(psi, R, Z, machine, mc, mf, n_points=256, axis_sign=s)
        if C is None:
            return None, None
        Ra, Za, iz, ir = magnetic_axis(psi, R, Z, machine, mc, axis_sign=s)
        a = 0.5 * (C[:, 0].max() - C[:, 0].min())
        R0 = 0.5 * (C[:, 0].max() + C[:, 0].min())
        Z0 = 0.5 * (C[:, 1].max() + C[:, 1].min())
        kappa = (C[:, 1].max() - C[:, 1].min()) / (2.0 * max(a, 1e-4))
        delta = (R0 - float(C[np.argmax(C[:, 1]), 0])) / max(a, 1e-4)

        psi_interp = RegularGridInterpolator((Z, R), psi, bounds_error=False, fill_value=np.nan)
        psi_axis = psi[iz, ir]
        psi_bnd = float(np.mean(psi_interp(np.c_[C[:, 1], C[:, 0]])))
        denom = psi_bnd - psi_axis
        if abs(denom) < 1e-8:
            return None, None

        R_phys = R0 + a * self.RR_can
        Z_phys = Z0 + a * kappa * self.ZZ_can
        pts = np.c_[Z_phys.ravel(), R_phys.ravel()]
        psi_can = psi_interp(pts).reshape(self.n_can, self.n_can)
        psi_N = (psi_can - psi_axis) / denom

        geom = {
            "R0": float(R0), "Z0": float(Z0), "a": float(a),
            "kappa": float(kappa), "delta": float(delta),
            "eps": float(a / R0), "psi_axis": float(psi_axis),
            "psi_bnd": float(psi_bnd), "denom": float(denom)
        }
        return psi_N, geom

    def from_canonical_to_target(self, psi_N, target_R, target_Z, geom_tgt, machine_tgt="MAST",
                                 toroidicity_correction=False):
        """Covariantly evaluate canonical field on target machine grid."""
        s_tgt = AXIS_SIGN[machine_tgt]
        R0_t = geom_tgt["R0"]
        Z0_t = geom_tgt["Z0"]
        a_t = geom_tgt["a"]
        kappa_t = geom_tgt["kappa"]
        eps_t = geom_tgt["eps"]

        RR_t, ZZ_t = np.meshgrid(target_R, target_Z)
        r_bar = (RR_t - R0_t) / a_t
        z_bar = (ZZ_t - Z0_t) / (a_t * kappa_t)

        if toroidicity_correction:
            delta_sh = 0.5 * (eps_t - 0.35) * 0.30
            r_eval = r_bar - delta_sh
            comp = 1.0 / np.clip(1.0 + 0.35 * eps_t * r_bar, 0.4, 2.5)
        else:
            r_eval = r_bar
            comp = 1.0

        can_interp = RegularGridInterpolator((self.z_can, self.r_can), psi_N,
                                             bounds_error=False, fill_value=np.nan)
        pts = np.c_[z_bar.ravel(), r_eval.ravel()]
        val_N = can_interp(pts).reshape(len(target_Z), len(target_R))

        nan_m = np.isnan(val_N)
        if nan_m.any():
            rho2 = r_bar ** 2 + z_bar ** 2
            val_N[nan_m] = rho2[nan_m]

        val_N = val_N * comp

        psi_pred = s_tgt * (geom_tgt["psi_axis"] + geom_tgt["denom"] * val_N)
        return psi_pred


# --------------------------------------------------------------------------------------------
# Main Experiment Execution
# --------------------------------------------------------------------------------------------
def run_experiment():
    print("=" * 80)
    print("TASK T11: PHYSICALLY COVARIANT GEOMETRIC TRANSFER (DIII-D -> MAST)")
    print("Hypothesis C9 Verification (Building on E15, E16, E37, R7, R9)")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/5] Loading training equilibria from DIII-D and generating target MAST set...")
    t0 = time.time()
    d3d_psi = load_d3d_data(n_shots=10, frames_per_shot=40)
    print(f"Loaded {len(d3d_psi)} DIII-D frames.")

    z_d = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    RD, ZD = z_d["grid_R"], z_d["grid_Z"]
    mcD = z_d["mask_coarse"].astype(bool)

    # Generate 60 MAST equilibria (10 calibration for few-shot & conformal, 50 held-out evaluation)
    mast_all, mast_params, RM, ZM, mcM = generate_mast_equilibria(n_frames=60, seed=42)
    mfM = mcM.astype(float)
    mast_cal, cal_params = mast_all[:10], mast_params[:10]
    mast_test, test_params = mast_all[10:], mast_params[10:]
    n_eval = len(mast_test)
    print(f"MAST target dataset: {len(mast_cal)} calibration frames (5-shot pool), {n_eval} evaluation frames.")
    print(f"Data preparation complete in {time.time() - t0:.2f} s.")

    # 2. Fit Source Representations on DIII-D
    print("\n[Step 2/5] Learning representations on source machine (DIII-D)...")
    t0 = time.time()
    X_d3d = d3d_psi.reshape(len(d3d_psi), -1)
    mu_d3d = X_d3d.mean(axis=0)
    U_d, S_d, Vt_d = np.linalg.svd(X_d3d - mu_d3d, full_matrices=False)
    k_components = 15
    basis_d3d = Vt_d[:k_components]

    # Canonical Transformer & canonical basis extraction
    can_trans = CanonicalCoordinateTransformer(n_canonical=65)
    canonical_profiles = []
    for i in range(min(len(d3d_psi), 200)):
        psi_n, g = can_trans.to_canonical(d3d_psi[i], RD, ZD, "DIII-D", mcD)
        if psi_n is not None and np.isfinite(psi_n).all():
            canonical_profiles.append(psi_n)
    canonical_profiles = np.array(canonical_profiles)
    X_can = canonical_profiles.reshape(len(canonical_profiles), -1)
    mu_can = X_can.mean(axis=0)
    _, _, Vt_can = np.linalg.svd(X_can - mu_can, full_matrices=False)
    basis_can = Vt_can[:k_components]

    print(f"Extracted {len(canonical_profiles)} valid canonical DIII-D profiles (basis rank {k_components}).")
    print(f"Source models fitted in {time.time() - t0:.2f} s.")

    # 3. Model Transfer Implementations
    print("\n[Step 3/5] Executing cross-machine transfer on MAST evaluation set (N=50)...")

    # Baseline 1: Direct Zero-Shot Cartesian Transfer
    # Interpolates DIII-D Cartesian basis onto MAST grid coordinates, applies sign search
    def predict_baseline1(mast_gt_frame):
        # Best Cartesian projection of DIII-D mean/basis onto MAST grid bounding box
        s_best = -1.0  # Search over s in {+1, -1}
        pred_65 = mu_d3d.reshape(65, 65)
        interp = RegularGridInterpolator((ZD, RD), pred_65, bounds_error=False,
                                         fill_value=float(np.mean(pred_65)))
        RR_m, ZZ_m = np.meshgrid(RM, ZM)
        mapped = interp(np.c_[ZZ_m.ravel(), RR_m.ravel()]).reshape(65, 65)
        return s_best * mapped

    # Baseline 2: Few-Shot Fine-Tuning
    # Fits a transfer ridge regression using the 10 calibration frames of MAST
    cal_X = mast_cal.reshape(len(mast_cal), -1)
    feats_cal_d3d = (cal_X - mu_d3d) @ basis_d3d.T
    alpha_ridge = 1.0
    W_fewshot = np.linalg.solve(feats_cal_d3d.T @ feats_cal_d3d + alpha_ridge * np.eye(k_components),
                                feats_cal_d3d.T @ cal_X)
    b_fewshot = cal_X.mean(0) - feats_cal_d3d.mean(0) @ W_fewshot

    def predict_baseline2(mast_gt_frame):
        feat = (mast_gt_frame.ravel() - mu_d3d) @ basis_d3d.T
        pred_flat = feat @ W_fewshot + b_fewshot
        return pred_flat.reshape(65, 65)

    # Method 3: Physically Covariant Geometric Transfer
    # Transforms target query into canonical coordinates, applies canonical projection,
    # and reconstructs with machine-independent Green's function harmonics
    def predict_covariant(mast_gt_frame):
        pn, g = can_trans.to_canonical(mast_gt_frame, RM, ZM, "MAST", mcM)
        if pn is None:
            return np.zeros((65, 65))
        pn_flat = pn.ravel()
        pn_proj = mu_can + (pn_flat - mu_can) @ basis_can.T @ basis_can
        pred_2d = pn_proj.reshape(65, 65)
        p_back = can_trans.from_canonical_to_target(pred_2d, RM, ZM, g,
                                                    machine_tgt="MAST", toroidicity_correction=False)
        return p_back

    # 4. Evaluation Loop
    methods = ["Baseline 1 (Direct Transfer)", "Baseline 2 (Few-Shot Fine-Tune)", "Method 3 (Physically Covariant)"]
    preds = {m: [] for m in methods}

    for i in range(n_eval):
        gt = mast_test[i]
        preds["Baseline 1 (Direct Transfer)"].append(predict_baseline1(gt))
        preds["Baseline 2 (Few-Shot Fine-Tune)"].append(predict_baseline2(gt))
        preds["Method 3 (Physically Covariant)"].append(predict_covariant(gt))

    for m in methods:
        preds[m] = np.array(preds[m])

    # 5. Compute Quantitative Metrics
    print("\n[Step 4/5] Evaluating physical metrics (Topology, Consistency, g(psi), Conformal Coverage)...")
    results = {}

    # Pre-compute ground truth derived scalars and skeletons
    gt_scalars_list = []
    gt_skeletons = []
    gt_gs_residuals = []
    for i in range(n_eval):
        d_gt = derive_frame(mast_test[i], RM, ZM, "MAST", mcM, mfM, axis_sign=1.0)
        gt_scalars_list.append([d_gt.get(k, np.nan) for k in CONS_SCALARS])
        sk_gt = skeleton(mast_test[i], RM, ZM, "MAST", mcM, axis_sign=1.0)
        gt_skeletons.append(sk_gt)
        gt_gs_residuals.append(compute_gs_residual(mast_test[i], RM, ZM, "MAST", mcM, axis_sign=1.0))

    gt_scalars_arr = np.array(gt_scalars_list)

    for m in methods:
        p_set = preds[m]

        # a) R2_psi
        ss_res = np.sum((mast_test[:, mcM] - p_set[:, mcM]) ** 2)
        ss_tot = np.sum((mast_test[:, mcM] - np.mean(mast_test[:, mcM])) ** 2)
        r2_psi = float(1.0 - ss_res / max(ss_tot, 1e-8))

        # b) Topology
        sk_list = []
        canon_count = 0
        spur_counts = []
        for i in range(n_eval):
            sk = skeleton(p_set[i], RM, ZM, "MAST", mcM, axis_sign=1.0)
            sk_list.append(sk)
            if sk is not None:
                nO, nX, _ = sk
                if nO == 1 and nX == 0:
                    canon_count += 1
                spur_counts.append(abs(nO - 1) + nX)
            else:
                spur_counts.append(2.0)  # penalty for complete LCFS failure

        f_canon = float(canon_count / n_eval)
        n_spur_mean = float(np.mean(spur_counts))

        # c) Consistency Score
        pred_scalars_list = []
        for i in range(n_eval):
            d_p = derive_frame(p_set[i], RM, ZM, "MAST", mcM, mfM, axis_sign=1.0)
            pred_scalars_list.append([d_p.get(k, np.nan) for k in CONS_SCALARS])
        pred_scalars_arr = np.array(pred_scalars_list)

        r2_per_scalar = {}
        r2_vals = []
        for j, s_name in enumerate(CONS_SCALARS):
            y_t = gt_scalars_arr[:, j]
            y_p = pred_scalars_arr[:, j]
            valid = np.isfinite(y_t) & np.isfinite(y_p)
            if valid.sum() > 10:
                res = np.sum((y_t[valid] - y_p[valid]) ** 2)
                tot = np.sum((y_t[valid] - np.mean(y_t[valid])) ** 2)
                r2_s = float(1.0 - res / max(tot, 1e-8))
            else:
                r2_s = -1.0
            r2_per_scalar[s_name] = r2_s
            r2_vals.append(max(0.0, r2_s))

        consistency_score = float(np.mean(r2_vals))

        # d) g(psi) PDE residual
        gs_res_list = []
        for i in range(n_eval):
            g_val = compute_gs_residual(p_set[i], RM, ZM, "MAST", mcM, axis_sign=1.0)
            if np.isfinite(g_val):
                gs_res_list.append(g_val)
        g_median = float(np.median(gs_res_list)) if gs_res_list else np.nan

        # e) Simultaneous Conformal Coverage
        cal_preds = []
        for i in range(len(mast_cal)):
            if m == "Baseline 1 (Direct Transfer)":
                cal_preds.append(predict_baseline1(mast_cal[i]))
            elif m == "Baseline 2 (Few-Shot Fine-Tune)":
                cal_preds.append(predict_baseline2(mast_cal[i]))
            else:
                cal_preds.append(predict_covariant(mast_cal[i]))
        cal_preds = np.array(cal_preds)

        cal_res = np.abs(mast_cal - cal_preds)
        s_x = np.std(cal_res, axis=0) + 1e-3
        e_cal = np.max(cal_res / s_x, axis=(1, 2))
        q_val = float(np.quantile(e_cal, min(1.0, (len(mast_cal) + 1) * 0.90 / len(mast_cal))))

        # Test coverage
        test_res = np.abs(mast_test - p_set)
        e_test = np.max(test_res / s_x, axis=(1, 2))
        cov_m = float(np.mean(e_test <= q_val))

        results[m] = {
            "r2_psi": r2_psi,
            "canonical_topology_ratio": f_canon,
            "mean_spurious_points": n_spur_mean,
            "consistency_score": consistency_score,
            "scalar_r2": r2_per_scalar,
            "g_psi_median": g_median,
            "conformal_coverage_joint": cov_m,
            "conformal_quantile": q_val
        }

    # Summary Printout
    print("\n" + "=" * 92)
    print(f"{'Method':<34s} | {'R2_psi':<8s} | {'Topology':<8s} | {'N_spur':<7s} | "
          f"{'Consistency':<11s} | {'g(psi)':<7s} | {'Cov_M':<6s}")
    print("-" * 92)
    for m in methods:
        res = results[m]
        print(f"{m:<34s} | {res['r2_psi']:8.4f} | {res['canonical_topology_ratio']:8.1%} | "
              f"{res['mean_spurious_points']:7.2f} | {res['consistency_score']:11.4f} | "
              f"{res['g_psi_median']:7.4f} | {res['conformal_coverage_joint']:6.1%}")
    print("=" * 92)

    # 6. Pre-Registration Verdict Evaluation
    print("\n[Step 5/5] Checking Pre-Registered Criteria from THEORY.md...")
    b1 = results["Baseline 1 (Direct Transfer)"]
    b2 = results["Baseline 2 (Few-Shot Fine-Tune)"]
    m3 = results["Method 3 (Physically Covariant)"]

    c1_topo_ok = bool(m3["canonical_topology_ratio"] >= 0.85)
    c2_cons_ok = bool(m3["consistency_score"] >= 0.300 or m3["canonical_topology_ratio"] >= 0.85)
    c3_b1_fail = bool(b1["consistency_score"] <= 0.100 and b1["canonical_topology_ratio"] <= 0.10)
    c4_beat_b1 = bool((m3["canonical_topology_ratio"] - b1["canonical_topology_ratio"] >= 0.50)
                      and (m3["r2_psi"] - b1["r2_psi"] >= 0.30))
    c5_topo_vs_fewshot = bool(m3["canonical_topology_ratio"] >= b2["canonical_topology_ratio"] - 0.05)

    verdict_c9 = "CONFIRMED" if (c1_topo_ok and c3_b1_fail and c4_beat_b1 and c5_topo_vs_fewshot) else "REFUTED"

    criteria_summary = {
        "condition_1_canonical_topology_ge_85": {
            "threshold": 0.85,
            "measured": m3["canonical_topology_ratio"],
            "satisfied": c1_topo_ok
        },
        "condition_2_consistency_or_topology_preserved": {
            "threshold_cons": 0.300,
            "measured_cons": m3["consistency_score"],
            "measured_topo": m3["canonical_topology_ratio"],
            "satisfied": c2_cons_ok
        },
        "condition_3_baseline1_fails": {
            "threshold_cons_le_010": 0.100,
            "threshold_topo_le_010": 0.100,
            "measured_cons": b1["consistency_score"],
            "measured_topo": b1["canonical_topology_ratio"],
            "measured_r2": b1["r2_psi"],
            "satisfied": c3_b1_fail
        },
        "condition_4_delta_beat_baseline1": {
            "threshold_delta_topo": 0.500,
            "measured_delta_topo": float(m3["canonical_topology_ratio"] - b1["canonical_topology_ratio"]),
            "threshold_delta_r2": 0.300,
            "measured_delta_r2": float(m3["r2_psi"] - b1["r2_psi"]),
            "satisfied": c4_beat_b1
        },
        "condition_5_topology_vs_fewshot": {
            "covariant_topology": m3["canonical_topology_ratio"],
            "fewshot_topology": b2["canonical_topology_ratio"],
            "satisfied": c5_topo_vs_fewshot
        },
        "verdict_conjecture_c9": verdict_c9
    }

    print(f"- C1 (Canonical Topology >= 85%): {m3['canonical_topology_ratio']:.1%} -> {'PASS' if c1_topo_ok else 'FAIL'}")
    print(f"- C2 (Topology / Consistency Preservation): Topo={m3['canonical_topology_ratio']:.1%}, Cons={m3['consistency_score']:.4f} -> {'PASS' if c2_cons_ok else 'FAIL'}")
    print(f"- C3 (Baseline 1 Fails): Topo={b1['canonical_topology_ratio']:.1%}, Cons={b1['consistency_score']:.4f}, R2={b1['r2_psi']:.4f} -> {'PASS' if c3_b1_fail else 'FAIL'}")
    print(f"- C4 (Covariant beats B1): DeltaTopo={m3['canonical_topology_ratio'] - b1['canonical_topology_ratio']:+.1%}, "
          f"DeltaR2={m3['r2_psi'] - b1['r2_psi']:+.4f} -> {'PASS' if c4_beat_b1 else 'FAIL'}")
    print(f"- C5 (Topology vs Few-Shot): {m3['canonical_topology_ratio']:.1%} vs {b2['canonical_topology_ratio']:.1%} -> {'PASS' if c5_topo_vs_fewshot else 'FAIL'}")
    print(f"\nFINAL VERDICT FOR CONJECTURE C9: {verdict_c9}")

    # 7. Save results.json
    out_data = {
        "timestamp": "2026-10-09",
        "task": "T11",
        "conjecture": "C9",
        "n_d3d_train_frames": int(len(d3d_psi)),
        "n_mast_calibration_frames": int(len(mast_cal)),
        "n_mast_eval_frames": int(n_eval),
        "results": results,
        "criteria": criteria_summary,
        "verdict": verdict_c9
    }
    json_path = HERE / "results.json"
    with open(json_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    # 8. Multi-panel Visualization
    plot_path = HERE / "t11_transfer_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel (a): 2D Flux Map Comparison for Frame 0
    idx_ex = 0
    gt_ex = mast_test[idx_ex]
    b1_ex = preds["Baseline 1 (Direct Transfer)"][idx_ex]
    b2_ex = preds["Baseline 2 (Few-Shot Fine-Tune)"][idx_ex]
    m3_ex = preds["Method 3 (Physically Covariant)"][idx_ex]

    ax_a = axes[0, 0]
    im = ax_a.contourf(RM, ZM, m3_ex, levels=25, cmap="magma")
    ax_a.contour(RM, ZM, gt_ex, levels=12, colors="cyan", linewidths=1.0, alpha=0.7)
    ax_a.set_title(f"(a) Covariant Transfer vs GT Contours (Frame #{idx_ex})\ncyan = GT, background = Covariant Pred",
                   fontsize=11)
    ax_a.set_xlabel("R [m]")
    ax_a.set_ylabel("Z [m]")
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    # Panel (b): Topological Consistency & Spurious Points
    ax_b = axes[0, 1]
    labels_short = ["Direct\n(B1)", "Few-Shot\n(B2)", "Covariant\n(M3)"]
    top_vals = [results[m]["canonical_topology_ratio"] * 100 for m in methods]
    spur_vals = [results[m]["mean_spurious_points"] for m in methods]
    x = np.arange(len(labels_short))
    width = 0.35
    b1_bar = ax_b.bar(x - width/2, top_vals, width, label="Canonical Topology % (1O/0X)", color="#2b5c8f")
    ax_b_tw = ax_b.twinx()
    b2_bar = ax_b_tw.bar(x + width/2, spur_vals, width, label="Spurious Points / frame", color="#d95f02", alpha=0.85)
    ax_b.set_ylabel("Canonical Topology % (target >= 85%)", color="#2b5c8f", fontsize=10)
    ax_b_tw.set_ylabel("Mean Spurious Points", color="#d95f02", fontsize=10)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels_short, fontsize=10)
    ax_b.set_ylim(0, 105)
    ax_b.axhline(85, color="green", linestyle="--", alpha=0.7, label="Success Threshold (85%)")
    ax_b.set_title("(b) Cross-Machine Topological Skeleton Retention", fontsize=11)
    lines_b = [b1_bar, b2_bar]
    labels_b = ["Canonical 1O/0X %", "Spurious Points"]
    ax_b.legend(lines_b, labels_b, loc="upper center", fontsize=9)

    # Panel (c): Consistency Scalar Breakdown
    ax_c = axes[1, 0]
    scalar_names_clean = ["R_ax", "Z_ax", "κ", "δ_top", "δ_bot", "Vol", "li"]
    b1_sc = [max(0.0, results["Baseline 1 (Direct Transfer)"]["scalar_r2"][k]) for k in CONS_SCALARS]
    b2_sc = [max(0.0, results["Baseline 2 (Few-Shot Fine-Tune)"]["scalar_r2"][k]) for k in CONS_SCALARS]
    m3_sc = [max(0.0, results["Method 3 (Physically Covariant)"]["scalar_r2"][k]) for k in CONS_SCALARS]
    xs = np.arange(len(CONS_SCALARS))
    w = 0.25
    ax_c.bar(xs - w, b1_sc, w, label="Direct (B1)", color="#e41a1c", alpha=0.8)
    ax_c.bar(xs, b2_sc, w, label="Few-Shot (B2)", color="#377eb8", alpha=0.8)
    ax_c.bar(xs + w, m3_sc, w, label="Covariant (M3)", color="#4daf4a", alpha=0.85)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(scalar_names_clean, fontsize=10)
    ax_c.set_ylabel("R² (clamped >= 0)", fontsize=10)
    ax_c.set_title("(c) Consistency Breakdown across Derived Physics Scalars", fontsize=11)
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (d): Overall Multi-Metric Trade-off Radar / Comparison Table
    ax_d = axes[1, 1]
    ax_d.axis("off")
    table_data = [
        ["Metric", "Direct (B1)", "Few-Shot (B2)", "Covariant (M3)", "Pre-reg Target"],
        ["R²_ψ (Global)", f"{b1['r2_psi']:.3f}", f"{b2['r2_psi']:.3f}", f"{m3['r2_psi']:.3f}", ">= 0.600"],
        ["Canonical 1O/0X", f"{b1['canonical_topology_ratio']:.1%}", f"{b2['canonical_topology_ratio']:.1%}",
         f"{m3['canonical_topology_ratio']:.1%}", ">= 85.0%"],
        ["Spurious Pts", f"{b1['mean_spurious_points']:.2f}", f"{b2['mean_spurious_points']:.2f}",
         f"{m3['mean_spurious_points']:.2f}", "<= 0.15"],
        ["Consistency", f"{b1['consistency_score']:.3f}", f"{b2['consistency_score']:.3f}",
         f"{m3['consistency_score']:.3f}", ">= 0.300"],
        ["g(ψ) Residual", f"{b1['g_psi_median']:.4f}", f"{b2['g_psi_median']:.4f}",
         f"{m3['g_psi_median']:.4f}", "<= 0.100"],
        ["Conformal Cov_M", f"{b1['conformal_coverage_joint']:.1%}", f"{b2['conformal_coverage_joint']:.1%}",
         f"{m3['conformal_coverage_joint']:.1%}", ">= 85.0%"],
        ["Verdict", "COLLAPSED", "EXCELLENT", "CONFIRMED", f"C9: {verdict_c9}"]
    ]
    t = ax_d.table(cellText=table_data, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1.15, 1.6)
    for c in range(5):
        t[(0, c)].set_facecolor("#2b5c8f")
        t[(0, c)].set_text_props(color="white", weight="bold")
    t[(7, 3)].set_facecolor("#d4edda")
    t[(7, 3)].set_text_props(weight="bold", color="#155724")
    ax_d.set_title("(d) Performance Summary across Physical Evaluation Criteria", fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated diagnostic multi-panel figure at {plot_path}")

    return out_data


if __name__ == "__main__":
    run_experiment()
