"""Розд. 7, практикум: операційні межі DIII-D за даними Fusion Equilibrium Challenge.
Для кожного часу EFIT: beta_N, l_i, q95 (мітки EFIT), a з LCFS, I_p (magnetics, кА),
<n_e> — середнє по каналах core-Томсона (нижня оцінка лінійно-усередненої густини),
n_e0 — максимум по каналах (верхня оцінка),
n_GW = I_p[МА]/(pi a^2) [1e20 м^-3].  Запуск (з будь-якої теки; шляхи відносно цього файлу):
  "fusion equilibrium challenge/starter/.venv/bin/python" manual/code/ch07_limits.py
"""
import glob, sys, warnings, pathlib, numpy as np, pandas as pd
PROJ = pathlib.Path(__file__).resolve().parents[2]  # корінь проєкту (над manual/)
FEC = PROJ / "fusion equilibrium challenge"
sys.path.insert(0, str(FEC / "starter"))
from data_fixes import fix_d3d_ip_times  # виправлення часової осі I_p (errata v1.1.0)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = str(FEC / "downloaded_huggingface/hf_dataset/data/diii_d_train") + "/"

for f in sorted(glob.glob(ROOT + "*.parquet")):
    r = pd.read_parquet(f).iloc[0]
    te = np.asarray(r["efit_times"], float)
    bn = np.asarray(r["efit_beta_n"], float)
    li = np.asarray(r["efit_li"], float)
    q = np.asarray(r["efit_q95"], float)
    n = np.asarray(r["efit_lcfs_n"], int)
    lr = np.asarray(r["efit_lcfs_r"].tolist(), float)
    a = np.array([(lr[i, :n[i]].max() - lr[i, :n[i]].min()) / 2 if n[i] > 3 else np.nan
                  for i in range(len(te))])
    ip = np.interp(te, np.asarray(fix_d3d_ip_times(r), float),
                   np.abs(np.asarray(r["magnetics_plasma_current"], float))) / 1e3  # МА
    tt = np.asarray(r["thomson_core_times"], float)
    ne = np.asarray(r["thomson_core_ne"].tolist(), float)
    ne_m = np.nanmean(np.where(ne > 0, ne, np.nan), axis=1)
    ne_e = np.interp(te, tt, ne_m) / 1e20
    ne_0 = np.interp(te, tt, np.nanmax(np.where(ne > 0, ne, np.nan), axis=1)) / 1e20  # максимум по каналах ~ n_e(0)
    ngw = ip / (np.pi * a**2)
    m = np.isfinite(bn) & (bn > 0.05) & np.isfinite(a) & (ip > 0.3)
    fgw = ne_e[m] / ngw[m]
    fgw0 = ne_0[m] / ngw[m]
    print(f"{f.split('/')[-1]}: N={m.sum()}, a={np.nanmedian(a[m]):.3f} м, "
          f"Ip={np.nanmedian(ip[m]):.2f} МА, q95={np.nanmedian(q[m]):.2f}, "
          f"beta_N med/max={np.nanmedian(bn[m]):.2f}/{np.nanmax(bn[m]):.2f}, "
          f"l_i={np.nanmedian(li[m]):.2f}, max beta_N/(4 l_i)={np.nanmax(bn[m]/(4*li[m])):.2f}, "
          f"f_GW med/max={np.nanmedian(fgw):.2f}/{np.nanmax(fgw):.2f}, n_e0/n_GW max={np.nanmax(fgw0):.2f}")
