#!/usr/bin/env python3
"""E8: GS-нев'язка після PCA-усічення (базис одного розряду 203702) — відтворення числа PCA-5.
Перевірка путівника 2026-09-29; зареєстровано 2026-09-29 як E8 (копія guide/figures/g04_pca5_check.py).
Запуск: cd "fusion equilibrium challenge/starter" && .venv/bin/python "../../Our try/04-novelty/pca5_check.py"
"""
import os, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "fusion equilibrium challenge" / "starter")
sys.path.insert(0, str(ROOT / "Our try" / "04-novelty"))
import gs_residual_probe as g
import pyarrow.parquet as pq
z=np.load(g.STARTER/"fusion_scoring"/"masks"/"d3d_envelope.npz"); R,Z=z["grid_R"],z["grid_Z"]; mc=z["mask_coarse"].astype(bool); mf=mc.astype(float)
a=np.asarray(pq.ParquetFile(str(g.STARTER/"parquet_data"/"d3d_shot_203702.parquet")).read(columns=["efit_psirz"]).column("efit_psirz")[0].as_py(),dtype=float)
a=a[np.isfinite(a).all(axis=(1,2))]
X=a.reshape(len(a),-1); mu=X.mean(0); _,_,Vt=np.linalg.svd(X-mu,full_matrices=False)
idx=np.linspace(0,len(a)-1,12).astype(int)
for k in (5,10,50):
    rec=(mu+(X-mu)@Vt[:k].T@Vt[:k]).reshape(a.shape)
    print(k, np.nanmedian([g.gs_inconsistency(rec[i],R,Z,mc,mf)[0] for i in idx]))
