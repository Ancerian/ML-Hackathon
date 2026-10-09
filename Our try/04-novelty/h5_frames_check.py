#!/usr/bin/env python3
"""H5/E16: покадрові скелети (N_O, N_X, сума) замість медіан.
Перевірка путівника 2026-09-29; зареєстровано 2026-09-29 як E16 (примітка k=50) (копія guide/figures/g06_h5_frames.py).
Запуск: cd "fusion equilibrium challenge/starter" && .venv/bin/python "../../Our try/04-novelty/h5_frames_check.py"
"""
import os, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "fusion equilibrium challenge" / "starter")
sys.path.insert(0, str(ROOT / "Our try" / "04-novelty"))
import h5_transfer_topology as h
psiD,RD,ZD,mcD=h.load("DIII-D"); psiM,RM,ZM,mcM=h.load("MAST")
idxM=np.linspace(0,len(psiM)-1,8).astype(int); idxD=np.linspace(0,len(psiD)-1,8).astype(int)
print("MAST truth", [h.skeleton(psiM[i],RM,ZM,"MAST",mcM) for i in idxM])
print("D3D truth", [h.skeleton(psiD[i],RD,ZD,"DIII-D",mcD) for i in idxD])
X=psiD.reshape(len(psiD),-1); mu=X.mean(0); _,_,Vt=np.linalg.svd(X-mu,full_matrices=False)
for k in [5,10,20,50]:
    recD=(mu+(X-mu)@Vt[:k].T@Vt[:k]).reshape(psiD.shape)
    print("D3D k",k,[h.skeleton(recD[i],RD,ZD,"DIII-D",mcD) for i in idxD])
print(f"{0.5:.0f}", np.median([1,0,0,1,0,1,1,0]))
# Додано в реєстровій копії (2026-09-29): покадрово MAST -> базис DIII-D зі знаком, який обирає
# h5_transfer_topology.project (мін. похибка). Саме цей рядок k=50 дає «медіана N_X = 0,5».
Y=psiM.reshape(len(psiM),-1)
for k in [5,10,20,50]:
    best=None
    for s in (1.0,-1.0):
        rec=mu+(s*Y-mu)@Vt[:k].T@Vt[:k]; e=np.sum((rec-s*Y)**2)
        if best is None or e<best[0]: best=(e,s,rec)
    _,s,rec=best; rec=rec.reshape(psiM.shape)
    sk=[h.skeleton(rec[i],RM,ZM,"MAST",mcM) for i in idxM]; ok=[x for x in sk if x]
    nX=[x[1] for x in ok]
    print("MAST->D3D k",k,"sign",int(s),sk, "median N_X =",np.median(nX) if ok else None,
          "-> '.0f' prints", f"{np.median(nX):.0f}" if ok else "-", "; frames with X:",sum(n>0 for n in nX))
