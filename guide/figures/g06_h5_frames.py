#!/usr/bin/env python3
"""H5/E16: покадрові скелети (N_O, N_X, сума) замість медіан.
Перевірка, виконана для путівника 2026-09-29; до реєстрів НЕ внесена.
Запуск: cd "fusion equilibrium challenge/starter" && .venv/bin/python ../../guide/figures/g06_h5_frames.py
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
