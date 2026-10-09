#!/usr/bin/env python3
"""C8: чи збігаються провали вилучення LCFS зі знаком s_M? Реконструкцію повертають у конвенцію MAST.
Перевірка путівника 2026-09-29; зареєстровано 2026-09-29 як R9 (колишня C8) (копія guide/figures/g06_c8_sign_check.py).
Запуск: cd "fusion equilibrium challenge/starter" && .venv/bin/python "../../Our try/04-novelty/c8_sign_check.py"
"""
import os, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "fusion equilibrium challenge" / "starter")
sys.path.insert(0, str(ROOT / "Our try" / "04-novelty"))
import h5_transfer_topology as h
psiD,RD,ZD,mcD=h.load("DIII-D"); psiM,RM,ZM,mcM=h.load("MAST")
X=psiD.reshape(len(psiD),-1); mu=X.mean(0); _,_,Vt=np.linalg.svd(X-mu,full_matrices=False)
idxM=np.linspace(0,len(psiM)-1,8).astype(int)
Y=psiM.reshape(len(psiM),-1)
for k in [5,10,20,50]:
    for s in (1.0,-1.0):
        Ys=s*Y; rec=(mu+(Ys-mu)@Vt[:k].T@Vt[:k]); e=np.sum((rec-Ys)**2)
        r2=1-np.sum((rec-Ys)**2)/np.sum((Ys-Ys.mean())**2)
        rec=rec.reshape(psiM.shape)
        a=[h.skeleton(rec[i],RM,ZM,"MAST",mcM) for i in idxM]
        b=[h.skeleton(s*rec[i],RM,ZM,"MAST",mcM) for i in idxM]   # back to MAST convention
        print(k, s, f"err={e:.4g} R2={r2:.4f}", "fail as-is:",sum(x is None for x in a), "fail sign-restored:",sum(x is None for x in b), [x for x in b if x])
