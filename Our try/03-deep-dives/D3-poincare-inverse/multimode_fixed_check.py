#!/usr/bin/env python3
"""R8: multimode.py проти виправленої твірної функції (h(T) = -sum a/(4 pi^2 m) cos), max|det J - 1|.
Перевірка путівника 2026-09-29; зареєстровано 2026-09-29 як E35 (корінь R8) (копія guide/figures/g07_multimode_check.py).
Запуск: cd "fusion equilibrium challenge/starter" && .venv/bin/python "../../Our try/03-deep-dives/D3-poincare-inverse/multimode_fixed_check.py"
"""
import os, sys
import numpy as np
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)   # multimode.py, tokamap.py live here
sys.path.insert(0, os.getcwd())
import multimode as mm
from tokamap import q_of
def step_fixed(psi, T, a, phi):
    P = psi - 1.0 - mm._V(T, a, phi)
    psi_n = 0.5 * (P + np.sqrt(P * P + 4.0 * psi))
    h = -np.sum(a / (4*np.pi**2 * mm.MODES) * np.cos(2*np.pi*mm.MODES*T + phi))   # h' = V
    T_n = T + 1.0 / q_of(psi_n) + h / (1.0 + psi_n) ** 2
    return psi_n, np.mod(T_n, 1.0)
def detJ(f, psi, T, a, phi, h=1e-7):
    p0,t0=f(psi,T,a,phi);p1,t1=f(psi+h,T,a,phi);p2,t2=f(psi,T+h,a,phi)
    return ((p1-p0)/h)*((t2-t0)/h)-((p2-p0)/h)*((t1-t0)/h)
rng=np.random.default_rng(1); phi0=np.array([0.0,0.7,1.9])
for amp in [0.0,0.2,0.5]:
    a=np.full(3,amp)
    pts=[(rng.uniform(0.1,2.0),rng.uniform(0,1)) for _ in range(400)]
    d1=np.array([detJ(mm.step,p,t,a,phi0) for p,t in pts]); d2=np.array([detJ(step_fixed,p,t,a,phi0) for p,t in pts])
    print(amp, np.nanmax(abs(d1-1)), np.nanmax(abs(d2-1)))
# single mode m=1 check equivalence with tokamap
