#!/usr/bin/env python3
"""Еталон практикуму розд. «Анатомія токамака: JET» (manual/chapters/ch04j.tex).

1) Гофрування TF-поля JET 1975 на Re(plasma) = 4,21 м для N = 32 і N = 16 котушок
   (той самий повний струм Ic = 41 МА, та сама оцифрована D-форма).
2) q95 і q(a) рівноваги Соловйова (Cerfon--Freidberg) при різній трикутності delta
   (R0, a, b, B0, Ip, beta_p -- як у проєкті 1975, табл. I.3-1, PDF с. 83).
3) Оцінка «сходинки» (R/R2)^N + (R1/R)^N для порівняння.

Запуск з кореня проєкту (потрібен venv челенджу: numpy, scipy, scikit-image, PyYAML):
    "fusion equilibrium challenge/starter/.venv/bin/python" manual/code/ch04j_ripple_q.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tokamak-3d-viz" / "src"))

from tokviz.machine import equilibrium_analytic as EA  # noqa: E402
from tokviz.machine import ripple as RP                # noqa: E402

RREF = 4.21          # м, Re(plasma), табл. I.3-1
IC = 4.1e7           # А, повний струм TF (basic)


def ripple_at(n_coils):
    c = RP.winding_centreline()
    loops = RP.tf_filaments(c, n_coils=n_coils, current_total=IC)
    d, e, bmax, bmin = RP.ripple(loops, np.array([RREF, EA.JET_R0]), 0.0, n_coils=n_coils)
    rc = c[:, 0]
    return d, e, rc.min(), rc.max()


def main():
    print("== гофрування на R = 4,21 м (і на R0) ==")
    for n in (32, 16):
        d, e, r1, r2 = ripple_at(n)
        est = (RREF / r2) ** n + (r1 / RREF) ** n
        print(f"N={n:2d}: delta={d[0]*100:6.2f} %  eps_src={e[0]*100:6.2f} %  "
              f"delta(R0)={d[1]:.2e}   оцінка (R/R2)^N+(R1/R)^N={est*100:6.2f} %  "
              f"(R1={r1:.3f}, R2={r2:.3f} м)")

    print("\n== q95, q(a) як функції трикутності delta (Ip=3,8 МА, B0=2,77 Тл, beta_p=0,9) ==")
    for delta in (0.0, 0.2, EA.JET_DELTA, 0.5):
        eq, sol, info = EA.build_equilibrium(delta=delta)
        lv = np.r_[np.linspace(0.01, 0.99, 99), 1.0]
        x, q = eq.q_profile(lv)
        print(f"delta={delta:6.4f}: q0={eq.q_axis():.3f}  q95={np.interp(0.95, x, q):.3f}  "
              f"q(a)={q[-1]:.3f}  shift={eq.r_axis - EA.JET_R0:.3f} м  "
              f"li(3)={info['li3']:.3f}  beta_t={info['beta_t']*100:.2f} %")


if __name__ == "__main__":
    main()
