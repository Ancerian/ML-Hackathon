#!/usr/bin/env python3
"""Розд. 8 (ch:wall): шар Дебая, коефіцієнт передачі тепла, двоточкова модель SOL,
заряджання пилинки (OML). Еталонні розрахунки для задач і практикуму.

Запуск: python3 manual/code/ch08_sheath_2pm.py   (лише stdlib)
"""
import math

e = 1.602176634e-19
me = 9.1093837e-31
amu = 1.66053907e-27
eps0 = 8.8541878128e-12
mD = 2.01410178 * amu
mH = 1.00782503 * amu


def sheath(Ti_Te, mi):
    """eV_sh/T_e = 0.5 ln[2π (m_e/m_i)(1+T_i/T_e)]  (плаваюча стінка, Z=1, γ_i=1)."""
    return 0.5 * math.log(2 * math.pi * me / mi * (1 + Ti_Te))


def gamma_heat(Ti_Te, mi, delta_e=0.0):
    """γ = 2.5 T_i/T_e + 2/(1-δ) - eV_sh/T_e  (стандартні складові, поза корпусом)."""
    vsh = 0.5 * math.log(2 * math.pi * me / mi * (1 + Ti_Te) / (1 - delta_e) ** 2)
    return 2.5 * Ti_Te + 2 / (1 - delta_e) - vsh


def two_point(Q, L, nu, gam=7.0, kappa0=2000.0, mi=mD):
    """Двоточкова модель (провідність, без втрат): T в еВ, n в м^-3, Q в Вт/м^2."""
    Tu = (3.5 * Q * L / kappa0) ** (2 / 7)  # T_t^{7/2} << T_u^{7/2}
    for _ in range(50):
        # Q = γ n_t e T_t c_s(T_t),  n_t T_t = n_u T_u / 2,  c_s = sqrt(2 e T_t / m_i)
        cs = 2 * Q / (gam * nu * e * Tu)
        Tt = mi * cs ** 2 / (2 * e)
        Tu = (Tt ** 3.5 + 3.5 * Q * L / kappa0) ** (2 / 7)
    nt = nu * Tu / (2 * Tt)
    return Tu, Tt, nt


def oml_float(Ti_Te, mi):
    """Плаваючий потенціал ізольованої сфери (OML, без емісії): x = eφ/T_e < 0.
    I_i = πa² e n sqrt(8T_i/πm_i)(1 - x T_e/T_i),  I_e = πa² e n sqrt(8T_e/πm_e) exp(x)."""
    f = lambda x: math.sqrt(Ti_Te * me / mi) * (1 - x / Ti_Te) - math.exp(x)
    lo, hi = -10.0, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:  # f спадає з x: корінь праворуч
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("== Шар ==")
    for name, mi in (("H", mH), ("D", mD)):
        for tau in (0.0, 1.0):
            v = sheath(tau, mi)
            print(f"{name} Ti/Te={tau}: eV_sh/Te={v:.3f}; повне падіння (з передшаром ln2)={v - math.log(2):.3f}; "
                  f"gamma={gamma_heat(tau, mi):.2f}")
    Te = Ti = 20.0
    vtot = -(sheath(1.0, mD) - math.log(2))
    print(f"T=20 еВ, D: |eΦ_total|={vtot*Te:.1f} еВ")
    # Стандартна оцінка енергії удару (Стенгбі, поза корпусом): E_imp ≈ 3 Z T_e + 2 T_i
    for name, Z in (("D+", 1), ("W4+", 4)):
        print(f"  E_imp({name}) = 3ZT_e+2T_i = {3*Z*Te + 2*Ti:.0f} еВ  "
              f"(з точним падінням {vtot:.2f}T_e: {Z*vtot*Te + 2*Ti:.0f} еВ)")

    print("== Двоточкова модель (Q=30 МВт/м², L=40 м, D, γ=7) ==")
    for nu in (1.0e19, 2.0e19, 3.0e19, 4.0e19):
        Tu, Tt, nt = two_point(30e6, 40.0, nu)
        print(f"n_u={nu:.2e}: T_u={Tu:.1f} еВ, T_t={Tt:.2f} еВ, n_t={nt:.3e} м^-3")

    print("== Пилинка C, a=1 мкм, T_e=T_i=10 еВ, n=1e18 м^-3, D-плазма ==")
    a, T, n = 1e-6, 10.0, 1e18
    lamD = math.sqrt(eps0 * T / (n * e))
    x = oml_float(1.0, mD)
    phi = x * T
    Qd = 4 * math.pi * eps0 * a * (1 + a / lamD) * phi
    print(f"lambda_D={lamD*1e6:.1f} мкм, a/lambda_D={a/lamD:.3f}, eφ/T_e={x:.3f}, φ={phi:.1f} В, "
          f"Q={Qd:.3e} Кл = {Qd/e:.0f} e")
    print(f"час подвоєння радіуса при 10 нм/с: {a/10e-9:.0f} с")
