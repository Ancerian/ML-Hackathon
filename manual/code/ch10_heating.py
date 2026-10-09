"""Еталонні розрахунки до розд. 10 (додатковий нагрів): задачі 1-2.
Формули: Fukuyama et al. 1992, (8), (9), (11), (12); Hanada et al. 1990 (WT-3)."""
import numpy as np
from scipy.integrate import quad

e, me, mp, eps0, c = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8
u = 1.66053907e-27
mD, mT, mA = 2.01410178*u - me, 3.0160492*u - me, 4.001506*u

def H(y):
    """Частка енергії швидкої частинки, що йде іонам (Fukuyama (11))."""
    return 2/y**2*quad(lambda x: x/(x**3+1), 0, y)[0]

def slowing(Eb_keV, mb, Zb, Te_keV, ne, ions, lnL=17.0):
    """ions: список (n_i/n_e, Z_i, m_i). Повертає tau_s, v_c, E_c, y=v_b/v_c, H, tau_b."""
    Te = Te_keV*1e3*e
    taus = 6*np.pi*np.sqrt(2*np.pi)*eps0**2*mb*Te**1.5/(ne*Zb**2*np.sqrt(me)*e**4*lnL)
    vc3 = sum(3*np.sqrt(np.pi/2)*f*Z**2*me/m*(Te/me)**1.5 for f, Z, m in ions)
    vc = vc3**(1/3)
    Ec = 0.5*mb*vc**2/(1e3*e)
    vb = np.sqrt(2*Eb_keV*1e3*e/mb)
    y = vb/vc
    h = H(y)
    return dict(tau_s=taus, v_c=vc, E_c_keV=Ec, y=y, H=h, tau_b=0.5*taus*(1-h))

if __name__ == "__main__":
    print("--- Задача 1: WT-3 ---")
    f = 56e9
    Bres = 2*np.pi*f*me/(2*e)
    print(f"B_res (2-га гармоніка) = {Bres:.4f} Тл")
    R0, a = 0.65, 0.20
    for R in (R0, R0-0.05, R0-a, R0+a):
        print(f"  R_res={R:.2f} м -> B0 = {Bres*R/R0:.3f} Тл")
    print(f"  при B0=1,75 Тл: R_res = {R0*1.75/Bres:.3f} м > R0+a = {R0+a:.2f} м (поза плазмою)")
    ncO = eps0*me*(2*np.pi*f)**2/e**2
    print(f"n_c(O, 56 ГГц) = {ncO:.3e} м^-3; n_e = 5e18 -> n_e/n_c = {5e18/ncO:.3f}")
    # X-мода: відсічка R: w = [Wce + sqrt(Wce^2+4wpe^2)]/2  -> wpe^2 = w(w - Wce)
    w = 2*np.pi*f
    for B in (1.0,):
        Wce = e*B/me
        ncR = eps0*me*w*(w-Wce)/e**2
        print(f"  X-відсічка R при B={B} Тл: n = {ncR:.3e} м^-3")
    Ip = 80e3
    for V in (1.5, 0.8):
        print(f"  P_OH = {V*Ip/1e3:.0f} кВт при V_L={V} В")
    print(f"  Спітцер: V_L2/V_L1 = (510/670)^1,5 = {(510/670)**1.5:.3f} -> V_L2 = {1.5*(510/670)**1.5:.2f} В")
    print(f"  200 кВт / 120 кВт = {200/120:.2f}; 200/64 = {200/64:.2f}")
    print("--- Задача 2: ITER (Fukuyama), T_e=10 кеВ, <n_e>=0,7e20 ---")
    ne = 0.7e20
    DT = [(0.5, 1, mD), (0.5, 1, mT)]
    for name, Eb, mb, Zb in (("D 1 МеВ", 1000, mD, 1), ("alpha 3,5 МеВ", 3500, mA, 2)):
        r = slowing(Eb, mb, Zb, 10.0, ne, DT)
        print(name, {k: (f"{v:.4g}") for k, v in r.items()})
    r = slowing(80, mD, 1, 10.0, 3.8e19, [(1, 1, mD)])
    print("JET-подібний D 80 кеВ у D, T_e=10 кеВ, n_e=3,8e19:", {k: f"{v:.4g}" for k, v in r.items()})
    print("E_c/T_e для D у D:", slowing(80, mD, 1, 1.0, 1e19, [(1,1,mD)])["E_c_keV"])
    # Марчук: НГР
    B, n = 2.5, 1e17
    Wi = e*B/mD; wpi2 = n*e**2/(eps0*mD); wpe2 = n*e**2/(eps0*me); We = e*B/me
    wLH = np.sqrt(Wi**2 + wpi2/(1+wpe2/We**2))
    print(f"Марчук: Omega_D={Wi:.3e}, omega_pi={np.sqrt(wpi2):.3e}, wpe2/We2={wpe2/We**2:.2e}, omega_LH={wLH:.3e} рад/с")
    nres = (3e8**2 - Wi**2)*eps0*mD/e**2
    print(f"  густина НГР для omega=3e8: {nres:.3e} м^-3")
    # перевірка Тищенка 26,7 МГц
    print(f"f_cD(3,5 Тл) = {e*3.5/mD/2/np.pi/1e6:.2f} МГц; f_c,alpha = {2*e*3.5/mA/2/np.pi/1e6:.2f} МГц")
    # Спітцер: eta/mu0 при 10 кеВ
    for Z in (1.0, 1.5, 2.0):
        eta = 1.65e-9*Z*17/10**1.5
        print(f"  Z={Z}: eta_Sp={eta:.2e} Ом·м, eta/mu0={eta/(4e-7*np.pi):.2e} м^2/с")
    print(f"ECR 2,45 ГГц: B = {2*np.pi*2.45e9*me/e:.4f} Тл; n_c = {eps0*me*(2*np.pi*2.45e9)**2/e**2:.3e}")
    print("ln sigma: nu для sigma=10:", 1/(np.log(10)-0.4228))
