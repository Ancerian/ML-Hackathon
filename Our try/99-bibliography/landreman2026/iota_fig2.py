import numpy as np
from scipy.integrate import solve_ivp
def iota(eps, S, k, nturn=300):
    def rhs(ze, c):
        chi = c[0]; X, Y = -k*np.cos(chi), k*np.sin(chi); nu = eps/2*np.sin(2*ze)
        sg = S + np.arctan(np.tanh(nu)*Y/np.sqrt(1-Y*Y)) - np.arcsin(X/np.sqrt(np.cosh(nu)**2 - Y*Y))
        G = (np.sqrt(4*sg*sg+eps**2) + eps*np.cos(2*ze))/2
        return [2*G*np.sqrt(1-k*k*np.sin(chi)**2)/np.sqrt(np.cosh(nu)**2 - k*k)]
    s = solve_ivp(rhs, [0, 2*np.pi*nturn], [0.0], rtol=1e-10, atol=1e-12)
    return s.y[0, -1]/(2*np.pi*nturn)
for name, eps, S, d in [("A", 1.08, 3, 0.245), ("B", 4, 3.5, 0.245), ("C", 5.6, 4, 0.32), ("(3.27)", 2, 1, 1/200)]:
    kb = np.sqrt(2*d); i0, ib = iota(eps, S, 1e-4), iota(eps, S, kb)
    print(f"{name}: iota(axis)={i0:.4f} iota(edge)={ib:.4f}  rel.shear={(ib-i0)/i0:+.2%}")
