import numpy as np
e=1.602e-19; me=9.109e-31; mp=1.673e-27; eps0=8.854e-12
def lnL(n,TeV): return 31.3-np.log(np.sqrt(n)/TeV)
def tau_e(n,TeV,Z=1):
    L=lnL(n,TeV); T=TeV*e
    return 6*np.sqrt(2)*np.pi**1.5*eps0**2*np.sqrt(me)*T**1.5/(L*e**4*Z*n)
def tau_i(n,TiV,A=2,L=17):
    T=TiV*e; mi=A*mp
    return 12*np.pi**1.5*eps0**2*np.sqrt(mi)*T**1.5/(L*e**4*n)
for (lab,n,T,R,q,eps) in [("core ITER-like",1e20,1e4,6.2,1.5,0.1),("JET core",3.8e19,1e4,2.96,1.2,0.1),("edge",3e19,200,6.2,3.5,0.3),("JET edge",2e19,100,2.96,3.5,0.35)]:
    L=lnL(n,T); te=tau_e(n,T); ti=tau_i(n,T,L=L)
    vte=np.sqrt(T*e/me); vti=np.sqrt(T*e/(2*mp))
    nse=q*R/(eps**1.5*vte*te); nsi=q*R/(eps**1.5*vti*ti)
    print(lab, "lnL=%.1f tau_e=%.2e nu_ei=%.2e tau_i=%.2e nu*e=%.3g nu*i=%.3g eps^-1.5=%.1f"%(L,te,1/te,ti,nse,nsi,eps**-1.5))
