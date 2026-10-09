import math as m
e=1.602e-19; me=9.109e-31; mp=1.6726e-27; eps0=8.854e-12; amu=1.6605e-27
def lnL(n,T): return 31.3-m.log(m.sqrt(n)/T)
def taue(n,T,L): TJ=T*e; return 6*m.sqrt(2)*m.pi**1.5*eps0**2*m.sqrt(me)*TJ**1.5/(L*e**4*n)
def taui(n,T,L,A=2): TJ=T*e; return 12*m.pi**1.5*eps0**2*m.sqrt(A*mp)*TJ**1.5/(L*e**4*n)
# P1 core
n=3.8e19; Te=9.9e3; Ti=13.35e3; R=3.0; q=1.0; eps=0.1; B=3.45
L=lnL(n,Te); te=taue(n,Te,L); vte=m.sqrt(Te*e/me); nse=q*R/(eps**1.5*vte*te)
Li=L; ti=taui(n,Ti,Li); vti=m.sqrt(Ti*e/(2*mp)); nsi=q*R/(eps**1.5*vti*ti)
rho=m.sqrt(2*mp*Ti*e)/(e*B)
chi=0.66*q**2*rho**2/(ti*eps**1.5)
print("core lnL=%.2f tau_e=%.3e nu*e=%.4f tau_i=%.3e nu*i=%.4f rho_i=%.3e chi_neo=%.3e ratio=%.0f"%(L,te,nse,ti,nsi,rho,chi,0.5/chi))
# edge
n=2e19; T=100.; q=3.5; eps=0.35
L=lnL(n,T); te=taue(n,T,L); vte=m.sqrt(T*e/me); print("edge lnL=%.2f tau_e=%.3e nu*e=%.2f eps^-1.5=%.2f"%(L,te,q*R/(eps**1.5*vte*te),eps**-1.5))
# Spitzer diffusivity
T=1e4; n=1e20; L=lnL(n,T); te=taue(n,T,L); eta=me/(n*e**2*te)*0.51
print("eta_par=%.3e eta/mu0=%.3e"%(eta,eta/(4e-7*m.pi)))
# ITER89
M=2.5;I=22;R=6;a=2;k=2;nn=1.0;B=4.85;P=100
t=0.048*M**0.5*I**0.85*R**1.2*a**0.3*k**0.5*nn**0.1*B**0.2*P**-0.5
print("ITER89 tau=%.3f x1.5=%.3f"%(t,1.5*t))
print("B doubling: scaling x%.3f, model x2 ; Ip doubling scaling x%.3f"%(2**0.2,2**0.85))
# Wong factor
mGe=72.63*amu; w=1e5; Rr=2.5; mi=2*mp
E=mGe*w**2*Rr**2/e/1e3; ZI=22; TeK=1.23; TiK=3.08
A=1-(mi/mGe)*ZI*TeK/(TiK+TeK)
F=(1+E*A/(2*TiK))**2
print("mGe w2R2=%.1f keV A=%.3f factor=%.1f"%(E,A,F))
# Hulse tau_q
print("tau_q=%.1e s"%(0.02**2/1.0))
# LH
PLH=3.24/2*2.5**0.75*0.4**0.6*1.65**0.98*0.5**0.81
S=4*m.pi**2*1.65*0.5*m.sqrt((1+1.7**2)/2)
print("PLH=%.2f S=%.1f Qn19=%.3f Qn20=%.4f"%(PLH,S,0.0011*4**1.07*2.5**0.76*S,0.0011*0.4**1.07*2.5**0.76*S))
# bootstrap fraction estimate
