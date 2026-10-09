import numpy as np
from scipy.special import i0
from scipy.optimize import brentq
e=1.602176634e-19; mD=2.014*1.66054e-27
a=0.82; R0=2.5; B=4.0
nfun=lambda r: 2e19/(1+4*(r/a)**2)
Tfun=lambda r: 10e3/(1+9*(r/a)**2)      # eV
qfun=lambda r: 1.1*(1+6.5455*(r/a)**2)
# ---- A: force balance at r=a/2
x=0.5; r=x*a
n,T,q=nfun(r),Tfun(r),qfun(r); om=2e5/(1+4*x**2)
Bth=r*B/(q*R0)
dlnn=-8*x/a/(1+4*x**2); dlnT=-18*x/a/(1+9*x**2)
dia=T*(dlnn+dlnT); tor=om*R0*Bth
print("A: n=%.2e T=%.0f q=%.3f Bth=%.4f dlnn=%.3f dlnT=%.3f"%(n,T,q,Bth,dlnn,dlnT))
for k in (1.17,-0.5,-1.83):
    pol=-k*T*dlnT
    print("  k=%5.2f dia=%.0f tor=%.0f pol=%.0f Er=%.0f  u_th=%.0f"%(k,dia,tor,pol,dia+tor+pol,k*T*dlnT/B))
# ---- B: steady  (1/r)(r chi mn u')' - nu mn u + S + (1/r)(r Pres)'*(-1) = 0
def solve(N,chi,nu,S,ua=0.0,CT=0.0):
    rf=np.linspace(0,a,N+1); rc=0.5*(rf[1:]+rf[:-1]); dr=a/N
    mn=lambda r: mD*nfun(r)
    ci=lambda r: np.sqrt(Tfun(r)*e/mD)
    dlnTf=lambda r: -18*r/a**2/(1+9*(r/a)**2)
    M=np.zeros((N,N)); b=np.zeros(N)
    for i in range(N):
        vol=rc[i]*dr
        # east face
        rE=rf[i+1]; DE=rE*chi(rE)*mn(rE)/dr
        PresE=-CT*chi(rE)*mn(rE)*ci(rE)*dlnTf(rE)
        if i<N-1:
            M[i,i]-=DE; M[i,i+1]+=DE
        else:
            M[i,i]-=2*DE; b[i]-=2*DE*ua
        b[i]+=rE*PresE   # -(r Pres)' contribution: -(rP)_E + (rP)_W ; moved to rhs => +
        if i>0:
            rW=rf[i]; DW=rW*chi(rW)*mn(rW)/dr
            PresW=-CT*chi(rW)*mn(rW)*ci(rW)*dlnTf(rW)
            M[i,i]-=DW; M[i,i-1]+=DW
            b[i]-=rW*PresW
        M[i,i]-=nu(rc[i])*mn(rc[i])*vol
        b[i]-=S(rc[i])*vol
    u=np.linalg.solve(M,b)
    return rc,u
# test: uniform n? use Bessel check with constant mn: override temporarily
N=400
# verification with constant density
nsave=nfun
nfun=lambda r: 1e19+0*r
chi0=1.0; nu0=10.0; S0=0.1
rc,u=solve(N,lambda r:chi0,lambda r:nu0,lambda r:S0)
mn0=mD*1e19; lam=np.sqrt(chi0/nu0)
ua=S0/(mn0*nu0)*(1-i0(rc/lam)/i0(a/lam))
print("B-test: lam=%.4f max rel err=%.2e u0=%.4e"%(lam,np.max(np.abs(u-ua))/np.max(ua),u[0]))
rc,u=solve(N,lambda r:chi0,lambda r:0.0,lambda r:S0)
ua=S0*(a**2-rc**2)/(4*chi0*mn0)
print("B-test nu=0: err=%.2e"%(np.max(np.abs(u-ua))/np.max(ua)))
nfun=nsave
# TFTR-like
chi=lambda r:1.0
nu=lambda r:2e3*np.exp((r-a)/0.05)
w=0.4
def u0(S0):
    rc,u=solve(N,chi,nu,lambda r:S0*np.exp(-(r/w)**2)); return rc,u
rc,u1=u0(1.0)
S0=5e5/u1[0]
rc,u=u0(S0)
torque=np.trapezoid(S0*np.exp(-(rc/w)**2)*R0*2*np.pi*R0*2*np.pi*rc,rc)
L=np.trapezoid(mD*nfun(rc)*u*R0*2*np.pi*R0*2*np.pi*rc,rc)
print("TFTR: S0=%.4f N/m3 torque=%.2f N m  L=%.3f  tau_phi=%.3f s u(a/2)=%.3e"%(S0,torque,L,L/torque,np.interp(a/2,rc,u)))
i=np.argmin(abs(rc-a/2)); print("  om(a/2)=%.3e Wong: 1e5"%(np.interp(a/2,rc,u)/R0))
# E_r with solved u at a/2
uh=np.interp(a/2,rc,u); print("  Er(k=1.17)=%.0f"%(dia+uh*Bth-1.17*T*dlnT))
# no-cx
rc,uu=solve(N,chi,lambda r:0*r,lambda r:S0*np.exp(-(r/w)**2)); print("  no cx u0=%.3e"%uu[0])
# intrinsic
for CT in (0.1,0.3):
    rc,ui=solve(N,chi,nu,lambda r:0*r,CT=CT)
    print("CT=%.1f intrinsic u(0)=%.3e u(a/2)=%.3e ratio ctr/co=%.3f"%(CT,ui[0],np.interp(a/2,rc,ui),(u[0]-ui[0])/(u[0]+ui[0])))
# CT needed for ratio 2
rc,ui1=solve(N,chi,nu,lambda r:0*r,CT=1.0)
print("CT for ratio2 (same S0):", (u[0]/3)/abs(ui1[0]))
