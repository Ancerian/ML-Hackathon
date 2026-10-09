import numpy as np, sys
e=1.602e-19; me=9.109e-31; mp=1.673e-27; eps0=8.854e-12
def run(a=1.25,R0=3.0,n0=3.8e19,chi_i=0.5,chi_e=0.27,P=10e6,fi=0.6,w=0.4,Tb=0.1,Zeff=1.5,A=2.0,N=201,brem=True,Palpha=0.0):
    x=np.linspace(0,1,N); r=x*a; dr=r[1]
    n=n0*(0.2+0.8*(1-x**2))
    V=2*np.pi*R0*np.pi*a**2
    g=np.exp(-(x/w)**2)
    norm=np.trapezoid(g*2*np.pi*r*2*np.pi*R0,r)
    Sh=P*g/norm  # W/m3
    Te=np.full(N,1.0); Ti=np.full(N,1.0)  # keV
    dt=1e-3
    for it in range(200000):
        TeJ=Te*1e3*e
        L=31.3-np.log(np.sqrt(n)/(Te*1e3))
        taue=6*np.sqrt(2)*np.pi**1.5*eps0**2*np.sqrt(me)*TeJ**1.5/(L*e**4*n)
        Qei=3*me/(A*mp)*n*(Te-Ti)*1e3*e/taue  # W/m3 from e to i
        Pbr=5.35e-37*Zeff*n**2*np.sqrt(Te) if brem else 0
        Se=(1-fi)*Sh-Qei-Pbr; Si=fi*Sh+Qei
        new=[]
        for T,chi,S in ((Te,chi_e,Se),(Ti,chi_i,Si)):
            # implicit diffusion: 1.5 n dT/dt = (1/r) d/dr(r n chi dT/dr) + S  (T in J)
            TJ=T*1e3*e
            rp=0.5*(r[1:]+r[:-1]); npf=0.5*(n[1:]+n[:-1])
            k=rp*npf*chi/dr**2
            Mat=np.zeros((N,N)); rhs=1.5*n*TJ/dt+S
            for i in range(N):
                if i==0:
                    # symmetry: use volume of first cell r in [0,dr/2]
                    c=(0.5*dr*npf[0]*chi/dr)/((dr/2)**2/2)
                    Mat[0,0]=1.5*n[0]/dt+c; Mat[0,1]=-c
                elif i==N-1:
                    Mat[i,i]=1; rhs[i]=Tb*1e3*e
                else:
                    Mat[i,i]=1.5*n[i]/dt+(k[i-1]+k[i])/r[i]
                    Mat[i,i-1]=-k[i-1]/r[i]; Mat[i,i+1]=-k[i]/r[i]
            new.append(np.linalg.solve(Mat,rhs)/(1e3*e))
        dT=max(abs(new[0]-Te).max(),abs(new[1]-Ti).max())
        Te,Ti=new
        dt=min(dt*1.5,1.0)
        if dT<1e-7 and it>50: break
    W=1.5*np.trapezoid(n*(Te+Ti)*1e3*e*2*np.pi*r*2*np.pi*R0,r)
    Pbrt=np.trapezoid((5.35e-37*Zeff*n**2*np.sqrt(Te) if brem else 0*n)*2*np.pi*r*2*np.pi*R0,r)
    return Te[0],Ti[0],W,W/P, Pbrt
def f_for_Ti0(target=13.35,fi=0.6,N=101):
    """Множник f до обох chi, за якого T_i0 = target (кеВ)."""
    from scipy.optimize import brentq
    g=lambda f: run(chi_i=0.5*f,chi_e=0.27*f,fi=fi,N=N)[1]-target
    f=brentq(g,1.5,4.0,xtol=1e-4)
    return f, run(chi_i=0.5*f,chi_e=0.27*f,fi=fi,N=N)

if __name__=="__main__":
    # базовий випадок тексту (JET: a=1,25 м, R0=3 м), збіжність за сіткою
    for N in (101,401):
        Te0,Ti0,W,tau,Pb=run(N=N)
        print("N=%d Te0=%.2f Ti0=%.2f keV W=%.2f MJ tauE=%.3f s Pbr=%.2f MW"%(N,Te0,Ti0,W/1e6,tau,Pb/1e6))
    # скан множника f: T_i0 = 13,35 кеВ (табл. 4.2 Тищенка)
    for fi in (0.6,0.7):
        f,(Te0,Ti0,W,tau,Pb)=f_for_Ti0(fi=fi)
        print("fi=%.1f f=%.3f Te0=%.2f Ti0=%.2f keV tauE=%.3f s"%(fi,f,Te0,Ti0,tau))
