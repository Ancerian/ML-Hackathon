"""Independent numerical check of Landreman, arXiv:2609.26742 (both families)."""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(0)
H = 1e-4  # 4th-order central differences


def grad(f, r):
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = H
        g[i] = (-f(r + 2*e) + 8*f(r + e) - 8*f(r - e) + f(r - 2*e)) / (12*H)
    return g


def jac(B, r):
    J = np.zeros((3, 3))  # J[j,i] = d B_j / d x_i
    for i in range(3):
        e = np.zeros(3); e[i] = H
        J[:, i] = (-B(r + 2*e) + 8*B(r + e) - 8*B(r - e) + B(r - 2*e)) / (12*H)
    return J


def check(B, p, psi, pts, name):
    div, fb, bpsi, scale = [], [], [], []
    for r in pts:
        J = jac(B, r); b = B(r)
        curl = np.array([J[2, 1]-J[1, 2], J[0, 2]-J[2, 0], J[1, 0]-J[0, 1]])
        gp = grad(p, r)
        div.append(abs(np.trace(J)))
        fb.append(np.linalg.norm(np.cross(curl, b) - gp))
        scale.append(np.linalg.norm(gp) + np.linalg.norm(np.cross(curl, b)))
        bpsi.append(abs(b @ grad(psi, r)) / (np.linalg.norm(b)*np.linalg.norm(grad(psi, r))))
    print(f"[{name}] n={len(pts)}  max|divB|={max(div):.1e}  max|JxB-grad p|={max(fb):.1e}"
          f" (typ |grad p| {np.median(scale)/2:.2e})  max|b.grad psi|/(|B||grad psi|)={max(bpsi):.1e}")


# ---------------- Family 1 (iota = 2) ----------------
def fam1(eps):
    a, b = np.sqrt(1+eps), np.sqrt(1-eps)

    def B(r):
        x, y, z = r
        s = x*x/a**2 + y*y/b**2
        F = np.sqrt(1 - (1-s)**2 - 4*z*z)
        return np.array([(2*z*x - (a/b)*F*y)/s, (2*z*y + (b/a)*F*x)/s, 1 - s])

    def psi(r):
        x, y, z = r; bb = B(r)
        return (x*x + y*y + 4*z*z + bb @ bb - 2 + eps**2) / 4

    p = lambda r: 1.0 - 2*psi(r)

    def pos(u, v, zeta):
        q2 = u*u + v*v
        L = np.sqrt((1 + np.sqrt(1 - 4*q2))/2)
        return np.array([a*(L*np.cos(zeta) + (u*np.cos(zeta) + v*np.sin(zeta))/L),
                         b*(L*np.sin(zeta) + (v*np.cos(zeta) - u*np.sin(zeta))/L),
                         v*np.cos(2*zeta) - u*np.sin(2*zeta)])
    return a, b, B, psi, p, pos


for eps in (0.25, 0.5, 0.7):
    a, b, B, psi, p, pos = fam1(eps)
    delta = 0.9*(1-eps)**2/4
    pts, err_psi, err_map = [], [], []
    for _ in range(40):
        k = np.sqrt(delta*rng.random()); al = 2*np.pi*rng.random(); ze = 2*np.pi*rng.random()
        u, v = -eps/2 + k*np.cos(al), k*np.sin(al)
        r = pos(u, v, ze); pts.append(r)
        err_psi.append(abs(psi(r) - ((u+eps/2)**2 + v*v)))           # eq 2.15
        dr = (pos(u, v, ze+1e-6) - pos(u, v, ze-1e-6))/2e-6          # B = d r / d zeta
        err_map.append(np.linalg.norm(dr - B(r)))
    check(B, p, psi, pts, f"F1 eps={eps}")
    print(f"   eq2.15 psi(u,v) err {max(err_psi):.1e};  B = dr/dzeta err {max(err_map):.1e}")
    # Jacobian (2.13)
    u, v, ze = -eps/2+0.05, 0.03, 0.7; h = 1e-6
    Jm = np.column_stack([(pos(u+h, v, ze)-pos(u-h, v, ze))/(2*h), (pos(u, v+h, ze)-pos(u, v-h, ze))/(2*h),
                          (pos(u, v, ze+h)-pos(u, v, ze-h))/(2*h)])
    print(f"   det d(x,y,z)/d(u,v,zeta) = {np.linalg.det(Jm):.8f}  vs -ab = {-a*b:.8f}")
    # stellarator symmetry & 2 field periods
    r = pts[0]; R1 = np.array([r[0], -r[1], -r[2]])
    print(f"   stell-sym |B(x,-y,-z)-(-Bx,By,Bz)| = {np.linalg.norm(B(R1)-np.array([-1, 1, 1])*B(r)):.1e};"
          f"  NFP=2 |B(-x,-y,z)-(-Bx,-By,Bz)| = {np.linalg.norm(B(np.array([-r[0], -r[1], r[2]]))-np.array([-1, -1, 1])*B(r)):.1e}")

# beta example eps=1/2, delta=1/64 : (2.29) and Monte-Carlo check of (2.27)
eps, delta = 0.5, 1/64
a, b, B, psi, p, pos = fam1(eps)
N = 20000; k = np.sqrt(delta*rng.random(N)); al = 2*np.pi*rng.random(N); ze = 2*np.pi*rng.random(N)
B2 = np.array([B(pos(-eps/2+kk*np.cos(aa), kk*np.sin(aa), zz)) @ B(pos(-eps/2+kk*np.cos(aa), kk*np.sin(aa), zz))
               for kk, aa, zz in zip(k, al, ze)])
print(f"[F1 beta] <|B|^2>_V MC = {B2.mean():.5f} (+-{B2.std()/np.sqrt(N):.5f}) vs 1-eps^2/2+delta = {1-eps**2/2+delta:.5f};"
      f"  beta_V = {2*delta/(1-eps**2/2+delta):.6f} vs 2/57 = {2/57:.6f}")

# iota=2 by tracing a field line, eps=0.5
def iota_trace(B, axis_fn, r0, nturn):
    # integrate in toroidal angle phi; theta measured around the axis point at same phi
    def rhs(phi, r):
        b = B(r); R = np.hypot(r[0], r[1])
        bphi = (-b[0]*r[1] + b[1]*r[0]) / R
        return b * R / bphi
    sol = solve_ivp(rhs, [0, 2*np.pi*nturn], r0, rtol=1e-11, atol=1e-12, dense_output=True, max_step=0.02)
    phis = np.linspace(0, 2*np.pi*nturn, 4000*nturn)
    rr = sol.sol(phis)
    Ra, Za = axis_fn(phis)
    th = np.unwrap(np.arctan2(-(rr[2]-Za), np.hypot(rr[0], rr[1]) - Ra))
    return (th[-1]-th[0]) / (phis[-1]-phis[0]), rr


# axis of fam1: gamma(zeta) — tabulate Z_a(phi) numerically
def axis1(eps):
    zz = np.linspace(0, 2*np.pi, 20001)
    g = np.array([np.sqrt(1-eps**2)*np.cos(zz), np.sqrt(1-eps**2)*np.sin(zz), eps/2*np.sin(2*zz)])
    ph = np.unwrap(np.arctan2(g[1], g[0]))
    return lambda phi: (np.sqrt(1-eps**2)*np.ones_like(phi), np.interp(np.mod(phi, 2*np.pi), np.mod(ph, 2*np.pi)[np.argsort(np.mod(ph, 2*np.pi))], g[2][np.argsort(np.mod(ph, 2*np.pi))]))


eps = 0.5; a, b, B, psi, p, pos = fam1(eps)
r0 = pos(-eps/2 + 0.1, 0.0, 0.0)
io, rr = iota_trace(B, axis1(eps), r0, 6)
print(f"[F1 iota] traced iota = {io:.6f} (claim 2); closure after 1 transit: {np.linalg.norm(rr[:, 4000]-rr[:, 0]):.1e}")

# ---------------- Family 2 (sheared iota) ----------------
def fam2(eps, S, lam):
    def parts(r):
        x, y, z = r
        w = x + 1j*y; wb = np.conj(w)
        K = wb*np.sqrt(1 + eps/wb**2)
        if (K/wb).real < 0: K = -K
        Xi = w*K + np.pi/2 - S
        return z, K, Xi

    def B(r):
        z, K, Xi = parts(r)
        Wh = 1j*np.exp(-1j*lam*z)*np.sin(Xi)/(2*K)
        return np.array([Wh.real, Wh.imag, (np.exp(-1j*lam*z)*np.cos(Xi)).real/lam])

    def psi(r):
        z, K, Xi = parts(r)
        return 0.5*(np.sin(lam*z)**2 + ((np.exp(-1j*lam*z)*np.cos(Xi)).real)**2)

    p = lambda r: 1.0 - psi(r)/lam**2

    def pos(X, Y, zeta):
        h = lambda sg: np.sqrt(4*sg*sg + eps**2)
        nu = eps/2*np.sin(2*zeta)
        sg = S + np.arctan(np.tanh(nu)*Y/np.sqrt(1-Y*Y)) - np.arcsin(X/np.sqrt(np.cosh(nu)**2 - Y*Y))
        ac, bc = np.sqrt((h(sg)-eps)/2), np.sqrt((h(sg)+eps)/2)
        return np.array([ac*np.cos(zeta), bc*np.sin(zeta), -np.arcsin(Y)/lam])
    return B, psi, p, pos


for (eps, S, delta, lam) in [(1.08, 3, 0.245, 3.5), (4, 3.5, 0.245, 3.5), (5.6, 4, 0.32, 2), (2, 1, 1/200, 1.7)]:
    B, psi, p, pos = fam2(eps, S, lam); kb = np.sqrt(2*delta)
    pts, e1, e2, e3 = [], [], [], []
    for _ in range(40):
        k = kb*np.sqrt(rng.random()); chi = 2*np.pi*rng.random(); ze = 2*np.pi*rng.random()
        X, Y = -k*np.cos(chi), k*np.sin(chi)
        r = pos(X, Y, ze); pts.append(r)
        e1.append(abs(psi(r) - (X*X+Y*Y)/2)); e2.append(abs(lam*B(r)[2] - X)); e3.append(abs(-np.sin(lam*r[2]) - Y))
    check(B, p, psi, pts, f"F2 eps={eps} S={S} lam={lam}")
    print(f"   map (3.12-3.13): |psi-(X^2+Y^2)/2| {max(e1):.1e}, |lam Bz - X| {max(e2):.1e}, |Y+sin(lam z)| {max(e3):.1e}")
    r = pts[0]
    print(f"   stell-sym {np.linalg.norm(B(np.array([r[0], -r[1], -r[2]]))-np.array([-1, 1, 1])*B(r)):.1e};"
          f"  NFP=2 {np.linalg.norm(B(np.array([-r[0], -r[1], r[2]]))-np.array([-1, -1, 1])*B(r)):.1e}")

# iota profile for (3.27): eps=2, S=1, kb=0.1 -- via (3.24) and via direct tracing
eps, S, lam = 2.0, 1.0, 1.7
B, psi, p, pos = fam2(eps, S, lam)
hS = np.sqrt(4*S*S + eps**2)
zz = np.linspace(0, 2*np.pi, 200001)[:-1]
Hm = np.mean(1/np.cosh(eps/2*np.sin(2*zz)))
print(f"[F2 iota] (3.26) iota(0) = h0*H = {hS*Hm:.6f}  (claim 2.28690)")


def iota_eq324(k, nturn=400):
    def rhs(ze, c):
        chi = c[0]; X, Y = -k*np.cos(chi), k*np.sin(chi)
        nu = eps/2*np.sin(2*ze)
        sg = S + np.arctan(np.tanh(nu)*Y/np.sqrt(1-Y*Y)) - np.arcsin(X/np.sqrt(np.cosh(nu)**2 - Y*Y))
        G = (np.sqrt(4*sg*sg+eps**2) + eps*np.cos(2*ze))/2
        return [2*G*np.sqrt(1-k*k*np.sin(chi)**2)/np.sqrt(np.cosh(nu)**2 - k*k)]
    s = solve_ivp(rhs, [0, 2*np.pi*nturn], [0.0], rtol=1e-11, atol=1e-12)
    return s.y[0, -1]/(2*np.pi*nturn)


for k in (1e-4, 0.05, 0.1):
    print(f"   (3.24) iota(k={k}) = {iota_eq324(k):.5f}")

# direct field-line tracing of iota at k = 0.1 (axis is planar ellipse z=0)
ac, bc = np.sqrt((hS-eps)/2), np.sqrt((hS+eps)/2)
def axis2(phi):
    zeta = np.arctan2(np.sin(phi)/bc, np.cos(phi)/ac)
    return np.hypot(ac*np.cos(zeta), bc*np.sin(zeta)), np.zeros_like(phi)
for k in (0.03, 0.1):
    r0 = pos(-k, 0.0, 0.0)
    io, _ = iota_trace(B, axis2, r0, 40)
    print(f"   traced iota (k={k}) = {io:.5f}")
