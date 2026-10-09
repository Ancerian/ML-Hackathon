#!/usr/bin/env python3
"""Re-check of the 'hidden symmetry' remark for the letter (2026-10-02).
Family 1: B(x) = A B0(A^-1 x) (paper eq. 2.9), A = diag(a, b, 1). B0 is axisymmetric, so B should be
equivariant under g_t = A R_t A^-1, generator v = A J A^-1 x = (-(a/b) y, (b/a) x, 0).
Test: Lie bracket [v, B] = (v.grad)B - (B.grad)v = 0, and whether |B|, J = curl B, p are invariant.
Family 2: search for ANY affine v = Mx + c with [v, B] = 0, on all four parameter sets used before."""
import io, contextlib, os
import numpy as np
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_landreman.py")).read(), ns)
fam1, fam2, jac, grad = ns["fam1"], ns["fam2"], ns["jac"], ns["grad"]
rng = np.random.default_rng(7)

def curl(F, r):
    J = jac(F, r); return np.array([J[2,1]-J[1,2], J[0,2]-J[2,0], J[1,0]-J[0,1]])

print("== Family 1: generator v = (-(a/b)y, (b/a)x, 0) ==")
for eps in (0.25, 0.5, 0.7):
    a, b, B, psi, p, pos = fam1(eps)
    Mv = np.array([[0, -a/b, 0], [b/a, 0, 0], [0, 0, 0]])
    v = lambda r: Mv @ r
    delta = 0.9*(1-eps)**2/4
    br, dB2, dJ, dp = [], [], [], []
    for _ in range(40):
        k = np.sqrt(delta*rng.uniform(0.05, 1)); al, ze = rng.uniform(0, 2*np.pi, 2)
        r = pos(-eps/2 + k*np.cos(al), k*np.sin(al), ze)
        bracket = jac(B, r) @ v(r) - Mv @ B(r)                     # [v,B]
        br.append(np.linalg.norm(bracket) / (np.linalg.norm(B(r)) * np.linalg.norm(Mv)))
        nv = np.linalg.norm(v(r))
        dB2.append(abs(grad(lambda s: B(s) @ B(s), r) @ v(r)) / (np.linalg.norm(grad(lambda s: B(s) @ B(s), r)) * nv))
        Jr = curl(B, r); bracketJ = jac(lambda s: curl(B, s), r) @ v(r) - Mv @ Jr
        dJ.append(np.linalg.norm(bracketJ) / (np.linalg.norm(Jr) * np.linalg.norm(Mv)))
        dp.append(abs(grad(p, r) @ v(r)) / (np.linalg.norm(grad(p, r)) * nv))
    print(f"eps={eps}: max rel |[v,B]| = {max(br):.1e} | median rel |v.grad|B|^2| = {np.median(dB2):.2f}"
          f" | median rel |[v,J]| = {np.median(dJ):.2f} | median rel |v.grad p| = {np.median(dp):.2f}")

print("\n== Family 2: affine search on every parameter set ==")
for (eps, S, d, lam) in [(1.08, 3, 0.245, 3.5), (4, 3.5, 0.245, 3.5), (5.6, 4, 0.32, 2), (2, 1, 1/200, 1.7)]:
    B, psi, p, pos = fam2(eps, S, lam); kb = np.sqrt(2*d); rows = []
    for _ in range(40):
        k = kb*np.sqrt(rng.uniform(0.05, 1)); chi, ze = rng.uniform(0, 2*np.pi, 2)
        r = pos(-k*np.cos(chi), k*np.sin(chi), ze); J = jac(B, r); bb = B(r); A = np.zeros((3, 12))
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3)); E[i, j] = 1; A[:, 3*i+j] = J @ (E @ r) - E @ bb
            e = np.zeros(3); e[i] = 1; A[:, 9+i] = J @ e
        rows.append(A)
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    print(f"eps={eps}, S={S}, lam={lam}: smallest/largest singular value = {s[-1]/s[0]:.1e}  -> "
          f"{'affine symmetry EXISTS' if s[-1]/s[0] < 1e-6 else 'no affine symmetry'}")
# control: family 2 at eps -> 0 is axisymmetric, so the search must find rotation about z
B, psi, p, pos = fam2(1e-12, 3, 3.5); rows = []
for _ in range(40):
    k = 0.6*np.sqrt(rng.uniform(0.05, 1)); chi, ze = rng.uniform(0, 2*np.pi, 2)
    r = pos(-k*np.cos(chi), k*np.sin(chi), ze); J = jac(B, r); bb = B(r); A = np.zeros((3, 12))
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3)); E[i, j] = 1; A[:, 3*i+j] = J @ (E @ r) - E @ bb
        e = np.zeros(3); e[i] = 1; A[:, 9+i] = J @ e
    rows.append(A)
s = np.linalg.svd(np.vstack(rows), compute_uv=False)
print(f"control eps=1e-12 (axisymmetric): smallest/largest = {s[-1]/s[0]:.1e} -> detector "
      f"{'WORKS' if s[-1]/s[0] < 1e-6 else 'FAILS'}")
