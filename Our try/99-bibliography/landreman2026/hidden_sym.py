"""Search for affine vector fields v = Mx + c commuting with B ([v,B]=0) and check v.grad p."""
import numpy as np, io, contextlib, os
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_landreman.py")).read(), ns)
jac, grad = ns["jac"], ns["grad"]
rng = np.random.default_rng(1)

def search(B, p, pts, name):
    rows = []
    for r in pts:
        J = jac(B, r); b = B(r)
        # [v,B] = J v - M b,  v = M r + c ; unknowns: M (9), c (3)
        A = np.zeros((3, 12))
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3)); E[i, j] = 1
                A[:, 3*i+j] = J @ (E @ r) - E @ b
            e = np.zeros(3); e[i] = 1
            A[:, 9+i] = J @ e
        rows.append(A)
    A = np.vstack(rows)
    U, s, Vt = np.linalg.svd(A)
    print(f"[{name}] smallest singular values: {np.array2string(s[-4:], precision=2)}")
    v = Vt[-1]; M = v[:9].reshape(3, 3); c = v[9:]
    if s[-1] < 1e-6 * s[0]:
        print("   commuting affine field found; M =\n", np.round(M / np.abs(M).max(), 6), "\n   c =", np.round(c, 6))
        vp = [abs(grad(p, r) @ (M @ r + c)) / np.linalg.norm(grad(p, r)) / np.linalg.norm(M @ r + c) for r in pts]
        print(f"   is p invariant under it?  max |v.grad p|/(|v||grad p|) = {max(vp):.3f}")
    else:
        print("   no affine symmetry of B")

a, b, B, psi, p, pos = ns["fam1"](0.5)
pts = [pos(-0.25 + 0.1*np.cos(t), 0.1*np.sin(t), z) for t, z in rng.random((30, 2))*2*np.pi]
search(B, p, pts, "F1 eps=0.5")
B, psi, p, pos = ns["fam2"](1.08, 3, 3.5)
kb = np.sqrt(0.49)
pts = [pos(-0.5*kb*np.cos(t), 0.5*kb*np.sin(t), z) for t, z in rng.random((30, 2))*2*np.pi]
search(B, p, pts, "F2 eps=1.08 S=3")
