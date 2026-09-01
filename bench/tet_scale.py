"""Scalable 3D tetrahedral hyperelasticity harness: SPARSE assembly + ANALYTIC element Hessian +
sparse-LU projected-Newton. Reaches 1e4-1e5 elements, where the dense/finite-difference path in
bench/tet.py caps out at a few hundred. This is the foundation that moves the benchmark off the 2D toy.

Energy (Neo-Hookean, matching bench/tet.py conventions):
    psi(F) = mu/2 (I1 - 3) - mu logJ + lam/2 log^2 J,   I1 = ||F||_F^2,  J = det F
    P(F)   = mu F + (lam logJ - mu) F^{-T}                             (first Piola-Kirchhoff)
Analytic material tangent C = dvec(P)/dvec(F) (row-major vec, index p = 3i+I for F_iI):
    C[3i+I, 3j+J] = mu delta_ij delta_IJ
                  + lam        (F^{-1})_{Ii} (F^{-1})_{Jj}
                  + (mu-lam logJ) (F^{-1})_{Ij} (F^{-1})_{Ji}
The element stiffness is H_e = vol * B^T C B (12x12); per-element clamp projection makes it SPD.
Everything is VECTORISED over elements (batched einsum + batched eigh), so 50k-100k tets are routine.

Conformance (`python -m bench.tet_scale`): analytic grad & Hessian vs finite differences, rigid
invariance, and agreement of this energy/grad with bench/tet.py on a shared mesh.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .tet import box_tet_mesh, element_B, rest_quantities, make as _make_dense


def _edofs_all(tets):
    """(N,12) global DOF indices per tet: [x0,y0,z0, x1,y1,z1, ...]."""
    N = len(tets)
    d = np.empty((N, 12), dtype=np.int64)
    for a in range(4):
        d[:, 3 * a + 0] = 3 * tets[:, a] + 0
        d[:, 3 * a + 1] = 3 * tets[:, a] + 1
        d[:, 3 * a + 2] = 3 * tets[:, a] + 2
    return d


class TetProblem:
    """Structured box of tets under an axial stretch, with two pinned faces (x=0 held, x=W pulled)."""

    def __init__(self, n=12, mu=1.0, lam=1.0, stretch=1.5, W=1.0, twist=0.0):
        self.mu, self.lam = float(mu), float(lam)
        verts, tets = box_tet_mesh(n, n, n, W=W, H=1.0, D=1.0)
        self.rest = verts.copy()
        self.tets = tets
        self.nv = verts.shape[0]
        Bs, vols = rest_quantities(verts, tets)
        self.B = np.array(Bs)                    # (N,9,12)
        self.vol = np.asarray(vols)              # (N,)
        self.edof = _edofs_all(tets)             # (N,12)
        # boundary conditions: pin the x=0 face; stretch the x=W face to x*stretch
        x0 = verts[:, 0]
        lo = np.isclose(x0, 0.0); hi = np.isclose(x0, W)
        self.pinned = lo | hi
        free_v = ~self.pinned
        self.free = np.repeat(free_v, 3)         # (3nv,) dof mask
        x_init = verts.copy()
        x_init[hi, 0] = W * stretch              # displace the pulled face
        if twist != 0.0:                         # torsion: rotate the x=W face about the bar axis (x)
            th = float(twist); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            yz = verts[hi, 1:3] - 0.5            # about the bar centre (y=z=0.5)
            x_init[hi, 1:3] = (yz @ R.T) + 0.5
        self.x0 = x_init.reshape(-1)
        # precompute sparse scatter indices (constant sparsity pattern)
        r = np.broadcast_to(self.edof[:, :, None], (len(tets), 12, 12))
        c = np.broadcast_to(self.edof[:, None, :], (len(tets), 12, 12))
        self._rows = r.reshape(-1); self._cols = c.reshape(-1)
        # index map for the free-free submatrix
        self._free_idx = np.where(self.free)[0]

    # ---- vectorised element kinematics ----
    def _F(self, x):
        vecF = np.einsum('nij,nj->ni', self.B, x[self.edof])     # (N,9)
        return vecF.reshape(-1, 3, 3)

    def energy(self, x):
        F = self._F(x)
        J = np.linalg.det(F)
        if np.any(J <= 0):
            return np.inf
        I1 = np.einsum('nij,nij->n', F, F)
        psi = 0.5 * self.mu * (I1 - 3.0) - self.mu * np.log(J) + 0.5 * self.lam * np.log(J) ** 2
        return float(np.sum(self.vol * psi))

    def grad(self, x):
        F = self._F(x)
        J = np.linalg.det(F)
        Finv = np.linalg.inv(F)
        FinvT = np.transpose(Finv, (0, 2, 1))
        P = self.mu * F + (self.lam * np.log(J) - self.mu)[:, None, None] * FinvT   # (N,3,3)
        ge = self.vol[:, None] * np.einsum('nij,ni->nj', self.B, P.reshape(-1, 9))  # (N,12)
        g = np.zeros(3 * self.nv)
        np.add.at(g, self.edof, ge)
        return g

    def _elem_hess(self, F, filt="clamp"):
        """Analytic element stiffness (N,12,12). filt: 'none' (raw), 'clamp' (max(λ,ε)),
        'absolute' (max(|λ|,ε)) — the World-2 eigenvalue filters, per element."""
        N = F.shape[0]
        J = np.linalg.det(F)
        logJ = np.log(J)
        A = np.linalg.inv(F)                                   # A[n,I,i] = (F^-1)_{Ii}
        eye9 = np.eye(9)
        # C (N,9,9), index p=3i+I, q=3j+J
        T2 = np.einsum('nIi,nJj->niIjJ', A, A).reshape(N, 9, 9)         # (F^-1)_{Ii}(F^-1)_{Jj}
        T3 = np.einsum('nIj,nJi->niIjJ', A, A).reshape(N, 9, 9)         # (F^-1)_{Ij}(F^-1)_{Ji}
        coef = (self.mu - self.lam * logJ)
        C = self.mu * eye9[None] + self.lam * T2 + coef[:, None, None] * T3
        He = self.vol[:, None, None] * np.einsum('nia,nij,njb->nab', self.B, C, self.B)
        He = 0.5 * (He + np.transpose(He, (0, 2, 1)))
        if filt == "none":
            return He
        w, V = np.linalg.eigh(He)
        if filt == "clamp":
            w = np.maximum(w, 1e-9)
        elif filt == "absolute":
            w = np.maximum(np.abs(w), 1e-9)
        else:
            raise ValueError(filt)
        return np.einsum('nab,nb,ncb->nac', V, w, V)

    def hess(self, x, project=True, filt="clamp"):
        F = self._F(x)
        He = self._elem_hess(F, filt=(filt if project else "none"))    # (N,12,12)
        data = He.reshape(-1)
        H = sp.coo_matrix((data, (self._rows, self._cols)),
                          shape=(3 * self.nv, 3 * self.nv)).tocsr()
        return H


def solve_newton(P, max_iter=200, tol=1e-6, c=1e-4, filt="clamp", verbose=False):
    """Sparse projected-Newton with backtracking line search. Returns iteration & timing log.
    filt selects the per-element SPD filter ('clamp' | 'absolute')."""
    x = P.x0.copy()
    fi = P._free_idx
    res = []
    status = "maxiter"
    t0 = time.perf_counter()
    nfac = 0
    for it in range(max_iter):
        g = P.grad(x)
        gnorm = float(np.max(np.abs(g[fi])))
        res.append(gnorm)
        if gnorm < tol:
            status = "converged"; break
        H = P.hess(x, project=True, filt=filt).tocsc()
        Hff = H[fi][:, fi]
        try:
            lu = spla.splu(Hff)                                # sparse LU (SPD after projection)
            step = lu.solve(-g[fi])
            nfac += 1
        except Exception:
            step = spla.cg(Hff, -g[fi], maxiter=2000)[0]
        d = np.zeros_like(x); d[fi] = step
        gd = float(g[fi] @ step)
        if gd >= 0:
            status = "nondescent"; break
        E0 = P.energy(x); a = 1.0; x0f = x.copy()
        while True:
            x = x0f + a * d
            En = P.energy(x)
            if np.isfinite(En) and En <= E0 + c * a * gd:
                break
            a *= 0.5
            if a < 1e-14:
                x = x0f; status = "linesearch"; break
        if status == "linesearch":
            break
        if verbose:
            print(f"  it={it} |g|={gnorm:.2e} a={a:.2f} E={En:.4f}")
    return {"status": status, "iters": len(res) - (1 if status == "converged" else 0),
            "res": res, "wall_s": time.perf_counter() - t0, "factorizations": nfac, "x": x}


def _conformance(seed=0, h=1e-6):
    """Analytic grad & Hessian vs FD; rigid invariance; agreement with the dense bench/tet.py path."""
    rng = np.random.default_rng(seed)
    P = TetProblem(n=3, mu=1.0, lam=1.0)
    x = (P.rest + 0.02 * rng.standard_normal(P.rest.shape)).reshape(-1)

    # analytic grad vs FD
    g = P.grad(x)
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (P.energy(xp) - P.energy(xm)) / (2 * h)
    grel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)

    # analytic (unprojected) global Hessian vs FD of the gradient
    H = P.hess(x, project=False).toarray()
    Hfd = np.zeros_like(H)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        Hfd[:, k] = (P.grad(xp) - P.grad(xm)) / (2 * h)
    Hfd = 0.5 * (Hfd + Hfd.T)
    hrel = np.max(np.abs(H - Hfd)) / (np.max(np.abs(Hfd)) + 1e-12)

    # rigid invariance: rotate+translate the rest mesh -> zero energy & grad
    th = 0.4; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    xr = (P.rest @ R.T + np.array([0.3, -0.2, 0.1])).reshape(-1)
    Erig = P.energy(xr); grig = float(np.max(np.abs(P.grad(xr))))

    # agreement with the dense bench/tet.py energy/grad on the same perturbed state
    et, _, _ = _make_dense(1.0, 1.0)
    from .tet import assemble as dense_assemble, rest_quantities as dqr
    quad = list(zip(*dqr(P.rest, P.tets)))
    Ed, gd, _ = dense_assemble(x, P.tets, quad, et, "none")
    de = abs(Ed - P.energy(x)) / (abs(Ed) + 1e-12)
    dg = np.max(np.abs(gd - P.grad(x))) / (np.max(np.abs(gd)) + 1e-12)
    return grel, hrel, abs(Erig), grig, de, dg


if __name__ == "__main__":
    import sys
    grel, hrel, erig, grig, de, dg = _conformance()
    ok = grel < 1e-5 and hrel < 1e-4 and erig < 1e-9 and grig < 1e-8 and de < 1e-10 and dg < 1e-10
    print(f"[tet_scale conformance] grad/FD {grel:.2e}  hess/FD {hrel:.2e}  rigidE {erig:.2e}  "
          f"rigidG {grig:.2e}  vs-dense E {de:.2e} g {dg:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
