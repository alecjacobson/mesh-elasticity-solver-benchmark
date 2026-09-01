"""Quadratic (P2, 10-node) tetrahedral element — the locking-RELIEVED 3D control that completes the
§8.1 headline in 3D. The P1 constant-strain tet volumetrically LOCKS near incompressibility (its
strain is constant, so it cannot represent near-isochoric bending); the P2 element's strain varies
within the element and relieves that locking, exactly as a standard P2 / Taylor–Hood element does in
computational mechanics.

Same Neo-Hookean material and analytic tangent as bench/tet_scale.py, integrated with a 4-point
(degree-3) tet quadrature over quadratic shape functions. Sparse assembly + sparse-LU projected-Newton
with per-element clamp/absolute filtering. Conformance (`python -m bench.tet_p2`): rigid invariance
(zero energy/grad on a rotated+translated rest mesh), and analytic element grad & Hessian vs finite
differences. The payoff — absolute vs clamp near ν→½ on P2 vs P1 — is in run_tet3d_filters (P2 mode).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .tet import box_tet_mesh

# 6 edges of the reference tet, in the order used for the 6 midside nodes (nodes 4..9)
_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

# 4-point degree-3 tetrahedron quadrature (reference tet volume 1/6; weights sum to 1/6)
_QA = (5.0 - np.sqrt(5.0)) / 20.0
_QB = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_QPTS = np.array([[_QA, _QA, _QA], [_QB, _QA, _QA], [_QA, _QB, _QA], [_QA, _QA, _QB]])
_QW = np.full(4, 1.0 / 24.0)


def _dN_ref(xi):
    """Gradients d N_a / d(ξ,η,ζ) of the 10 P2 shape functions at reference point xi=(ξ,η,ζ)."""
    x, y, z = xi
    L = np.array([1.0 - x - y - z, x, y, z])
    dL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dN = np.zeros((10, 3))
    for i in range(4):                                   # corners: N=L_i(2L_i-1)
        dN[i] = (4.0 * L[i] - 1.0) * dL[i]
    for e, (i, j) in enumerate(_EDGES):                  # midsides: N=4 L_i L_j
        dN[4 + e] = 4.0 * (L[j] * dL[i] + L[i] * dL[j])
    return dN


def build_p2(nx, ny, nz, W=1.0):
    """P2 mesh from the P1 box: corners + one node at every unique tet edge midpoint (straight edges,
    so the rest map is affine and the rest Jacobian is the P1 Dm — constant per element)."""
    Vc, tets = box_tet_mesh(nx, ny, nz, W=W)
    nC = Vc.shape[0]
    mid = {}
    extra = []
    conn = np.zeros((len(tets), 10), dtype=int)
    for t, tet in enumerate(tets):
        conn[t, :4] = tet
        for e, (a, b) in enumerate(_EDGES):
            i, j = int(tet[a]), int(tet[b]); key = (min(i, j), max(i, j))
            if key not in mid:
                mid[key] = nC + len(extra)
                extra.append(0.5 * (Vc[i] + Vc[j]))
            conn[t, 4 + e] = mid[key]
    V = np.vstack([Vc, np.array(extra)]) if extra else Vc
    return V, conn


# ---- Neo-Hookean material (same as tet_scale) ----
def _psi(F, mu, lam):
    J = np.linalg.det(F)
    if J <= 0:
        return np.inf
    return 0.5 * mu * (np.sum(F * F) - 3.0) - mu * np.log(J) + 0.5 * lam * np.log(J) ** 2


def _PK1(F, mu, lam):
    J = np.linalg.det(F); Finv = np.linalg.inv(F)
    return mu * F + (lam * np.log(J) - mu) * Finv.T


def _tangent(F, mu, lam):
    """C (9x9), dvec(P)/dvec(F), row-major vec index p=3i+I (vectorised)."""
    J = np.linalg.det(F); logJ = np.log(J); A = np.linalg.inv(F)   # A[I,i]=(F^-1)_{Ii}
    coef = mu - lam * logJ
    T2 = np.einsum('Ii,Jj->iIjJ', A, A).reshape(9, 9)             # A[I,i]A[J,j]
    T3 = np.einsum('Ij,Ji->iIjJ', A, A).reshape(9, 9)             # A[I,j]A[J,i]
    return mu * np.eye(9) + lam * T2 + coef * T3


class P2Problem:
    def __init__(self, n=3, mu=1.0, lam=1.0, stretch=1.3, W=1.0):
        self.mu, self.lam = float(mu), float(lam)
        V, conn = build_p2(n, n, n, W=W)
        self.rest = V; self.conn = conn; self.nv = V.shape[0]
        # precompute per-element: Dm^-1 (from 4 corners), |det Dm|, and G_a(qp)=Dm^-T dN_ref (10x3) per qp
        self.DmInv = []; self.detDm = []; self.Gqp = []
        for c in conn:
            X = V[c]
            Dm = np.column_stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]))
            di = np.linalg.inv(Dm); self.DmInv.append(di); self.detDm.append(abs(np.linalg.det(Dm)))
            self.Gqp.append([di.T @ _dN_ref(qp).T for qp in _QPTS])   # each: (3,10) columns G_a
        # BCs: pin x=0 face, stretch x=W face (all nodes incl. midsides on those planes)
        x0 = V[:, 0]
        lo = np.isclose(x0, 0.0); hi = np.isclose(x0, W)
        self.pinned = lo | hi
        self.free = np.repeat(~self.pinned, 3)
        xi = V.copy(); xi[hi, 0] = W * stretch
        self.x0 = xi.reshape(-1)
        self._fidx = np.where(self.free)[0]
        # constant sparsity scatter
        edof = np.zeros((len(conn), 30), dtype=np.int64)
        for a in range(10):
            edof[:, 3 * a:3 * a + 3] = 3 * conn[:, a][:, None] + np.array([0, 1, 2])
        self.edof = edof
        r = np.broadcast_to(edof[:, :, None], (len(conn), 30, 30))
        c = np.broadcast_to(edof[:, None, :], (len(conn), 30, 30))
        self._rows = r.reshape(-1); self._cols = c.reshape(-1)

    def _Bqp(self, G):
        """B (9x30): vec(F)=B x_elem, F_{ab}=Σ_a x_a[.] ; B[3a+b,3i+a]=G_i[b] with G (3,10)."""
        B = np.zeros((9, 30))
        for a in range(3):
            for b in range(3):
                for i in range(10):
                    B[3 * a + b, 3 * i + a] = G[b, i]
        return B

    def _elem(self, xe, t, want_H=True, filt="clamp"):
        E = 0.0; g = np.zeros(30); H = np.zeros((30, 30)) if want_H else None
        for qi in range(4):
            G = self.Gqp[t][qi]; B = self._Bqp(G)
            F = (B @ xe).reshape(3, 3)
            if np.linalg.det(F) <= 0:
                return np.inf, None, None
            w = self.detDm[t] * _QW[qi]
            E += w * _psi(F, self.mu, self.lam)
            g += w * (B.T @ _PK1(F, self.mu, self.lam).reshape(9))
            if want_H:
                H += w * (B.T @ _tangent(F, self.mu, self.lam) @ B)
        if want_H:
            H = 0.5 * (H + H.T)
            wv, U = np.linalg.eigh(H)
            wv = np.maximum(wv, 1e-9) if filt == "clamp" else np.maximum(np.abs(wv), 1e-9)
            H = (U * wv) @ U.T
        return E, g, H

    def energy(self, x):
        E = 0.0
        for t, c in enumerate(self.conn):
            Ee = self._elem(x[self.edof[t]], t, want_H=False)[0]
            if not np.isfinite(Ee):
                return np.inf
            E += Ee
        return E

    def grad(self, x):
        g = np.zeros(3 * self.nv)
        for t in range(len(self.conn)):
            _, ge, _ = self._elem(x[self.edof[t]], t, want_H=False)
            g[self.edof[t]] += ge
        return g

    def assemble(self, x, filt="clamp"):
        g = np.zeros(3 * self.nv); data = np.empty(len(self.conn) * 900)
        for t in range(len(self.conn)):
            Ee, ge, He = self._elem(x[self.edof[t]], t, want_H=True, filt=filt)
            g[self.edof[t]] += ge
            data[t * 900:(t + 1) * 900] = He.reshape(-1)
        H = sp.coo_matrix((data, (self._rows, self._cols)),
                          shape=(3 * self.nv, 3 * self.nv)).tocsc()
        return g, H


def solve_newton(P, filt="clamp", max_iter=200, tol=1e-6, c=1e-4):
    x = P.x0.copy(); fi = P._fidx; res = []; status = "maxiter"; t0 = time.perf_counter()
    for it in range(max_iter):
        g, H = P.assemble(x, filt)
        gn = float(np.max(np.abs(g[fi]))); res.append(gn)
        if gn < tol:
            status = "converged"; break
        Hff = H[fi][:, fi]
        try:
            step = spla.splu(Hff.tocsc()).solve(-g[fi])
        except Exception:
            step = spla.cg(Hff, -g[fi], maxiter=3000)[0]
        d = np.zeros_like(x); d[fi] = step
        gd = float(g[fi] @ step)
        if gd >= 0:
            status = "nondescent"; break
        E0 = P.energy(x); a = 1.0; x0f = x.copy()
        while a > 1e-14:
            x = x0f + a * d
            if np.isfinite(P.energy(x)) and P.energy(x) <= E0 + c * a * gd:
                break
            a *= 0.5
        else:
            x = x0f; status = "linesearch"; break
    return {"status": status, "iters": len(res) - (1 if status == "converged" else 0),
            "res": res, "wall_s": time.perf_counter() - t0, "x": x}


def _conformance(seed=0, h=1e-6):
    rng = np.random.default_rng(seed)
    P = P2Problem(n=2, mu=1.0, lam=1.0)
    # rigid invariance: rotate+translate rest -> F=R at every qp -> psi=0, grad=0
    th = 0.4; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    xr = (P.rest @ R.T + np.array([0.2, -0.1, 0.3])).reshape(-1)
    Erig = P.energy(xr); grig = float(np.max(np.abs(P.grad(xr))))
    # analytic grad & (unprojected) Hessian vs FD on a perturbed state
    x = (P.rest + 0.02 * rng.standard_normal(P.rest.shape)).reshape(-1)
    g = P.grad(x); gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (P.energy(xp) - P.energy(xm)) / (2 * h)
    grel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    # element Hessian vs FD of element grad (single element, unprojected)
    t = 0; xe = x[P.edof[t]]
    _, _, He = P._elem(xe, t, want_H=True, filt="none") if False else (None, None, None)
    # build unprojected element H by summing qp contributions
    Hn = np.zeros((30, 30))
    for qi in range(4):
        G = P.Gqp[t][qi]; B = P._Bqp(G); F = (B @ xe).reshape(3, 3)
        w = P.detDm[t] * _QW[qi]; Hn += w * (B.T @ _tangent(F, P.mu, P.lam) @ B)
    Hfd = np.zeros((30, 30))
    for k in range(30):
        xp = xe.copy(); xp[k] += h; xm = xe.copy(); xm[k] -= h
        gp = P._elem(xp, t, want_H=False)[1]; gm = P._elem(xm, t, want_H=False)[1]
        Hfd[:, k] = (gp - gm) / (2 * h)
    Hfd = 0.5 * (Hfd + Hfd.T)
    hrel = np.max(np.abs(Hn - Hfd)) / (np.max(np.abs(Hfd)) + 1e-12)
    return abs(Erig), grig, grel, hrel


if __name__ == "__main__":
    import sys
    erig, grig, grel, hrel = _conformance()
    ok = erig < 1e-9 and grig < 1e-8 and grel < 1e-5 and hrel < 1e-4
    print(f"[tet_p2 conformance] rigidE {erig:.1e} rigidG {grig:.1e}  grad/FD {grel:.1e}  "
          f"elemHess/FD {hrel:.1e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
