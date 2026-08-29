"""Mass-spring incremental-potential testbed (V2.2 — constraint-projection dynamics edges).

The constraint-projection family (PBD, XPBD, Projective Dynamics, fast-mass-spring, nonlinear
Gauss-Seidel/pbng) is defined on MASS-SPRING systems, where the elastic energy is a sum of squared
distance constraints  E(x) = Σ_e (k/2)(‖x_i−x_j‖ − L_e)². On this substrate the methods are
*faithful* (unlike the FEM fixed-proxy stand-in of bench/incremental.py): local/global == exact
Projective Dynamics (Liu 2013), and XPBD uses the exact Macklin-2016 compliance update. One
implicit-Euler step minimizes  Φ(x) = 1/(2h²)(x−x̃)ᵀM(x−x̃) + E(x). NB XPBD's constraint
Gauss-Seidel is consistent only in the harmonic/small-violation limit — its fixed point does NOT in
general equal the Φ-minimum (it satisfies the compliant constraints but omits the momentum pull),
which is precisely what the primal-xpbd/pbng→xpbd edges measure.

Two claims this settles cleanly, both hardware-independent:
  (1) XPBD/PBD STAGNATE on the incremental-potential residual (they satisfy the constraints but omit
      the momentum coupling), while local/global, Newton, and nonlinear Gauss-Seidel drive it to 0 —
      tests primal-xpbd→xpbd and pbng→xpbd ("reaches tolerance where XPBD stagnates").
  (2) XPBD stiffness is iteration-count independent (compliance α=1/(k h²)); PBD over-stiffens with
      more iterations — tests xpbd→pbd ("PBD stiffens with iteration count").

Conformance-gated (`python -m bench.massspring`): ∇Φ vs FD; PD global matrix SPD.
"""
import numpy as np
from .mesh import grid_mesh, boundary_mask


def _edges(tris):
    s = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            s.add((min(a, b), max(a, b)))
    return np.array(sorted(s), int)


class MSProblem:
    def __init__(self, n=8, dt=1.0 / 30, k=1.0e3, overshoot=1.6, density=1.0):
        rest, tris = grid_mesh(n, n)
        E = _edges(tris)
        L = np.linalg.norm(rest[E[:, 0]] - rest[E[:, 1]], axis=1)     # rest lengths
        pin = rest[:, 0] < 1e-9
        free = ~np.repeat(pin, 2)
        # lumped mass ~ incident edge count (uniform-ish)
        m = np.full(rest.shape[0], density)
        Md = np.repeat(m, 2)
        xtil = rest.copy()
        xtil[:, 0] = rest[:, 0] + overshoot * rest[:, 0]              # stretch prediction
        xtil[:, 1] = rest[:, 1] - 0.1
        xtil[pin] = rest[pin]                                         # pinned vertices are HELD at rest
        self.rest = rest; self.tris = tris; self.E = E; self.L = L; self.k = k
        self.free = free; self.pin = pin; self.Md = Md; self.dt = dt
        self.inv_dt2 = 1.0 / (dt * dt); self.nv = rest.shape[0]
        self.xtil = xtil.reshape(-1); self.x0 = rest.reshape(-1).copy()

    # ---- energy / gradient / hessian ----
    def elastic(self, x):
        X = x.reshape(-1, 2)
        d = X[self.E[:, 0]] - X[self.E[:, 1]]
        l = np.linalg.norm(d, axis=1)
        return 0.5 * self.k * np.sum((l - self.L) ** 2)

    def phi(self, x):
        dx = x - self.xtil
        return self.elastic(x) + 0.5 * self.inv_dt2 * float(dx @ (self.Md * dx))

    def grad(self, x):
        X = x.reshape(-1, 2); g = np.zeros_like(X)
        d = X[self.E[:, 0]] - X[self.E[:, 1]]
        l = np.linalg.norm(d, axis=1) + 1e-15
        coef = (self.k * (l - self.L) / l)[:, None] * d      # k(l-L) u
        np.add.at(g, self.E[:, 0], coef)
        np.add.at(g, self.E[:, 1], -coef)
        g = g.reshape(-1) + self.inv_dt2 * (self.Md * (x - self.xtil))
        return g

    def hess(self, x, spd=True):
        X = x.reshape(-1, 2); H = np.zeros((2 * self.nv, 2 * self.nv))
        for e, (i, j) in enumerate(self.E):
            d = X[i] - X[j]; l = float(np.linalg.norm(d)) + 1e-15; u = d / l
            uu = np.outer(u, u); Le = self.L[e]
            Ke = self.k * (uu + max(0.0, (l - Le) / l) * (np.eye(2) - uu) if spd
                           else uu + ((l - Le) / l) * (np.eye(2) - uu))
            for (a, sa) in ((i, 1), (j, -1)):
                for (b, sb) in ((i, 1), (j, -1)):
                    H[2 * a:2 * a + 2, 2 * b:2 * b + 2] += sa * sb * Ke
        H += np.diag(self.inv_dt2 * self.Md)
        return H

    def resid(self, x):
        return float(np.max(np.abs(self.grad(x)[self.free])))

    # ---- PD global system (constant): M/h^2 + Σ_e k A_e^T A_e (weighted graph Laplacian ⊗ I2) ----
    def pd_system(self):
        A = np.diag(self.inv_dt2 * self.Md).copy()
        for (i, j) in self.E:
            for (a, sa) in ((i, 1), (j, -1)):
                for (b, sb) in ((i, 1), (j, -1)):
                    A[2 * a:2 * a + 2, 2 * b:2 * b + 2] += self.k * sa * sb * np.eye(2)
        return A


def _iters_to(res, rtol=1e-3):
    for k, r in enumerate(res):
        if r <= rtol * res[0]:
            return k
    return None


# ---- solvers ---------------------------------------------------------------------------------
def solve_newton(P, max_iter=200, rtol=1e-3):
    x = P.xtil.copy(); free = P.free; res = []
    for _ in range(max_iter):
        g = P.grad(x); res.append(float(np.max(np.abs(g[free]))))
        if res[-1] <= rtol * res[0]:
            break
        H = P.hess(x, spd=True)
        d = np.zeros_like(x)
        d[free] = np.linalg.solve(H[np.ix_(free, free)], -g[free])
        a = 1.0; E0 = P.phi(x); gd = float(g[free] @ d[free]); x0 = x.copy()
        while a > 1e-14:
            x = x0 + a * d
            if P.phi(x) <= E0 + 1e-4 * a * gd:
                break
            a *= 0.5
    return {"name": "newton", "res": res, "it": _iters_to(res, rtol), "x": x}


def solve_pd(P, max_iter=600, rtol=1e-3):
    """Projective Dynamics / fast-mass-spring (Liu 2013): local projection to rest length + global
    solve of the CONSTANT system. Exact local/global for mass-spring."""
    from scipy.linalg import cho_factor, cho_solve
    free = P.free; pin = ~free
    A = P.pd_system()
    Aff = cho_factor(A[np.ix_(free, free)], lower=True)
    Afp_xpin = A[np.ix_(free, pin)] @ P.x0[pin]          # constant: pinned dofs stay at rest
    x = P.xtil.copy(); res = []
    for _ in range(max_iter):
        res.append(P.resid(x))
        if res[-1] <= rtol * res[0]:
            break
        X = x.reshape(-1, 2)
        d = X[P.E[:, 0]] - X[P.E[:, 1]]
        l = np.linalg.norm(d, axis=1)[:, None] + 1e-15
        p = P.L[:, None] * (d / l)                       # local step: projected rest-length vectors
        rhs = (P.inv_dt2 * P.Md * P.xtil).reshape(-1, 2)  # global rhs: M/h^2 xtil + Σ k A_e^T p_e
        np.add.at(rhs, P.E[:, 0], P.k * p)
        np.add.at(rhs, P.E[:, 1], -P.k * p)
        rhs = rhs.reshape(-1)
        xn = x.copy()
        xn[free] = cho_solve(Aff, rhs[free] - Afp_xpin)   # subtract pinned coupling A_fp x_pin
        x = xn
    return {"name": "pd", "res": res, "it": _iters_to(res, rtol), "x": x}


def _pbd_sweep(P, x, lam, xpbd):
    """One Gauss-Seidel sweep over spring constraints. xpbd=True uses compliance (α̃=1/(k h²));
    xpbd=False is plain PBD (no compliance -> stiffness grows with iteration count)."""
    X = x.reshape(-1, 2)
    w = 1.0 / P.Md.reshape(-1, 2)[:, 0]                  # inverse mass per vertex (isotropic)
    a_tilde = (1.0 / P.k) * P.inv_dt2 if xpbd else 0.0   # XPBD compliance; PBD -> 0
    for e, (i, j) in enumerate(P.E):
        d = X[i] - X[j]; l = float(np.linalg.norm(d)) + 1e-15; u = d / l
        C = l - P.L[e]
        wi = w[i] if P.free[2 * i] else 0.0
        wj = w[j] if P.free[2 * j] else 0.0
        if wi + wj == 0:
            continue
        dl = (-C - a_tilde * lam[e]) / (wi + wj + a_tilde)
        X[i] += wi * u * dl
        X[j] += -wj * u * dl
        lam[e] += dl
    return x, lam


def solve_pbd(P, xpbd=True, max_iter=400, rtol=1e-3):
    x = P.x0.copy(); lam = np.zeros(len(P.E)); res = []
    # PBD/XPBD start each step from the inertial prediction x̃ (standard), not rest
    x = P.xtil.copy()
    for _ in range(max_iter):
        res.append(P.resid(x))
        if res[-1] <= rtol * res[0]:
            break
        x, lam = _pbd_sweep(P, x, lam, xpbd)
    return {"name": "xpbd" if xpbd else "pbd", "res": res, "it": _iters_to(res, rtol), "x": x}


def solve_pbng(P, max_iter=600, rtol=1e-3):
    """Nonlinear Gauss-Seidel (pbng-style): per-vertex local Newton on Φ over incident springs."""
    from .incremental import _iters_to as _it
    free_v = [i for i in range(P.nv) if P.free[2 * i]]
    inc = [[] for _ in range(P.nv)]
    for e, (i, j) in enumerate(P.E):
        inc[i].append((e, 1)); inc[j].append((e, -1))
    x = P.xtil.copy(); res = []
    for _ in range(max_iter):
        res.append(P.resid(x))
        if res[-1] <= rtol * res[0]:
            break
        X = x.reshape(-1, 2)
        for vi in free_v:
            gi = P.inv_dt2 * P.Md[2 * vi:2 * vi + 2] * (X[vi] - P.xtil.reshape(-1, 2)[vi])
            Hi = np.diag(P.inv_dt2 * P.Md[2 * vi:2 * vi + 2]).astype(float)
            for (e, s) in inc[vi]:
                i, j = P.E[e]; other = j if vi == i else i
                d = X[vi] - X[other]; l = float(np.linalg.norm(d)) + 1e-15; u = d / l
                gi = gi + P.k * (l - P.L[e]) * u
                uu = np.outer(u, u)
                Hi = Hi + P.k * (uu + max(0.0, (l - P.L[e]) / l) * (np.eye(2) - uu))
            X[vi] = X[vi] - np.linalg.solve(Hi, gi)
    return {"name": "pbng", "res": res, "it": _iters_to(res, rtol), "x": x}


# ---- conformance -----------------------------------------------------------------------------
def _conformance():
    P = MSProblem(n=5)
    rng = np.random.default_rng(0)
    x = P.x0 + 0.05 * rng.standard_normal(P.x0.shape)
    g = P.grad(x); gfd = np.zeros_like(g); h = 1e-6
    for kk in range(g.size):
        xp = x.copy(); xp[kk] += h; xm = x.copy(); xm[kk] -= h
        gfd[kk] = (P.phi(xp) - P.phi(xm)) / (2 * h)
    gerr = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    a0min = float(np.linalg.eigvalsh(P.pd_system()[np.ix_(P.free, P.free)]).min())
    return gerr, a0min


if __name__ == "__main__":
    import sys
    gerr, a0min = _conformance()
    ok = gerr < 1e-5 and a0min > 0
    print(f"[massspring conformance] gradPhi vs FD: {gerr:.1e} | PD system min eig: {a0min:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
