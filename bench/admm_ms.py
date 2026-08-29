"""ADMM and Anderson-accelerated ADMM on the mass-spring incremental potential (V2.3).

Reuses the conformance-gated mass-spring substrate (bench/massspring.MSProblem). ADMM-PD (Overby
et al. 2017) splits the incremental potential  Φ(x)=½h⁻²‖x−x̃‖²_M + Σ_e (k/2)(‖A_e x‖−L_e)²  by
introducing per-spring auxiliaries z_e = A_e x:
  x-update: (M/h² + ρ DᵀD) x = M/h² x̃ + ρ Dᵀ(z − u)     [same constant global system as PD, prefactored]
  z-update: per spring, prox of (k/2)(‖z‖−L)² + (ρ/2)‖z−(A_e x + u_e)‖²   [1-D along the edge]
  dual:     u += A_e x − z
This tests:
  - admm-pd → projective-dynamics (convergence): ADMM vs PD (=local/global) iterations to the same tol.
  - aa-admm → admm (convergence): Anderson acceleration of the ADMM fixed point vs plain ADMM.

Speed/wall-clock headlines are NOT adjudicated (hardware-confounded). Writes results/admm_ms.md.
Run: `python -m bench.run_admm_ms`.
"""
import numpy as np
from .massspring import MSProblem, _iters_to


def _prox_and_dual(P, x, u):
    """z-update (per-spring prox) + return z and the current constraint vectors A_e x."""
    X = x.reshape(-1, 2)
    Ax = X[P.E[:, 0]] - X[P.E[:, 1]]                 # (m,2) spring vectors
    y = Ax + u                                       # target for the prox
    ny = np.linalg.norm(y, axis=1) + 1e-15
    # magnitude prox: s* = argmin (k/2)(s-L)^2 + (rho/2)(s-‖y‖)^2  = (k L + rho ‖y‖)/(k+rho)
    return y, ny, Ax


def solve_admm(P, rho=None, max_iter=600, rtol=1e-3, accel=False, m=5):
    from scipy.linalg import cho_factor, cho_solve
    free = P.free; pin = ~free
    if rho is None:
        rho = P.k                                    # w=½√k <-> a balanced penalty; ρ≈k here
    # constant global system M/h^2 + rho D^T D  (D^T D = weighted graph Laplacian with unit weights)
    A = np.diag(P.inv_dt2 * P.Md).astype(float)
    for (i, j) in P.E:
        for (a, sa) in ((i, 1), (j, -1)):
            for (b, sb) in ((i, 1), (j, -1)):
                A[2 * a:2 * a + 2, 2 * b:2 * b + 2] += rho * sa * sb * np.eye(2)
    Aff = cho_factor(A[np.ix_(free, free)], lower=True)
    Afp_xpin = A[np.ix_(free, pin)] @ P.x0[pin]
    Mxtil = (P.inv_dt2 * P.Md * P.xtil).reshape(-1, 2)

    def x_update(z, u):
        rhs = Mxtil.copy()
        DT = rho * (z - u)                           # D^T (z-u): +to i, -to j
        np.add.at(rhs, P.E[:, 0], DT)
        np.add.at(rhs, P.E[:, 1], -DT)
        rhs = rhs.reshape(-1)
        x = P.x0.copy()
        x[free] = cho_solve(Aff, rhs[free] - Afp_xpin)
        return x

    # init
    x = P.xtil.copy()
    X = x.reshape(-1, 2)
    z = X[P.E[:, 0]] - X[P.E[:, 1]]
    u = np.zeros_like(z)
    res = []

    def admm_step(zu):
        """One ADMM sweep as a fixed-point map on the stacked (z,u); returns new (x, z, u)."""
        z_, u_ = zu[0].copy(), zu[1].copy()
        xx = x_update(z_, u_)
        y, ny, Ax = _prox_and_dual(P, xx, u_)
        s = (P.k * P.L + rho * ny) / (P.k + rho)     # prox magnitude per spring
        zn = s[:, None] * (y / ny[:, None])
        un = u_ + (Ax - zn)
        return xx, zn, un

    if not accel:
        for _ in range(max_iter):
            res.append(P.resid(x))
            if res[-1] <= rtol * res[0]:
                break
            x, z, u = admm_step((z, u))
        return {"name": "admm", "res": res, "it": _iters_to(res, rtol), "x": x}

    # Anderson acceleration of the ADMM fixed point on v=(z,u) with a combined-residual safeguard.
    from .massspring import MSProblem as _  # keep import local
    v = np.concatenate([z.reshape(-1), u.reshape(-1)])
    mnE = len(P.E)

    def split(v):
        return v[:2 * mnE].reshape(mnE, 2), v[2 * mnE:].reshape(mnE, 2)

    def G(v):
        zz, uu = split(v)
        xx, zn, un = admm_step((zz, uu))
        return np.concatenate([zn.reshape(-1), un.reshape(-1)]), xx

    dF, dG = [], []
    Fprev = Gprev = None
    for _ in range(max_iter):
        # current x from a z,u (do one map to get x for residual)
        Gv, xx = G(v)
        res.append(P.resid(xx))
        if res[-1] <= rtol * res[0]:
            x = xx; break
        Fk = Gv - v
        if m > 0 and Fprev is not None:
            dF.append(Fk - Fprev); dG.append(Gv - Gprev)
            if len(dF) > m:
                dF.pop(0); dG.pop(0)
        if dF:
            D = np.array(dF).T
            theta, *_ = np.linalg.lstsq(D, Fk, rcond=None)
            v_new = Gv - np.array(dG).T @ theta
        else:
            v_new = Gv
        # safeguard: accept AA only if the ADMM residual ‖F‖ does not increase, else plain step
        _, xacc = G(v_new)
        if not np.all(np.isfinite(v_new)) or P.resid(xacc) > 1.5 * P.resid(xx):
            v_new = Gv
        Fprev, Gprev = Fk.copy(), Gv.copy()
        v = v_new; x = xx
    return {"name": "aa-admm", "res": res, "it": _iters_to(res, rtol), "x": x}
