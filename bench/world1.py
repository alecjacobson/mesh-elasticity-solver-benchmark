"""World-1 distortion accelerators on the shared symmetric-Dirichlet energy.

All minimize the SAME energy (bench.energy) so they are directly comparable -- the point of the
E2/E3 decomposition experiments. This module holds the Laplacian-proxy family: the cotan
Laplacian, AQP (Accelerated Quadratic Proxy), and a Sobolev-preconditioned gradient step (the
BCQN proxy core). Each is conformance-checked by reaching the same minimizer as projected Newton.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .energy import element_eg
from .descent import assemble_eg


def cotan_laplacian(rest, tris):
    """Cotan Laplacian (nv x nv) of the rest mesh (SPD up to the constant nullspace)."""
    nv = rest.shape[0]
    I, J, V = [], [], []
    for tri in tris:
        p = rest[tri]                      # 3x2
        for a in range(3):
            i, j, k = tri[a], tri[(a + 1) % 3], tri[(a + 2) % 3]
            # cotangent of the angle at vertex a (between edges a->j and a->k), weights edge (j,k)
            u = rest[j] - rest[a] if False else p[(a + 1) % 3] - p[a]
            v = p[(a + 2) % 3] - p[a]
            cross = abs(u[0] * v[1] - u[1] * v[0]) + 1e-30
            cot = float(u @ v) / cross
            w = 0.5 * cot
            for (r, c, val) in ((j, j, w), (k, k, w), (j, k, -w), (k, j, -w)):
                I.append(r); J.append(c); V.append(val)
    L = sp.coo_matrix((V, (I, J)), shape=(nv, nv)).tocsr()
    return L


def _vfree(free_dof):
    """vertex free-mask from the dof free-mask (vertex free iff its x-dof is free)."""
    return free_dof[0::2]


def solve_aqp(x0, tris, rest, free_dof, eta=100.0, max_iter=3000, tol=1e-6, c=1e-4):
    """Accelerated Quadratic Proxy (Kovalsky-Galun-Lipman 2016). Proxy H = cotan-Laplacian(rest)
    (tensor I2), fixed Nesterov theta from eta, Armijo line search from the accelerated point,
    momentum restart on energy increase. Minimizes symmetric Dirichlet to the Newton minimum."""
    from .mesh import rest_quantities
    nv = rest.shape[0]
    vf = _vfree(free_dof)
    L = cotan_laplacian(rest, tris)
    Lff = (L[vf][:, vf] + 1e-9 * sp.eye(int(vf.sum()))).tocsc()
    solveL = spla.factorized(Lff)
    Bs, areas = rest_quantities(rest, tris)
    theta = (1.0 - np.sqrt(1.0 / eta)) / (1.0 + np.sqrt(1.0 / eta))

    def eg(xx):
        return assemble_eg(xx.reshape(-1), tris, Bs, areas, element_eg)   # (E, g)

    x = x0.copy().reshape(nv, 2)
    x_prev = x.copy()
    Ecur, _ = eg(x)
    log = []; t0 = time.perf_counter(); status = "maxiter"
    counts = {"grad_evals": 0, "energy_evals": 0}
    beta = 0.0
    for it in range(max_iter):
        y = x + beta * (x - x_prev)
        Ey, g = eg(y); counts["grad_evals"] += 1
        if not np.isfinite(Ey):                      # accelerated point inverted -> drop momentum
            y = x.copy(); Ey, g = eg(y); beta = 0.0
        gv = g.reshape(nv, 2)
        gnorm = float(np.max(np.abs(gv[vf])))
        log.append({"iter": it, "energy": Ecur, "grad_inf": gnorm, "wall_s": time.perf_counter() - t0})
        if gnorm < tol:
            status = "converged"; break
        d = np.zeros((nv, 2))
        d[vf, 0] = solveL(-gv[vf, 0]); d[vf, 1] = solveL(-gv[vf, 1])
        gd = float((gv[vf] * d[vf]).sum())           # = -g^T L^-1 g < 0 (descent)
        alpha = 1.0
        while True:
            xn = y + alpha * d
            En, _ = eg(xn); counts["energy_evals"] += 1
            if np.isfinite(En) and En <= Ey + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-14:
                xn = x.copy(); En = Ecur; status = "linesearch"; break
        if status == "linesearch":
            break
        # restart momentum on energy increase (function restart) or if momentum points uphill
        # at the accelerated point (O'Donoghue-Candes gradient restart) -> settles the tail
        uphill = float(g @ (x - x_prev).reshape(-1)) > 0.0
        beta = 0.0 if (En > Ecur or uphill) else theta
        x_prev, x, Ecur = x, xn, En
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": "aqp", "status": status, "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": time.perf_counter() - t0,
            "counts": counts, "x": x.reshape(-1)}
