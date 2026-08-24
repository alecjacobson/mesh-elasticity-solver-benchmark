"""Alternative search-direction slots: gradient descent, L-BFGS, Adam (first-order / QN).

These are the World-0 honesty baselines for experiment E4 (first- vs second-order): they need
only the gradient (no Hessian assembly), so wall-clock is compared fairly. Adam is full-batch
here (no minibatch noise) -> it behaves as sign/normalized descent and is expected to plateau
above tight tolerances -- the informative control (docs/metrics.md, corpus World-0).
"""
import time
import numpy as np


def _dofs(tri):
    return np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                     2 * tri[2], 2 * tri[2] + 1])


def assemble_eg(x, tris, Bs, areas, eg):
    nv = x.size // 2
    g = np.zeros(2 * nv)
    E = 0.0
    for t, tri in enumerate(tris):
        dofs = _dofs(tri)
        Ee, ge, _ = eg(x[dofs], Bs[t], areas[t])
        if not np.isfinite(Ee):
            return np.inf, None
        E += Ee
        g[dofs] += ge
    return E, g


def _result(method, status, log, wall, counts, x):
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": method, "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": wall,
            "counts": counts, "x": x, "log": log}


def _armijo(x, free, d, E0, gd, tris, Bs, areas, eg, counts, c=1e-4):
    alpha = 1.0
    xf0 = x[free].copy()
    while True:
        x[free] = xf0 + alpha * d
        En, _ = assemble_eg(x, tris, Bs, areas, eg); counts["energy_evals"] += 1
        if np.isfinite(En) and En <= E0 + c * alpha * gd:
            return True
        alpha *= 0.5
        if alpha < 1e-14:
            x[free] = xf0
            return False


def solve_gd(x0, tris, Bs, areas, free, eg, max_iter=5000, tol=1e-6):
    x = x0.copy(); log = []; counts = {"grad_evals": 0, "energy_evals": 0}
    t0 = time.perf_counter(); status = "maxiter"
    for it in range(max_iter):
        E, g = assemble_eg(x, tris, Bs, areas, eg); counts["grad_evals"] += 1
        gf = g[free]; gn = float(np.max(np.abs(gf)))
        log.append({"iter": it, "energy": E, "grad_inf": gn, "wall_s": time.perf_counter() - t0})
        if gn < tol:
            status = "converged"; break
        if not _armijo(x, free, -gf, E, -float(gf @ gf), tris, Bs, areas, eg, counts):
            status = "linesearch"; break
    return _result("gradient-descent", status, log, time.perf_counter() - t0, counts, x)


def solve_lbfgs(x0, tris, Bs, areas, free, eg, m=8, max_iter=5000, tol=1e-6):
    x = x0.copy(); log = []; counts = {"grad_evals": 0, "energy_evals": 0}
    S, Y, rho = [], [], []
    t0 = time.perf_counter(); status = "maxiter"; g_prev = None; x_prev = None
    for it in range(max_iter):
        E, g = assemble_eg(x, tris, Bs, areas, eg); counts["grad_evals"] += 1
        gf = g[free]; gn = float(np.max(np.abs(gf)))
        log.append({"iter": it, "energy": E, "grad_inf": gn, "wall_s": time.perf_counter() - t0})
        if gn < tol:
            status = "converged"; break
        if g_prev is not None:
            s = x[free] - x_prev; y = gf - g_prev
            sy = float(s @ y)
            if sy > 1e-12:
                if len(S) == m:
                    S.pop(0); Y.pop(0); rho.pop(0)
                S.append(s); Y.append(y); rho.append(1.0 / sy)
        q = gf.copy(); alphas = []
        for i in range(len(S) - 1, -1, -1):
            a = rho[i] * float(S[i] @ q); alphas.append(a); q -= a * Y[i]
        gamma = (float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])) if S else 1.0
        r = gamma * q
        for i in range(len(S)):
            b = rho[i] * float(Y[i] @ r); r += S[i] * (alphas[len(S) - 1 - i] - b)
        d = -r
        gd = float(gf @ d)
        if gd >= 0.0:
            d = -gf; gd = -float(gf @ gf)          # safeguard: fall back to steepest
        x_prev = x[free].copy(); g_prev = gf.copy()
        if not _armijo(x, free, d, E, gd, tris, Bs, areas, eg, counts):
            status = "linesearch"; break
    return _result("l-bfgs", status, log, time.perf_counter() - t0, counts, x)


def solve_adam(x0, tris, Bs, areas, free, eg, lr=0.01, b1=0.9, b2=0.999, eps=1e-8,
               max_iter=8000, tol=1e-6):
    x = x0.copy(); log = []; counts = {"grad_evals": 0, "energy_evals": 0}
    mt = np.zeros(int(free.sum())); vt = np.zeros_like(mt)
    t0 = time.perf_counter(); status = "maxiter"
    for it in range(1, max_iter + 1):
        E, g = assemble_eg(x, tris, Bs, areas, eg); counts["grad_evals"] += 1
        gf = g[free]; gn = float(np.max(np.abs(gf)))
        log.append({"iter": it - 1, "energy": E, "grad_inf": gn, "wall_s": time.perf_counter() - t0})
        if gn < tol:
            status = "converged"; break
        mt = b1 * mt + (1 - b1) * gf
        vt = b2 * vt + (1 - b2) * gf * gf
        step = lr * (mt / (1 - b1 ** it)) / (np.sqrt(vt / (1 - b2 ** it)) + eps)
        xf0 = x[free].copy(); k = 0
        while True:                                 # minimal inversion guard (Adam has no line search)
            x[free] = xf0 - step
            En, _ = assemble_eg(x, tris, Bs, areas, eg); counts["energy_evals"] += 1
            if np.isfinite(En):
                break
            step *= 0.5; k += 1
            if k > 25:
                x[free] = xf0; status = "infeasible"; break
        if status == "infeasible":
            break
    return _result("adam", status, log, time.perf_counter() - t0, counts, x)
