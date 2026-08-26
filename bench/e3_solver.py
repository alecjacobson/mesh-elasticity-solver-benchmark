"""Unified L-BFGS solver for the E3 2³ factorial (docs/experiments.md §E3, issue #96).

One code path parametrized by BCQN's three components so every cell shares identical logic and only
the toggled factor differs:
  - direction  H0 ∈ {"lbfgs" = γI ,  "sobolev" = cotan-Laplacian⁻¹}   (the blended-Sobolev proxy)
  - line_search ∈ {"backtrack" , "barrier"}                            (barrier = inversion-free cap)
  - criterion   evaluated POST-HOC from the log: ‖g‖∞  and  the area-weighted RMS gradient
                (a mesh/scale-invariant "characteristic gradient" — the BCQN-style criterion).

Each solve logs BOTH gradient norms every iteration, so the criterion factor is applied by
re-reading the log (as in E5): iters-to-target for each criterion, no re-solve. Minimizes symmetric
Dirichlet (element_eg). Used by run_e3 (extended to the full factorial).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .energy import element_eg
from .descent import assemble_eg
from .world1 import cotan_laplacian
from .barrier_ls import max_step_to_inversion


def vertex_areas(rest, tris):
    """Barycentric vertex areas a_v = (1/3) Σ incident triangle areas (rest metric)."""
    nv = rest.shape[0]
    av = np.zeros(nv)
    for i, j, k in tris:
        e1 = rest[j] - rest[i]; e2 = rest[k] - rest[i]
        A = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        av[i] += A / 3.0; av[j] += A / 3.0; av[k] += A / 3.0
    return av


def char_gradnorm(g_full, free_dof, av):
    """Area-weighted RMS gradient over free vertices: sqrt(Σ a_v ‖g_v‖² / Σ a_v). Mesh-invariant."""
    G = g_full.reshape(-1, 2)
    vfree = free_dof[0::2]
    w = av[vfree]
    gv2 = np.sum(G[vfree] ** 2, axis=1)
    return float(np.sqrt(np.sum(w * gv2) / (np.sum(w) + 1e-30)))


def solve_unified(x0, tris, rest, Bs, areas, free, direction="lbfgs", line_search="backtrack",
                  m=8, max_iter=3000, tol=1e-7, c=1e-4):
    """L-BFGS with selectable H0 and line search; logs grad_inf AND characteristic gradient norm."""
    av = vertex_areas(rest, tris)
    fidx = np.where(free)[0]
    nvf = len(fidx) // 2
    if direction == "sobolev":
        vf = free[0::2]
        L = cotan_laplacian(rest, tris)
        Lff = (L[vf][:, vf] + 1e-9 * sp.eye(int(vf.sum()))).tocsc()
        solveL = spla.factorized(Lff)

        def apply_H0(q, gamma):
            qm = q.reshape(nvf, 2); r = np.empty_like(qm)
            r[:, 0] = solveL(qm[:, 0]); r[:, 1] = solveL(qm[:, 1])
            return r.reshape(-1)
    else:
        def apply_H0(q, gamma):
            return gamma * q

    x = x0.copy(); S, Y, rho = [], [], []
    log = []; t0 = time.perf_counter(); status = "maxiter"
    counts = {"grad_evals": 0, "energy_evals": 0}
    g_prev = xf_prev = None
    for it in range(max_iter):
        E, g = assemble_eg(x, tris, Bs, areas, element_eg); counts["grad_evals"] += 1
        gf = g[fidx]; gn = float(np.max(np.abs(gf)))
        log.append({"iter": it, "energy": E, "grad_inf": gn,
                    "char": char_gradnorm(g, free, av), "wall_s": time.perf_counter() - t0})
        if gn < tol:
            status = "converged"; break
        if g_prev is not None:
            s = x[fidx] - xf_prev; y = gf - g_prev; sy = float(s @ y)
            if sy > 1e-12:
                if len(S) == m:
                    S.pop(0); Y.pop(0); rho.pop(0)
                S.append(s); Y.append(y); rho.append(1.0 / sy)
        q = gf.copy(); al = []
        for i in range(len(S) - 1, -1, -1):
            a = rho[i] * float(S[i] @ q); al.append(a); q -= a * Y[i]
        gamma = (float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])) if S else 1.0
        r = apply_H0(q, gamma)
        for i in range(len(S)):
            b = rho[i] * float(Y[i] @ r); r += S[i] * (al[len(S) - 1 - i] - b)
        d = -r
        gd = float(gf @ d)
        if gd >= 0:
            d = -apply_H0(gf, gamma); gd = float(gf @ d)
        xf_prev = x[fidx].copy(); g_prev = gf.copy()
        # line search
        if line_search == "barrier":
            d_full = np.zeros(x.size); d_full[fidx] = d
            alpha = min(1.0, max_step_to_inversion(x, d_full, tris))
        else:
            alpha = 1.0
        xf0 = x[fidx].copy()
        while True:
            x[fidx] = xf0 + alpha * d
            En, _ = assemble_eg(x, tris, Bs, areas, element_eg); counts["energy_evals"] += 1
            if np.isfinite(En) and En <= E + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-16:
                x[fidx] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    return {"direction": direction, "line_search": line_search, "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": log[-1]["energy"] if log else np.inf,
            "counts": counts, "x": x, "log": log}


def iters_to(log, key, target):
    """First iteration where log[key] < target (None if never)."""
    for e in log:
        if e[key] < target:
            return e["iter"]
    return None
