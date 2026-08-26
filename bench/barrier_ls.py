"""Barrier-aware line search — BCQN's cheapest component (E3 triple-split, docs/experiments.md).

BCQN (Zhu–Bridson–Kaufman 2018) attributes much of its robustness/speed to a line search that
caps the step at the largest inversion-free length before Armijo backtracking (its Fig.6 suggests
this single component carries most of the win). This implements that component faithfully as a
harness slot so E3 can isolate it from the blended-Sobolev proxy and the characteristic-gradient
criterion.

`max_step_to_inversion(x, d, tris)`: along x+αd every 2D triangle's signed area is the quadratic
A(α)=a α²+b α+c (c = current area > 0). The first positive root is where that element would flip;
the maximum inversion-free step is the min over elements, times a shrink < 1. This is the standard
CCD-for-inversion used by SLIM/BCQN/IPC (2D triangle case, exact — a quadratic, no cubic solve).

`python -m bench.barrier_ls` runs the conformance gate. Used by run_e3 (the 2³ factorial).
"""
import time
import numpy as np
from .descent import assemble_eg, _dofs, _result


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def max_step_to_inversion(x, d, tris, shrink=0.9):
    """Largest α such that no triangle inverts on [0, α], × shrink (∞ if none ever would).

    x, d are flat 2·nv arrays (d already zero on pinned DOFs). Assumes A(0) > 0 for all tris."""
    X = x.reshape(-1, 2); D = d.reshape(-1, 2)
    amin = np.inf
    for t in tris:
        i, j, k = t
        e1x, e1y = X[j] - X[i]
        e2x, e2y = X[k] - X[i]
        f1x, f1y = D[j] - D[i]
        f2x, f2y = D[k] - D[i]
        a = _cross(f1x, f1y, f2x, f2y)
        b = _cross(e1x, e1y, f2x, f2y) + _cross(f1x, f1y, e2x, e2y)
        c = _cross(e1x, e1y, e2x, e2y)          # 2× current signed area (> 0)
        # smallest positive root of a α² + b α + c = 0
        if abs(a) < 1e-14:
            if abs(b) > 1e-14:
                r = -c / b
                if r > 0:
                    amin = min(amin, r)
            continue
        disc = b * b - 4 * a * c
        if disc < 0:
            continue                             # never crosses zero (c>0) → no inversion
        sq = np.sqrt(disc)
        for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
            if r > 1e-15:
                amin = min(amin, r)
    return np.inf if amin == np.inf else shrink * amin


def barrier_armijo(x, free, d, E0, gd, tris, Bs, areas, eg, counts, c=1e-4):
    """Armijo backtracking whose INITIAL step is capped at the inversion-free maximum."""
    nv = x.size // 2
    d_full = np.zeros(2 * nv); d_full[free] = d
    a_max = max_step_to_inversion(x, d_full, tris)
    alpha = min(1.0, a_max)
    xf0 = x[free].copy()
    while True:
        x[free] = xf0 + alpha * d
        En, _ = assemble_eg(x, tris, Bs, areas, eg); counts["energy_evals"] += 1
        if np.isfinite(En) and En <= E0 + c * alpha * gd:
            return True
        alpha *= 0.5
        if alpha < 1e-16:
            x[free] = xf0
            return False


def _tri_areas(x, tris):
    X = x.reshape(-1, 2)
    return np.array([0.5 * _cross(*(X[j] - X[i]), *(X[k] - X[i])) for i, j, k in tris])


def _conformance(seed=0):
    """Gate: at the returned α the min area is ~0⁺ (about to flip) and a slightly larger step
    actually inverts some element — i.e. the step is the tight inversion boundary."""
    from .mesh import grid_mesh
    rng = np.random.default_rng(seed)
    rest, tris = grid_mesh(6, 6)
    x = rest.reshape(-1).copy()
    d = rng.standard_normal(x.size)                     # random direction that will flip something
    a = max_step_to_inversion(x, d, tris, shrink=1.0)   # exact boundary (no shrink)
    if not np.isfinite(a):
        return False, np.inf, np.inf
    A_at = _tri_areas(x + a * d, tris)
    A_past = _tri_areas(x + a * 1.001 * d, tris)
    min_at = float(A_at.min())                           # ~0 (an element at the flip boundary)
    min_past = float(A_past.min())                       # < 0 (that element inverted)
    ok = abs(min_at) < 1e-6 and min_past < 0 and float(_tri_areas(x, tris).min()) > 0
    return ok, min_at, min_past


def solve_lbfgs_barrier(x0, tris, Bs, areas, free, eg, m=8, max_iter=5000, tol=1e-6):
    """L-BFGS with the barrier-aware line search (BCQN component in isolation)."""
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
            s = x[free] - x_prev; y = gf - g_prev; ys = float(y @ s)
            if ys > 1e-12:
                S.append(s); Y.append(y); rho.append(1.0 / ys)
                if len(S) > m:
                    S.pop(0); Y.pop(0); rho.pop(0)
        q = gf.copy(); al = []
        for si, yi, ri in zip(reversed(S), reversed(Y), reversed(rho)):
            a_i = ri * float(si @ q); al.append(a_i); q = q - a_i * yi
        gamma = (float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])) if S else 1.0
        z = gamma * q
        for si, yi, ri, a_i in zip(S, Y, rho, reversed(al)):
            b_i = ri * float(yi @ z); z = z + (a_i - b_i) * si
        d = -z
        gd = float(gf @ d)
        if gd >= 0:
            d = -gf; gd = float(gf @ d)
        x_prev = x[free].copy(); g_prev = gf.copy()
        if not barrier_armijo(x, free, d, E, gd, tris, Bs, areas, eg, counts):
            status = "linesearch"; break
    return _result("l-bfgs-barrier", status, log, time.perf_counter() - t0, counts, x)


def run():
    ok, min_at, min_past = _conformance()
    print(f"[barrier-ls] conformance: min-area at α = {min_at:.2e} (≈0), just past α = {min_past:.2e} "
          f"(<0)  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
