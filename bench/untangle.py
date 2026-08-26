"""Classical untangling objective — maximize the minimum signed area (issue #97).

The graphics injectivity cohort (TLC / foldover-free / progressive-embedding / simplex-assembly) are
barrier-FREE untangling energies: unlike a distortion barrier (symmetric Dirichlet, +∞ at J≤0) they
are finite for folded/degenerate meshes, so they can pull a NON-injective initialization to an
injective one. This implements their shared classical ancestor — the one-sided signed-area penalty
(Freitag–Plassmann 2000; Escobar 2003; Toulorge 2013) —

    E(x) = Σ_t w_t · max(0, δ_t − a_t)² ,   a_t = signed area of triangle t,  δ_t = target > 0

which is C¹, barrier-free, and (when a feasible boundary admits one) has an injective global min with
all a_t ≥ δ_t. This is NOT TLC — it is TLC's ancestor, used here as a faithful, gated stand-in for
"a barrier-free untangling energy". `da_t/dx` is constant per element (signed area is bilinear), so
the gradient is exact and cheap.

`python -m bench.untangle` runs the gradient conformance gate. Used by run_injectivity.
"""
import numpy as np


def signed_areas(V, tris):
    """Per-triangle signed area (2× would be the cross; this is the true area with sign)."""
    out = np.empty(len(tris))
    for t, (i, j, k) in enumerate(tris):
        e1 = V[j] - V[i]; e2 = V[k] - V[i]
        out[t] = 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
    return out


def _grad_area(V, i, j, k):
    """∇ of signed area a=0.5·((Vj−Vi)×(Vk−Vi)) w.r.t. the three vertices (constant in V)."""
    xi, xj, xk = V[i], V[j], V[k]
    # da/dVi, da/dVj, da/dVk  (2-vectors each)
    gi = 0.5 * np.array([xj[1] - xk[1], xk[0] - xj[0]])
    gj = 0.5 * np.array([xk[1] - xi[1], xi[0] - xk[0]])
    gk = 0.5 * np.array([xi[1] - xj[1], xj[0] - xi[0]])
    return gi, gj, gk


def energy_grad(x, tris, delta, w=None):
    """E = Σ w_t max(0, δ_t − a_t)² and its gradient (flat 2·nv). δ may be scalar or per-triangle."""
    V = x.reshape(-1, 2)
    g = np.zeros_like(V)
    E = 0.0
    delta = np.full(len(tris), delta) if np.isscalar(delta) else delta
    w = np.ones(len(tris)) if w is None else w
    for t, (i, j, k) in enumerate(tris):
        e1 = V[j] - V[i]; e2 = V[k] - V[i]
        a = 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        viol = delta[t] - a
        if viol > 0:
            E += w[t] * viol * viol
            gi, gj, gk = _grad_area(V, i, j, k)
            coef = -2.0 * w[t] * viol          # dE/da · da/dx, dE/da = -2 w viol
            g[i] += coef * gi; g[j] += coef * gj; g[k] += coef * gk
    return E, g.reshape(-1)


def solve(x0, tris, free, delta, w=None, max_iter=2000, tol=1e-8):
    """Minimize the untangling energy over free DOFs with scipy L-BFGS-B (barrier-free → robust)."""
    from scipy.optimize import minimize
    x = x0.copy()
    fidx = np.where(free)[0]

    def fun(xf):
        x[fidx] = xf
        E, g = energy_grad(x, tris, delta, w)
        return E, g[fidx]

    res = minimize(fun, x[fidx], jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-16, "gtol": tol})
    x[fidx] = res.x
    a = signed_areas(x.reshape(-1, 2), tris)
    return {"x": x, "min_area": float(a.min()), "n_inverted": int((a <= 0).sum()),
            "iters": int(res.nit), "success": bool(a.min() > 0), "fun": float(res.fun)}


def _conformance(seed=0, h=1e-6):
    from .mesh import grid_mesh
    rng = np.random.default_rng(seed)
    rest, tris = grid_mesh(5, 5)
    x = rest.reshape(-1) + 0.2 * rng.standard_normal(rest.size)   # some triangles violate δ
    delta = 0.3 * (1.0 / 5) ** 2
    E0, g = energy_grad(x, tris, delta)
    worst = 0.0
    for k in range(x.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gk = (energy_grad(xp, tris, delta)[0] - energy_grad(xm, tris, delta)[0]) / (2 * h)
        worst = max(worst, abs(gk - g[k]) / (abs(g[k]) + 1e-8))
    return worst


def run():
    err = _conformance()
    ok = err < 1e-5
    print(f"[untangle] grad vs FD: max rel err {err:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
