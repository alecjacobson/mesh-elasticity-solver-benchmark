"""FAITHFUL TLC — Total Lifted Content ("Lifting Simplices to Find Injectivity", Du, Aigerman, Zhou,
Kovalsky, Yan, Kaufman, Ju, SIGGRAPH 2020, arXiv:2010.08551), reimplemented from the paper + the
authors' reference code (github.com/duxingyi-charles/lifting_simplices_to_find_injectivity).

For a 2D triangle with current squared edge lengths (d12,d13,d23), lift each edge by a fixed auxiliary
(Tutte / uniform form: add the same alpha to every squared edge length), and sum the *lifted content*
(area) of each lifted triangle:

    E_TLC(x) = Σ_t area( d12+alpha, d13+alpha, d23+alpha ),
    16·area(p,q,r)² = 2(pq+qr+rp) − (p²+q²+r²)   (Cayley–Menger on squared edge lengths, Eq. 16),
    alpha = 1e-6 · |signed area of the target/init| / (nF · √3/4)   (Tutte form, Sec. 6.1).

Because the auxiliary block is non-degenerate, the lifted content is strictly positive and C^∞
EVERYWHERE — including at inverted/degenerate triangles (Prop. 4.1). So, unlike a barrier distortion
energy (+∞ at a fold), TLC can start from a folded map and slide across folds; for small enough alpha
its global minima are injective (Prop. 4.3). Minimized by L-BFGS, stopping the instant every triangle
is positively oriented (Sec. 5). Fixed boundary: handle vertices are removed from the variables.

Conformance (`python -m bench.tlc`): energy FINITE at a folded config (barrier-free — the defining
property), analytic gradient vs finite differences, the alpha→0 limit equals total unsigned area, and
TLC untangles a folded initialization to fully injective. Adjudicates the injectivity-cohort edges
(§8.4). Drives: `run_tlc.py`.
"""
import numpy as np
from .untangle import signed_areas


def alpha_tutte(V, tris, ratio=1e-6):
    """alpha = ratio · |target signed area| / (nF · √3/4)  (Tutte/uniform auxiliary, Sec. 6.1)."""
    a = float(abs(signed_areas(V, tris).sum()))
    rest = len(tris) * np.sqrt(3.0) / 4.0
    return ratio * max(a, 1e-30) / rest


def energy_grad(x, tris, alpha):
    """TLC energy (lifted content, Tutte form) and its analytic gradient (flat 2·nv vector)."""
    V = x.reshape(-1, 2)
    g = np.zeros_like(V)
    E = 0.0
    for (i, j, k) in tris:
        eij = V[j] - V[i]; eik = V[k] - V[i]; ejk = V[k] - V[j]
        p = float(eij @ eij) + alpha            # lifted squared edge lengths
        q = float(eik @ eik) + alpha
        r = float(ejk @ ejk) + alpha
        S = 2.0 * (p * q + q * r + r * p) - (p * p + q * q + r * r)   # = 16·area²  (>0, lifted)
        S = max(S, 1e-300)
        area = np.sqrt(S) / 4.0
        E += area
        # dE/dp = (dS/dp)/(8√S);  dS/dp = 2(q+r) − 2p, etc.
        c = 1.0 / (8.0 * np.sqrt(S))
        dp = c * (2.0 * (q + r) - 2.0 * p)
        dq = c * (2.0 * (p + r) - 2.0 * q)
        dr = c * (2.0 * (p + q) - 2.0 * r)
        # d(squared edge)/d(vertex): p=|Vj−Vi|² -> dp/dVi=2(Vi−Vj), dp/dVj=2(Vj−Vi)
        g[i] += dp * (2.0 * (V[i] - V[j])) + dq * (2.0 * (V[i] - V[k]))
        g[j] += dp * (2.0 * (V[j] - V[i])) + dr * (2.0 * (V[j] - V[k]))
        g[k] += dq * (2.0 * (V[k] - V[i])) + dr * (2.0 * (V[k] - V[j]))
    return E, g.reshape(-1)


def solve(x0, tris, free, alpha=None, ratio=1e-6, max_iter=4000, tol=1e-9):
    """Minimize TLC over free DOFs with L-BFGS-B; track the iteration at which all triangles first
    become positively oriented (`first_injective`) — the metric comparable across untangling energies.
    Handles (pinned vertices) are held fixed by excluding them from the variables."""
    from scipy.optimize import minimize
    x = x0.copy()
    fidx = np.where(free)[0]
    if alpha is None:
        alpha = alpha_tutte(x.reshape(-1, 2), tris, ratio)
    state = {"it": 0, "first_inj": None}

    def fun(xf):
        x[fidx] = xf
        E, g = energy_grad(x, tris, alpha)
        return E, g[fidx]

    def cb(xf):
        state["it"] += 1
        if state["first_inj"] is None:
            x[fidx] = xf
            if signed_areas(x.reshape(-1, 2), tris).min() > 0:
                state["first_inj"] = state["it"]

    res = minimize(fun, x[fidx], jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": max_iter, "ftol": 1e-16, "gtol": tol})
    x[fidx] = res.x
    a = signed_areas(x.reshape(-1, 2), tris)
    return {"x": x, "alpha": alpha, "min_area": float(a.min()), "n_inverted": int((a <= 0).sum()),
            "iters": int(res.nit), "first_injective": state["first_inj"],
            "success": bool(a.min() > 0), "fun": float(res.fun)}


def _conformance(seed=0, h=1e-6):
    """Barrier-free (finite at a fold), analytic grad vs FD, alpha→0 == total unsigned area, untangles."""
    from .run_injectivity import folded_init
    _rest, tris, _Bs, _areas, free, x0, _target = folded_init(strength=1.0, seed=seed)
    V0 = x0.reshape(-1, 2)
    alpha = alpha_tutte(V0, tris)
    # (1) barrier-free: energy finite at the folded (inverted) start
    E0, g0 = energy_grad(x0, tris, alpha)
    finite = np.isfinite(E0) and np.all(np.isfinite(g0))
    # (2) analytic grad vs FD (on the free dofs)
    fidx = np.where(free)[0]
    rng = np.random.default_rng(seed)
    xt = x0.copy(); xt[fidx] += 0.01 * rng.standard_normal(fidx.size)
    _, g = energy_grad(xt, tris, alpha)
    gfd = np.zeros_like(g)
    for d in fidx:
        xp = xt.copy(); xp[d] += h; xm = xt.copy(); xm[d] -= h
        gfd[d] = (energy_grad(xp, tris, alpha)[0] - energy_grad(xm, tris, alpha)[0]) / (2 * h)
    grel = np.max(np.abs((g - gfd)[fidx])) / (np.max(np.abs(gfd[fidx])) + 1e-12)
    # (3) alpha→0 limit == total unsigned area
    E_small, _ = energy_grad(x0, tris, 1e-14)
    tua = float(np.sum(np.abs(signed_areas(V0, tris))))
    tua_rel = abs(E_small - tua) / (tua + 1e-12)
    # (4) untangles the folded init to fully injective
    r = solve(x0, tris, free, max_iter=4000)
    return finite, grel, tua_rel, r["success"], r["first_injective"], r["n_inverted"]


if __name__ == "__main__":
    import sys
    finite, grel, tua_rel, ok_inj, fi, ninv = _conformance()
    ok = finite and grel < 1e-5 and tua_rel < 1e-6 and ok_inj
    print(f"[tlc conformance] barrier-free={finite}  grad/FD={grel:.2e}  alpha->0==TUA={tua_rel:.2e}  "
          f"untangles={ok_inj} (first-injective it={fi}, {ninv} left) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
