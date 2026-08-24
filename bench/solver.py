"""Search-direction + line-search + linear-solver + criterion slots: projected Newton.

Emits a per-iteration telemetry log (harness.md: metrics are computed from the log, never
hand-reported). One config = (energy, filter, line search, solver, criterion); E1 swaps only
`filter` and holds the rest fixed. The energy slot is pluggable via `eterms`
(a callable (x_elem, B, area) -> (E, g6, H6x6, detF)); defaults to symmetric Dirichlet.
"""
import time
import numpy as np
from .energy import element_terms as _sd_element_terms
from .filters import project_element


def _dofs(tri):
    return np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                     2 * tri[2], 2 * tri[2] + 1])


def assemble(x, tris, Bs, areas, filt, eterms=_sd_element_terms):
    """Global (E, g, H) with per-element filter applied during assembly.
    Returns E=inf (g=H=None) if any element is inverted."""
    nv = x.size // 2
    g = np.zeros(2 * nv)
    H = np.zeros((2 * nv, 2 * nv))
    E = 0.0
    for t, tri in enumerate(tris):
        dofs = _dofs(tri)
        Ee, ge, He, _ = eterms(x[dofs], Bs[t], areas[t])
        if not np.isfinite(Ee):
            return np.inf, None, None
        E += Ee
        if filt in ("clamp", "absolute"):
            He = project_element(He, filt)
        g[dofs] += ge
        H[np.ix_(dofs, dofs)] += He
    return E, g, H


def energy_only(x, tris, Bs, areas, eterms=_sd_element_terms):
    E = 0.0
    for t, tri in enumerate(tris):
        Ee, _, _, _ = eterms(x[_dofs(tri)], Bs[t], areas[t])
        if not np.isfinite(Ee):
            return np.inf
        E += Ee
    return E


def _spd_shift_solve(Hff, gf):
    """Levenberg identity-shift: smallest tau (doubling) making Hff+tau I PD; return d, tau."""
    n = Hff.shape[0]
    base = 1e-10 * (np.trace(Hff) / n + 1.0)
    tau = 0.0
    for _ in range(60):
        try:
            L = np.linalg.cholesky(Hff + tau * np.eye(n))
            y = np.linalg.solve(L, -gf)
            return np.linalg.solve(L.T, y), tau
        except np.linalg.LinAlgError:
            tau = base if tau == 0.0 else tau * 2.0
    return np.linalg.lstsq(Hff + tau * np.eye(n), -gf, rcond=None)[0], tau


def solve(x0, tris, Bs, areas, free, filt, eterms=_sd_element_terms,
          max_iter=300, tol=1e-6, c=1e-4):
    """Run projected Newton with the given filter + energy. Returns a result dict + per-iter log."""
    x = x0.copy()
    log = []
    t0 = time.perf_counter()
    status = "maxiter"
    for it in range(max_iter):
        E, g, H = assemble(x, tris, Bs, areas, filt, eterms)
        if not np.isfinite(E):
            status = "infeasible"; break
        gf = g[free]
        gnorm = float(np.max(np.abs(gf)))
        Hff = H[np.ix_(free, free)]
        log.append({"iter": it, "energy": E, "grad_inf": gnorm,
                    "wall_s": time.perf_counter() - t0})
        if gnorm < tol:
            status = "converged"; break

        if filt == "identity-shift":
            d, _ = _spd_shift_solve(Hff, gf)
        else:
            try:
                d = np.linalg.solve(Hff, -gf)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(Hff, -gf, rcond=None)[0]
        gd = float(gf @ d)
        if gd >= 0.0:                        # not a descent direction (raw/none can hit this)
            status = "nondescent"; break

        alpha = 1.0
        xf0 = x[free].copy()
        while True:
            x[free] = xf0 + alpha * d
            En = energy_only(x, tris, Bs, areas, eterms)
            if np.isfinite(En) and En <= E + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-14:
                x[free] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    wall = time.perf_counter() - t0
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": filt, "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": wall, "log": log}
