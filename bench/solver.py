"""Search-direction + line-search + linear-solver + criterion slots: projected Newton.

Emits a per-iteration telemetry log AND hardware-independent cost counters
(assemblies, energy evals, linear solves, factorizations) alongside wall-clock -- the
pairing docs/metrics.md mandates so hardware can't masquerade as algorithm. One config =
(energy, filter, line search, solver, criterion); E1 swaps only `filter`. The energy slot is
pluggable via `eterms` (callable (x_elem,B,area)->(E,g6,H6x6,detF)); default symmetric Dirichlet.
"""
import time
import numpy as np
from .energy import element_terms as _sd_element_terms
from .filters import project_element


def _dofs(tri):
    return np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                     2 * tri[2], 2 * tri[2] + 1])


def assemble(x, tris, Bs, areas, filt, eterms=_sd_element_terms):
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
        if filt in ("clamp", "absolute", "project-on-demand"):
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
    """Levenberg identity-shift; returns (d, tau, n_factorizations)."""
    n = Hff.shape[0]
    base = 1e-10 * (np.trace(Hff) / n + 1.0)
    tau = 0.0
    for k in range(60):
        try:
            L = np.linalg.cholesky(Hff + tau * np.eye(n))
            y = np.linalg.solve(L, -gf)
            return np.linalg.solve(L.T, y), tau, k + 1
        except np.linalg.LinAlgError:
            tau = base if tau == 0.0 else tau * 2.0
    return np.linalg.lstsq(Hff + tau * np.eye(n), -gf, rcond=None)[0], tau, 60


def solve(x0, tris, Bs, areas, free, filt, eterms=_sd_element_terms,
          max_iter=400, tol=1e-6, c=1e-4):
    x = x0.copy()
    log = []
    counts = {"assemblies": 0, "energy_evals": 0, "lin_solves": 0, "factorizations": 0}
    t0 = time.perf_counter()
    status = "maxiter"
    for it in range(max_iter):
        E, g, H = assemble(x, tris, Bs, areas, filt, eterms); counts["assemblies"] += 1
        if not np.isfinite(E):
            status = "infeasible"; break
        gf = g[free]
        gnorm = float(np.max(np.abs(gf)))
        Hff = H[np.ix_(free, free)]
        log.append({"iter": it, "energy": E, "grad_inf": gnorm,
                    "wall_s": time.perf_counter() - t0,
                    "assemblies": counts["assemblies"], "lin_solves": counts["lin_solves"]})
        if gnorm < tol:
            status = "converged"; break

        if filt == "identity-shift":
            d, _, nf = _spd_shift_solve(Hff, gf)
            counts["factorizations"] += nf; counts["lin_solves"] += 1
        elif filt == "global-pdn":
            counts["lin_solves"] += 1
            try:                                    # try true Newton; project (shift) only if indefinite
                L = np.linalg.cholesky(Hff); counts["factorizations"] += 1
                d = np.linalg.solve(L.T, np.linalg.solve(L, -gf))
            except np.linalg.LinAlgError:
                d, _, nf = _spd_shift_solve(Hff, gf); counts["factorizations"] += nf
        else:
            counts["factorizations"] += 1; counts["lin_solves"] += 1
            try:
                d = np.linalg.solve(Hff, -gf)
            except np.linalg.LinAlgError:
                d = np.linalg.lstsq(Hff, -gf, rcond=None)[0]
        gd = float(gf @ d)
        if gd >= 0.0:
            status = "nondescent"; break

        alpha = 1.0
        xf0 = x[free].copy()
        while True:
            x[free] = xf0 + alpha * d
            En = energy_only(x, tris, Bs, areas, eterms); counts["energy_evals"] += 1
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
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": wall,
            "counts": counts, "x": x, "log": log}
