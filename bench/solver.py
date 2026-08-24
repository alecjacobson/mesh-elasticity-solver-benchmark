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


def _cg_solve(Hff, gf, counts, rtol=1e-9, maxiter=5000):
    """Conjugate-gradient inner solve (SPD Hff), counting matrix-vector products (metric #15)."""
    import scipy.sparse.linalg as spla
    n = Hff.shape[0]
    mv = [0]

    def matvec(v):
        mv[0] += 1
        return Hff @ v

    A = spla.LinearOperator((n, n), matvec=matvec)
    d, _info = spla.cg(A, -gf, rtol=rtol, maxiter=maxiter)
    counts["mat_vecs"] += mv[0]
    return d


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
          linsolver="direct", linesearch="backtracking", max_iter=400, tol=1e-6, c=1e-4):
    x = x0.copy()
    log = []
    counts = {"assemblies": 0, "energy_evals": 0, "lin_solves": 0,
              "factorizations": 0, "mat_vecs": 0}
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
            counts["lin_solves"] += 1
            if linsolver == "cg":            # iterative inner solve (SPD-filtered Hff)
                d = _cg_solve(Hff, gf, counts)
            else:
                counts["factorizations"] += 1
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
            if linesearch == "full-step":         # accept once feasible; NO Armijo (no descent guarantee)
                if np.isfinite(En):
                    break
            else:                                  # backtracking Armijo (default)
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


# ---------------------------------------------------------------------------
# Sparse backend (separate from the dense path above so the verified dense
# experiments/smoke test are unaffected). Supports the per-element filters with
# a sparse direct (SuperLU) or CG inner solve -- the linear-solver slot at scale.
# ---------------------------------------------------------------------------

def assemble_sparse(x, tris, Bs, areas, filt, eterms=_sd_element_terms):
    import scipy.sparse as sp
    nv = x.size // 2
    g = np.zeros(2 * nv)
    E = 0.0
    ne = len(tris)
    rows = np.empty(ne * 36, dtype=np.int64)
    cols = np.empty(ne * 36, dtype=np.int64)
    data = np.empty(ne * 36, dtype=float)
    k = 0
    for t, tri in enumerate(tris):
        dofs = _dofs(tri)
        Ee, ge, He, _ = eterms(x[dofs], Bs[t], areas[t])
        if not np.isfinite(Ee):
            return np.inf, None, None
        E += Ee
        if filt in ("clamp", "absolute", "project-on-demand"):
            He = project_element(He, filt)
        g[dofs] += ge
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows[k:k + 36] = rr.ravel(); cols[k:k + 36] = cc.ravel(); data[k:k + 36] = He.ravel()
        k += 36
    H = sp.coo_matrix((data, (rows, cols)), shape=(2 * nv, 2 * nv)).tocsr()
    return E, g, H


def solve_sparse(x0, tris, Bs, areas, free, filt, eterms=_sd_element_terms,
                 linsolver="direct", max_iter=400, tol=1e-6, c=1e-4):
    import scipy.sparse.linalg as spla
    x = x0.copy()
    log = []
    counts = {"assemblies": 0, "energy_evals": 0, "lin_solves": 0,
              "factorizations": 0, "mat_vecs": 0, "nnz": 0}
    fidx = np.where(free)[0]
    t0 = time.perf_counter()
    status = "maxiter"
    for it in range(max_iter):
        E, g, H = assemble_sparse(x, tris, Bs, areas, filt, eterms); counts["assemblies"] += 1
        if not np.isfinite(E):
            status = "infeasible"; break
        gf = g[fidx]
        gnorm = float(np.max(np.abs(gf)))
        Hff = H[fidx][:, fidx]
        counts["nnz"] = int(Hff.nnz)
        log.append({"iter": it, "energy": E, "grad_inf": gnorm,
                    "wall_s": time.perf_counter() - t0, "nnz": int(Hff.nnz)})
        if gnorm < tol:
            status = "converged"; break
        counts["lin_solves"] += 1
        if linsolver in ("cg", "pcg-jacobi"):
            mv = [0]
            def matvec(v, _Hff=Hff, _mv=mv):
                _mv[0] += 1
                return _Hff @ v
            A = spla.LinearOperator(Hff.shape, matvec=matvec)
            M = None
            if linsolver == "pcg-jacobi":                # diagonal (Jacobi) preconditioner
                dinv = 1.0 / Hff.diagonal()
                M = spla.LinearOperator(Hff.shape, matvec=lambda v, _di=dinv: _di * v)
            d, _ = spla.cg(A, -gf, rtol=1e-9, maxiter=10000, M=M)
            counts["mat_vecs"] += mv[0]
        else:
            counts["factorizations"] += 1
            d = spla.spsolve(Hff.tocsc(), -gf)
        gd = float(gf @ d)
        if gd >= 0.0:
            status = "nondescent"; break
        alpha = 1.0; xf0 = x[fidx].copy()
        while True:
            x[fidx] = xf0 + alpha * d
            En = energy_only(x, tris, Bs, areas, eterms); counts["energy_evals"] += 1
            if np.isfinite(En) and En <= E + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-14:
                x[fidx] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    wall = time.perf_counter() - t0
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": filt, "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": wall,
            "counts": counts, "x": x, "log": log}


# ---------------------------------------------------------------------------
# Trust-region (Steihaug-CG) Newton: handles indefinite Hessians INTRINSICALLY
# (negative-curvature-aware truncated CG in a trust radius) -- no eigenvalue
# filter. Tests the survey thesis that graphics filtering ~ classical modified
# Newton / trust region. Uses the raw (unfiltered) Hessian.
# ---------------------------------------------------------------------------

def _steihaug(gf, Hff, Delta, cg_tol, maxit):
    """Truncated CG for min gf.p + 0.5 p.Hff.p s.t. ||p|| <= Delta (Nocedal-Wright Alg 7.2)."""
    p = np.zeros_like(gf)
    r = gf.copy()
    d = -r
    r0 = float(np.sqrt(r @ r))
    if r0 < cg_tol:
        return p, 0
    for j in range(maxit):
        Hd = Hff @ d
        dHd = float(d @ Hd)
        if dHd <= 0:                                   # negative curvature -> to boundary
            a, b, cc = float(d @ d), 2 * float(p @ d), float(p @ p) - Delta * Delta
            tau = (-b + np.sqrt(max(b * b - 4 * a * cc, 0.0))) / (2 * a)
            return p + tau * d, j + 1
        alpha = float(r @ r) / dHd
        p_new = p + alpha * d
        if np.sqrt(p_new @ p_new) >= Delta:            # crossed boundary -> intersect
            a, b, cc = float(d @ d), 2 * float(p @ d), float(p @ p) - Delta * Delta
            tau = (-b + np.sqrt(max(b * b - 4 * a * cc, 0.0))) / (2 * a)
            return p + tau * d, j + 1
        r_new = r + alpha * Hd
        if np.sqrt(r_new @ r_new) < cg_tol * r0:
            return p_new, j + 1
        beta = float(r_new @ r_new) / float(r @ r)
        d = -r_new + beta * d
        p, r = p_new, r_new
    return p, maxit


def solve_trust_region(x0, tris, Bs, areas, free, eterms=_sd_element_terms,
                       max_iter=400, tol=1e-6, Delta0=1.0, Delta_max=1e3, eta=0.1):
    x = x0.copy()
    log = []
    counts = {"assemblies": 0, "energy_evals": 0, "mat_vecs": 0, "lin_solves": 0}
    Delta = Delta0
    t0 = time.perf_counter()
    status = "maxiter"
    for it in range(max_iter):
        E, g, H = assemble(x, tris, Bs, areas, "none", eterms); counts["assemblies"] += 1
        if not np.isfinite(E):
            status = "infeasible"; break
        gf = g[free]
        gnorm = float(np.max(np.abs(gf)))
        Hff = H[np.ix_(free, free)]
        log.append({"iter": it, "energy": E, "grad_inf": gnorm, "wall_s": time.perf_counter() - t0})
        if gnorm < tol:
            status = "converged"; break
        p, nmv = _steihaug(gf, Hff, Delta, cg_tol=1e-6, maxit=2 * gf.size)
        counts["mat_vecs"] += nmv; counts["lin_solves"] += 1
        pred = -(float(gf @ p) + 0.5 * float(p @ (Hff @ p)))
        xf0 = x[free].copy(); x[free] = xf0 + p
        E_new = energy_only(x, tris, Bs, areas, eterms); counts["energy_evals"] += 1
        ared = (E - E_new) if np.isfinite(E_new) else -np.inf
        rho = ared / pred if pred > 0 else -np.inf
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and np.sqrt(p @ p) > 0.99 * Delta:
            Delta = min(2 * Delta, Delta_max)
        if not (rho > eta):
            x[free] = xf0                              # reject step
    wall = time.perf_counter() - t0
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": "trust-region", "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": wall,
            "counts": counts, "x": x, "log": log}
