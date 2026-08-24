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
            "counts": counts, "x": x.reshape(-1), "log": log}


def solve_sobolev_lbfgs(x0, tris, rest, free_dof, m=5, max_iter=3000, tol=1e-6, c=1e-4):
    """BCQN proxy core (Zhu-Bridson-Kaufman 2018): L-BFGS with the Sobolev initial inverse-Hessian
    D0 = L^-1 (cotan Laplacian) instead of a scaled identity. beta=0 (no secant blend) -- isolates
    the Sobolev-preconditioning component. Reaches the Newton minimizer."""
    from .mesh import rest_quantities
    nv = rest.shape[0]
    vf = _vfree(free_dof)
    nvf = int(vf.sum())
    L = cotan_laplacian(rest, tris)
    Lff = (L[vf][:, vf] + 1e-9 * sp.eye(nvf)).tocsc()
    solveL = spla.factorized(Lff)
    Bs, areas = rest_quantities(rest, tris)
    # map: free dofs (interleaved per free vertex) <-> (nvf, 2)
    fidx = np.where(free_dof)[0]

    def apply_D0(q):                                   # D0 = L^-1 tensor I2, per coordinate
        qm = q.reshape(nvf, 2)
        r = np.empty_like(qm)
        r[:, 0] = solveL(qm[:, 0]); r[:, 1] = solveL(qm[:, 1])
        return r.reshape(-1)

    x = x0.copy()
    S, Y, rho = [], [], []
    log = []; t0 = time.perf_counter(); status = "maxiter"
    counts = {"grad_evals": 0, "energy_evals": 0}
    g_prev = xf_prev = None
    for it in range(max_iter):
        E, g = assemble_eg(x, tris, Bs, areas, element_eg); counts["grad_evals"] += 1
        gf = g[fidx]; gnorm = float(np.max(np.abs(gf)))
        log.append({"iter": it, "energy": E, "grad_inf": gnorm, "wall_s": time.perf_counter() - t0})
        if gnorm < tol:
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
        r = apply_D0(q)                                # Sobolev initial inverse Hessian
        for i in range(len(S)):
            b = rho[i] * float(Y[i] @ r); r += S[i] * (al[len(S) - 1 - i] - b)
        d = -r
        gd = float(gf @ d)
        if gd >= 0:
            d = -apply_D0(gf); gd = float(gf @ d)      # safeguard: Sobolev steepest descent
        xf_prev = x[fidx].copy(); g_prev = gf.copy()
        alpha = 1.0; xf0 = x[fidx].copy()
        while True:
            x[fidx] = xf0 + alpha * d
            En, _ = assemble_eg(x, tris, Bs, areas, element_eg); counts["energy_evals"] += 1
            if np.isfinite(En) and En <= E + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-14:
                x[fidx] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    Efin = log[-1]["energy"] if log else np.inf
    gfin = log[-1]["grad_inf"] if log else np.inf
    return {"filter": "sobolev-lbfgs", "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": Efin, "final_grad_inf": gfin, "wall_s": time.perf_counter() - t0,
            "counts": counts, "x": x, "log": log}


# ---------------------------------------------------------------------------
# ARAP local-global + Anderson acceleration (minimize the ARAP energy
# sum_e a_e ||F_e - R(F_e)||^2). Separate energy from symmetric Dirichlet, so
# these two are compared to each other (anderson->local-global edge).
# ---------------------------------------------------------------------------

def _arap_setup(rest, tris, free_dof):
    from .mesh import rest_quantities
    nv = rest.shape[0]
    Bs, areas = rest_quantities(rest, tris)
    ndof = 2 * nv
    M = sp.lil_matrix((ndof, ndof))
    for t, tri in enumerate(tris):
        dofs = np.array([2*tri[0],2*tri[0]+1,2*tri[1],2*tri[1]+1,2*tri[2],2*tri[2]+1])
        Me = areas[t] * (Bs[t].T @ Bs[t])
        for a in range(6):
            for b in range(6):
                M[dofs[a], dofs[b]] += Me[a, b]
    M = M.tocsr()
    free = np.where(free_dof)[0]; pin = np.where(~free_dof)[0]
    Mff = M[free][:, free].tocsc()
    Mfp = M[free][:, pin]
    solveM = spla.factorized(Mff + 1e-10 * sp.eye(len(free)).tocsc())
    return Bs, areas, M, free, pin, Mff, Mfp, solveM


def _arap_rhs(x, tris, Bs, areas):
    nv = x.size // 2
    b = np.zeros(2 * nv)
    for t, tri in enumerate(tris):
        dofs = np.array([2*tri[0],2*tri[0]+1,2*tri[1],2*tri[1]+1,2*tri[2],2*tri[2]+1])
        F = (Bs[t] @ x[dofs]).reshape(2, 2)
        U, s, Vt = np.linalg.svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:                        # ensure proper rotation
            U[:, -1] *= -1; R = U @ Vt
        b[dofs] += areas[t] * (Bs[t].T @ R.reshape(4))
    return b


def arap_energy(x, tris, Bs, areas):
    E = 0.0
    for t, tri in enumerate(tris):
        dofs = np.array([2*tri[0],2*tri[0]+1,2*tri[1],2*tri[1]+1,2*tri[2],2*tri[2]+1])
        s = np.linalg.svd((Bs[t] @ x[dofs]).reshape(2, 2), compute_uv=False)
        E += areas[t] * ((s[0]-1)**2 + (s[1]-1)**2)
    return E


def _lg_step(x, tris, Bs, areas, free, pin, Mfp, solveM, xpin):
    b = _arap_rhs(x, tris, Bs, areas)
    rhs = b[free] - Mfp @ xpin
    xn = x.copy(); xn[free] = solveM(rhs)
    return xn


def solve_local_global(x0, tris, rest, free_dof, max_iter=2000, tol=1e-6):
    """ARAP local-global (Sorkine-Alexa / Liu 2008): local rotation fit + global cotan-Laplacian
    solve. First-order fixed point; convergence by ARAP-gradient norm."""
    Bs, areas, M, free, pin, Mff, Mfp, solveM = _arap_setup(rest, tris, free_dof)
    x = x0.copy(); xpin = x[pin]
    log = []; t0 = time.perf_counter(); status = "maxiter"
    for it in range(max_iter):
        b = _arap_rhs(x, tris, Bs, areas)
        grad = 2.0 * (M @ x - b); gnorm = float(np.max(np.abs(grad[free])))
        log.append({"iter": it, "energy": arap_energy(x, tris, Bs, areas), "grad_inf": gnorm})
        if gnorm < tol:
            status = "converged"; break
        x = _lg_step(x, tris, Bs, areas, free, pin, Mfp, solveM, xpin)
    return {"filter": "local-global", "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": log[-1]["energy"], "final_grad_inf": log[-1]["grad_inf"],
            "wall_s": time.perf_counter() - t0, "x": x}


def solve_anderson(x0, tris, rest, free_dof, m=5, max_iter=2000, tol=1e-6):
    """Anderson acceleration (Peng et al. 2018) of the ARAP local-global fixed point, with the
    energy-decrease safeguard (fall back to plain local-global on failure)."""
    Bs, areas, M, free, pin, Mff, Mfp, solveM = _arap_setup(rest, tris, free_dof)
    xpin = x0[pin].copy()

    def G(x):
        return _lg_step(x, tris, Bs, areas, free, pin, Mfp, solveM, xpin)

    def En(x):
        return arap_energy(x, tris, Bs, areas)

    def gnorm(x):
        return float(np.max(np.abs((2.0 * (M @ x - _arap_rhs(x, tris, Bs, areas)))[free])))

    x = x0.copy()
    Gx = G(x); Fk = (Gx - x)[free]
    dF, dG = [], []                                    # rolling buffers of last m differences
    F_prev, G_prev = Fk.copy(), Gx.copy()
    log = []; t0 = time.perf_counter(); status = "maxiter"; E_acc = En(x)
    for it in range(max_iter):
        gn = gnorm(x)
        log.append({"iter": it, "energy": En(x), "grad_inf": gn})
        if gn < tol:
            status = "converged"; break
        Gx = G(x); Fk = (Gx - x)[free]
        if dF:
            D = np.array(dF).T                          # (nfree, k)
            theta, *_ = np.linalg.lstsq(D, Fk, rcond=None)
            x_aa = Gx.copy()
            x_aa[free] = Gx[free] - np.array(dG).T @ theta
        else:
            x_aa = Gx
        # safeguard: accept AA only if it decreases energy, else plain local-global step
        if np.isfinite(En(x_aa)) and En(x_aa) <= E_acc:
            x_new = x_aa
        else:
            x_new = Gx
        # update difference buffers
        dF.append(Fk - F_prev); dG.append((Gx - G_prev)[free])
        if len(dF) > m:
            dF.pop(0); dG.pop(0)
        F_prev, G_prev, E_acc = Fk.copy(), Gx.copy(), En(x_new)
        x = x_new
    return {"filter": "anderson", "status": status,
            "iters": len(log) - (1 if status == "converged" else 0),
            "final_energy": log[-1]["energy"], "final_grad_inf": log[-1]["grad_inf"],
            "wall_s": time.perf_counter() - t0, "x": x}
