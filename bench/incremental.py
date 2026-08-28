"""Incremental-potential inner-solver testbed (V2.1 — dynamics-solver convergence edges).

One implicit-Euler timestep minimizes the incremental potential
    Phi(x) = 1/(2 h^2) (x - xtil)^T M (x - xtil) + E(x),                 (E = Neo-Hookean elastic)
and a large fraction of the corpus's simulation methods are simply DIFFERENT inner minimizers of
this SAME Phi. Comparing them on iterations-to-residual-tolerance (a hardware-independent count, per
docs/metrics.md) faithfully tests their *convergence* claims — the ones that are not purely GPU/wall-
clock (those stay hardware-confounded). Every solver here shares the exact Phi, gradient, and
residual, so the comparison is apples-to-apples.

Solvers (each faithful to the paper's algorithm, contact-free 2D):
  newton     : projected-Newton on Phi (clamp filter). Reference.
  pd         : generalized Projective Dynamics == Liu-2017 quasi-Newton with m=0 history: a
               FIXED-metric descent x <- x - A0^{-1} grad Phi with A0 = M/h^2 + H_rest (SPD,
               prefactored once), globalized by a line search. (For a fitting energy this is exactly
               local/global; for general E it is the standard fixed-proxy generalization.)
  cheby      : Chebyshev semi-iterative acceleration (Wang 2015) of the pd fixed point.
  lbfgs_lap  : L-BFGS with initial inverse-Hessian A0^{-1}, A0 = M/h^2 + H_rest (quasi-newton-liu2017).
  lbfgs_id   : L-BFGS with a scaled-identity initial inverse-Hessian (the "plain L-BFGS" baseline).
  vbd_gs     : Vertex Block Descent (Chen 2024), Gauss-Seidel sweep — per-vertex 2x2 local Newton,
               using already-updated neighbours.
  vbd_jacobi : same per-vertex 2x2 local Newton but block-Jacobi (all vertices from the old state).

Conformance-gated (`python -m bench.incremental`): grad Phi vs finite differences; A0 SPD; the VBD
per-vertex block equals the diagonal 2x2 block of the assembled Phi-Hessian.
"""
import numpy as np
from .mesh import grid_mesh, rest_quantities, boundary_mask
from .solver import assemble, energy_only
from .filters import project_element
from . import energy_neohookean as nh


# ----- incremental-potential problem -----------------------------------------------------------
def lumped_mass(rest, tris, areas, density=1.0):
    m = np.zeros(rest.shape[0])
    for t, tri in enumerate(tris):
        m[tri] += density * areas[t] / 3.0
    return np.repeat(m, 2)


def _incidence(tris, nv):
    """vertex -> list of (element index, local corner 0/1/2)."""
    inc = [[] for _ in range(nv)]
    for t, tri in enumerate(tris):
        for a in range(3):
            inc[tri[a]].append((t, a))
    return inc


class Problem:
    """One implicit-Euler timestep: everything a solver needs to minimize Phi over the free dofs."""
    def __init__(self, n=8, dt=1.0 / 60, nu=0.3, overshoot=2.4, seed=0, stiffness=1.0):
        rest, tris = grid_mesh(n, n)
        Bs, areas = rest_quantities(rest, tris)
        bmask = boundary_mask(rest)
        pin = (rest[:, 0] < 1e-9)                       # pin the left edge only (a hanging beam)
        free = ~np.repeat(pin, 2)
        et, psi, gp, _ = nh.make(mu=stiffness, lam=nh.lam_from_nu(nu))
        Md = lumped_mass(rest, tris, areas)
        # predicted (inertial) target: a big pull to the right + gravity => a hard timestep ("overshoot")
        rng = np.random.default_rng(seed)
        xtil = rest.copy()
        xtil[:, 0] = rest[:, 0] + overshoot * (rest[:, 0])      # stretch prediction
        xtil[:, 1] = rest[:, 1] - 0.15                          # gravity-ish drop
        xtil = xtil.reshape(-1)
        x0 = rest.reshape(-1).copy()                    # start each solve from the previous state (rest)
        self.rest = rest; self.tris = tris; self.Bs = Bs; self.areas = areas
        self.et = et; self.free = free; self.Md = Md; self.dt = dt
        self.xtil = xtil; self.x0 = x0; self.nv = rest.shape[0]
        self.inv_dt2 = 1.0 / (dt * dt)
        self.inc = _incidence(tris, self.nv)
        # fixed PD/quasi-Newton proxy A0 = M/h^2 I + SPD(H_elastic at rest)
        _, _, Hr = assemble(x0, tris, Bs, areas, "clamp", et)
        self.A0 = np.diag(self.inv_dt2 * Md) + Hr

    def phi(self, x):
        Ee = energy_only(x, self.tris, self.Bs, self.areas, self.et)
        if not np.isfinite(Ee):
            return np.inf
        dx = x - self.xtil
        return Ee + 0.5 * self.inv_dt2 * float(dx @ (self.Md * dx))

    def grad_hess(self, x, filt="none"):
        Ee, g, H = assemble(x, self.tris, self.Bs, self.areas, filt, self.et)
        if not np.isfinite(Ee):
            return np.inf, None, None
        dx = x - self.xtil
        g = g + self.inv_dt2 * (self.Md * dx)
        if H is not None:
            H = H + np.diag(self.inv_dt2 * self.Md)
        return Ee, g, H

    def resid(self, x):
        _, g, _ = self.grad_hess(x)
        if g is None:
            return np.inf
        return float(np.max(np.abs(g[self.free])))


def _iters_to(reslog, rtol=1e-3):
    r0 = reslog[0]
    for k, r in enumerate(reslog):
        if r <= rtol * r0:
            return k
    return None


def _ls(P, x, free, d, g, gtol_gd=None):
    """Backtracking Armijo line search on Phi along d (descent dir). Returns new x, ok."""
    gd = float(g[free] @ d[free]) if gtol_gd is None else gtol_gd
    if gd >= 0:
        return x, False
    E0 = P.phi(x); a = 1.0; x0 = x.copy()
    for _ in range(60):
        xn = x0 + a * d
        if np.isfinite(P.phi(xn)) and P.phi(xn) <= E0 + 1e-4 * a * gd:
            return xn, True
        a *= 0.5
    return x0, False


# ----- solvers ---------------------------------------------------------------------------------
def solve_newton(P, max_iter=200, rtol=1e-3):
    x = P.x0.copy(); free = P.free; res = []
    for _ in range(max_iter):
        _, g, H = P.grad_hess(x, "clamp")
        if g is None:
            break
        res.append(float(np.max(np.abs(g[free]))))
        if res[-1] <= rtol * res[0]:
            break
        d = np.zeros_like(x)
        d[free] = np.linalg.solve(H[np.ix_(free, free)], -g[free])
        x, ok = _ls(P, x, free, d, g)
        if not ok:
            break
    return {"name": "newton", "res": res, "it": _iters_to(res, rtol), "x": x}


def _fixed_metric_dir(P, x, Aff_chol):
    """PD/quasi-Newton m=0 step direction d_free = -A0_ff^{-1} grad Phi_free."""
    _, g, _ = P.grad_hess(x)
    free = P.free
    from scipy.linalg import cho_solve
    d = np.zeros_like(x)
    d[free] = cho_solve(Aff_chol, -g[free])
    return d, g


def _chol(A):
    from scipy.linalg import cho_factor
    return cho_factor(A, lower=True)


def solve_pd(P, max_iter=400, rtol=1e-3):
    x = P.x0.copy(); free = P.free; res = []
    Aff = _chol(P.A0[np.ix_(free, free)])
    for _ in range(max_iter):
        d, g = _fixed_metric_dir(P, x, Aff)
        res.append(float(np.max(np.abs(g[free]))))
        if res[-1] <= rtol * res[0]:
            break
        x, ok = _ls(P, x, free, d, g)
        if not ok:
            break
    return {"name": "pd", "res": res, "it": _iters_to(res, rtol), "x": x}


def solve_cheby(P, rho=0.9, max_iter=400, rtol=1e-3):
    """Chebyshev semi-iterative acceleration (Wang 2015) of the PD fixed point G(x)=x+d_pd(x)."""
    free = P.free
    Aff = _chol(P.A0[np.ix_(free, free)])
    x = P.x0.copy(); xprev = x.copy(); res = []
    omega = 1.0
    for k in range(max_iter):
        d, g = _fixed_metric_dir(P, x, Aff)   # G(x)-x = d (one PD step, unit metric step)
        res.append(float(np.max(np.abs(g[free]))))
        if res[-1] <= rtol * res[0]:
            break
        Gx = x + d
        if k == 0:
            omega = 1.0
        elif k == 1:
            omega = 2.0 / (2.0 - rho * rho)
        else:
            omega = 4.0 / (4.0 - rho * rho * omega)
        xn = omega * (Gx - xprev) + xprev     # Chebyshev combination
        # safeguard: accept only if Phi does not blow up, else fall back to the plain PD step
        if not np.isfinite(P.phi(xn)) or P.phi(xn) > P.phi(Gx):
            xn = Gx
        xprev = x; x = xn
    return {"name": "cheby-pd", "res": res, "it": _iters_to(res, rtol), "x": x}


def solve_lbfgs(P, mode="lap", m=5, max_iter=400, rtol=1e-3):
    """L-BFGS on Phi. mode='lap' uses A0^{-1} as the initial inverse-Hessian (quasi-newton-liu2017);
    mode='id' uses the scaled-identity initial inverse-Hessian (plain L-BFGS)."""
    free = P.free
    Aff = _chol(P.A0[np.ix_(free, free)]) if mode == "lap" else None
    from scipy.linalg import cho_solve
    x = P.x0.copy(); res = []
    S, Y, rho = [], [], []
    gprev = None; xprev = None
    for _ in range(max_iter):
        _, g, _ = P.grad_hess(x); gf = g[free]
        res.append(float(np.max(np.abs(gf))))
        if res[-1] <= rtol * res[0]:
            break
        if gprev is not None:
            s = (x - xprev)[free]; y = (g - gprev)[free]; sy = float(s @ y)
            if sy > 1e-12:
                S.append(s); Y.append(y); rho.append(1.0 / sy)
                if len(S) > m:
                    S.pop(0); Y.pop(0); rho.pop(0)
        q = gf.copy(); alphas = []
        for i in range(len(S) - 1, -1, -1):
            a = rho[i] * float(S[i] @ q); alphas.append(a); q = q - a * Y[i]
        if mode == "lap":
            r = cho_solve(Aff, q)
        else:
            gamma = (float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])) if S else 1.0
            r = gamma * q
        for i in range(len(S)):
            b = rho[i] * float(Y[i] @ r); r = r + S[i] * (alphas[len(S) - 1 - i] - b)
        d = np.zeros_like(x); d[free] = -r
        gprev = g.copy(); xprev = x.copy()
        x, ok = _ls(P, x, free, d, g)
        if not ok:
            break
    return {"name": f"lbfgs-{mode}", "res": res, "it": _iters_to(res, rtol), "x": x}


def _vertex_local(P, x, i):
    """Local 2-gradient and SPD 2x2 Hessian of Phi at vertex i (inertia + incident elements)."""
    gi = P.inv_dt2 * P.Md[2 * i:2 * i + 2] * (x[2 * i:2 * i + 2] - P.xtil[2 * i:2 * i + 2])
    Hi = np.diag(P.inv_dt2 * P.Md[2 * i:2 * i + 2])
    for (t, a) in P.inc[i]:
        tri = P.tris[t]
        dofs = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                         2 * tri[2], 2 * tri[2] + 1])
        xe = x[dofs]
        Ee, ge, He, _ = P.et(xe, P.Bs[t], P.areas[t])
        if not np.isfinite(Ee):
            return None, None
        He = project_element(He, "clamp")
        gi = gi + ge[2 * a:2 * a + 2]
        Hi = Hi + He[2 * a:2 * a + 2, 2 * a:2 * a + 2]
    return gi, Hi


def _vertex_phi(P, x, i):
    """LOCAL incremental potential seen by vertex i: its inertia term + the energy of its incident
    elements. Moving only vertex i changes Phi by exactly the change in this quantity, so a VBD local
    line search can use it (O(incident) not O(all elements))."""
    dxi = x[2 * i:2 * i + 2] - P.xtil[2 * i:2 * i + 2]
    val = 0.5 * P.inv_dt2 * float(dxi @ (P.Md[2 * i:2 * i + 2] * dxi))
    for (t, a) in P.inc[i]:
        tri = P.tris[t]
        dofs = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                         2 * tri[2], 2 * tri[2] + 1])
        Ee, _, _, _ = P.et(x[dofs], P.Bs[t], P.areas[t])
        if not np.isfinite(Ee):
            return np.inf
        val += Ee
    return val


def _vbd_step_vertex(P, xread, i):
    """One vertex's local 2x2 Newton step with a local Armijo line search evaluated against xread's
    neighbourhood. Returns the new 2-vector for vertex i (writes into a scratch copy for the search)."""
    gi, Hi = _vertex_local(P, xread, i)
    if gi is None:
        return xread[2 * i:2 * i + 2].copy()
    di = np.linalg.solve(Hi, -gi)
    xi0 = xread[2 * i:2 * i + 2].copy(); gd = float(gi @ di)
    scratch = xread                                  # only slot 2i:2i+2 is mutated then restored
    base = _vertex_phi(P, scratch, i); a = 1.0; out = xi0
    for _ in range(30):
        scratch[2 * i:2 * i + 2] = xi0 + a * di
        lp = _vertex_phi(P, scratch, i)
        if np.isfinite(lp) and lp <= base + 1e-4 * a * gd:
            out = xi0 + a * di; break
        a *= 0.5
    scratch[2 * i:2 * i + 2] = xi0                   # restore (caller decides how to apply)
    return out


def _vbd(P, gauss_seidel, max_iter=800, rtol=1e-3, jac_omega=None):
    """VBD sweeps. Gauss-Seidel is self-stabilising (sequential). Block-Jacobi applies all steps
    simultaneously and MUST be under-relaxed (ω<1) or it overshoots -- as every real block-Jacobi/
    parallel-VBD scheme does; jac_omega defaults to 1/(1+max_valence) among free vertices, the
    standard diagonal-dominance-safe choice (review-V2.1 #2)."""
    free_v = [i for i in range(P.nv) if P.free[2 * i]]
    if jac_omega is None:
        maxval = max(len(P.inc[i]) for i in free_v)
        jac_omega = 1.0 / (1.0 + maxval)
    x = P.x0.copy(); res = []
    for _ in range(max_iter):
        res.append(P.resid(x))
        if res[-1] <= rtol * res[0]:
            break
        if gauss_seidel:                             # sequential: each vertex sees updated neighbours
            for i in free_v:
                x[2 * i:2 * i + 2] = _vbd_step_vertex(P, x, i)
        else:                                        # block-Jacobi: all from old x, UNDER-RELAXED
            new = {i: _vbd_step_vertex(P, x, i) for i in free_v}
            for i, xi in new.items():
                xi0 = x[2 * i:2 * i + 2]
                x[2 * i:2 * i + 2] = xi0 + jac_omega * (xi - xi0)
    name = "vbd-gs" if gauss_seidel else "vbd-jacobi"
    return {"name": name, "res": res, "it": _iters_to(res, rtol), "x": x, "jac_omega": jac_omega}


def solve_vbd_gs(P, **kw):
    return _vbd(P, True, **kw)


def solve_vbd_jacobi(P, **kw):
    return _vbd(P, False, **kw)


# ----- conformance -----------------------------------------------------------------------------
def _conformance():
    P = Problem(n=5)
    rng = np.random.default_rng(1)
    x = P.x0 + 0.02 * rng.standard_normal(P.x0.shape)
    # (1) grad Phi vs FD
    _, g, H = P.grad_hess(x)
    gfd = np.zeros_like(g); h = 1e-6
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (P.phi(xp) - P.phi(xm)) / (2 * h)
    gerr = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    # (2) A0 SPD
    a0min = float(np.linalg.eigvalsh(P.A0[np.ix_(P.free, P.free)]).min())
    # (3) VBD local block == diagonal 2x2 block of the (clamp-projected) assembled Phi-Hessian
    _, _, Hc = P.grad_hess(x, "clamp")
    i = [v for v in range(P.nv) if P.free[2 * v]][3]
    _, Hi = _vertex_local(P, x, i)
    blockerr = np.max(np.abs(Hi - Hc[2 * i:2 * i + 2, 2 * i:2 * i + 2]))
    return gerr, a0min, blockerr


if __name__ == "__main__":
    import sys
    gerr, a0min, blockerr = _conformance()
    ok = gerr < 1e-5 and a0min > 0 and blockerr < 1e-9
    print(f"[incremental conformance] gradPhi vs FD: {gerr:.1e} | A0 min eig: {a0min:.2e} | "
          f"VBD block == Hessian block: {blockerr:.1e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
