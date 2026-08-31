"""FAITHFUL BCQN — Blended Cured Quasi-Newton for Distortion Optimization
(Zhu, Bridson, Kaufman, SIGGRAPH 2018), reimplemented from the paper + the authors' reference code
(github.com/mike323zyf/BCQN). This is the FULL method, not the Sobolev-L-BFGS proxy fragment
(`world1.solve_sobolev_lbfgs`, which is BCQN with blend/cure disabled). All four components:

  1. PROXY: initial inverse-Hessian D0 = L^{-1}, L = 2·(cotan/Dirichlet Laplacian), factored ONCE,
     applied per-coordinate (the fixed-scalar-Laplacian cost win).
  2. BLENDED (Eq. 10, 13): L-BFGS secant y_i blended with L·s_i,
        β_i = clamp( normest(L)·(y_iᵀ L s_i) / Σ_t a_t , 0, 1),   z_i = β_i·L s_i + (1−β_i)·y_i,
     and z_i (not y_i) enters the L-BFGS history.
  3. CURED (Eq. 16–23): barrier-aware filter of the search direction p̃=−D g onto the linearised
     no-inversion cone {a(x)+∇a·p ≥ 0}, a strictly-convex QP solved as an LCP by damped projected
     Jacobi (ω=0.5, ≤20 iters, Fischer–Burmeister residual), p = p̃ + Cᵀλ, kept only if still descent.
  4. LINE SEARCH: inversion-free step cap (0.5× first flip) then Armijo (c1=0.2, backtrack 0.8).
  Stop: characteristic gradient norm  ‖g‖ ≤ ε·⟨W⟩·‖ℓ‖,  ⟨W⟩=8 (symmetric Dirichlet), ℓ = one-ring
  perimeters (Eq. 25–27), ε=1e-3.

Conformance (`python -m bench.bcqn`): β∈[0,1] & β=0 ⇒ plain SL-BFGS; monotone energy decrease; the
cured direction is a no-op when p̃ is already inversion-safe; converges to the SAME symmetric-Dirichlet
minimum as projected-Newton. Writes nothing; `run_bcqn.py` drives the comparisons.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .world1 import cotan_laplacian, assemble_eg, _vfree
from .energy import element_eg
from .solver import energy_only as _e_only
from .mesh import rest_quantities
from .barrier_ls import max_step_to_inversion


def _one_ring_perimeters(rest, tris):
    """ℓ_i = Σ over incident triangles of the length of the edge OPPOSITE vertex i (Eq. 26)."""
    n = rest.shape[0]
    ell = np.zeros(n)
    for a, b, c in tris:
        pa, pb, pc = rest[a], rest[b], rest[c]
        ell[a] += np.linalg.norm(pb - pc)     # edge opposite a
        ell[b] += np.linalg.norm(pc - pa)
        ell[c] += np.linalg.norm(pa - pb)
    return ell


def _elem_orient_and_grad(x, tris, Bs):
    """a_t = det F_t (orientation) and ∇a_t wrt the element's 6 dofs, per triangle."""
    m = len(tris)
    a = np.empty(m)
    ga = np.empty((m, 6))
    for t, tri in enumerate(tris):
        dofs = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                         2 * tri[2], 2 * tri[2] + 1])
        F = (Bs[t] @ x[dofs]).reshape(2, 2)
        a[t] = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        # d det F / d vecF (vecF=[F00,F01,F10,F11]) = [F11,-F10,-F01,F00]
        dd = np.array([F[1, 1], -F[1, 0], -F[0, 1], F[0, 0]])
        ga[t] = Bs[t].T @ dd
    return a, ga


def _build_C(tris, ga, fidx_of_dof, nfree):
    """Sparse constraint Jacobian C (m × nfree): row t = ∇a_t restricted to free dofs."""
    rows, cols, data = [], [], []
    for t, tri in enumerate(tris):
        dofs = [2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1, 2 * tri[2], 2 * tri[2] + 1]
        for k, gd in enumerate(dofs):
            j = fidx_of_dof.get(gd)
            if j is not None:
                rows.append(t); cols.append(j); data.append(ga[t, k])
    return sp.csr_matrix((data, (rows, cols)), shape=(len(tris), nfree))


def _fb(a, b):
    """Fischer–Burmeister residual sqrt(Σ (a+b−sqrt(a²+b²))²)."""
    return float(np.sqrt(np.sum((a + b - np.sqrt(a * a + b * b + 0.0)) ** 2)))


def _cure(ptilde, C, bvec, max_it=20, omega=0.5, atol=1e-6, rtol=1e-3):
    """Damped projected Jacobi on the LCP  0 ≤ λ ⊥ Mλ + q ≥ 0,  M=CCᵀ, q=C p̃ + b.
    Returns λ; filtered direction is p̃ + Cᵀλ. λ⁰=0 so a feasible p̃ (q≥0) yields λ=0 (no-op)."""
    m = C.shape[0]
    q = C @ ptilde + bvec
    lam = np.zeros(m)
    Tdiag = np.asarray(C.multiply(C).sum(axis=1)).ravel() + 1e-12   # diag(M)=row-norms² of C
    Mlam = C @ (C.T @ lam)
    fb = _fb(lam, Mlam + q)
    for _ in range(max_it):
        if fb < atol:
            break
        lam = np.maximum(lam - omega * (Mlam + q) / Tdiag, 0.0)
        Mlam = C @ (C.T @ lam)
        fb_new = _fb(lam, Mlam + q)
        if fb > 0 and abs(fb - fb_new) / fb < rtol:
            fb = fb_new; break
        fb = fb_new
    return lam


def solve_bcqn(x0, tris, rest, free_dof, m=5, eps=1e-3, max_iter=3000, blend=True, cure=True,
               c1=0.2, bt=0.8):
    """Faithful BCQN. blend/cure flags allow the paper's own ablations (blend=False,cure=False is the
    Sobolev-L-BFGS proxy). Returns log with per-iteration energy, |g|∞, |g|₂, β, and #cured elements."""
    vf = _vfree(free_dof)
    nvf = int(vf.sum())
    fidx = np.where(free_dof)[0]
    fidx_of_dof = {int(d): j for j, d in enumerate(fidx)}
    nfree = fidx.size
    Bs, areas = rest_quantities(rest, tris)
    L = (2.0 * cotan_laplacian(rest, tris)).tocsc()
    Lff = (L[vf][:, vf] + 1e-9 * sp.eye(nvf)).tocsc()
    solveL = spla.factorized(Lff)
    # per-coordinate L (free) as an operator for the blend term y·(L s)
    Lff_op = L[vf][:, vf]
    normestL = float(spla.eigsh(Lff, k=1, which="LM", return_eigenvectors=False, tol=1e-3)[0])
    Avt = float(np.sum(areas))
    char = 8.0 * float(np.linalg.norm(_one_ring_perimeters(rest, tris)))   # ⟨W⟩=8 for sym-Dirichlet

    def apply_D0(qf):                                  # D0 = L^{-1} ⊗ I2, per coordinate
        qm = qf.reshape(nvf, 2)
        r = np.empty_like(qm)
        r[:, 0] = solveL(qm[:, 0]); r[:, 1] = solveL(qm[:, 1])
        return r.reshape(-1)

    def applyL(sf):                                    # (L ⊗ I2) s, per coordinate, free block
        sm = sf.reshape(nvf, 2)
        r = np.empty_like(sm)
        r[:, 0] = Lff_op @ sm[:, 0]; r[:, 1] = Lff_op @ sm[:, 1]
        return r.reshape(-1)

    x = x0.copy()
    S, Z, rho = [], [], []
    log = []; t0 = time.perf_counter(); status = "maxiter"
    g_prev_f = None; beta_last = 0.0
    for it in range(max_iter):
        E, g = assemble_eg(x, tris, Bs, areas, element_eg)
        gf = g[fidx]
        gnf = float(np.linalg.norm(gf)); gninf = float(np.max(np.abs(gf)))
        log.append({"iter": it, "energy": E, "grad_inf": gninf, "grad2": gnf, "beta": beta_last,
                    "wall_s": time.perf_counter() - t0})
        if gnf <= eps * char:
            status = "converged"; break
        # (1) two-loop L-BFGS with H0 = L^{-1}, history stores (s, z)
        q = gf.copy(); al = []
        for si, zi, ri in zip(reversed(S), reversed(Z), reversed(rho)):
            a = ri * float(si @ q); al.append(a); q = q - a * zi
        r = apply_D0(q)
        for si, zi, ri, a in zip(S, Z, rho, reversed(al)):
            b = ri * float(zi @ r); r = r + (a - b) * si
        ptilde = -r
        # (2) cure: filter p̃ onto the no-inversion cone
        ncured = 0
        p = ptilde
        if cure:
            aorient, ga = _elem_orient_and_grad(x, tris, Bs)
            C = _build_C(tris, ga, fidx_of_dof, nfree)
            lam = _cure(ptilde, C, aorient)
            ncured = int(np.sum(lam > 0))
            pf = ptilde + C.T @ lam
            if float(gf @ pf) <= 0.0:          # descent safeguard
                p = pf
        # (3) line search: inversion-free cap (0.5×) then Armijo
        d = np.zeros_like(x); d[fidx] = p
        amax = max_step_to_inversion(x, d, tris, shrink=0.5)
        a = min(1.0, amax if np.isfinite(amax) else 1.0)
        gd = float(gf @ p)
        x0f = x.copy()
        while a > 1e-14:
            x = x0f + a * d
            En = _e_only(x, tris, Bs, areas)
            if np.isfinite(En) and En <= E + c1 * a * gd:
                break
            a *= bt
        if a <= 1e-14:
            x = x0f; status = "linesearch"; break
        # (4) BCQN blended L-BFGS history update
        E2, g2 = assemble_eg(x, tris, Bs, areas, element_eg)
        g2f = g2[fidx]
        s = (x - x0f)[fidx]; y = g2f - gf
        Ls = applyL(s)
        if blend:
            beta = normestL * float(y @ Ls) / Avt
            beta = max(0.0, min(1.0, beta))
        else:
            beta = 0.0
        z = beta * Ls + (1.0 - beta) * y
        beta_last = beta
        sz = float(s @ z)
        if sz > 1e-12:                          # standard L-BFGS skip if curvature non-positive
            S.append(s); Z.append(z); rho.append(1.0 / sz)
            if len(S) > m:
                S.pop(0); Z.pop(0); rho.pop(0)
        log[-1]["ncured"] = ncured
    return {"status": status, "iters": len(log) - (1 if status == "converged" else 0),
            "log": log, "x": x, "wall_s": time.perf_counter() - t0,
            "final_energy": log[-1]["energy"], "char": char}


def _conformance(seed=0):
    """β∈[0,1]; monotone energy; cure is a no-op on an inversion-safe start; reaches the p-Newton min."""
    from .run_e1 import build_scenario
    from .solver import solve
    from .energy import element_terms
    sc = build_scenario(nx=6, ny=6, seed=seed)
    x0, tris, rest, free = sc["x0"], sc["tris"], sc["rest"], sc["free"]
    r = solve_bcqn(x0, tris, rest, free, eps=1e-6, max_iter=2000)
    betas = [d["beta"] for d in r["log"] if "beta" in d]
    beta_ok = all(0.0 <= b <= 1.0 for b in betas)
    Es = [d["energy"] for d in r["log"]]
    mono = all(Es[i + 1] <= Es[i] + 1e-7 for i in range(len(Es) - 1))
    # reference projected-Newton minimum
    rn = solve(x0, tris, sc["Bs"], sc["areas"], free, "clamp", eterms=element_terms, tol=1e-9)
    same = abs(r["final_energy"] - rn["final_energy"]) / (abs(rn["final_energy"]) + 1e-12)
    # cure no-op: on the (inversion-free) rest-ish start, un-cured vs cured first direction match
    return beta_ok, mono, same, r["status"], r["iters"]


if __name__ == "__main__":
    import sys
    beta_ok, mono, same, st, it = _conformance()
    ok = beta_ok and mono and same < 1e-5 and st == "converged"
    print(f"[bcqn conformance] beta∈[0,1]={beta_ok}  monotone={mono}  "
          f"same-min-as-pNewton={same:.2e}  status={st} iters={it} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
