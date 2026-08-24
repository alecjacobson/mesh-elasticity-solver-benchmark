"""3D linear (P1) tetrahedral Neo-Hookean element. Self-contained (own mesh + assembly + solve)
so the 2D path is untouched. Formulas numerically verified (subagent spec) in the harness's
conventions: F=Ds Dm^-1 (3x3), psi = mu/2(I1-3) - mu logJ + lam/2 logJ^2, PK1 = muF + (lam logJ - mu)F^-T.
Conformance-gated: `python -m bench.tet` runs FD grad + rigid-invariance checks.
"""
import time
import numpy as np
from .filters import project_element


def box_tet_mesh(nx, ny, nz, W=1.0, H=1.0, D=1.0):
    xs = np.linspace(0, W, nx + 1); ys = np.linspace(0, H, ny + 1); zs = np.linspace(0, D, nz + 1)
    verts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)

    def idx(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    # 6-tet Freudenthal decomposition of each cube (all share the main diagonal c000-c111)
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = {(a, b, d): idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)}
                c000, c100, c010, c001 = c[0, 0, 0], c[1, 0, 0], c[0, 1, 0], c[0, 0, 1]
                c110, c101, c011, c111 = c[1, 1, 0], c[1, 0, 1], c[0, 1, 1], c[1, 1, 1]
                tets += [
                    [c000, c100, c110, c111], [c000, c110, c010, c111],
                    [c000, c010, c011, c111], [c000, c011, c001, c111],
                    [c000, c001, c101, c111], [c000, c101, c100, c111],
                ]
    return verts, np.array(tets, int)


def element_B(Minv):
    """9x12: vec(F)=B x_elem, x_elem=[x0x,x0y,x0z, x1..., x2..., x3...]. F=Ds Minv, Ds=[x1-x0,x2-x0,x3-x0]."""
    B = np.zeros((9, 12))
    for d in range(12):
        e = np.zeros(12); e[d] = 1.0
        xt = e.reshape(4, 3)
        Ds = np.column_stack((xt[1] - xt[0], xt[2] - xt[0], xt[3] - xt[0]))
        B[:, d] = (Ds @ Minv).reshape(9)
    return B


def rest_quantities(rest, tets):
    Bs, vols = [], []
    for t in tets:
        X = rest[t]
        Dm = np.column_stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]))
        vols.append(abs(np.linalg.det(Dm)) / 6.0)
        Bs.append(element_B(np.linalg.inv(Dm)))
    return Bs, np.array(vols)


def make(mu=1.0, lam=1.0):
    def psi(F):
        J = np.linalg.det(F)
        if J <= 0:
            return np.inf
        return 0.5 * mu * (float(np.sum(F * F)) - 3.0) - mu * np.log(J) + 0.5 * lam * np.log(J) ** 2

    def grad_psi(F):
        J = np.linalg.det(F)
        return mu * F + (lam * np.log(J) - mu) * np.linalg.inv(F).T

    def hess_psi(F, h=1e-6):
        Ff = F.reshape(9).astype(float); Hm = np.zeros((9, 9))
        for k in range(9):
            fp = Ff.copy(); fp[k] += h; fm = Ff.copy(); fm[k] -= h
            Hm[:, k] = (grad_psi(fp.reshape(3, 3)).reshape(9)
                        - grad_psi(fm.reshape(3, 3)).reshape(9)) / (2 * h)
        return 0.5 * (Hm + Hm.T)

    def element_terms(x_elem, B, vol):
        F = (B @ x_elem).reshape(3, 3)
        J = float(np.linalg.det(F))
        if J <= 0:
            return np.inf, None, None, J
        return vol * psi(F), vol * (B.T @ grad_psi(F).reshape(9)), vol * (B.T @ hess_psi(F) @ B), J

    return element_terms, psi, grad_psi


def _edofs(t):
    d = np.empty(12, int)
    for a in range(4):
        d[3 * a:3 * a + 3] = [3 * t[a], 3 * t[a] + 1, 3 * t[a] + 2]
    return d


def assemble(x, tets, quad, eterms, filt):
    nv = x.size // 3
    g = np.zeros(3 * nv); H = np.zeros((3 * nv, 3 * nv)); E = 0.0
    for i, t in enumerate(tets):
        dofs = _edofs(t)
        Ee, ge, He, _ = eterms(x[dofs], *quad[i])
        if not np.isfinite(Ee):
            return np.inf, None, None
        E += Ee
        if filt in ("clamp", "absolute", "project-on-demand"):
            He = project_element(He, filt)
        g[dofs] += ge; H[np.ix_(dofs, dofs)] += He
    return E, g, H


def energy(x, tets, quad, eterms):
    E = 0.0
    for i, t in enumerate(tets):
        Ee, _, _, _ = eterms(x[_edofs(t)], *quad[i])
        if not np.isfinite(Ee):
            return np.inf
        E += Ee
    return E


def solve(x0, tets, quad, free, eterms, filt, max_iter=400, tol=1e-6, c=1e-4):
    x = x0.copy(); it_done = 0; status = "maxiter"; t0 = time.perf_counter()
    for it in range(max_iter):
        E, g, H = assemble(x, tets, quad, eterms, filt)
        if not np.isfinite(E):
            status = "infeasible"; break
        gf = g[free]; it_done = it
        if float(np.max(np.abs(gf))) < tol:
            status = "converged"; break
        Hff = H[np.ix_(free, free)]
        try:
            d = np.linalg.solve(Hff, -gf)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(Hff, -gf, rcond=None)[0]
        if float(gf @ d) >= 0:
            status = "nondescent"; break
        alpha = 1.0; xf0 = x[free].copy()
        while True:
            x[free] = xf0 + alpha * d
            En = energy(x, tets, quad, eterms)
            if np.isfinite(En) and En <= E + c * alpha * float(gf @ d):
                break
            alpha *= 0.5
            if alpha < 1e-14:
                x[free] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    return {"filter": filt, "status": status, "iters": it_done, "final_energy": E,
            "wall_s": time.perf_counter() - t0, "x": x}


def _conformance(seed=0, h=1e-6):
    rng = np.random.default_rng(seed)
    verts, tets = box_tet_mesh(2, 2, 2)
    quad = list(zip(*rest_quantities(verts, tets)))
    et, psi, grad_psi = make(1.0, 1.0)
    # rigid invariance: proper rotation + translation -> F=R, psi=0, g=0
    th = 0.4; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    xr = (verts @ R.T + np.array([0.3, -0.2, 0.1])).reshape(-1)
    Erig = energy(xr, tets, quad, et)
    # grad vs FD
    x = (verts + 0.02 * rng.standard_normal(verts.shape)).reshape(-1)
    _, g, _ = assemble(x, tets, quad, et, "none")
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (energy(xp, tets, quad, et) - energy(xm, tets, quad, et)) / (2 * h)
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    return rel, abs(Erig)


if __name__ == "__main__":
    import sys
    rel, erig = _conformance()
    print(f"[tet conformance] grad vs FD: {rel:.2e}  |  rigid energy: {erig:.2e}  -> "
          f"{'PASS' if rel < 1e-5 and erig < 1e-9 else 'FAIL'}")
    sys.exit(0 if rel < 1e-5 and erig < 1e-9 else 1)
