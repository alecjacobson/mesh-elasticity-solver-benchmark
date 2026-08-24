"""Quadratic (P2, 6-node) triangle element -- a locking-mitigating discretization.

P1 constant-strain triangles lock volumetrically as ν→½ (results/e1_nu.md, results/locking.md).
P2 displacement has enough kinematic freedom to substantially relieve that locking, so it is the
right tool to test whether the "absolute underperforms clamp" ν-result was a locking artifact.

Self-contained (own mesh, isoparametric assembly, and Newton solve) so it doesn't touch the
verified P1 path. Conformance-gated: `python -m bench.p2` runs a finite-difference gradient check.
"""
import numpy as np
from .filters import project_element

# 3-point quadrature on the reference triangle (area 1/2), exact to degree 2.
_QP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
_QW = np.array([1 / 6, 1 / 6, 1 / 6])


def _dNref(r, s):
    """Gradient of the 6 P2 shape functions wrt (r,s). Nodes: 3 corners then mids (12,23,31)."""
    L1, L2, L3 = 1 - r - s, r, s
    return np.array([
        [(4 * L1 - 1) * (-1), (4 * L1 - 1) * (-1)],   # N1 = L1(2L1-1)
        [(4 * L2 - 1) * (1),  0.0],                    # N2 = L2(2L2-1)
        [0.0, (4 * L3 - 1) * (1)],                     # N3 = L3(2L3-1)
        [4 * (L1 - L2), 4 * (-L2)],                    # N4 = 4 L1 L2
        [4 * L3, 4 * L2],                              # N5 = 4 L2 L3
        [-4 * L3, 4 * (L1 - L3)],                      # N6 = 4 L3 L1
    ])


def grid_mesh_p2(nx, ny, W=1.0, H=1.0):
    xs = np.linspace(0, W, nx + 1); ys = np.linspace(0, H, ny + 1)
    corners = [[x, y] for y in ys for x in xs]

    def ci(i, j):
        return j * (nx + 1) + i

    tris3 = []
    for j in range(ny):
        for i in range(nx):
            v00, v10, v01, v11 = ci(i, j), ci(i + 1, j), ci(i, j + 1), ci(i + 1, j + 1)
            tris3 += [[v00, v10, v11], [v00, v11, v01]]

    nodes = list(corners)
    edge_mid = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_mid:
            edge_mid[key] = len(nodes)
            nodes.append([0.5 * (nodes[a][0] + nodes[b][0]), 0.5 * (nodes[a][1] + nodes[b][1])])
        return edge_mid[key]

    elems = []
    for a, b, c in tris3:
        elems.append([a, b, c, mid(a, b), mid(b, c), mid(c, a)])
    return np.array(nodes, float), np.array(elems, int)


def rest_quantities_p2(rest, elems):
    """Per element: list of (B_q [4x12], w_eff) over the 3 quadrature points, from REST config."""
    out = []
    for e in elems:
        Xr = rest[e]                      # 6x2 rest node coords
        qd = []
        for (r, s), w in zip(_QP, _QW):
            dNref = _dNref(r, s)          # 6x2
            Jref = Xr.T @ dNref           # 2x2  dX/dxi
            detJ = np.linalg.det(Jref)
            dNdX = dNref @ np.linalg.inv(Jref)   # 6x2  dN/dX (rest spatial gradient)
            # B_q: vec(F) = B_q @ x_elem (12,), F = x^T @ dNdX ; build columns via unit dofs
            B = np.zeros((4, 12))
            for d in range(12):
                ue = np.zeros(12); ue[d] = 1.0
                F = ue.reshape(6, 2).T @ dNdX
                B[:, d] = F.reshape(4)
            qd.append((B, w * abs(detJ)))
        out.append(qd)
    return out


def make_element_terms(psi, grad_psi, hess_psi):
    def element_terms(x_elem, quaddata):
        E = 0.0; g = np.zeros(12); H = np.zeros((12, 12)); minJ = np.inf
        for B, w in quaddata:
            F = (B @ x_elem).reshape(2, 2)
            J = float(np.linalg.det(F)); minJ = min(minJ, J)
            if J <= 0.0:
                return np.inf, None, None, J
            E += w * psi(F)
            g += w * (B.T @ grad_psi(F).reshape(4))
            H += w * (B.T @ hess_psi(F) @ B)
        return E, g, H, minJ
    return element_terms


def _edofs(e):
    d = np.empty(12, int)
    for a in range(6):
        d[2 * a] = 2 * e[a]; d[2 * a + 1] = 2 * e[a] + 1
    return d


def assemble_p2(x, elems, quad, eterms, filt):
    nv = x.size // 2
    g = np.zeros(2 * nv); H = np.zeros((2 * nv, 2 * nv)); E = 0.0
    for t, e in enumerate(elems):
        dofs = _edofs(e)
        Ee, ge, He, _ = eterms(x[dofs], quad[t])
        if not np.isfinite(Ee):
            return np.inf, None, None
        E += Ee
        if filt in ("clamp", "absolute", "project-on-demand"):
            He = project_element(He, filt)
        g[dofs] += ge; H[np.ix_(dofs, dofs)] += He
    return E, g, H


def energy_p2(x, elems, quad, eterms):
    E = 0.0
    for t, e in enumerate(elems):
        Ee, _, _, _ = eterms(x[_edofs(e)], quad[t])
        if not np.isfinite(Ee):
            return np.inf
        E += Ee
    return E


def solve_p2(x0, elems, quad, free, eterms, filt, max_iter=400, tol=1e-6, c=1e-4):
    import time
    x = x0.copy(); it_done = 0; status = "maxiter"; t0 = time.perf_counter()
    for it in range(max_iter):
        E, g, H = assemble_p2(x, elems, quad, eterms, filt)
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
        gd = float(gf @ d)
        if gd >= 0:
            status = "nondescent"; break
        alpha = 1.0; xf0 = x[free].copy()
        while True:
            x[free] = xf0 + alpha * d
            En = energy_p2(x, elems, quad, eterms)
            if np.isfinite(En) and En <= E + c * alpha * gd:
                break
            alpha *= 0.5
            if alpha < 1e-14:
                x[free] = xf0; status = "linesearch"; break
        if status == "linesearch":
            break
    return {"filter": filt, "status": status, "iters": it_done,
            "final_energy": E, "wall_s": time.perf_counter() - t0, "x": x}


def _conformance(seed=0, h=1e-6):
    """FD gradient check on a small P2 mesh (grounding gate)."""
    from .energy import psi, grad_psi, hess_psi
    rng = np.random.default_rng(seed)
    nodes, elems = grid_mesh_p2(3, 3)
    quad = rest_quantities_p2(nodes, elems)
    et = make_element_terms(psi, grad_psi, hess_psi)
    x = (nodes + 0.02 * rng.standard_normal(nodes.shape)).reshape(-1)
    _, g, _ = assemble_p2(x, elems, quad, et, "none")
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (energy_p2(xp, elems, quad, et) - energy_p2(xm, elems, quad, et)) / (2 * h)
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    return rel


if __name__ == "__main__":
    import sys
    rel = _conformance()
    print(f"[p2 conformance] global grad vs FD: max rel err {rel:.2e} -> "
          f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    sys.exit(0 if rel < 1e-5 else 1)
