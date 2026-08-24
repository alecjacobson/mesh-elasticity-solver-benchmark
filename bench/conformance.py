"""Conformance / grounding tests (harness.md admissibility gate).

For the classical analytic energy, correctness of the derivatives IS the reference: we verify
(1) the analytic dpsi/dF against a finite-difference of psi, and (2) the assembled global
gradient against a finite-difference of the global energy. This stands in for official-code
regression until an official reference (e.g. TinyAD / libigl) is ported; it is the gate that a
component -- human- or agent-written -- must pass before entering a comparison.
"""
import numpy as np
from .energy import psi, grad_psi
from .mesh import grid_mesh, rest_quantities
from .solver import energy_only, assemble


def check_grad_psi(n=200, h=1e-6, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        F = np.eye(2) + 0.3 * rng.standard_normal((2, 2))
        if np.linalg.det(F) <= 0.2:
            continue
        G = grad_psi(F).reshape(4)
        Ff = F.reshape(4)
        Gfd = np.zeros(4)
        for k in range(4):
            fp = Ff.copy(); fp[k] += h
            fm = Ff.copy(); fm[k] -= h
            Gfd[k] = (psi(fp.reshape(2, 2)) - psi(fm.reshape(2, 2))) / (2 * h)
        worst = max(worst, np.max(np.abs(G - Gfd)) / (np.max(np.abs(Gfd)) + 1e-12))
    return worst, worst < tol


def check_global_gradient(nx=4, ny=4, h=1e-6, tol=1e-5, seed=1):
    rng = np.random.default_rng(seed)
    rest, tris = grid_mesh(nx, ny)
    Bs, areas = rest_quantities(rest, tris)
    x = (rest + 0.05 * rng.standard_normal(rest.shape)).reshape(-1)
    _, g, _ = assemble(x, tris, Bs, areas, "none")
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        gfd[k] = (energy_only(xp, tris, Bs, areas) - energy_only(xm, tris, Bs, areas)) / (2 * h)
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    return rel, rel < tol


def run():
    r1, ok1 = check_grad_psi()
    r2, ok2 = check_global_gradient()
    print(f"[conformance] dpsi/dF vs FD:        max rel err {r1:.2e}  -> {'PASS' if ok1 else 'FAIL'}")
    print(f"[conformance] global grad vs FD:    max rel err {r2:.2e}  -> {'PASS' if ok2 else 'FAIL'}")
    ok = ok1 and ok2
    print(f"[conformance] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
