"""OFFICIAL-CODE regression (D3): our symmetric-Dirichlet solve vs libigl's SLIM.

Runs the SAME parametrization problem (same mesh, same pinned boundary, same feasible init)
through (a) our harness and (b) libigl's SLIM (igl.SYMMETRIC_DIRICHLET) and checks that the two
reach the same minimum energy. This upgrades the D3 grounding for the World-1 energy from
"matches the canonical definition" (bench/conformance.py) to "matches an official reference
implementation's minimizer". OPTIONAL: needs `pip install libigl`; skips cleanly if absent, so
core CI stays light. Run: `python -m bench.check_libigl`.
"""
import numpy as np


def main():
    try:
        import igl
    except ImportError:
        print("[libigl-regression] SKIP: libigl not installed (pip install libigl)")
        return True
    from .mesh import grid_mesh, rest_quantities
    from .solver import solve, energy_only
    from .energy import element_terms as sd

    N, s = 8, 1.6
    rest, tris = grid_mesh(N, N)
    Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]
    pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    bidx = np.where(pin)[0].astype(np.int32)
    target = rest.copy(); target[:, 0] = s * rest[:, 0]        # stretch targets on pinned cols
    bc = target[bidx].astype(np.float64)

    # non-trivial feasible init: uniform stretch + small interior perturbation
    rng = np.random.default_rng(0)
    Vinit = target.copy()
    pert = rng.standard_normal(rest.shape) * (0.05 / N)
    pert[pin] = 0.0
    Vinit = Vinit + pert

    # (a) our harness
    free = ~np.repeat(pin, 2)
    r = solve(Vinit.reshape(-1).copy(), tris, Bs, areas, free, "clamp", eterms=sd,
              tol=1e-9, max_iter=500)
    E_ours = r["final_energy"]

    # (b) libigl SLIM (official) on the same problem
    V3 = np.hstack([rest, np.zeros((rest.shape[0], 1))]).astype(np.float64)
    data = igl.slim_precompute(V3, tris.astype(np.int32), Vinit.astype(np.float64),
                               igl.SYMMETRIC_DIRICHLET, bidx, bc, 1e8)
    UV = igl.slim_solve(data, 300)
    E_slim = energy_only(UV.reshape(-1), tris, Bs, areas, sd)

    rel = abs(E_ours - E_slim) / abs(E_ours)
    ok = rel < 1e-5
    print(f"[libigl-regression] our harness converged E = {E_ours:.6f} ({r['iters']} it)")
    print(f"[libigl-regression] libigl SLIM  converged E = {E_slim:.6f}")
    print(f"[libigl-regression] relative difference       = {rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
