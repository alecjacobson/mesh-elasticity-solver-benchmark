"""Fast smoke test: asserts the harness invariants hold (run in CI).

Plain asserts (no pytest dependency): `python -m bench.test_smoke` exits 0 on pass, 1 on fail.
Keeps to tiny problems so it runs in a few seconds.
"""
import numpy as np
from . import conformance
from .mesh import grid_mesh, boundary_mask, rest_quantities
from .solver import solve, energy_only
from .energy import element_terms as sd_terms


def _tiny_scenario(n=5, amp_frac=0.3, seed=3):
    rest, tris = grid_mesh(n, n)
    Bs, areas = rest_quantities(rest, tris)
    bmask = boundary_mask(rest)
    rng = np.random.default_rng(seed)
    amp = amp_frac / n
    for _ in range(40):
        pert = rng.standard_normal(rest.shape); pert[bmask] = 0.0
        x = (rest + amp * pert).reshape(-1)
        if np.isfinite(energy_only(x, tris, Bs, areas, sd_terms)):
            break
        amp *= 0.8
    return rest, tris, Bs, areas, x, ~np.repeat(bmask, 2)


def main():
    ok = True

    # 1. conformance (derivatives)
    r1, p1 = conformance.check_grad_psi()
    r2, p2 = conformance.check_global_gradient()
    assert p1 and p2, f"conformance failed: {r1:.1e}, {r2:.1e}"
    print(f"[smoke] conformance PASS ({r1:.1e}, {r2:.1e})")

    rest, tris, Bs, areas, x0, free = _tiny_scenario()
    args = (x0, tris, Bs, areas, free)

    # 2. clamp converges to the identity-map minimum (E = 4.0 for symmetric Dirichlet, area 1)
    rc = solve(*args, "clamp", eterms=sd_terms, tol=1e-6)
    assert rc["status"] == "converged", f"clamp did not converge: {rc['status']}"
    assert abs(rc["final_energy"] - 4.0) < 1e-3, f"clamp energy {rc['final_energy']}"
    print(f"[smoke] clamp converged in {rc['iters']} it to E={rc['final_energy']:.4f}")

    # 3. absolute and project-on-demand also converge to the same minimum
    for f in ("absolute", "project-on-demand", "identity-shift", "global-pdn"):
        r = solve(*args, f, eterms=sd_terms, tol=1e-6)
        assert r["status"] == "converged", f"{f} did not converge: {r['status']}"
        assert abs(r["final_energy"] - 4.0) < 1e-3, f"{f} energy {r['final_energy']}"
    print("[smoke] absolute/project-on-demand/identity-shift/global-pdn all converge to E=4.0")

    # 4. telemetry counters are populated and monotone-ish
    assert rc["counts"]["assemblies"] >= rc["iters"] >= 1
    assert rc["counts"]["lin_solves"] >= 1
    print("[smoke] telemetry counters present")

    print("[smoke] ALL PASS")
    return ok


if __name__ == "__main__":
    import sys
    try:
        sys.exit(0 if main() else 1)
    except AssertionError as e:
        print("[smoke] FAIL:", e); sys.exit(1)
