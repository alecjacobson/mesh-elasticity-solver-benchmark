"""Conformance for World-1 accelerators: each must reach the SAME minimizer as projected Newton
on symmetric Dirichlet (that's what makes them comparable). `python -m bench.test_world1`.
"""
import numpy as np
from .solver import solve
from .energy import element_terms as sd
from .run_e1 import build_scenario
from . import world1


def main():
    ok = True
    sc = build_scenario(nx=8, ny=8)
    ref = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], "clamp", eterms=sd, tol=1e-6)
    E_ref = ref["final_energy"]

    methods = [("aqp", lambda: world1.solve_aqp(sc["x0"], sc["tris"], sc["rest"], sc["free"],
                                                max_iter=4000, tol=1e-6))]
    if hasattr(world1, "solve_slim"):
        methods.append(("slim", lambda: world1.solve_slim(sc["x0"], sc["tris"], sc["rest"],
                                                          sc["free"], max_iter=2000, tol=1e-6)))
    for name, run in methods:
        r = run()
        d = abs(r["final_energy"] - E_ref)
        good = r["status"] == "converged" and d < 1e-5
        ok = ok and good
        print(f"[world1] {name:6s} status={r['status']:10s} iters={r['iters']:5d} "
              f"dE_vs_Newton={d:.1e} -> {'PASS' if good else 'FAIL'}")
    print(f"[world1] {'ALL PASS' if ok else 'FAILED'} (Newton ref E={E_ref:.6f}, {ref['iters']} it)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
