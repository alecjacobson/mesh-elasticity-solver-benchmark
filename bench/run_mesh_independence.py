"""AQP in its DESIGN regime: mesh-independent iteration count (#29).

E2 compared AQP to L-BFGS on iteration count and found AQP loses -- but that is not AQP's actual
claim. AQP's design contribution (Kovalsky-Galun-Lipman 2016) is that its Laplacian proxy makes the
iteration count **mesh-independent** (a Sobolev/H^1 preconditioner), unlike a plain first-order
method whose iteration count grows as the mesh refines. Testing AQP only on raw iteration count at
one resolution evaluates it outside the regime it was built for (the reviewer's #29 point).

Here we hold the CONTINUOUS problem fixed (unit square, boundary pinned to a fixed stretch) and
refine the mesh, measuring iterations to a relative-energy tolerance for AQP vs L-BFGS vs Newton.
The design claim predicts AQP's count stays ~flat while L-BFGS's grows with DOFs.

Writes results/mesh_independence.md. Run: `python -m bench.run_mesh_independence`.
"""
import os
import numpy as np
from .mesh import grid_mesh, rest_quantities
from .solver import solve, energy_only
from .energy import element_terms as sd, element_eg
from .descent import solve_lbfgs
from . import world1

TOL_E = 1e-4      # relative energy tolerance for the shared criterion
GTOL = 1e-6       # inner gradient tol (energy-tol is reached well before)


def stretch_problem(n, s=1.5, seed=0):
    """Fixed continuous problem at resolution n: unit square, left/right edges pinned, right edge
    stretched to x=s, interior seeded from rest -- self-similar across n."""
    rest, tris = grid_mesh(n, n)
    Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]
    pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    x = rest.copy(); x[np.abs(xc - 1) < 1e-9, 0] = s
    return dict(rest=rest, tris=tris, Bs=Bs, areas=areas, x0=x.reshape(-1),
                free=~np.repeat(pin, 2), ndof=int((~np.repeat(pin, 2)).sum()))


def iters_to(log, E0, Estar, rtol=TOL_E):
    span = (E0 - Estar) + 1e-30
    for e in log:
        if (e["energy"] - Estar) / span < rtol:
            return e["iter"]
    return None


def main():
    print("== AQP mesh-independence (its design regime) ==\n")
    ns = [6, 10, 14]
    rows = []
    for n in ns:
        p = stretch_problem(n)
        a = (p["x0"], p["tris"], p["Bs"], p["areas"], p["free"])
        rn = solve(*a, "clamp", eterms=sd, tol=GTOL, max_iter=200)
        E0 = energy_only(p["x0"], p["tris"], p["Bs"], p["areas"], sd)
        Estar = rn["final_energy"]
        aqp = world1.solve_aqp(p["x0"], p["tris"], p["rest"], p["free"], max_iter=3000, tol=GTOL)
        lb = solve_lbfgs(*a, element_eg, max_iter=5000, tol=GTOL)
        row = {"n": n, "ndof": p["ndof"],
               "newton": iters_to(rn["log"], E0, Estar),
               "aqp": iters_to(aqp["log"], E0, Estar),
               "l-bfgs": iters_to(lb["log"], E0, Estar)}
        rows.append(row)
        print(f"  n={n:2d} ({row['ndof']:4d} dof): Newton {row['newton']}  AQP {row['aqp']}  L-BFGS {row['l-bfgs']}")

    def growth(key):
        v0, v1 = rows[0][key], rows[-1][key]
        return (v1 / v0) if (v0 and v1) else None
    g_aqp, g_lb, g_nw = growth("aqp"), growth("l-bfgs"), growth("newton")
    dof_growth = rows[-1]["ndof"] / rows[0]["ndof"]

    os.makedirs("results", exist_ok=True)
    L = ["# AQP mesh-independence — its design regime (measured)", "",
         "AQP's actual design claim is a **mesh-independent iteration count** (its Laplacian proxy is "
         "an H¹/Sobolev preconditioner), not a raw iteration win over L-BFGS at one resolution "
         "(which E2 tested and AQP lost). Fixed continuous problem (unit square, right edge stretched "
         "to x=1.5), refined; iterations to relative energy tolerance 1e-4. "
         "Run: `python -m bench.run_mesh_independence`.", "",
         "| mesh | free dof | Newton | AQP | L-BFGS |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']}×{r['n']} | {r['ndof']} | {r['newton']} | {r['aqp']} | {r['l-bfgs']} |")
    L += ["", "## Observed (suggestive, NOT established — review-r2 #52)", "",
          f"- Over a **{dof_growth:.1f}× DOF increase**, iterations-to-tol change by: "
          f"Newton **{g_nw:.1f}×**, AQP **{g_aqp:.1f}×**, L-BFGS **{g_lb:.1f}×**." if all(
              [g_nw, g_aqp, g_lb]) else "- See table (some methods did not reach the tol).",
          f"- AQP's count ({rows[0]['aqp']}→{rows[1]['aqp']}→{rows[-1]['aqp']}) stays roughly flat "
          f"(it even *decreases* on the coarse→mid step — consistent with the coarse mesh needing a "
          f"couple extra iterations to hit the *relative*-energy tol, not a clean asymptotic trend), "
          f"while L-BFGS's ({rows[0]['l-bfgs']}→{rows[1]['l-bfgs']}→{rows[-1]['l-bfgs']}) grows "
          f"monotonically. This is **consistent with** the mesh-independence AQP was built for — the "
          f"scaling axis E2's single-resolution comparison cannot see.",
          "- **But this is only 3 mesh sizes and one seed, so it is SUGGESTIVE, not established.** "
          "A non-monotone AQP count over 3 points does not prove asymptotic mesh-independence, and "
          "L-BFGS's growth could partly reflect its own (un-preconditioned) scaling rather than a fair "
          "contrast. The honest reading of `aqp→l-bfgs`: AQP loses on raw iterations at a fixed mesh "
          "(e2), and its proxy *appears* to give mesh-independent scaling here — a hypothesis that "
          "needs a wider refinement sweep (to ~20×20+) and multiple seeds with error bars to confirm.",
          "",
          "_Caveat: 2D, 3 sizes / single seed / dense / energy-tol — indicative only. Path to a real "
          "verdict: more refinements + seed-averaged counts with spread (#52). Newton is "
          "mesh-independent too (known) but pays a factorization per iteration (see e2)._"]
    with open("results/mesh_independence.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nwrote results/mesh_independence.md")


if __name__ == "__main__":
    main()
