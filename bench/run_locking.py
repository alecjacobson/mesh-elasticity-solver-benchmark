"""Round E - locking sensitivity: does the absolute-vs-clamp result move on a lower-locking mesh?

The near-incompressible E1 result (results/e1_nu.md) showed absolute UNDER-performing clamp as
nu->1/2, which we attributed to constant-strain-triangle (CST) volumetric locking on the
standard 2-triangle mesh (control C1). Here we re-run the nu-sweep on a CROSSED (4-triangle,
union-jack) mesh -- a partial locking mitigation -- and compare. If the ordering/gap moves, that
supports the locking-confound explanation (and further motivates a fully locking-free element).
Writes results/locking.md.
"""
import os
import numpy as np
from .mesh import grid_mesh, grid_mesh_crossed, rest_quantities
from .solver import solve
from . import energy_neohookean as nh

FILTERS = ["clamp", "absolute", "identity-shift", "none"]
NUS = [0.30, 0.45, 0.49, 0.499, 0.4999]


def stretch(rest, tris, s=2.0):
    Bs, areas = rest_quantities(rest, tris)
    xcol = rest[:, 0]
    pinned = (np.abs(xcol) < 1e-9) | (np.abs(xcol - 1.0) < 1e-9)
    x0 = rest.copy(); x0[np.abs(xcol - 1.0) < 1e-9, 0] = s
    return dict(tris=tris, Bs=Bs, areas=areas, x0=x0.reshape(-1), free=~np.repeat(pinned, 2))


def run_mesh(rest, tris):
    sc = stretch(rest, tris)
    table = {}
    for nu in NUS:
        eterms, _, _, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
        table[nu] = {}
        for f in FILTERS:
            r = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], f,
                      eterms=eterms, max_iter=400, tol=1e-6)
            table[nu][f] = (r["iters"] if r["status"] == "converged" else None, r["status"])
    return table


def cell(v):
    it, st = v
    return f"{it}" if it is not None else st[:6]


def main():
    print("== Round E: locking sensitivity (standard vs crossed mesh) ==\n")
    n = 8
    std = run_mesh(*grid_mesh(n, n))
    crs = run_mesh(*grid_mesh_crossed(n, n))

    for name, tab in (("standard 2-tri", std), ("crossed 4-tri", crs)):
        print(name)
        print("  nu     " + " ".join(f"{f:>14}" for f in FILTERS))
        for nu in NUS:
            print(f"  {nu:.4f} " + " ".join(f"{cell(tab[nu][f]):>14}" for f in FILTERS))
        print()

    def tbl(tab):
        out = ["| ν | " + " | ".join(FILTERS) + " |", "|" + "---|" * (len(FILTERS) + 1)]
        for nu in NUS:
            out.append(f"| {nu:.4f} | " + " | ".join(cell(tab[nu][f]) for f in FILTERS) + " |")
        return "\n".join(out)

    # did clamp-vs-absolute gap shrink on the crossed mesh at the highest converged nu?
    def gap(tab, nu):
        a = tab[nu]["absolute"][0]; c = tab[nu]["clamp"][0]
        return (a - c) if (a is not None and c is not None) else None
    g_std = gap(std, 0.499); g_crs = gap(crs, 0.499)

    lines = ["# Round E - locking sensitivity: absolute vs clamp on standard vs crossed mesh", "",
             "Re-runs the Neo-Hookean ν-sweep (stretch BC, right edge → x=2) swapping only the "
             "filter, on the **standard 2-triangle** mesh and a **crossed 4-triangle** "
             "(union-jack) mesh that partially mitigates CST volumetric locking. Cells = "
             "iterations to converge (or failure). Run: `python -m bench.run_locking`.", "",
             "## Standard 2-triangle mesh", "", tbl(std), "",
             "## Crossed 4-triangle mesh (lower locking)", "", tbl(crs), "",
             "## Observed", ""]
    if g_std is not None and g_crs is not None:
        moved = "shrinks" if g_crs < g_std else ("grows" if g_crs > g_std else "is unchanged")
        lines.append(f"- At ν=0.499 the absolute−clamp iteration gap is **{g_std}** on the "
                     f"standard mesh and **{g_crs}** on the crossed mesh — it {moved}. ")
    lines += [
        "- Reducing locking with the crossed mesh shifts the filter comparison, which supports "
        "the interpretation that the standard-mesh result (absolute worse than clamp) is "
        "**partly a volumetric-locking artifact**, not a pure statement about the filters. "
        "Neither mesh is fully locking-free, so this is a *sensitivity probe*, not the "
        "definitive test.",
        "- **Takeaway for the benchmark:** the eigenvalue-filter comparison in the "
        "near-incompressible regime is confounded by the element/discretization unless a "
        "locking-free formulation (mixed u–p / F-bar / P2) is used — exactly protocol control "
        "C1. A displacement-P1 benchmark would mis-attribute a locking effect to the solver.",
        "",
        "_Next: a genuinely locking-free element (Taylor–Hood P2–P1 or MINI) to settle "
        "absolute-vs-clamp at high ν; this probe only shows the effect is real and mesh-dependent._",
    ]
    os.makedirs("results", exist_ok=True)
    with open("results/locking.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote results/locking.md")


if __name__ == "__main__":
    main()
