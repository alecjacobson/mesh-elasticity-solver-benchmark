"""AQP vs ARAP local-global — P5.2 edge #6 (aqp -> local-global, speed).

Claim (Kovalsky et al. 2016): "AQP terminates faster than ARAP local/global alternation"
(0.38s vs 1.55s in 2D). Wall-clock is implementation-confounded, so we adjudicate on the
HW-INDEPENDENT cost both share: each does ONE prefactorization (AQP: fixed Laplacian proxy;
local-global: cotan-Laplacian) then ONE global BACK-SOLVE per iteration, so **back-solves ==
iterations** is a fair per-iteration-cost-matched proxy. We count back-solves each method needs to
reach its OWN minimum to a relative-energy tolerance.

Honest confound (why this is 'indicative', not 'validated'): AQP minimizes SYMMETRIC-DIRICHLET
while local-global minimizes ARAP — different energies with different minima. We therefore compare
each method's back-solves to ITS OWN relative-energy tol on the SAME deformation instance; this
measures "which alternation reaches its target faster," not a same-objective race. Wall-clock is
reported diagnostically (both pure NumPy here, so it is at least not cross-language).
Writes results/aqp_localglobal.md. Run: `python -m bench.run_aqp_localglobal`.
"""
import os
import numpy as np
from .mesh import grid_mesh, boundary_mask, rest_quantities
from .solver import solve, energy_only
from .energy import element_terms as sd
from . import world1


def _bend(n, k=0.7, seed=0):
    """Whole-mesh non-affine bend (rotate each vertex by angle k·(x−½)); injective (no flips) but a
    non-trivial minimum, so BOTH methods must iterate. An affine BC would make the affine interior
    the exact minimiser (1 step, no headroom); a small non-flipping jitter varies the seed."""
    rest, tris = grid_mesh(n, n)
    bmask = boundary_mask(rest)
    xr = rest[:, 0] - 0.5; yr = rest[:, 1] - 0.5; th = k * xr
    x = np.stack([np.cos(th) * xr - np.sin(th) * yr, np.sin(th) * xr + np.cos(th) * yr], 1)
    rng = np.random.default_rng(seed)
    interior = ~bmask
    x[interior] += (0.03 / n) * rng.standard_normal((int(interior.sum()), 2))
    return rest, tris, x.reshape(-1), ~np.repeat(bmask, 2)


def _iters_to(energies, rtol=1e-4):
    E0, Estar = energies[0], min(energies)
    span = (E0 - Estar) + 1e-30
    for k, E in enumerate(energies):
        if (E - Estar) / span < rtol:
            return k
    return len(energies) - 1


def main():
    N, SEEDS = 10, [0, 1, 2, 3, 4]
    aqp_its, lg_its, an_its = [], [], []
    aqp_wall, lg_wall, an_wall = [], [], []
    for s in SEEDS:
        rest, tris, x0, free = _bend(N, seed=s)
        Bs, areas = rest_quantities(rest, tris)
        # AQP on symmetric Dirichlet (hard BC)
        ra = world1.solve_aqp(x0, tris, rest, free, max_iter=3000, tol=1e-7)
        aqp_E = [e["energy"] for e in ra["log"]]
        aqp_its.append(_iters_to(aqp_E)); aqp_wall.append(ra["wall_s"])
        # ARAP local-global (hard BC)
        rl = world1.solve_local_global(x0, tris, rest, free, max_iter=3000, tol=1e-7)
        lg_E = [e["energy"] for e in rl["log"]]
        lg_its.append(_iters_to(lg_E)); lg_wall.append(rl["wall_s"])
        # Anderson-accelerated local-global (ARAP) — the anderson->aqp edge #5, same instance
        ran = world1.solve_anderson(x0, tris, rest, free, m=5, max_iter=3000, tol=1e-7)
        an_E = [e["energy"] for e in ran["log"]]
        an_its.append(_iters_to(an_E)); an_wall.append(ran["wall_s"])

    aqp_m, lg_m, an_m = (float(np.mean(aqp_its)), float(np.mean(lg_its)), float(np.mean(an_its)))
    aqp_faster = aqp_m < lg_m
    an_beats_aqp = an_m < aqp_m
    L = ["# AQP vs ARAP local-global vs Anderson-LG — back-solves to each method's own minimum "
         "(measured, P5.2 #6 & #5)", "",
         f"{N}×{N} non-affine bend, {len(SEEDS)} seeds. Each method does 1 prefactorization + 1 global "
         "**back-solve per iteration**, so back-solves == iterations is a per-iteration-cost-matched "
         "HW-independent proxy. Iterations to reach each method's OWN "
         "`(E-E*)/(E0-E*) < 1e-4` (AQP on symmetric-Dirichlet; local-global & Anderson-LG on ARAP). "
         "Run: `python -m bench.run_aqp_localglobal`.", "",
         "| method (energy) | back-solves to tol, mean [min–max] | wall (ms) mean |",
         "|---|---:|---:|",
         f"| AQP (symmetric-Dirichlet) | {aqp_m:.1f} [{min(aqp_its)}–{max(aqp_its)}] | "
         f"{1e3*np.mean(aqp_wall):.0f} |",
         f"| local-global (ARAP) | {lg_m:.1f} [{min(lg_its)}–{max(lg_its)}] | "
         f"{1e3*np.mean(lg_wall):.0f} |",
         f"| Anderson-LG, m=5 (ARAP) | {an_m:.1f} [{min(an_its)}–{max(an_its)}] | "
         f"{1e3*np.mean(an_wall):.0f} |", "",
         "## Observed", "",
         (f"- **#5 `anderson->aqp` (smaller cost):** Anderson-accelerated local-global reaches its "
          f"ARAP minimum in **{an_m:.1f}** back-solves vs AQP's **{aqp_m:.1f}** to its "
          f"symmetric-Dirichlet minimum — Anderson is cheaper on the shared back-solve axis, "
          "indicative support (cross-energy: 'lower final energy' is not comparable across the two "
          "objectives, only cost is)."
          if an_beats_aqp else
          f"- **#5 `anderson->aqp`:** Anderson-LG needs **{an_m:.1f}** back-solves vs AQP's "
          f"**{aqp_m:.1f}** — not cheaper here; claim not reproduced on the back-solve axis."), ""]
    if aqp_faster:
        L.append(f"- **`aqp->local-global` reproduces on back-solves:** AQP reaches its "
                 f"symmetric-Dirichlet minimum in **{aqp_m:.1f}** back-solves vs local-global's "
                 f"**{lg_m:.1f}** to its ARAP minimum — AQP terminates in fewer equally-priced global "
                 "solves, consistent with the paper's faster-termination claim.")
    else:
        L.append(f"- **Not reproduced here:** AQP needs **{aqp_m:.1f}** back-solves vs local-global's "
                 f"**{lg_m:.1f}** — local-global reaches its target in fewer/equal global solves on "
                 "this instance, so the faster-termination claim does not hold on the back-solve axis.")
    L += [f"- Wall-clock (both pure NumPy, diagnostic): AQP {1e3*np.mean(aqp_wall):.0f}ms vs "
          f"local-global {1e3*np.mean(lg_wall):.0f}ms — "
          + ("consistent with" if (np.mean(aqp_wall) < np.mean(lg_wall)) == aqp_faster
             else "note the wall-clock ordering differs from the back-solve ordering (per-iter "
                  "overheads: AQP's line search vs local-global's rotation fits), so") +
          " the count carries the verdict, not the millisecond number.",
          "",
          "_Caveat: CROSS-ENERGY — AQP minimizes symmetric-Dirichlet, local-global minimizes ARAP; "
          "each is scored to its own minimum, so this is 'which reaches its target in fewer equal-cost "
          "solves,' not a same-objective race. Single mesh size, moderate shear; indicative, not a "
          "validated same-energy head-to-head._"]

    os.makedirs("results", exist_ok=True)
    with open("results/aqp_localglobal.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  AQP back-solves {aqp_its} (mean {aqp_m:.1f})")
    print(f"  LG  back-solves {lg_its} (mean {lg_m:.1f})")
    print(f"  AndersonLG {an_its} (mean {an_m:.1f})")
    print(f"  aqp_faster={aqp_faster}; anderson<aqp={an_beats_aqp}; wrote results/aqp_localglobal.md")
    return True


if __name__ == "__main__":
    main()
