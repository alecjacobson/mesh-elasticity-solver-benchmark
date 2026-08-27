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
         "back-solve per iteration (AQP additionally runs an Armijo line search, so its per-iteration "
         "work is a little higher). Iterations to reach each method's OWN "
         "`(E-E*)/(E0-E*) < 1e-4` (AQP on symmetric-Dirichlet; local-global & Anderson-LG on ARAP). "
         "Run: `python -m bench.run_aqp_localglobal`.", "",
         "> ⚠️ **This is NOT a fair head-to-head, and does not adjudicate `aqp→local-global` (#6) or "
         "`anderson→aqp` (#5).** The methods minimize **different energies** (symmetric-Dirichlet vs "
         "ARAP) to **different minima**, so 'back-solves to each own tol' compares distances to two "
         "unrelated basins — on this bend the ARAP minimum simply sits nearer the start. A fair "
         "same-objective race is not constructible without committing all three to one energy (which "
         "the source papers do not specify). The numbers below are **descriptive only**; both edges "
         "stay `self-claimed`.", "",
         "| method (energy) | back-solves to own tol, mean [min–max] | wall (ms) mean |",
         "|---|---:|---:|",
         f"| AQP (symmetric-Dirichlet) | {aqp_m:.1f} [{min(aqp_its)}–{max(aqp_its)}] | "
         f"{1e3*np.mean(aqp_wall):.0f} |",
         f"| local-global (ARAP) | {lg_m:.1f} [{min(lg_its)}–{max(lg_its)}] | "
         f"{1e3*np.mean(lg_wall):.0f} |",
         f"| Anderson-LG, m=5 (ARAP) | {an_m:.1f} [{min(an_its)}–{max(an_its)}] | "
         f"{1e3*np.mean(an_wall):.0f} |", "",
         "## Observed (descriptive — not a verdict)", "",
         f"- Each method reaches ITS OWN minimum in: AQP **{aqp_m:.1f}**, local-global **{lg_m:.1f}**, "
         f"Anderson-LG **{an_m:.1f}** back-solves. Anderson-LG's lower count vs local-global is "
         "consistent with the *validated* `anderson→local-global` edge (acceleration of the same ARAP "
         "map). But comparing AQP's symmetric-Dirichlet count against the two ARAP counts does **not** "
         "adjudicate `aqp→local-global` or `anderson→aqp`: the ARAP minimum being nearer on this bend "
         "is an instance property, not a convergence win, and 'lower final energy' is not even "
         "comparable across the two objectives."]
    L += [f"- Wall-clock (both pure NumPy, diagnostic only): AQP {1e3*np.mean(aqp_wall):.0f}ms, "
          f"local-global {1e3*np.mean(lg_wall):.0f}ms, Anderson-LG {1e3*np.mean(an_wall):.0f}ms — "
          "reported for completeness; it cannot rank cross-energy methods either.",
          "",
          "_Caveat: CROSS-ENERGY, single mesh size, one bend family (5 seeds differing only by tiny "
          "jitter ≈ one instance). This runner exists to DOCUMENT WHY `aqp→local-global` (#6) and "
          "`anderson→aqp` (#5) are not adjudicable in this harness, not to score them — both edges "
          "remain `self-claimed`. A fair test needs all methods committed to a single shared energy._"]

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
