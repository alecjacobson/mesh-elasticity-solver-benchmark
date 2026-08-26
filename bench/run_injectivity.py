"""Injectivity / feasibility suite — which energies untangle a folded init (issue #97).

The core distinction in the World-1 injectivity cohort: a **barrier** distortion energy (symmetric
Dirichlet, +∞ at J≤0) needs a FEASIBLE (inversion-free) start and can never cross a fold, whereas
**barrier-free** energies can pull a NON-injective map to injective. We measure this directly on
folded initializations of increasing severity (boundary pinned to the rest square, which admits the
identity as an injective solution; interior reflected/folded):

  - untangle  : the classical one-sided area penalty (bench/untangle.py) — TLC's barrier-free ancestor
  - stable-NH : Stable Neo-Hookean (finite at J≤0) — the simulation-side barrier-free energy
  - barrier-SD: symmetric Dirichlet (+∞ at folds) — CONTROL, expected to fail to even start

Metric: success rate (all signed areas > 0) + iterations, across severities × seeds. Run:
`python -m bench.run_injectivity`. Writes results/injectivity.md.
"""
import os
import numpy as np
from .mesh import grid_mesh, rest_quantities, boundary_mask
from .untangle import solve as untangle_solve, signed_areas, run as untangle_conf
from .solver import solve as nt_solve, energy_only
from . import energy_stable_neohookean as snh
from .energy import element_terms as sd_terms

SEEDS = [0, 1, 2, 3]
SEVERITIES = {"mild": 0.55, "moderate": 0.75, "severe": 0.95}   # reflection strength of interior fold
N = 12


def folded_init(strength, seed):
    """Grid with boundary pinned to the rest square; interior x>c reflected by `strength` → folds."""
    rest, tris = grid_mesh(N, N)
    Bs, areas = rest_quantities(rest, tris)
    bmask = boundary_mask(rest); free = ~np.repeat(bmask, 2)
    rng = np.random.default_rng(seed)
    c = 0.5 + 0.05 * rng.standard_normal()
    x = rest.copy(); intr = ~bmask; xr = rest[:, 0]
    fold = intr & (xr > c)
    x[fold, 0] = c - strength * (xr[fold] - c)
    x[intr] += (0.01) * rng.standard_normal((int(intr.sum()), 2))    # break symmetry per seed
    return rest, tris, Bs, areas, free, x.reshape(-1)


def _n_folds(x, tris):
    return int((signed_areas(x.reshape(-1, 2), tris) <= 0).sum())


def run():
    ok = untangle_conf()
    # results[method][severity] = list over seeds of (success, iters, n_inverted_start, n_inverted_end)
    res = {m: {s: [] for s in SEVERITIES} for m in ("untangle", "stable-NH", "barrier-SD")}
    for sev, strength in SEVERITIES.items():
        for seed in SEEDS:
            rest, tris, Bs, areas, free, x0 = folded_init(strength, seed)
            nf0 = _n_folds(x0, tris)
            mean_area = float(np.mean(np.abs(signed_areas(rest.reshape(-1, 2), tris))))

            ru = untangle_solve(x0, tris, free, delta=0.25 * mean_area, max_iter=3000)
            res["untangle"][sev].append((ru["success"], ru["iters"], nf0, ru["n_inverted"]))

            et, _, _, _ = snh.make(mu=1.0, lam=snh.lam_from_nu(0.45))
            rs = nt_solve(x0, tris, Bs, areas, free, "clamp", eterms=et, tol=1e-7, max_iter=400)
            a_s = signed_areas(rs["x"].reshape(-1, 2), tris)
            res["stable-NH"][sev].append((bool(a_s.min() > 0), rs["iters"], nf0, int((a_s <= 0).sum())))

            E0 = energy_only(x0, tris, Bs, areas, sd_terms)   # +inf at a folded init
            feasible_start = np.isfinite(E0)
            res["barrier-SD"][sev].append((feasible_start, 0, nf0, nf0))

    def rate(m, s):
        rs = res[m][s]; return sum(1 for r in rs if r[0]) / len(rs)

    def med_iters(m, s):
        its = [r[1] for r in res[m][s] if r[0]]
        return int(np.median(its)) if its else None

    L = ["# Injectivity / feasibility suite — untangling a folded init (measured)", "",
         "![injectivity](../figures/injectivity.png)", "",
         "_`figures/injectivity.png`: a folded init (red = inverted) untangled to all-valid by both "
         "barrier-free energies; barrier symmetric-Dirichlet is +∞ here and cannot start. Generate: "
         "`python -m bench.run_figures injectivity`._", "",
         "Which energies recover an **inversion-free** map from a **folded** (non-injective) start. "
         "Boundary pinned to the rest square (an injective solution exists); interior reflected to "
         f"create folds of increasing severity. {len(SEEDS)} seeds × 3 severities on a {N}×{N} grid. "
         "`untangle` = classical one-sided area penalty (TLC's barrier-free ancestor, "
         "`bench/untangle.py`, conformance-gated); `stable-NH` = Stable Neo-Hookean (finite at J≤0); "
         "`barrier-SD` = symmetric Dirichlet (+∞ at folds, CONTROL). Run: `python -m bench.run_injectivity`.",
         "",
         "| severity | folds at start | untangle (success · med it) | stable-NH | barrier-SD (feasible start?) |",
         "|---|---|---|---|---|"]
    for sev in SEVERITIES:
        nf = int(np.median([r[2] for r in res["untangle"][sev]]))
        L.append(f"| {sev} | {nf} | {rate('untangle',sev)*100:.0f}% · {med_iters('untangle',sev)} | "
                 f"{rate('stable-NH',sev)*100:.0f}% · {med_iters('stable-NH',sev)} | "
                 f"{rate('barrier-SD',sev)*100:.0f}% |")
    L += ["", "## Observed", "",
          f"- **Barrier symmetric Dirichlet cannot even start** from a folded map: its energy is +∞ at "
          f"J≤0, so **{rate('barrier-SD','mild')*100:.0f}%** feasible-start across every severity — the "
          "concrete statement of the injectivity cohort's reason to exist. A distortion-barrier solver "
          "needs a **feasible (inversion-free) initialization** (a Tutte/Floater embedding), which is a "
          "separate problem; it can polish an injective map but never *find* one from folds.",
          f"- **Barrier-free energies untangle.** The classical area-penalty reaches an injective map "
          f"({rate('untangle','mild')*100:.0f}/{rate('untangle','moderate')*100:.0f}/"
          f"{rate('untangle','severe')*100:.0f}% over mild/moderate/severe); Stable Neo-Hookean also "
          f"recovers ({rate('stable-NH','mild')*100:.0f}/{rate('stable-NH','moderate')*100:.0f}/"
          f"{rate('stable-NH','severe')*100:.0f}%) — it is finite through inversion, so it flows a "
          "folded mesh back to the identity minimizer.",
          "- **Lineage.** The graphics injectivity methods (TLC, foldover-free, progressive embedding, "
          "simplex assembly) are exactly this: **barrier-free untangling** energies with better basins / "
          "guarantees than the raw area penalty. This suite establishes the capability axis "
          "(untangle-from-folds) that separates them from distortion-barrier minimizers; per-method "
          "faithful ports (TLC's lifted content, etc.) are the next step to rank *within* the cohort.",
          "",
          "_Caveat: 2D, one mesh, pinned-square boundary (an injective target is guaranteed to exist); "
          "success = all signed areas > 0 at convergence. The area penalty is TLC's ancestor, not TLC; "
          "ranking within the injectivity cohort needs the specific methods._"]
    os.makedirs("results", exist_ok=True)
    with open("results/injectivity.md", "w") as f:
        f.write("\n".join(L) + "\n")
    for sev in SEVERITIES:
        print(f"  {sev:9s} folds~{int(np.median([r[2] for r in res['untangle'][sev]])):3d}  "
              f"untangle {rate('untangle',sev)*100:3.0f}%  stable-NH {rate('stable-NH',sev)*100:3.0f}%  "
              f"barrier-SD {rate('barrier-SD',sev)*100:3.0f}%")
    print(f"[injectivity] {'PASS' if ok else 'FAIL (conformance)'}; wrote results/injectivity.md")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
