# Injectivity / feasibility suite — untangling a folded init (measured)

![injectivity](../figures/injectivity.png)

_`figures/injectivity.png`: a folded init (red = inverted) untangled to all-valid by both barrier-free energies; barrier symmetric-Dirichlet is +∞ here and cannot start. Generate: `python -m bench.run_figures injectivity`._

Which energies recover an **inversion-free** map from a **folded** (non-injective) start. Boundary pinned to the rest square (an injective solution exists); interior reflected to create folds of increasing severity. 4 seeds × 3 severities on a 12×12 grid. `untangle` = classical one-sided area penalty (TLC's barrier-free ancestor, `bench/untangle.py`, conformance-gated); `stable-NH` = Stable Neo-Hookean (finite at J≤0); `barrier-SD` = symmetric Dirichlet (+∞ at folds, CONTROL). Run: `python -m bench.run_injectivity`.

| severity | folds at start | untangle (success · med it) | stable-NH | barrier-SD (feasible start?) |
|---|---|---|---|---|
| mild | 107 | 100% · 30 | 100% · 8 | 0% |
| moderate | 108 | 100% · 39 | 100% · 9 | 0% |
| severe | 109 | 100% · 52 | 100% · 9 | 0% |

## Observed

- **Barrier symmetric Dirichlet cannot even start** from a folded map: its energy is +∞ at J≤0, so **0%** feasible-start across every severity — the concrete statement of the injectivity cohort's reason to exist. A distortion-barrier solver needs a **feasible (inversion-free) initialization** (a Tutte/Floater embedding), which is a separate problem; it can polish an injective map but never *find* one from folds.
- **Barrier-free energies untangle.** The classical area-penalty reaches an injective map (100/100/100% over mild/moderate/severe); Stable Neo-Hookean also recovers (100/100/100%) — it is finite through inversion, so it flows a folded mesh back to the identity minimizer.
- **Lineage.** The graphics injectivity methods (TLC, foldover-free, progressive embedding, simplex assembly) are exactly this: **barrier-free untangling** energies with better basins / guarantees than the raw area penalty. This suite establishes the capability axis (untangle-from-folds) that separates them from distortion-barrier minimizers; per-method faithful ports (TLC's lifted content, etc.) are the next step to rank *within* the cohort.

_Caveat: 2D, one mesh, pinned-square boundary (an injective target is guaranteed to exist); success = all signed areas > 0 at convergence. The area penalty is TLC's ancestor, not TLC; ranking within the injectivity cohort needs the specific methods._
