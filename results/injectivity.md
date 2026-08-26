# Injectivity / feasibility suite — untangling a folded init (measured)

![injectivity](../figures/injectivity.png)

_`figures/injectivity.png`: a folded init (red = inverted) untangled to all-valid by both barrier-free energies; barrier symmetric-Dirichlet is +∞ here and cannot start. Generate: `python -m bench.run_figures injectivity`._

Which energies recover an **inversion-free** map from a **folded** (non-injective) start. Boundary pinned to the rest square (so the identity is a guaranteed injective solution); interior reflected to create folds of increasing severity. 4 seeds × 4 severities on a 12×12 grid. Two **barrier-free** energies are compared: `untangle` = classical one-sided area penalty (TLC's barrier-free *ancestor*, `bench/untangle.py`, conformance-gated) and `stable-NH` = Stable Neo-Hookean (finite at J≤0). The shared, energy-independent metric is **iters-to-first-injective** (first iterate with all signed areas > 0); each method's *final* iters-to-tol are on different energies/criteria and are **not** comparable. Run: `python -m bench.run_injectivity`.

| severity | folds at start | untangle: success · first-inj · [final it] | stable-NH: success · first-inj · [final it] |
|---|---|---|---|
| mild | 107 | 100% · 18 · [30] | 100% · 2 · [8] |
| moderate | 108 | 100% · 27 · [39] | 100% · 3 · [9] |
| severe | 109 | 100% · 40 · [52] | 100% · 3 · [9] |
| extreme | 109 | 100% · 36 · [51] | 100% · 3 · [9] |

### Hard boundary — a provably-injective **non-convex** target (wavy warp A=0.5, Jacobian≡1), 4 seeds

The pinned-square target above is trivially the identity, so success saturates. This case pins the boundary to φ(rest) with φ(x,y)=(x+A·sin πy, y) — a **unit-Jacobian bijection**, so φ(grid) is a *guaranteed* injective solution, but the boundary is **non-convex** (Tutte gives no guarantee) and stable-NH's elastic minimizer need not be injective.

| method | success | first-inj (median) |
|---|---|---|
| untangle | 100% | 610 |
| stable-NH | 100% | 6 |

## Observed

- **Barrier-free energies untangle; the axis is capability, not speed.** Both reach an injective map **100%** across every severity (mild→extreme, up to ~109 folds), because with the boundary pinned to the rest square the identity is the unique injective minimizer and both energies are finite through inversion. On the shared **iters-to-first-injective** metric Stable NH reaches injectivity faster (3 vs 40 it at severe) — a better basin from the elastic energy — but the suite does **not separate them on success** here; the hard non-convex boundary below is what does.
- **The hard non-convex boundary discriminates on SPEED, not success.** Both still reach injectivity (100%), but iters-to-first-injective **blows apart**: the raw area-penalty needs **610** iters vs Stable NH's **6** (~102×) — on the square it was only 40 vs 3. The raw penalty's basin degrades severely on a non-convex boundary while the elastic energy stays efficient — the concrete reason the injectivity cohort (TLC's lifted content, foldover-free's regularization) invests in better formulations than 'penalize negative areas'. Success still saturates (an injective target exists); a boundary with no injective solution is the feasibility-detection probe.
- **Barrier symmetric Dirichlet is a definitional non-starter, stated as such (not scored).** At a folded init SD is +∞ by construction (`finite=False`), so we do **not** run it — reporting a per-severity '0%' would be measuring the initialization, not a solver. The honest control is the **asymmetry**: given a *feasible* start SD is fine — from the rest square it needs **0 iters** (the identity already IS the minimizer) and from a feasible distorted start it converges normally (`e1`). SD can *polish* an injective map but can never *find* one from folds; that feasible-start requirement — not a 0% score — is the injectivity cohort's reason to exist.
- **Lineage.** The graphics injectivity methods (TLC, foldover-free, progressive embedding, simplex assembly) **share this barrier-free-untangling core** — finite content through inversion — but with materially different machinery (TLC's lifted content, progressive-embedding's edge-collapses) and stronger basins/guarantees than the raw area penalty. This suite establishes the *capability axis* (untangle-from-folds) that separates the cohort from distortion-barrier minimizers; faithful per-method ports are the next step to rank *within* it.

_Caveat: 2D, one mesh, pinned-square boundary (an injective target is guaranteed to exist, so success saturates at 100% — the suite ranks by first-injective, not success); the area penalty is TLC's ancestor, not TLC. A harder boundary and faithful cohort ports are the discriminating follow-up._
