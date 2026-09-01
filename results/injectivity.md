# Injectivity / feasibility suite — untangling a folded init (measured)

![injectivity](../figures/injectivity.png)

_`figures/injectivity.png`: a folded init (red = inverted) untangled to all-valid by both barrier-free energies; barrier symmetric-Dirichlet is +∞ here and cannot start. Generate: `python -m bench.run_figures injectivity`._

Which energies recover an **inversion-free** map from a **folded** (non-injective) start. Boundary pinned to the rest square (so the identity is a guaranteed injective solution); interior reflected to create folds of increasing severity. 4 seeds × 4 severities on a 12×12 grid. **Three barrier-free** energies compete (the injectivity cohort of §8.4): `untangle` = classical one-sided area penalty (TLC's barrier-free *ancestor*, `bench/untangle.py`); `stable-NH` = Stable Neo-Hookean (finite at J≤0); and `TLC` = Total Lifted Content (`bench/tlc.py`, reimplemented from paper + code, gated) — the three all conformance-gated. The shared, energy-independent metric is **iters-to-first-injective** (first iterate with all signed areas > 0); each method's *final* iters-to-tol are on different energies/criteria and are **not** comparable. Run: `python -m bench.run_injectivity`.

| severity | folds | untangle: succ · first-inj | stable-NH: succ · first-inj | TLC: succ · first-inj |
|---|---|---|---|---|
| mild | 107 | 100% · 18 | 100% · 2 | 100% · 17 |
| moderate | 108 | 100% · 27 | 100% · 3 | 100% · 29 |
| severe | 109 | 100% · 40 | 100% · 3 | 100% · 112 |
| extreme | 109 | 100% · 36 | 100% · 3 | 100% · 84 |

### Hard boundary — a **non-convex** target (wavy warp A=0.5), 4 seeds

This pins the boundary to φ(rest), φ(x,y)=(x+A·sin πy, y). φ(grid) is a guaranteed injective solution for a **discrete** reason: on the row-aligned grid every triangle has two vertices in the same y-row and φ's x-shift depends only on y, so those two shift identically → each triangle's area is preserved EXACTLY for any A (not merely the continuum unit-Jacobian, which would not guarantee a triangulation stays unfolded). **Consequence: A tunes boundary non-convexity but NOT the target's feasibility** — so this case cannot separate the methods on *success* (an injective target always exists); it can only probe *speed of first crossing*. stable-NH's elastic minimizer need not equal φ(grid) nor be injective.

| method | success | first-inj (median) | first-inj unit |
|---|---|---|---|
| untangle | 100% | 610 | scipy L-BFGS-B outer iters |
| stable-NH | 100% | 6 | projected-Newton iters |
| TLC | 0% | None | scipy L-BFGS-B outer iters |

## Observed

- **All three barrier-free energies untangle the easy target; the axis is capability, not speed.** untangle, Stable NH, and TLC each reach an injective map **100%** across every severity (mild→extreme, up to ~109 folds) with the boundary pinned to the rest square (where the identity is the injective minimizer and all three energies are finite through inversion). On the shared **iters-to-first-injective** metric they rank Stable NH (3) < untangle (40) ≈ TLC (112) at severe — Stable NH's elastic basin is fastest; the classical area penalty and faithful TLC are comparable.
- **The injectivity-cohort ranking, with faithful TLC (§8.4).** TLC (`bench/tlc.py`, reimplemented from paper + reference code) is a *reliable but slow* untangler here: on the HARD non-convex boundary it reaches injectivity **0%** within budget, while its classical ancestor the one-sided area penalty reaches it **100%** (median 610 iters) and Stable NH **100%** (6 Newton iters). This is TLC's own documented trade-off, not an implementation flaw: the paper's default lifting `α=1e-6` buys high untangling reliability with *very flat gradients* that slow first-order convergence, so on a hard non-convex target TLC does not cross within a budget where the penalty and elastic energies do. The honest cohort verdict: faithful TLC matches the classical area penalty on easy folds but its small-α gradient makes it slower/stuck on a hard non-convex boundary here — exactly the α reliability-vs-speed tension the TLC paper describes.
- **The hard non-convex boundary does not separate the methods on SUCCESS** (both 100% — by construction, since an injective target provably exists for any A). It *does* show the raw area-penalty needs far more first-order steps to first crossing (610 L-BFGS-B outer iters) than Stable NH does Newton iters (6), vs only 40 vs 3 on the easy square. **But these are iteration counts of DIFFERENT algorithms (scipy L-BFGS-B vs projected Newton) and are NOT work-comparable** — the same non-comparability the 1a suite flags; a Newton iter is a factorization. So this is suggestive (the raw penalty's first-order basin is shallow on a non-convex boundary) but NOT a clean ratio; a work-comparable ranking needs wall-clock / eval-counts, and a genuine *capability* (success) discrimination needs a boundary whose elastic minimizer is provably folded — which this unit-Jacobian warp cannot produce (deferred).
- **Barrier symmetric Dirichlet is a definitional non-starter, stated as such (not scored).** At a folded init SD is +∞ by construction (`finite=False`), so we do **not** run it — reporting a per-severity '0%' would be measuring the initialization, not a solver. The honest control is the **asymmetry**: given a *feasible* start SD is fine — from the rest square it needs **0 iters** (the identity already IS the minimizer) and from a feasible distorted start it converges normally (`e1`). SD can *polish* an injective map but can never *find* one from folds; that feasible-start requirement — not a 0% score — is the injectivity cohort's reason to exist.
- **Lineage.** The graphics injectivity methods (TLC, foldover-free, progressive embedding, simplex assembly) **share this barrier-free-untangling core** — finite content through inversion — but with materially different machinery (TLC's lifted content, progressive-embedding's edge-collapses) and stronger basins/guarantees than the raw area penalty. This suite establishes the *capability axis* (untangle-from-folds) that separates the cohort from distortion-barrier minimizers; faithful per-method ports are the next step to rank *within* it.

_Caveat: 2D, one mesh, pinned-square boundary (an injective target is guaranteed to exist, so success saturates at 100% — the suite ranks by first-injective, not success); the area penalty is TLC's ancestor, not TLC. A harder boundary and faithful cohort ports are the discriminating follow-up._
