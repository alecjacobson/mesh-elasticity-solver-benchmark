# Decomposition Experiments

The scientific payoff: the "untangling" runs the benchmark exists to produce. Each is a
**config diff** in the harness (`harness.md`) — vary one (or a controlled few) component slot(s),
hold the rest fixed — measured with the per-cell orthogonal core (`metrics.md`) on the frozen
strata (`protocol.md`). Each experiment names the **claims-graph edges** (`claims/claims.yaml`)
it can harden from `self-claimed` → `validated` / `qualified` / `refuted` (issue #5).

Format per experiment: **Hypothesis · Config diff · Strata · Metrics · Edges hardened ·
Expected outcome / failure mode it would expose.**

---

## E1 — Filter isolation (the v1 headline)

- **Hypothesis:** with energy / direction / line-search / solver / criterion fixed, the
  eigenvalue-filter choice changes convergence; *absolute* beats *clamp* mainly in the
  near-incompressible regime; *trust-region* dominates by adapting.
- **Config diff:** `hessian_filter ∈ {none(full-Newton), identity-shift, clamp, absolute,
  trust-region, project-on-demand, progressive, analytic-eigensystem, spectral-shift, blending,
  modified-Cholesky}`; everything else fixed (Stable Neo-Hookean, projected-Newton skeleton,
  one line search, direct Cholesky, Newton-decrement τ).
- **Strata:** 1b, full resolution × element-order × stiffness grid **and the ν sweep** — with
  **control C1** (locking-free element; also run P1 to expose the confound).
- **Metrics:** iterations-to-τ + #linear-solves (HW-indep) paired with wall-clock; #projected-
  elements (PPN's axis); success-rate; per-stratum data profiles. τ ∈ {1e-3,1e-6}.
- **Edges hardened:** `absolute-filtering→clamp-filtering` (convergence, robustness);
  `trust-region-filtering→{clamp,absolute,full-newton}`; `eigenvalue-blending→{clamp,absolute}`;
  `progressively-projected-newton→clamp-filtering`; `pitfalls-projection→clamp-filtering`;
  `analytic-eigensystems→numeric-eigendecomposition`.
- **Would expose:** whether absolute's edge is *only* at high ν (and whether it survives a
  locking-free element — if it vanishes on locking-free but persists on P1, the "win" was
  fixing a bad element, not the solver). Whether trust-region uniformly dominates or ties
  blending. Whether classical modified-Cholesky matches the graphics filters (the "rebranding?"
  question).

## E2 — Seed-claim decomposition

- **Hypothesis:** each seed paper's headline delta shrinks once the confounds it *also* changed
  (line search, criterion, energy) are pinned; the residual is the true contribution.
- **Config diff:** for each seed method, run (a) *as-published* (all its changes on) vs (b)
  *isolated* (only its named axis swapped into the common skeleton, everything else at harness
  default). The gap (a)−(b) = the confound-borne portion of the claim.
- **Strata:** the cell matching each seed (1a for AQP/SLIM/BCQN/CompMajor; 1b for the filters).
- **Metrics:** the delta on the paper's own headline metric, plus the orthogonal core; report
  attributed vs confound-borne fractions.
- **Edges hardened:** the outgoing edges of each seed node (e.g. `slim→aqp`, `composite-
  majorization→{aqp,slim}`, and the filter edges via E1).
- **Would expose:** claims whose advantage is mostly a co-changed criterion or line search
  rather than the advertised mechanism.

## E3 — BCQN triple-split

- **Hypothesis:** BCQN's "fastest + most robust" is the *product* of three independent changes;
  no single one explains it.
- **Config diff:** a 2³ factorial over BCQN's three components as harness slots —
  `line_search ∈ {backtracking, barrier-aware-filter}` × `search_direction ∈ {L-BFGS,
  blended-Sobolev+L-BFGS}` × `criterion ∈ {gradient-norm, characteristic-gradient-norm}` — on
  the fixed 1a energy.
- **Strata:** 1a typical + adversarial (folded init).
- **Metrics:** wall-clock-to-τ, success-rate, iterations; main-effects + interactions of the
  three factors.
- **Edges hardened:** `bcqn→{aqp,slim,composite-majorization,l-bfgs}` (attribute each to a
  factor); confirms/【qualifies】the `bcqn→gradient-descent` edge (line-search-filter alone
  claimed >10×).
- **Would expose:** whether the barrier-aware line-search filter (BCQN's cheapest component)
  carries most of the win — the paper's own Fig.6 suggests it does; the factorial settles it.

## E4 — First- vs second-order honesty (the wall-clock inversion)

- **Hypothesis:** many "N× fewer iterations" claims invert under wall-clock because per-iteration
  cost differs by orders; and many "N× faster than Newton" claims are fixed-budget/per-iteration,
  not converged.
- **Config diff:** hold energy + scenario; compare first-order/proxy directions (AQP, SLIM,
  L-BFGS, gradient descent, Adam) vs second-order (projected Newton variants) across a
  **mesh-size sweep**; plot both iterations-to-τ and wall-clock-to-τ, and both *converged* and
  *fixed-budget* readings.
- **Strata:** 1a and 1b, resolution sweep (mesh-independence, metric #69/#70).
- **Metrics:** the paired HW-independent/HW-dependent cost; crossover mesh size where the
  ranking flips.
- **Edges hardened:** re-reads the fixed-budget-flagged sim edges (`jgs2→full-newton`,
  `quasi-newton-liu2017→full-newton`, `vertex-block-descent→full-newton`, `projective-dynamics→
  full-newton`) — converts "per-iteration/FPS" claims into converged statements or marks them
  `qualified`.
- **Would expose:** where Adam/first-order plateaus (the honesty control) and where iteration-
  count wins are erased by per-iteration cost.

## E5 — Criterion sensitivity

- **Hypothesis:** solver rankings flip with the convergence criterion — the silent confound
  behind most published speed claims.
- **Config diff:** *no re-run needed* — re-score the E1–E4 per-iteration logs offline under
  `criterion ∈ {Newton-decrement, characteristic-gradient-norm, backward-Euler-residual,
  fixed-budget}` (the harness emits every criterion's value per iter, `harness.md` §2).
- **Strata:** all cells (it's a re-scoring).
- **Metrics:** rank-correlation of solver orderings across criteria; count of ordering
  inversions.
- **Edges hardened:** meta — annotates every speed/convergence edge with its criterion-
  sensitivity, flagging which published claims are criterion-dependent.
- **Would expose:** claims that hold only under the criterion their authors happened to choose.

---

## Cross-experiment protocol

- Every varied component must **pass its conformance suite** (harness.md §3) before its results
  enter the closed division — a measured win is attributable to one *validated* component.
- Report all five under the frozen `protocol.md` (equal budget, no per-problem tuning, controls
  C1/C2, data-profile aggregation, hidden tier).
- **Hardening bookkeeping (#5):** each experiment writes its edge-status updates back to
  `claims/claims.yaml` (`status`, `assessed_by: benchmark`, `notes` = regime) so the claims
  graph becomes the living record of what survived.
