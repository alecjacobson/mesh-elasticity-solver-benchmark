# Harness Architecture

Architecture (contracts, not code) for the common framework of **hot-swappable components**.
Per D3 (`design.md` §10): components are reimplemented **official-code-first** and
**regression-tested** against the official reference or an oracle; they will increasingly be
**agent-generated** under the same rule. This document defines the interfaces that make a swap
*fair* (one thing changes) and *admissible* (it provably matches a trusted reference).

## 1. The core idea: a config is a point in component space

A benchmark run = a **scenario** solved by a **solver config**. The solver config selects one
implementation per **slot**; a *closed-division* comparison swaps exactly one slot and holds the
rest fixed (this is the fairness gate from `taxonomy.md`, made executable).

```
solve(scenario, config) → { iterates, per-iter telemetry, final state, metrics }

config = {
  energy:          <slot impl>   # ψ(F): value, gradient (forces), Hessian (tangent stiffness)
  search_direction:<slot impl>   # full-Newton | Gauss-Newton | L-BFGS | AQP-proxy | local-global | ADMM | ...
  hessian_filter:  <slot impl>   # none | identity-shift | clamp | absolute | trust-region | analytic | ...
  line_search:     <slot impl>   # none | Armijo | Wolfe | injectivity-barrier | CCD-filtered
  linear_solver:   <slot impl>   # Cholesky | PCG | MINRES | multigrid | additive-Schwarz | ...
  criterion:       <slot impl>   # Newton-decrement | char-grad-norm | backward-Euler-residual | fixed-budget
  # tuning: a fixed hyperparameter block per config, held constant across the suite (Lever 4, metrics.md)
}
```

The six slots are exactly the taxonomy's method axes (`taxonomy.md` Face A). Not every slot is
independent for every method (e.g. local-global bundles direction+solver; trust-region bundles
filter+direction) — such couplings are declared by the impl (see §3 `couples_with`) so the
harness never presents an incoherent config.

## 2. Slot contracts

Each slot is an interface with a minimal, telemetry-instrumented contract. Sketch:

- **Energy `ψ`** — `value(x)`, `gradient(x)` (= −forces), `hessian(x)` (tangent stiffness, sparse),
  and, where available, `analytic_eigensystem(F)` (per-element, for the filter slot). Declares:
  isotropic? least-squares-form? inversion-safe? Counts: #value/#grad/#hessian evals (metrics
  #11–13).
- **Search direction** — `step(x, grad, hessian_op) → d`. May be matrix-free (consumes a
  Hessian-vector product) or matrix-based. Declares order (first/second), memory (L-BFGS window).
- **Hessian filter** — `filter(H_or_block) → SPD-ish operator`. Per-element or global. Counts:
  #eigendecompositions / #projected elements (metrics #13, PPN's axis).
- **Line search** — `alpha(x, d, energy) → step length`, with feasibility hooks
  (injectivity / CCD truncation). Counts: #backtracks, #energy evals (metrics #17, #11).
- **Linear solver** — `solve(A, b) → x`, direct or iterative. Counts: #factorizations,
  #linear-solves, #mat-vecs (metrics #16, #14, #15) — the HW-independent cost proxies.
- **Criterion** — `converged?(state) → bool` + emits the criterion value each iter (so a run can
  be re-scored under a *different* criterion offline — this powers decomposition experiment #5,
  "rankings flip with the criterion").

**Telemetry is mandatory and uniform:** every slot emits its counters into a per-iteration
record `{iter, energy, grad_norm, newton_decrement, step_len, #mat-vecs, #factorizations,
#evals, wall_clock_ns, ...}`. Metrics (`metrics.md`) are computed *from this log*, never
hand-reported — this is the "numbers are always measured" invariant.

## 3. Component descriptor (admissibility)

Every component ships a descriptor:

```yaml
id: absolute-filtering
slot: hessian_filter
official_code: "github.com/honglin-c/abs-psd"   # or "" if none exists
oracle: ""                                        # fallback trusted reference if no official code
couples_with: []                                  # slots this impl bundles (must co-vary)
conformance:                                       # THE admissibility gate (D3)
  - input: "canonical/armadillo_stretch.json"
    reference: "official"                          # official_code | oracle | analytic
    match: "newton_decrement trajectory within 1e-8 rel for 20 iters"
  - input: "canonical/incompressible_nu0.4999.json"
    reference: "official"
    match: "final energy within 1e-10 rel; #projected-elements within 2%"
provenance: "reimplemented from official; validated <commit/date>"
```

**Admissibility rule (D3):** a component — human- or agent-written — may enter a comparison
**only** if its conformance suite passes against `official_code` (preferred) or a named `oracle`.
No official code and no oracle ⇒ the component is `experimental` and quarantined to the *open
division*, never the *closed* one. Agent-generated components are ordinary components under this
rule: they must pass the same conformance suite.

## 4. Official-code-first mapping (port targets + oracles)

| slot | official-code exemplars (port targets) | independent oracle |
|---|---|---|
| energy ψ | Stable Neo-Hookean / HOBAK; TinyAD; libigl | FEniCS/deal.II (symbolic tangent) |
| search direction | PD (ShapeOp), ADMM-elastic, AASolver, BCQN, SLIM | — |
| Hessian filter | abs-psd, trust-region-newton, Analytic Eigensystems | modified-Cholesky (Gill–Murray) |
| line search | IPC line search; BCQN barrier filter | — |
| linear solver | Eigen/CHOLMOD, AMGCL, PETSc KSP | PETSc SNES/TAO |
| criterion | BCQN char-grad-norm; PPN residual | — |
| **whole-solve oracle** | — | **PETSc SNES/TAO**, FEniCS, deal.II step-44, Trilinos NOX/LOCA |

Whole-solve oracles validate a *full config* (not just one slot): the harness can assert its
Newton-LS + Cholesky config reproduces PETSc SNES on a fixed BVP before any component swap is
trusted. deal.II step-44 is the locking-free oracle for the near-incompressible ν-sweep (control
C1).

## 5. Scenario layer (keeps v2 contact a drop-in, not a fork)

```
scenario = {
  mesh, boundary_conditions, energy_params (ψ, material, ν),
  initial_state, target_tolerance, budget,
  load_schedule:  none | stepping | arc-length      # control C2 (design.md)
  contact:        none | { barrier d̂, CCD tol, friction ε_v }   # v1: none; v2: this block
  element:        P1 | P2 | mixed-u-p | F-bar        # control C1 (locking-free for ν-sweep)
}
```

Contact and load-schedule are **scenario** properties, not solver properties. This is why v2
(Track 2) adds no new axis: the same slots run; `contact` just becomes non-null and the
capability-track metrics (`metrics.md` W3: non-penetration #52, friction #54) switch on.

## 6. Divisions & reproducibility (governance)

- **Closed division** — identical skeleton, one slot swapped, all conformance suites passing.
  This is where fair *convergence numbers* live.
- **Open division** — best-effort / experimental / not-yet-conformant components; reported for
  breadth, never mixed into closed-division rankings.
- **Determinism:** fixed seeds; report run-to-run variance as error bars (metric #76); record
  hardware, threads, precision, library versions, and the full solver set with every profile
  (metric #80). Bitwise reproducibility is not required (parallel FP) — agreement-to-tolerance is.
- **Hidden/rotating tier:** a scenario subset withheld to detect overfitting (`design.md` §11 W7).

## 7. What this unblocks

The architecture makes each decomposition experiment (`design.md` §11) a *config diff*:
filter-isolation = vary `hessian_filter` only; BCQN triple-split = vary `line_search` /
`search_direction` / `criterion` independently; criterion-sensitivity = re-score one run log
under different `criterion`s. Every such diff is admissible only when the varied component passes
its conformance suite — so a measured win is attributable to *one* named component that provably
matches its official reference.
