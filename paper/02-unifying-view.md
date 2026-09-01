# 2. The Unifying View: Everything Is Metric Descent

The starting point for honest comparison is a single observation: nearly every solver in this
literature — graphics, classical optimization, and machine learning alike — is an instance of
**metric descent**,

$$\mathbf{x}' = \mathbf{x} - \alpha\, M^{-1} \nabla E(\mathbf{x}),$$

an iterate that moves along the gradient of the elastic energy `E`, preconditioned by a metric `M`
and globalized by a step length `α` (a line search) and, when `M` is only positive-semidefinite, a
regularization. The methods differ almost entirely in their *choice of, or modification to*, `M`, and
in how they enforce descent. Reading the field through this lens makes the swapped components explicit
and turns "method A versus method B" into "which axis differs, and by how much."

The literature's clashing notations are reconciled into one, used throughout (Table 2.0).

| symbol | meaning |
|---|---|
| `x` | (free) vertex positions — the variable |
| `E(x)`, `ψ` | total elastic energy; per-element density |
| `F`, `J = det F` | element deformation gradient; its Jacobian |
| `Σ, σ` | max / min singular values of `F` |
| `M` | descent *metric* / preconditioner |
| `∇E`, `∇²E` | energy gradient and Hessian |
| `α` | line-search step length |
| `ν`; `λ, μ` | Poisson ratio; Lamé parameters |
| `τ` | convergence tolerance |
| `d`, `dhat` | contact distance; barrier threshold |
| `h` | implicit-Euler time step |

*Table 2.0. Unified notation. Where a source paper uses a different symbol, we translate to this one.*

## 2.1 The metric table

| metric `M` | method | world |
|---|---|---|
| `I` (identity) | gradient descent; with momentum, Nesterov / heavy-ball [cite:nesterov-1983] | World-0 baseline |
| `∇²E` (energy Hessian) | Newton; projected Newton once the Hessian is filtered to SPD | all |
| Laplacian / H¹ (Sobolev) | **AQP** [cite:aqp], **BCQN**'s proxy [cite:bcqn] — a fixed graph-Laplacian preconditioner | World-1 |
| Killing operator | **AKVF** [cite:akvf] — an isometry-aware Riemannian metric | World-1 |
| reweighted energy Hessian | **SLIM** [cite:slim] — iteratively-reweighted Gauss–Newton | World-1 |
| Fisher information | natural gradient (Amari) [cite:amari-1998] | ML |
| a fixed factorized proxy | projective dynamics [cite:projective-dynamics] / local–global [cite:local-global] (ADMM [cite:boyd-2011-admm] under a quadratic proxy) | World-2 |

The payoff is a precise cross-field statement: **"Sobolev preconditioning" in graphics and "natural
gradient" in machine learning are the same idea under different metrics** — both replace the Euclidean
inner product on the update with one adapted to the problem's geometry. The accelerated quadratic
proxy is Nesterov acceleration applied over a Laplacian metric; SLIM is a reweighted Gauss–Newton; the
eigenvalue *filters* of World-2 (§4.3) are precisely the choice of how to turn an indefinite `∇²E`
into a usable `M`.

## 2.2 Honest boundaries of the analogy

A unifying view earns its keep only if it states where it breaks. Two caveats:

- **The Fisher/natural-gradient analogy is partial.** For mesh elasticity the unknowns are *positions*,
  not the parameters of an output distribution, so there is no likelihood and the Fisher metric is
  undefined in the strict sense. Where a "natural gradient" is invoked (e.g. in physics-informed
  learning with an energy loss), it collapses to Gauss–Newton on the residual — which is a legitimate
  member of the table, but not the information-geometric object of Amari's original.

- **Not everything is a single global step.** The template `x' = x − α M⁻¹ ∇E(x)` covers methods that
  take one *global* update per iteration. It does **not** cover **block-coordinate / Gauss–Seidel**
  sweeps — Vertex Block Descent, JGS2, PBNG — whose update solves per-vertex (or per-block) local
  problems in a coloring-dependent order. Their effective operator is triangular and sweep-dependent,
  not `M⁻¹` for any fixed `M`. These belong to a separate "relaxation / coordinate-descent" family;
  we scope the unifying claim to global-step methods and treat the relaxation family as a sibling
  branch rather than forcing it into the mold.

## 2.3 Reductions the view makes precise

The metric-descent lens does more than tabulate; it exposes *reductions* — statements that a method
thought sui generis is another method with one component swapped. Beyond the cross-field equalities of
§2.1 (Sobolev preconditioning = natural gradient; AQP = Nesterov over a Laplacian metric; the World-2
filters = the choice of how to turn an indefinite Hessian into `M`), two reductions are load-bearing
for the benchmark's simulation track (§8.7) and worth stating outright:

- **Projective Dynamics is a constant-metric quasi-Newton.** Its global solve is Newton's step with the
  true Hessian replaced by a *fixed*, prefactored SPD proxy (the rest-state Laplacian of the
  constraint set) — the Hessian evaluated once and reused, i.e. a *lagged-Hessian* Newton
  [cite:quasi-newton-liu2017]. This single reduction predicts the measured behavior: PD needs **more**
  iterations than a refactoring Newton but each is far cheaper (§8.7), exactly as a fixed-metric method
  must. Chebyshev-, Anderson-, and L-BFGS-accelerated PD are then just different *globalizations* of
  that same fixed-metric iteration.

- **XPBD is compliant implicit Euler.** Adding a per-constraint *compliance* `1/(k·h²)` to Position
  Based Dynamics makes the constraint stiffness time-step-consistent — which is precisely
  backward-Euler on the elastic potential at finite stiffness [cite:xpbd]; plain PBD is the
  infinitely-stiff, iteration-count-dependent limit (compliance → 0). This reduction is why the
  benchmark can put XPBD, PBD, PD, ADMM, and Newton on the *same* incremental potential and residual
  (§8.7), and why XPBD's constraint sweep — which realizes the compliant elastic solve but drops the
  momentum-coupling term — *stagnates* on that potential's gradient while a primal solve of the same
  potential converges.

Each reduction is a testable prediction, not just a re-labeling: §8.7 confirms both on the shared
testbed. Naming them is where the survey stops cataloguing and starts adjudicating.

With these boundaries and reductions stated, the metric-descent view organizes the rest of the report:
the taxonomy (§3) enumerates the axes on which `M` and the globalization are chosen; the survey (§4)
catalogs the choices; the lineage map (§5) names their classical origins; and the benchmark (§7–§8)
changes one choice at a time.
