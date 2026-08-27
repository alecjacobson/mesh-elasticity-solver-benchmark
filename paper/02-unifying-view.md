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

## 2.1 The metric table

| metric `M` | method | world |
|---|---|---|
| `I` (identity) | gradient descent; with momentum, Nesterov / heavy-ball | World-0 baseline |
| `∇²E` (energy Hessian) | Newton; projected Newton once the Hessian is filtered to SPD | all |
| Laplacian / H¹ (Sobolev) | **AQP**, **BCQN**'s proxy — a fixed graph-Laplacian preconditioner | World-1 |
| Killing operator | **AKVF** — an isometry-aware Riemannian metric | World-1 |
| reweighted energy Hessian | **SLIM** — iteratively-reweighted Gauss–Newton | World-1 |
| Fisher information | natural gradient (Amari) | ML |
| a fixed factorized proxy | projective dynamics / local–global (ADMM under a quadratic proxy) | World-2 |

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

With these boundaries stated, the metric-descent view organizes the rest of the report: the taxonomy
(§3) enumerates the axes on which `M` and the globalization are chosen; the survey (§4) catalogs the
choices; the lineage map (§5) names their classical origins; and the benchmark (§7–§8) changes one
choice at a time.
