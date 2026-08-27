# 1. Introduction

Simulating and optimizing the deformation of triangle and tetrahedral meshes is a workhorse of
computer graphics. The same mathematical object recurs across seemingly distinct subfields: given a
rest mesh and some boundary conditions, find vertex positions `x` that minimize a nonlinear elastic
energy

$$E(\mathbf{x}) = \sum_e V_e\, \psi\!\left(F_e(\mathbf{x})\right),$$

a sum over elements of a stored-energy density `ψ` of the per-element deformation gradient `F_e`.
UV parametrization minimizes a *distortion* energy (symmetric Dirichlet, MIPS, ARAP); flesh and cloth
simulation minimize a *hyperelastic* energy (Neo-Hookean, corotational) plus inertia; and
contact-rich animation adds barrier and friction terms. Across all of them the computational core is
the same: iteratively descend a nonconvex energy whose Hessian is indefinite far from the minimum.

**The entanglement problem.** A decade of papers has proposed "faster" or "more robust" solvers for
this core. Almost universally, a new method is a *component swap* inside one shared iteration — it
changes the energy, the way the indefinite Hessian is made positive-definite (the *filter*), the
search direction, the line search, the linear solver, or the convergence criterion. The difficulty is
that papers typically change **two or three of these components at once and attribute the resulting
speed-up or robustness to a single one**. A method might blend a new preconditioner *and* a new
line-search *and* a new stopping criterion, then report an order-of-magnitude win against a baseline
that shares none of them. The reader cannot tell how much of the advantage is the headline idea, how
much is the other bundled changes, and how much is a weak or mismatched baseline. Superiority claims
accumulate that are individually plausible and collectively irreconcilable.

**Two failure poles.** A survey of this literature can fail in two opposite ways. It can be *insular
and narrow* — comparing only within one clique of graphics papers, inheriting their baselines and
their confounds, and reproducing rather than testing their claims. Or it can be *broad and unfair* —
racing methods across problem classes they were never designed for, or against implementations of
wildly different maturity (a compiled C++ library versus a research prototype), so that hardware and
engineering masquerade as algorithm. A useful account must be broad enough to place graphics work
against its classical-optimization and computational-mechanics roots, yet disciplined enough to
compare only what is comparable, one component at a time.

**This report.** We reorganize the field around its shared structure and ask, component by component,
which published superiority claims survive confound control. Our contributions are:

- **A unifying view (§2).** Graphics, classical-optimization, and machine-learning solvers are all
  *metric descent* `x' = x − α M⁻¹ ∇E(x)`, differing only in the metric `M` and its globalization.
  This makes the swapped components explicit and lets us say precisely, for example, that "Sobolev
  preconditioning" in graphics and "natural gradient" in machine learning are the same idea under
  different metrics — with the honest caveats where the analogy breaks.

- **A taxonomy over three worlds (§3).** Six method axes (energy, filter, direction, line search,
  linear solver, criterion) crossed with problem-class *capability cells*, grouped into three worlds
  — static distortion, hyperelastic simulation, contact dynamics — that share machinery but not
  comparability. Comparability is governed by the problem class, not the method.

- **A survey by axis and a lineage map (§4–§5).** We organize the annotated corpus by *component*
  rather than chronology, and we trace each graphics "innovation" to its named classical ancestor:
  eigenvalue filtering to modified-Cholesky Hessian modification, the accelerated quadratic proxy to
  Nesterov acceleration, projective dynamics to ADMM, IPC barriers to primal interior-point methods.
  Cited as adaptations rather than inventions, the lineage points each method at the classical analysis
  that explains — and often anticipates — its measured behavior (§8).

- **A superiority-claims graph (§6).** A machine-readable directed graph — 81 methods, 160 claimed
  wins — recording who claims to beat whom, on what dimension, with what evidentiary status
  (self-claimed / qualified / validated). It exposes the recurring honesty patterns: fixed-budget
  versus converged comparisons, hardware confounds, entanglement, and the authors' own regime
  disclaimers.

- **A component-factored benchmark (§7–§8).** A conformance-gated harness in which a configuration is
  a *point in component space* and each decomposition experiment changes exactly one axis, holding the
  rest fixed. Applied to the contact-free solver track, it overturns, qualifies, or contextualizes
  several well-cited claims (§8).

**The honest core.** Of the 160 extracted superiority edges, only two are independently validated and
23 qualified by our measurements; the remainder stay the papers' own word pending faithful
re-measurement. We regard this ledger, and the **adversarial review loop** that produced it — in
which the benchmark's confound-untangling was turned reflexively on our *own* draft conclusions,
forcing repeated retractions of our own overreach — as the report's real deliverable: not a
leaderboard, but a reproducible *method for honest attribution*.

**Scope.** The v1 benchmark measurements are a 2D prototype (dense solves, small meshes, few seeds);
they are *indicative, not definitive*, and every headline below is reported with its regime of
validity. The contact "world," larger-scale studies, and faithful ports of a handful of methods that
require their source papers are explicitly deferred. What is offered now is the attribution method,
the taxonomy/lineage/claims scaffolding, and a released harness that seeds a living benchmark.
