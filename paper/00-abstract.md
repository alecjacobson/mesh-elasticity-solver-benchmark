# Untangling a Decade of Mesh-Elasticity Solvers: A Component-Factored Survey and Benchmark

## Abstract

Over the past decade, computer graphics has produced a steady stream of "faster" and "more robust"
solvers for mesh-elasticity problems — parametrization and distortion optimization, projected-Newton
hyperelastic simulation, and contact-coupled dynamics. Yet nearly every such paper is a **component
swap inside one shared iteration**: minimize a nonlinear elastic energy over vertex positions by some
form of metric descent `x' = x − α M⁻¹ ∇E(x)`. Papers routinely change two or three components at
once — the energy, the Hessian filter, the search direction, the line search, the linear solver, the
convergence criterion — and credit a single one, making the literature's superiority claims difficult
to trust or reproduce.

This State-of-the-Art Report reorganizes the field around that shared structure and asks, component
by component, **which published superiority claims survive confound control**. We contribute (i) a
*unifying view* that casts graphics, classical-optimization, and machine-learning solvers as metric
descent under different choices of `M`; (ii) a six-axis *taxonomy* over three "worlds" (static
distortion, hyperelastic simulation, contact dynamics) that share machinery but not comparability;
(iii) a *lineage map* showing that many graphics "innovations" are adaptations of named classical
technique (eigenvalue filtering ⇐ modified Cholesky, accelerated quadratic proxy ⇐ Nesterov,
projective dynamics ⇐ ADMM, IPC ⇐ interior-point); (iv) a machine-readable *superiority-claims graph*
(81 methods, 160 claimed wins) recording who claims to beat whom with what evidence; and (v) a
*component-factored benchmark* — a conformance-gated harness in which a configuration is a point in
component space and each experiment changes exactly one axis.

Applied to the contact-free solver track (2D prototype), the benchmark's decomposition experiments
overturn, qualify, or contextualize several well-cited claims. Our headline case study — a recent
near-incompressibility filtering claim — *reverses* on standard constant-strain elements (the
filter's advantage becomes a failure — *a volumetric-locking artifact of the element, not a property
of the filter*) and then *re-validates* only once **two entangled confounds, the element and the
energy, are separately controlled**; four independent locking treatments concur.
We further find that a flagship quasi-Newton method's three components **entangle rather than add**,
that a proxy method's celebrated mesh-independence is a **loose-tolerance artifact**, and that the
entire clamp-versus-absolute filtering question reduces to **one analytic scalar** — the sole
sign-indefinite eigenmode of the element Hessian.

Of the 160 extracted superiority edges, only **2 are independently validated** and **59 qualified**
by our measurements; the rest remain the papers' own word pending faithful re-measurement. This is
the honest core: rather than a leaderboard, the benchmark and its **adversarial review loop** — in
which the harness's confound-untangling is applied reflexively to our *own* conclusions, forcing
repeated retractions — offer a reproducible *method for honest attribution*. We release the harness,
claims graph, and figures as the seed of a living benchmark.

**Scope.** The v1 measurements are a 2D prototype: dense solves, small meshes, indicative not
definitive. Every headline is reported with its regime of validity; the contact track and larger-scale
studies are future work. The contribution is the attribution *method* and the survey scaffolding, not
a settled ranking.
