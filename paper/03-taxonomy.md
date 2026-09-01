# 3. Taxonomy: Six Axes over Three Worlds

If methods are metric-descent iterations that differ by component, the natural organizing structure is
the **product of component axes with problem-class capability cells**. We use six method axes and
three problem "worlds."

## 3.1 The six method axes

Every configuration in the benchmark is a choice on each of:

1. **Energy** `ψ` — symmetric Dirichlet / MIPS (distortion); Neo-Hookean, corotational, Stable
   Neo-Hookean (hyperelastic); barrier + friction (contact). The energy fixes what "distortion" means
   and whether the potential is finite through element inversion.
2. **Hessian filter** — how the indefinite `∇²E` is made SPD: none (raw Newton), eigenvalue *clamp*
   (`max(λ,ε)`), *absolute* (`max(|λ|,ε)`), trust-region-style blends, or none-with-a-globalization.
3. **Search direction** — Newton, quasi-Newton (L-BFGS), Sobolev-preconditioned (AQP), reweighted
   Gauss–Newton (SLIM), local–global, Anderson-accelerated, or first-order (GD, Adam).
4. **Line search / feasibility** — backtracking Armijo, exact, or **injectivity-barrier-aware /
   CCD-filtered** (the step is capped at the largest inversion-free length).
5. **Linear solver** — dense or sparse direct factorization, conjugate gradient, or a
   preconditioned Krylov method, plus the inner tolerance.
6. **Convergence criterion** — gradient ∞-norm, a mesh-invariant *characteristic* gradient norm,
   Newton decrement, energy-relative, or a backward-error residual.

The central methodological point of the report is that a paper's headline typically moves *several* of
these axes at once, and the benchmark's job is to move exactly one.

## 3.2 The three worlds

Comparability is governed by the *problem class*, not the method. We group problems into three worlds
that share the projected-Newton machinery but not the metrics that make a fair race:

- **World 1 — static distortion / parametrization.** No inertia, no contact. Symmetric-Dirichlet UV
  maps, ARAP, MIPS. Methods: AQP, SLIM, BCQN, Composite Majorization, TLC, GOSS, Anderson-PD.
- **World 2 — quasistatic / dynamic hyperelasticity.** Inertia, no contact. The eigenvalue-filtering
  cohort (clamp, absolute, trust-region), projected Newton, L-BFGS, ADMM / projective dynamics,
  Vertex Block Descent.
- **World 3 — contact-coupled dynamics.** Barriers, continuous collision detection, friction. IPC,
  ABD, GIPC, OGC.

Worlds 1 and 2 share the entire projected-Newton skeleton, so a method from one can often be *run* in
the other — but the comparison is only fair within a world, because the energies, conditioning, and
success criteria differ. World 3 additionally introduces **four parameters that belong to no solver**
— barrier stiffness `d̂`, CCD tolerance, friction regularizer `ε_v`, and time step `Δt` — which
confound any cross-method comparison unless held fixed by protocol. This report's benchmark measures
Worlds 1–2 (the contact-free solver track); World 3 is surveyed, and a minimal faithful IPC now opens
it far enough to settle IPC's intersection-free guarantee (§8.8), with its mesh–mesh/GPU-scale edges
deferred to v2.

## 3.3 Orthogonality and the fairness gate

The axes are *nearly* orthogonal but interact, and the benchmark must respect that. A strong Hessian
filter, for example, can make the line-search axis inert (§8); the barrier-aware line search interacts
with the search direction (§8). We therefore do not claim the axes are independent; we claim they are
*separately controllable*, and we report interactions where the factorial exposes them rather than
averaging them away.

The **fairness gate** is a conformance suite (§7): a configuration is admissible into the benchmark
only if its element derivatives pass a finite-difference check, its filter reproduces the reference
projection, and — for ported official code — it reproduces the source implementation's energy on a
shared instance. This prevents a mis-implemented component from masquerading as an algorithmic
difference.

## 3.4 The method matrix

The taxonomy's payoff is that it makes otherwise-incommensurable methods line up as points in one
design space. Table 3.1 places the load-bearing solvers on the axes that actually govern their cost
and behavior — the *local model* they descend on, whether the *system matrix is reused*, the
*convergence guarantee*, and *inversion handling* — and, in the last column, this report's
benchmark verdict. Every entry is a checkable fact, not an adjective; the verdict column is where the
benchmark earns its keep, replacing each method's self-reported headline with a measured, regime-scoped
status (§8–§9).

| method (world) | local model | matrix | convergence | inversion | benchmark verdict |
|---|---|---|---|---|---|
| gradient descent · W0 | identity | none | linear | energy-set | reference |
| Nesterov / AGD · W0 | identity + momentum | none | accel. 1st-order | energy-set | reference |
| L-BFGS · W0 | quasi-Newton (m-secant) | matrix-free | superlinear | energy-set | fair 1st-class baseline |
| projected Newton · W1–2 | filtered Hessian | refac/iter | quadratic (post-filter) | barrier | filter *necessary*; choice = one twist scalar (§8.5) |
| AQP · W1 | fixed Laplacian proxy + Nesterov | prefac once | linear tail | needs feasible start | qualified — mesh-indep. is loose-τ only (§8.2) |
| SLIM · W1 | reweighted Gauss–Newton | refac/iter | few-iter (superlin-like) | needs feasible start | **validated** — 6 vs 19 iters vs AQP (§8.2) |
| BCQN · W1 | Sobolev-L-BFGS + blend/cure | prefac once + history | superlinear-like | inversion-aware LS | qualified — bundle entangles; 1 factor moves it (§8.3) |
| Composite Majorization · W1 | convex majorizer Hessian | refac/iter (analytic) | monotone MM | needs feasible start | qualified — 9 vs ~780 AQP; ties p-Newton (§8.5) |
| Anderson-accel. · W1 | multisecant on fixed point | LS over history | accel. → superlin | inherits base map | **validated** — 1.85× local–global, multi-seed×mesh (§8.6) |
| local–global / PD · W1–2 | fixed quadratic proxy | prefac once | linear | constraint-set | reference / proxy |
| TLC, foldover-free · W1 | Newton on lifted energy | refac/iter | quad on smooth energy | **finite through folds** | qualified — untangles 100%; a *capability* axis (§8.4) |
| Projective Dynamics · W2 | fixed quadratic majorizer | prefac once | linear | constraint-set | qualified — ~5× *more* iters than Newton; edge is factor-reuse (§8.7) |
| quasi-Newton (Liu ’17) · W2 | L-BFGS + PD-proxy init | prefac once + history | superlinear-like | constraint-set | qualified — 6 vs 78 vs scaled-id init (§8.7) |
| Chebyshev-PD · W2 | Chebyshev on PD fixed point | prefac once | accel. linear (needs ρ) | constraint-set | qualified — shaves proxy tail; ties Anderson (§8.7) |
| ADMM-PD · W2 | operator splitting | prefac once | linear (ADMM) | constraint-set | qualified — reduces to PD; no iter win (§8.7) |
| AA-ADMM · W2 | Anderson on ADMM map | prefac once + LS | accel. linear | constraint-set | qualified — 11 → 6 iters (§8.7) |
| XPBD · W2 | compliant-constraint G–S | matrix-free | none (stagnates) | compliance | qualified — 0.7–65% off Newton by regime (§8.7) |
| PBD · W2 | constraint Gauss–Seidel | matrix-free | none (iter-stiffening) | — | reference — over-stiffens ~23× (§8.7) |
| Vertex Block Descent · W2 | per-vertex block Newton | matrix-free | G–S convergent | local | qualified — G–S beats relaxed Jacobi (§8.7) |
| Stable Neo-Hookean · W2 | *energy* (finite at J≤0) | — | — | **finite through inversion** | qualified — inversion recovery, partly definitional (§9.1) |
| IPC · W3 | Newton + log-barrier | refac/iter | quadratic (smoothed) | **intersection-free (CCD)** | qualified — guarantee holds vs a penalty that tunnels (§8.8); speed edges unmeasured |

*Table 3.1. The method matrix. "matrix" = global-system reuse (refactor each iteration vs. prefactor
once and reuse vs. matrix-free); "inversion" = behavior at a degenerate/inverted element (barrier =
`+∞`, needs a feasible start; finite = passes through folds; constraint-set = position-based, no
elastic barrier). W0–W3 are the three worlds of §3.2 plus classical baselines. The verdict column is
this report's measured status (§8–§9); "reference" marks a World-0 baseline or proxy that carries no
first-party superiority edge. Contested cells — where a self-reported headline meets a benchmark
qualification — are the report's contribution.*
