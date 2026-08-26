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
Worlds 1–2 (the contact-free solver track); World 3 is surveyed but deferred to v2.

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
