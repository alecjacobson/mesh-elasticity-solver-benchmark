# 4. Survey by Axis

We organize the corpus by *component* rather than chronology. Each subsection covers the graphics
methods on one axis together with their World-0 (classical-optimization) baselines — the honest
reference against which a graphics "innovation" should be measured. The full annotated corpus (~180
entries across Worlds 0–3) is in the supplementary `corpus.md`; here we give the load-bearing methods.

## 4.1 Energy

The energy fixes the geometry of the problem and, crucially, its behavior *through inversion*.
Distortion energies (symmetric Dirichlet `‖F‖² + ‖F⁻¹‖²`, MIPS) are **barriers**: infinite at a
degenerate or inverted element, so a solver cannot cross a fold. Classical hyperelastic Neo-Hookean
(`μ/2(‖F‖²−d) − μ log J + λ/2 log²J`) is likewise a log-barrier at `J ≤ 0`. **Stable Neo-Hookean**
(Smith–de Goes–Kim) [cite:stable-neo-hookean] removes the log barrier so the potential is finite for all `J`, enabling recovery
from inverted initializations — a capability, not a speed, distinction (§8.4). The choice of energy is
frequently entangled with the solver comparison; our headline case study (§8.1) turns on separating
the two.

## 4.2 Search direction

This is the axis with the most graphics activity and the clearest metric-descent reading (§2):

- **Newton / projected Newton** — the second-order baseline; `M = ∇²E` filtered to SPD.
- **L-BFGS** — the quasi-Newton World-0 baseline; a well-implemented L-BFGS is the fair reference
  that several graphics accelerators are, in fact, measured against too weakly.
- **Accelerated Quadratic Proxy (AQP)** [cite:aqp] — Nesterov acceleration over a fixed Laplacian
  metric; claims mesh-independent iteration counts and large speed-ups.
- **SLIM** [cite:slim] — iteratively-reweighted Gauss–Newton; a second-order-like descent that reaches
  the symmetric-Dirichlet minimum in very few iterations.
- **BCQN** [cite:bcqn] — a *blend* of a Sobolev/L-BFGS proxy with a barrier-aware line search and a
  characteristic-gradient criterion (the archetypal entangled method, §8.3).
- **Composite Majorization** [cite:composite-majorization] — a convex-majorizer Hessian that is SPD by
  construction rather than by eigenvalue clamping.
- **Anderson acceleration** [cite:anderson-geometry] — a multisecant quasi-Newton wrapper applied to a
  fixed-point iteration (e.g. ARAP local–global [cite:local-global]).

## 4.3 Hessian filter (World-2)

Far from the minimum, `∇²E` is indefinite and Newton's step is not a descent direction. The *filter*
axis decides how to fix this per element. Given the analytic eigensystem of an isotropic energy
(Smith–de Goes–Kim [cite:analytic-eigensystems]), the element Hessian has closed-form eigenpairs —
two *stretching* modes, a *flip* mode, and a *twist* mode. **Clamping** [cite:clamp-filtering]
replaces each eigenvalue `λ` with `max(λ,ε)`; **absolute** filtering [cite:absolute-filtering] uses
`max(|λ|,ε)`; **trust-region** blends [cite:trust-region-filtering]. As we show in §8.5, only the twist
mode is ever sign-indefinite, so the filters differ *only there* — the entire World-2 filter debate is
one scalar per element.

## 4.4 Line search / feasibility

Classical backtracking Armijo is the World-0 baseline. The graphics addition is the
**injectivity-barrier-aware** line search: cap the step at the largest length that keeps every element
inversion-free (a per-element signed-area root, the CCD-for-inversion of SLIM/BCQN/IPC). For the
*feasibility* sub-problem — recovering an injective map from a folded one — this axis is decisive: a
barrier energy needs a feasible start, while barrier-free untangling energies (the classical one-sided
area penalty and its descendants TLC, foldover-free, progressive embedding) can cross folds (§8.4).

## 4.5 Linear solver

Dense direct, sparse Cholesky, and preconditioned conjugate gradient, with an inner tolerance. This
axis is where **hardware masquerades as algorithm**: an unpreconditioned CG's matrix-vector count
grows with mesh conditioning, and wall-clock ranks two solvers oppositely across scenarios while the
hardware-independent count stays consistent — so the count, not the clock, must carry the verdict.

## 4.6 Convergence criterion

Gradient ∞-norm, mesh-invariant characteristic gradient norm, Newton decrement, energy-relative, and
backward-error residuals. This is the *silent* confound behind many published speed claims: the
"fastest" method can change with the criterion, because a criterion sets *when* you stop, not the
descent trajectory (§8.6). BCQN's characteristic-gradient criterion is one instance; the benchmark
treats the criterion as a first-class, swappable axis and re-times every result under more than one.
