# The twist eigenvalue is the whole clamp-vs-absolute-vs-CM story (analytic, gated)

![twist phase](../figures/twist_phase.png)

_`figures/twist_phase.png`: (left) the twist eigenvalue λ_t over the singular-value plane — blue = negative = indefinite, all of it under compression, vanishing at the isometry; (right) the clamp↔absolute gap |λ_t|, i.e. exactly where and how much the filter choice matters. Generate with `python -m bench.run_figures twist_phase`._

Built on the **validated** analytic eigensystem (`results/analytic_eig.md`; eigenpairs match a finite-difference Hessian to ~1e-10). Closed-form modes cross-checked against `analytic_eig._eigpairs` here to **0.0e+00**. Run: `python -m bench.run_twist_analysis`.

The 2D symmetric-Dirichlet element Hessian ∂²ψ/∂F² has four analytic eigenvalues in the SVD F=UΣVᵀ (g(σ)=2σ−2σ⁻³):

| mode | eigenvalue | sign |
|---|---|---|
| stretch ×2 | 2 + 6/σᵢ⁴ | **always > 0** |
| flip | (g(σ₁)−g(σ₂))/(σ₁−σ₂) | **always > 0** (g monotone ↑) |
| twist | (g(σ₁)+g(σ₂))/(σ₁+σ₂) | **can be < 0** (compression) |

Sampling 250000 points over σ∈[0.3,2.6]²: **stretch<0 in 0, flip<0 in 0, twist<0 in 94552 (37.8%)**. The twist is the ONLY sign-indefinite mode, it vanishes exactly at the isometry σ₁=σ₂=1, and it is negative only under compression (small singular values).

## Why this settles what the filters are actually doing

Every projected-Newton filter in this benchmark is **identical except on the twist mode**:

- **clamp** → replaces λ_t (when <0) with ε → drops the mode.
- **absolute** → replaces λ_t with |λ_t| → keeps the mode's magnitude, flips its sign.
- **plain Newton** → keeps λ_t<0 → indefinite step (the affine-invariant but non-descent one, `results/pitfalls.md`).
- **Composite Majorization (#14)** → builds a *convex majorizer* of exactly this twist term (a global upper bound, not a local clamp), giving full-step monotone descent.

So the entire absolute-vs-clamp verdict (`results/world2_filters.md`, `results/p2_nu.md`) lives in **one scalar per element**, active **only under compression** — which is exactly the regime a near-incompressible material enters as it necks (`results/p2_nu.md`, the ν→½ locking story). See `figures/twist_phase.png` for the σ-plane map: the λ_t<0 region and the clamp↔absolute gap |λ_t|.

## On #14 (Composite Majorization)

This pins down, gated and end-to-end, the mode CM acts on — but it is **not** a CM implementation. CM's contribution is a *specific convex majorizer* of the twist term (a global MM upper bound giving full steps), whose exact form needs the source paper (Shtengel et al. 2017). The substrate is now in place: the validated eigensystem, the isolated twist scalar, and the acceptance gates (FD conformance + the majorize-minimize property: full-step monotone decrease). The `cm→{aqp,slim,projected-newton}` edges stay `self-claimed` until that majorizer is implemented — projected-Newton **clamps** this mode, CM **majorizes** it, so they are not interchangeable.

_Caveat: 2D symmetric Dirichlet; the analytic eigenstructure generalizes to 3D (9×9, three twist/flip pairs) — the same 'twist is the indefinite mode' story, not re-derived here._
