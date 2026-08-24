# Pitfalls of Projection — affine invariance + asymptotic rate (measured)

Tests the Pitfalls-of-Projection thesis on its **actual** claims, which an iteration-count-to-tolerance comparison cannot reach (review-r1 #39): eigenvalue projection (a) breaks **affine invariance** of the Newton step and (b) can degrade the **asymptotic rate**. Run: `python -m bench.run_pitfalls`.

## Part 1 — affine invariance (definitive)

At an indefinite-Hessian point (3/50 negative eigenvalues), we rescale coordinates by a non-uniform per-DOF diagonal `S` (scales spanning 0.1–10, i.e. a change of *units* per coordinate) and measure the covariance residual `||S·d_y − d_x|| / ||d_x||` — zero iff the step is affine-covariant.

| filter | covariance residual | affine-invariant? |
|---|---|---|
| none | 3.10e-13 | **yes** |
| clamp | 6.08e+01 | no |
| absolute | 2.12e-01 | no |
| global-pdn | 6.08e+01 | no |

- **Unfiltered Newton is affine-invariant** (residual ~1e-13): `−H⁻¹g` transforms as `d = S·d_y` exactly, independent of the coordinate units.
- **Every eigenvalue projection that actually acts breaks it** — clamp, absolute, *and* the faithful assembled global-PDN all give an O(1) covariance residual, because clamping the eigenvalues of `SᵀHS` is not the congruence of the clamped `H` (`P(SᵀHS) ≠ SᵀP(H)S`). This is the Pitfalls thesis, shown directly: the projected step *depends on the coordinate system*, so a filtered Newton solver is not invariant to a reparametrization/units change that plain Newton is blind to.
- **Nuance on the paper's 'PDN recovers affine invariance':** it does — but only in the **SPD regime**, where project-on-demand/PDN is *inert* (it leaves the Hessian raw = plain Newton = invariant). Our probe is at an **indefinite** point, where PDN *must* project the negative eigenvalues, and there it loses invariance just like clamp (hence global-PDN's 60.8 here). So the exact statement is: **affine invariance is preserved iff no eigenvalue is actually projected**; PDN's advantage is that it projects *less often* (only when indefinite), not that a projection ever becomes affine-invariant.

## Part 2 — asymptotic rate to 1e-11

Tail residual ratios `r_{k+1}/r_k` near the solution (→0 = super-linear/quadratic; →const = linear):

| filter | status | iters | tail ratios |
|---|---|---|---|
| none | nondescent | 2 | 4.2e-01 |
| clamp | converged | 8 | 1.1e-01, 1.4e-02, 1.9e-04, 4.6e-08 |
| global-pdn | converged | 8 | 1.1e-01, 1.4e-02, 2.0e-04, 5.0e-08 |

- **Clamp and global-PDN converge super-linearly in the tail** (ratios shrink toward 0: …1.9e-4, 4.6e-8), i.e. no rate degradation is visible on *this* trajectory — an honest null. Unfiltered Newton (`none`) is **non-descent** here (the raw Hessian keeps a few negative eigenvalues even near the solution), so it provides no tail baseline — which is itself why some projection is needed for global convergence.
- The rate-degradation the paper warns about needs the projection to remain **active in the tail** (a solution at a near-degenerate/indefinite point, or projection-forced detours), not a clean SPD minimum. The **mechanism**, though, is already established in Part 1: a non-affine-invariant step's convergence depends on conditioning/coordinates, unlike Newton's — which is *why* projection can slow the asymptotic rate. Part 1 is the coordinate-free evidence; Part 2 is the honest note that a clean SPD minimum does not trigger it.

_Caveat: 2D, single scenario/seed, dense. Part 1 is the definitive, coordinate-free result; it holds for global-PDN too, so it is a statement about **projection itself**, not about per-element vs assembled variants._
