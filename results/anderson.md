# Anderson acceleration of ARAP local-global (measured)

Hardens the `anderson-geometry -> local-global` edge with a *reproducible* runner. Config: ARAP energy, boundary pinned to an affine **shear** (interior initialized at rest, so the minimum is a genuine non-zero-energy deformation — not rest-recovery), same init for both methods, only the accelerator swapped. Criterion `|ARAP-grad|inf < 1e-8`. Run: `python -m bench.run_anderson`.

**Cost model (HW-independent, per docs/metrics.md Lever 1):** both prefactor the cotan-Laplacian once (1 factorization) and do one global back-solve per iteration, so `#back-solves == #iters` for both; Anderson adds a small (nfree×m) least-squares + one safeguard energy-evaluation per iteration, visible only in wall-clock. Iterations, wall-clock, and the derived back-solve count are all reported.

| mesh | free dof | method | status | iters (= back-solves) | wall (ms) | final E |
|---|---|---|---|---|---|---|
| 6×6 | 50 | local-global | converged | 23 | 88.7 | 1.2689e-01 |
| 6×6 | 50 | anderson | converged | 12 | 74.5 | 1.2689e-01 |
| 9×9 | 128 | local-global | converged | 23 | 195.1 | 1.2689e-01 |
| 9×9 | 128 | anderson | converged | 12 | 164.2 | 1.2689e-01 |
| 12×12 | 242 | local-global | converged | 24 | 359.9 | 1.2689e-01 |
| 12×12 | 242 | anderson | converged | 13 | 336.9 | 1.2689e-01 |

## Observed

- On the headline instance (n=9, 128 dof), Anderson reaches the same ARAP minimum in **12 it vs 23 it** (1.92× fewer iterations / back-solves), to the same energy. Because each iteration is one back-solve for both, the iteration ratio *is* the HW-independent work ratio; wall-clock includes Anderson's per-iteration least-squares overhead, so the wall-clock speedup is smaller than the iteration speedup.
- The iteration counts are **mesh-independent** for both methods across the sweep (the acceleration factor does not wash out as the mesh refines).

## Generality — the same core wraps a different fixed-point map (#36)

Anderson's defining property is that it accelerates an *arbitrary* fixed-point iteration, not just local-global. We factored the AA core into a map-agnostic `anderson_accelerate(G, energy, resid, x0, free, m)` (`bench/world1.py`) and applied the **identical** core to a completely different map: a damped **Jacobi** stationary iteration for a linear SPD system `A x = b` (A = the 98×98 cotan-stiffness free block) — the kind of iteration Anderson acceleration was originally invented for. `m=0` is plain Jacobi (a fair same-map baseline); `m∈{5,10}` is Anderson-accelerated.

| Anderson history m | status | iterations to `|b−Ax|∞ < 1e-8` |
|---|---|---|
| 0 (plain Jacobi) | converged | 374 |
| 5 | converged | 62 |
| 10 | converged | 37 |

- The same core cuts plain Jacobi's **374** iterations to **62** (m=5, 6.0×) and **37** (m=10, 10.1×) on a map that has *nothing* to do with ARAP — confirming the acceleration is a property of the **generic Anderson core**, not of the local-global map. This is the faithful, general Anderson (Peng et al.): m history, min-norm `lstsq`, energy-decrease safeguard, applied to whatever `G` you hand it.

_Scope: 2D, single seed; the two maps (ARAP local-global + Jacobi linear solve) exercise the generality. Wrapping the official SLIM reweighting as a third map is a natural extension._
