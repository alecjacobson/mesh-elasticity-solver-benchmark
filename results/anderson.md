# Anderson acceleration of ARAP local-global (measured)

Hardens the `anderson-geometry -> local-global` edge with a *reproducible* runner. Config: ARAP energy, boundary pinned to an affine **shear** (interior initialized at rest, so the minimum is a genuine non-zero-energy deformation — not rest-recovery), same init for both methods, only the accelerator swapped. Criterion `|ARAP-grad|inf < 1e-8`. Run: `python -m bench.run_anderson`.

**Cost model (HW-independent, per docs/metrics.md Lever 1):** both prefactor the cotan-Laplacian once (1 factorization) and do one global back-solve per iteration, so `#back-solves == #iters` for both; Anderson adds a small (nfree×m) least-squares + one safeguard energy-evaluation per iteration, visible only in wall-clock. Iterations, wall-clock, and the derived back-solve count are all reported.

| mesh | free dof | method | status | iters (= back-solves) | wall (ms) | final E |
|---|---|---|---|---|---|---|
| 6×6 | 50 | local-global | converged | 23 | 896.7 | 1.2689e-01 |
| 6×6 | 50 | anderson | converged | 12 | 1008.1 | 1.2689e-01 |
| 9×9 | 128 | local-global | converged | 23 | 1802.9 | 1.2689e-01 |
| 9×9 | 128 | anderson | converged | 12 | 1506.5 | 1.2689e-01 |
| 12×12 | 242 | local-global | converged | 24 | 2917.2 | 1.2689e-01 |
| 12×12 | 242 | anderson | converged | 13 | 2405.5 | 1.2689e-01 |

## Observed

- On the headline instance (n=9, 128 dof), Anderson reaches the same ARAP minimum in **12 it vs 23 it** (1.92× fewer iterations / back-solves), to the same energy. Because each iteration is one back-solve for both, the iteration ratio *is* the HW-independent work ratio; wall-clock includes Anderson's per-iteration least-squares overhead, so the wall-clock speedup is smaller than the iteration speedup.
- The iteration counts are **mesh-independent** for both methods across the sweep (the acceleration factor does not wash out as the mesh refines).

_Scope: 2D, ARAP energy, single shear/seed, dense-ish prototype; Anderson's generality claim (it wraps *any* fixed-point map — SLIM/PD/physics) is only exercised here on the local-global map. Applying the same AA core to a second map to harden a second edge is tracked in #36._
