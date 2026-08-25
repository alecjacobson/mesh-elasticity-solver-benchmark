# Anderson acceleration of ARAP local-global (measured, multi-seed)

Hardens the `anderson-geometry -> local-global` edge. Config: ARAP energy, boundary pinned to an affine **shear** (interior from rest + a small seeded perturbation, so the minimum is a genuine non-zero-energy deformation — not rest-recovery), same init for both methods, only the accelerator swapped. Criterion `|ARAP-grad|inf < 1e-8`. **Multi-seed** (3 seeds) with min–max spread (review-r2 #47). Run: `python -m bench.run_anderson`.

**Cost model (HW-independent, per docs/metrics.md Lever 1):** both prefactor the cotan-Laplacian once (1 factorization) and do one global back-solve per iteration, so `#back-solves == #iters` for both; Anderson adds a small (nfree×m) least-squares + one safeguard energy-evaluation per iteration, visible only in wall-clock.

| mesh | free dof | local-global iters | anderson iters | iter speedup | LG wall (ms) | AA wall (ms) |
|---|---|---|---|---|---|---|
| 6×6 | 50 | 23.3 [23–24] | 12.0 [12–12] | 1.94× [1.92–2.00] | 92 | 77 |
| 9×9 | 128 | 24.0 [24–24] | 13.0 [13–13] | 1.85× [1.85–1.85] | 208 | 181 |
| 12×12 | 242 | 24.0 [24–24] | 12.7 [12–13] | 1.90× [1.85–2.00] | 365 | 310 |

## Observed

- On the headline mesh (n=9, 128 dof), Anderson reaches the same ARAP minimum in **13.0 it [13–13] vs 24.0 it [24–24]** over 3 seeds — a **1.85× [1.85–1.85]** iteration speedup. Each iteration is one back-solve for both, so the iteration ratio is the HW-independent work ratio; the wall-clock speedup is smaller (Anderson's per-iter lstsq).
- **The speedup holds across all seeds and meshes** (see the min–max spread — it never collapses to 1×), so the acceleration is not a single-seed artifact and does not wash out as the mesh refines. This upgrades the earlier single-seed result (review-r2 #47). The larger seeded interior perturbation gives modest but **non-degenerate** seed variance (ratios span 1.85–2.00, not a single value); the shear target dominates the problem, so the ~1.9× speedup is robust rather than noise (review-r3 #R6).

## Generality — the same core wraps a different fixed-point map (#36)

Anderson's defining property is that it accelerates an *arbitrary* fixed-point iteration, not just local-global. We factored the AA core into a map-agnostic `anderson_accelerate(G, energy, resid, x0, free, m)` (`bench/world1.py`) and applied the **identical** core to a completely different map: a damped **Jacobi** stationary iteration for a linear SPD system `A x = b` (A = the 98×98 cotan-stiffness free block) — the kind of iteration Anderson acceleration was originally invented for. `m=0` is plain Jacobi (a fair same-map baseline); `m∈{5,10}` is Anderson-accelerated.

| Anderson history m | status | iterations to `|b−Ax|∞ < 1e-8` |
|---|---|---|
| 0 (plain Jacobi) | converged | 374 |
| 5 | converged | 62 |
| 10 | converged | 37 |

- On this **single instance** (one RHS, fixed ω=0.6, one tol — *illustrative*, not a measured speedup with spread), the same core cuts plain Jacobi's **374** iterations to **62** (m=5) and **37** (m=10) on a map that has *nothing* to do with ARAP — the point is **map-agnosticism** (the acceleration is a property of the generic Anderson core, not the local-global map), a smoke test of generality rather than a benchmarked ratio. This is the faithful, general Anderson (Peng et al.): m history, min-norm `lstsq`, energy-decrease safeguard, applied to whatever `G` you hand it.

_Scope: 2D, single seed; the two maps (ARAP local-global + Jacobi linear solve) exercise the generality. Wrapping the official SLIM reweighting as a third map is a natural extension. NB the Jacobi map is an SPD quadratic, so the energy-decrease safeguard is near-trivially satisfied -- this demonstrates map-agnosticism of the core, not a stress test of the safeguard (the non-convex ARAP map exercises that)._
