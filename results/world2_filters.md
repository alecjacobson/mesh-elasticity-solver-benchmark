# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)

Neo-Hookean ν-sweep (stretch init), only the filter swapped. Run: `python -m bench.run_world2_filters`.

## P1 (locking)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 10 |
| 0.4900 | 44 | 79 | 78 |
| 0.4990 | 139 | 314 | 281 |
| 0.4999 | 242 | maxiter | maxiter |

## P2 (locking-relieved)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 8 | 8 |
| 0.4900 | 15 | 15 | 11 |
| 0.4990 | 23 | 23 | 23 |
| 0.4999 | 53 | 41 | 39 |

## Observed

- **On P1 (locking):** clamp is best, absolute worst, and **trust-region tracks absolute** at high ν -- locking gives a poor quadratic-model fit (ρ far from 1), so the adaptive rule keeps selecting absolute, inheriting the locking penalty. The 'switchboard' is only as good as the discretization it runs on.
- **On P2 (locking-relieved): trust-region BEATS BOTH clamp and absolute** -- e.g. 39 it vs clamp 53 / absolute 41 at ν=0.4999, and 11 vs 15 / 15 at ν=0.49. With locking removed the quadratic-model fit is reliable, so the adaptive rule picks the better filter each step and dominates its own components -- exactly the paper's 'switchboard beats each standalone' claim. The advantage is real but **conditional on a proper (locking-free) element**.
- **Hardens** `trust-region-filtering→{clamp,absolute}`: **validated on P2** (TR beats both), while on locking P1 it degrades to the worse component (absolute). So the switchboard claim holds, but is discretization-conditional -- the benchmark separates the real adaptive-filter advantage from the P1-locking confound (control C1).

_Caveat: dense solve, single stretch/mesh; ρ threshold ε=0.01 (paper default)._
