# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)

Neo-Hookean ν-sweep (stretch init), only the filter swapped. **Three axes** per cell (docs/metrics.md): iterations, wall-clock, and — where available — global factorizations. Run: `python -m bench.run_world2_filters`.

### Iterations to converge

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 8 |
| 0.4900 | 44 | 79 | 46 |
| 0.4990 | 139 | 314 | 62 |
| 0.4999 | 242 | maxiter | 188 |

_(P2, locking-relieved)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 8 | 8 |
| 0.4900 | 15 | 15 | 11 |
| 0.4990 | 23 | 23 | 23 |
| 0.4999 | 53 | 41 | 39 |

### Wall-clock (ms)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 228 | 203 | 760 |
| 0.4500 | 437 | 427 | 1642 |
| 0.4900 | 2022 | 3538 | 9529 |
| 0.4990 | 6186 | 14008 | 13709 |
| 0.4999 | 10788 | 17754 | 43360 |

_(P2)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 599 | 592 | 593 |
| 0.4500 | 1287 | 1191 | 1143 |
| 0.4900 | 2189 | 2129 | 1538 |
| 0.4990 | 3397 | 3098 | 3268 |
| 0.4999 | 9007 | 5903 | 5643 |

### Global factorizations (P1 solver; P2 solver does not expose counts)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 9 |
| 0.4900 | 44 | 79 | 49 |
| 0.4990 | 139 | 314 | 66 |
| 0.4999 | 242 | 400 | 194 |

## Observed (corrected, review-r2 #42/#43/#44/#45)

Trust-region is the three-state blend λ_eff=(1−w)λ+w|λ|, w∈{0,0.5,1} → {full Newton, clamp, absolute} driven by the model-fit ratio ρ. **Two honest corrections to the round-1 write-up:**
- **Fewer iterations is NOT cheaper here.** P1 at ν=0.4999: trust-region 188 it / 43360 ms (194 fac) vs clamp 242 it / 10788 ms (242 fac) vs absolute maxiter it. So TR uses ~1.3× FEWER iterations but ~4.0× MORE wall-clock than clamp. On P1 the trust-region step does a **full eigendecomposition of the assembled Hessian** every iteration (plus an extra one on each non-descent escalation), whereas clamp/absolute do cheap per-element 6×6 projections — so TR's few iterations cost more wall-clock than clamp's many. The earlier 'trust-region dominates / validated on both' verdict was drawn on **iteration count alone** and is withdrawn; on the paired (iterations, wall-clock, factorizations) view TR trades iterations for per-step cost.
- **P1 and P2 use different trust-region implementations.** P1 (this solver) uses the assembled three-state eigen-blend; P2 (`bench/p2.py`) still uses a *per-element two-state* clamp/absolute switch. So the P1 and P2 trust-region columns are **not the same operator**, and the cross-element comparison is apples-to-oranges — flagged rather than hidden. A per-element three-state blend (same cost as clamp/absolute) is the right unification and is future work.
- What DOES hold on iteration count: on the locking-free P2 the adaptive rule is ≤ both standalone filters at every ν, and on P1 it uses fewer iterations than both at the most incompressible ν — but **not uniformly** (e.g. P1 ν=0.49 it slightly trails clamp), so 'dominates everywhere' was an overclaim. The eps floor is now 1e-9, matching the standalone filters (so the w=0.5/w=1 states ARE those filters — conformance-gated against `filters.project_element`; note this changed the P1 counts vs the round-1 run, which used a 0.01 floor). The absolute-vs-clamp gap on P1 stays locking-confounded (control C1).

_Caveat: dense solve, single stretch/mesh/seed, single τ=1e-6 — an **indicative** head-to-head, not a validated verdict (review-r2). ρ→w thresholds ρ≥0.75→Newton, ≤0→absolute, else clamp. No official-code regression (code unavailable); the operator is conformance-gated to reproduce full-Newton and the real clamp/absolute filters at eps=1e-9._
