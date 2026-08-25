# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)

Neo-Hookean ν-sweep (stretch init), only the filter swapped. **Three axes** per cell (docs/metrics.md): iterations, wall-clock, and — where available — global factorizations. Run: `python -m bench.run_world2_filters`.

### Iterations to converge

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 7 |
| 0.4900 | 44 | 79 | 9 |
| 0.4990 | 139 | 314 | 100 |
| 0.4999 | 242 | maxiter | 114 |

_(P2, locking-relieved)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 8 | 12 |
| 0.4900 | 15 | 15 | 22 |
| 0.4990 | 23 | 23 | 50 |
| 0.4999 | 53 | 41 | 94 |

### Wall-clock (ms)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 238 | 201 | 187 |
| 0.4500 | 425 | 426 | 339 |
| 0.4900 | 1961 | 3518 | 448 |
| 0.4990 | 6126 | 13918 | 6309 |
| 0.4999 | 10737 | 17634 | 7327 |

_(P2)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 604 | 587 | 557 |
| 0.4500 | 1282 | 1177 | 1825 |
| 0.4900 | 2236 | 2122 | 3567 |
| 0.4990 | 3464 | 3111 | 9978 |
| 0.4999 | 9031 | 5901 | 20988 |

### Global factorizations (P1 solver; P2 solver does not expose counts)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 8 |
| 0.4900 | 44 | 79 | 11 |
| 0.4990 | 139 | 314 | 173 |
| 0.4999 | 242 | 400 | 135 |

## Observed (round-2 fair-cost re-implementation)

Trust-region is now the **faithful PER-ELEMENT three-state blend** λ_eff=(1−w)λ+w|λ|, w∈{0,0.5,1} → {full Newton, clamp, absolute} driven by the global model-fit ratio ρ. It is the **same per-iteration cost** as clamp/absolute (one per-element projection + one factorization) — conformance-gated to equal `filters.project_element` exactly — and **P1 and P2 now use the identical implementation** (the round-1 P1-assembled / P2-per-element split, and the expensive global `eigh`, are gone; review-r2 #42/#44). This changes the verdict:
- **On P1 (locking): trust-region wins on BOTH axes.** P1 at ν=0.4999: **TR 114 it / 7327 ms** · clamp 242 it / 10737 ms · absolute maxiter it / 17634 ms. TR ≤ both filters on iterations **and** wall-clock here — with a fair per-step cost, the adaptive back-off to raw Newton genuinely helps escape the locking element.
- **On P2 (locking-relieved): trust-region LOSES to both.** P2 at ν=0.4999: **TR 94 it / 20988 ms** · clamp 53 it / 9031 ms · absolute 41 it / 5901 ms. TR is worse than both clamp and absolute on iterations **and** wall-clock — where the model already fits well, ρ picks w=0 (Newton), which is indefinite at high ν, so each step wastes a failed-Newton attempt before escalating; plain clamp/absolute just converge.
- **This REVERSES the round-1 P2 story.** Round 1 reported 'TR beats both on P2' — but that used the expensive global **assembled**-`eigh` operator, a *different and costlier* projection than per-element filtering. With the faithful, fair-cost per-element operator the P2 win disappears. So the switchboard's benefit is **discretization-dependent**: it helps on the ill-conditioned/locking element and *hurts* on the well-conditioned one (its ρ-driven adaptivity is counter-productive when the plain filter already converges fast). The `trust-region→{clamp,absolute}` edges stay **qualified/indicative**.

_Caveat: dense solve, single stretch/mesh/seed, single τ=1e-6 — indicative. ρ→w thresholds ρ≥0.75→Newton, ≤0→absolute, else clamp (untuned; a better schedule might help P2). No official-code regression (code unavailable); the per-element operator is conformance-gated to equal the real clamp/absolute filters (eps=1e-9) exactly._
