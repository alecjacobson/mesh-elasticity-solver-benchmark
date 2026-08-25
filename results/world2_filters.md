# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)

Neo-Hookean ν-sweep (stretch init), only the filter swapped. **Three axes** per cell (docs/metrics.md): iterations, wall-clock, and — where available — global factorizations. Run: `python -m bench.run_world2_filters`.

### Iterations to converge

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 6 |
| 0.4900 | 44 | 79 | 9 |
| 0.4990 | 139 | 314 | 69 |
| 0.4999 | 242 | maxiter | 120 |

_(P2, locking-relieved)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 8 | 9 |
| 0.4900 | 15 | 15 | 15 |
| 0.4990 | 23 | 23 | 22 |
| 0.4999 | 53 | 41 | 52 |

### Wall-clock (ms)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 205 | 204 | 192 |
| 0.4500 | 429 | 429 | 322 |
| 0.4900 | 2369 | 3693 | 511 |
| 0.4990 | 6433 | 14453 | 4795 |
| 0.4999 | 11058 | 18285 | 8036 |

_(P2)_

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 642 | 639 | 870 |
| 0.4500 | 1345 | 1252 | 2084 |
| 0.4900 | 3275 | 2377 | 3754 |
| 0.4990 | 3607 | 3301 | 5372 |
| 0.4999 | 9408 | 6208 | 17088 |

### Global factorizations (P1 solver; P2 solver does not expose counts)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 6 |
| 0.4900 | 44 | 79 | 9 |
| 0.4990 | 139 | 314 | 69 |
| 0.4999 | 242 | 400 | 120 |

## Observed (round-3 refined: faithful per-element blend + principled schedule)

This filter labelled **trust-region** is a ρ-driven **switchboard** over the per-element eigenvalue blend λ_eff=(1−w)λ+w|λ|, w∈{0,0.5,1} → {Newton, clamp, absolute} — **not** a trust-region *radius* method (that is `solve_trust_region`, `results/tr.md`); it is named by analogy to Chen et al. 2024. Same per-iteration cost as clamp/absolute (conformance-gated to equal `filters.project_element` exactly). The schedule is now principled (review-r3): an **SPD-probe** uses raw Newton only when the assembled Hessian is actually SPD (Cholesky), escalation tries **clamp before absolute**, and pred≤0 keeps clamp — so the verdict below is not an artifact of implementation shortcuts.
- **On P1 (locking): the switchboard wins.** P1 at ν=0.4999: **TR 120 it / 8036 ms** · clamp 242 it / 11058 ms · absolute maxiter it / 18285 ms. TR ≤ both filters on iterations **and** (now, per-element) wall-clock — the adaptive back-off to raw Newton (when SPD) genuinely helps escape the locking element.
- **On P2 (locking-relieved): a wash on iterations, penalized on wall-clock.** P2 at ν=0.4999: **TR 52 it / 17088 ms** · clamp 53 it / 9408 ms · absolute 41 it / 6208 ms. The **SPD-probe brought TR's iteration count to parity with clamp** (52 vs 53) — so the *earlier* larger P2 loss was partly bad raw-Newton steps, now fixed. But TR is still ~1.8× **slower than clamp in wall-clock** (the SPD-probe + clamp-before-absolute escalation adds extra per-iteration assemblies), and **absolute is best outright**. So on the **locking-relieved** P2 element — where the plain fixed filters already converge in tens of iterations — the adaptive switchboard buys nothing over a fixed filter and costs more per step.
- **Discretization-dependent verdict, and the axis is LOCKING, not conditioning.** The switchboard clearly **wins on the locking P1** element (where the plain filters struggle, 242 it / maxiter) — fewer iterations *and* less wall-clock than both — but on the **locking-relieved P2** (where they already converge fast) it is **at best a wash** (iteration-parity with clamp, wall-clock-penalized, beaten by absolute). NB the operative variable is **volumetric locking** (a kinematic over-constraint), NOT Hessian conditioning: measured, P2's Hessian is in fact *worse*-conditioned than P1's (κ≈4e6 vs 6e5 at ν=0.4999, largely because P2 has more DOFs), yet it converges faster — for these direct-solve projected-Newton filters iteration count tracks nonlinearity/locking, not κ(H). (Round 1's 'TR beats both on P2' was an artifact of a costlier global assembled-`eigh` operator, reversed here.) The `trust-region→{clamp,absolute}` edges stay **qualified/indicative** — a real, regime-dependent finding, not a decisive win.

_Caveat: dense solve, single stretch/mesh/seed, single τ=1e-6 — indicative. The ρ→w thresholds (≥0.75→Newton, ≤0→absolute, else clamp) are ONE untuned schedule; a smarter schedule could change the P2 outcome, but the ρ-switching *mechanism* is what's measured. Named-by-analogy filter, not a radius-TR; no official-code regression (code unavailable)._
