# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)

Neo-Hookean ν-sweep (stretch init), only the filter swapped. Run: `python -m bench.run_world2_filters`.

## P1 (locking)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 8 |
| 0.4900 | 44 | 79 | 31 |
| 0.4990 | 139 | 314 | 62 |
| 0.4999 | 242 | maxiter | 139 |

## P2 (locking-relieved)

| ν | clamp | absolute | trust-region |
|---|---|---|---|
| 0.3000 | 4 | 4 | 4 |
| 0.4500 | 9 | 8 | 8 |
| 0.4900 | 15 | 15 | 11 |
| 0.4990 | 23 | 23 | 23 |
| 0.4999 | 53 | 41 | 39 |

## Observed

Trust-region here is the FAITHFUL three-state blend λ_eff=(1−w)λ+w|λ|, w∈{0,0.5,1} → {full Newton, clamp, absolute} driven by the model-fit ratio ρ (review-r1 #38); the operator reproduces the three named filters exactly (conformance-gated) and adds the **w=0 full-Newton branch the old two-state version lacked**.
- **On P2 (locking-relieved): trust-region BEATS BOTH clamp and absolute** -- 39 it vs clamp 53 / absolute 41 at ν=0.4999, and 11 vs 15 / 15 at ν=0.49. With locking removed the quadratic-model fit is reliable, so the adaptive rule picks the better state each step and dominates its own components -- exactly the 'switchboard beats each standalone' claim.
- **On P1 (locking): trust-region now also beats BOTH** (139 vs clamp 242 / absolute maxiter at ν=0.4999; 62 vs 139 / 314 at ν=0.499). This is the payoff of restoring the full-Newton branch: the old two-state switchboard was stuck choosing between clamp and absolute and inherited absolute's locking penalty; the three-state rule can back off to raw Newton when the model fits and only project when it doesn't, so it dominates even on the locking element. **NB:** the absolute-vs-clamp *gap* on P1 is still **locking-confounded and non-attributable** (control C1) -- but trust-region's win over *both* is a genuine adaptive-solver effect, not a locking artifact.
- **Hardens** `trust-region-filtering→{clamp,absolute}`: **validated on BOTH P1 and P2** (TR ≤ both filters everywhere, strictly better in most rows). The switchboard dominates its components; what remains discretization-dependent is only the clamp-vs-absolute ordering, not the trust-region advantage.

_Caveat: dense solve, single stretch/mesh; ρ→w thresholds (ρ≥0.75→Newton, ≤0→absolute, else clamp), eigenvalue floor ε=0.01 (paper default). No official-code regression (code unavailable); the three-state operator is instead conformance-gated to reproduce full-Newton/clamp/absolute exactly._
