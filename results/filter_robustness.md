# Filter robustness — success rate from inverted starts (measured, P5.2 #2 & #3 & #12)

Stable Neo-Hookean (nu=0.45), P1 8x8 grid, `inverted_scenario` starts swept over severity amp in [1.3, 1.7, 2.2, 3.0, 4.0] x 20 seeds. Success = **converged AND inversion-free** (tol 1e-6, max_iter 400). 100 genuinely-inverted starts. Run: `python -m bench.run_filter_robustness`.

> ⚠️ **Read the baseline honestly.** `none` is NOT a well-globalized Newton — in `solver.solve` it HARD-TERMINATES the instant the raw Hessian yields a non-descent direction (its failures are **{'nondescent': 95}**, i.e. essentially all `nondescent`), with no negative-curvature fallback or trust-region radius. So a big margin over `none` measures **presence of *any* indefiniteness handling**, not a filter's edge over a competently globalized Newton. The starts are also **not independent** (correlated severities on one 8×8 mesh, one energy, one ν), so a rate like `100/100` is a saturated point estimate, not a statistically-powered success rate. These results are `qualified`, not `validated`.

### Success rate per method

| method | successes / starts | rate | failure modes |
|---|---:|---:|---|
| `absolute` | 100 / 100 | 100% | — |
| `clamp` | 100 / 100 | 100% | — |
| `none` | 5 / 100 | 5% | {'nondescent': 95} |
| `project-on-demand` | 100 / 100 | 100% | — |
| `trust-region` | 100 / 100 | 100% | — |

### Paired comparison (does the claimed-more-robust method win?)

| edge | A succ | B succ | A-only | B-only | verdict |
|---|---:|---:|---:|---:|---|
| #2 absolute->clamp | 100 | 100 | 0 | 0 | tie (equal success) — claim not distinguished here |
| #3 trust-region->full-newton | 100 | 5 | 95 | 0 | **A (trust-region) more robust** (+95) |
| #12 pitfalls-PDN->full-newton | 100 | 5 | 95 | 0 | **A (project-on-demand) more robust** (+95) |

## Observed

- **#2 absolute == clamp:** equal success (100/100); on this battery neither filter recovers a start the other cannot (disagreement 0+0). The robustness edge is **not distinguished** here — both line-search-safeguarded Newton variants land the same basin.
- **#3 trust-region ≫ unfiltered Newton (qualified):** the rho-switchboard recovers 95 more starts than the `none` baseline (95 TR-only wins). The direction is real (an unhandled indefinite Hessian fails; the SPD switchboard does not) — but recall the baseline hard-terminates on the first non-descent direction, so this measures *presence of indefiniteness handling*, not TR's edge over a competently globalized Newton. Qualified, not validated.
- **#12 project-on-demand ≫ unfiltered Newton (qualified):** PDN (project a block's Hessian only when indefinite → PSD assembled Hessian → always a descent direction) recovers 95 more starts than the `none` baseline (95 PDN-only wins). Same reading as #3: real direction, but the margin is against a baseline with *no* indefiniteness handling, so qualified, not validated.

_Caveat: 2D, single energy/nu, one 8×8 mesh, NON-INDEPENDENT correlated starts; the `none` baseline has no negative-curvature fallback (hard-terminates on first non-descent), so these compare 'any indefiniteness handling vs none', not a filter's edge over a competently globalized Newton. #2 (absolute vs clamp) is the one apples-to-apples pair here and it is a tie. All three edges are `qualified`, none `validated`._
