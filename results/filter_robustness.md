# Filter robustness — success rate from inverted starts (measured, P5.2 #2 & #3)

Stable Neo-Hookean (nu=0.45), P1 8x8 grid, `inverted_scenario` starts swept over severity amp in [1.3, 1.7, 2.2, 3.0, 4.0] x 20 seeds. Success = **converged AND inversion-free** (tol 1e-6, max_iter 400). 100 genuinely-inverted starts. Run: `python -m bench.run_filter_robustness`.

### Success rate per method

| method | successes / starts | rate |
|---|---:|---:|
| `absolute` | 100 / 100 | 100% |
| `clamp` | 100 / 100 | 100% |
| `none` | 5 / 100 | 5% |
| `project-on-demand` | 100 / 100 | 100% |
| `trust-region` | 100 / 100 | 100% |

### Paired comparison (does the claimed-more-robust method win?)

| edge | A succ | B succ | A-only | B-only | verdict |
|---|---:|---:|---:|---:|---|
| #2 absolute->clamp | 100 | 100 | 0 | 0 | tie (equal success) — claim not distinguished here |
| #3 trust-region->full-newton | 100 | 5 | 95 | 0 | **A (trust-region) more robust** (+95) |
| #12 pitfalls-PDN->full-newton | 100 | 5 | 95 | 0 | **A (project-on-demand) more robust** (+95) |

## Observed

- **#2 absolute == clamp:** equal success (100/100); on this battery neither filter recovers a start the other cannot (disagreement 0+0). The robustness edge is **not distinguished** here — both line-search-safeguarded Newton variants land the same basin.
- **#3 trust-region > full-newton:** the rho-switchboard recovers 95 more starts than raw Newton (95 TR-only wins vs 0) — supports the edge: unfiltered Newton takes indefinite steps and fails where the fallback survives.
- **#12 project-on-demand > full-newton:** Project-on-Demand Newton (project a block's Hessian only when it is indefinite) recovers 95 more starts than unprojected Newton (95 PDN-only wins) — supports 'PDN keeps Projected-Newton robustness where classical Newton fails'.

_Caveat: 2D, single energy/nu, one mesh; success is basin-of-attraction under a backtracking line search (which itself safeguards raw Newton), so this measures the FILTER's marginal robustness on top of a globalized solver, not filter-vs-nothing._
