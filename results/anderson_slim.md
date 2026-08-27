# Anderson acceleration OF SLIM — does it cut iterations? (measured, P5.2 #9)

Wraps the **official libigl SLIM** fixed-point map in the map-agnostic `anderson_accelerate` core (Peng et al. 2018). `m=0` is plain SLIM (same-map baseline); `m>0` is Anderson-accelerated. Hard instance (12×12, non-affine bend k=0.8, injective start) so plain SLIM's fixed point contracts slowly. We report iterations to cut the **fixed-point residual** (symmetric-Dirichlet gradient ‖·‖∞) to 1e-3 of its start AND to an **absolute** tol ‖·‖∞<1e-4 (the criterion the validated anderson→local-global edge uses, for consistency). The SD *energy* saturates in ~1 SLIM step here, so an energy criterion shows nothing — the residual carries the tail acceleration acts on; we report both so the metric choice is transparent, not selected. The SLIM map is made pure by re-`slim_precompute`-ing each step; we VERIFY this equals continuous SLIM: max coordinate discrepancy over 20 steps = **0.0e+00** (so the plain-SLIM baseline is faithful, not an inflated re-precompute artifact). Run: `python -m bench.run_anderson_slim`.

| Anderson history m | iters to residual-tol (1e-3 rel) | iters to ‖g‖<1e-4 (abs) | speedup (rel/abs) |
|---|---:|---:|---:|
| 0 (plain SLIM) | 380 | 325 | 1.0× / 1.0× |
| 5 | 10 | 9 | 38.0× / 36.1× |

## Observed

- **Anderson accelerates SLIM (instance-dependent):** on this deliberately slow-contracting instance plain SLIM needs **380** iterations to the residual tol; Anderson (m=5) needs **10** — a **38× reduction** on the SAME official-SLIM map, confirmed on the absolute-tol criterion too (325→9). The DIRECTION (Anderson is a wrapper that speeds SLIM up, not a replacement) is the result; the MAGNITUDE is instance-selected — we chose a bend that makes plain SLIM's tail long, so this is an *up-to* figure, not a typical speedup.
- Why the effect exists here and vanishes elsewhere: this instance sits on SLIM's **slowly-contracting linear tail** (residual creeps down over hundreds of iterations), which is exactly what Anderson's history extrapolation collapses; on an easy instance where SLIM is already near-quadratic there is nothing to accelerate. So the edge is regime-dependent: `qualified/indicative` on a single hand-picked instance (m∈{0,5} only, no multi-seed/mesh sweep, no m-profile) — NOT the multi-condition evidence the repo reserves for `validated`.

_Caveat: single hard 2D instance; SLIM's soft-BC boundary drift is not re-checked here (see results/slim.md, drift 4.4e-16); official libigl SLIM grounds the base map (D3). Wall-clock is not the metric — each Anderson step adds a small least-squares over the SLIM step, and re-precompute inflates our Python wall-clock but not the iteration count._
