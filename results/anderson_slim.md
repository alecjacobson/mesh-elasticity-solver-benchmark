# Anderson acceleration OF SLIM — does it cut iterations? (measured, P5.2 #9)

Wraps the **official libigl SLIM** fixed-point map in the map-agnostic `anderson_accelerate` core (Peng et al. 2018). `m=0` is plain SLIM (same-map baseline); `m>0` is Anderson-accelerated. Hard instance (12×12, non-affine bend k=0.8, injective start) so plain SLIM's fixed point contracts slowly (380 it to 1e-3 residual). Metric: iterations to cut the **fixed-point residual** (symmetric-Dirichlet gradient ‖·‖∞) to 1e-3 of its start — the SD *energy* saturates in ~1 SLIM step here, so the residual, not the energy, carries the long tail that acceleration acts on. The SLIM map is made pure by re-`slim_precompute`-ing each step (reproduces continuous SLIM to 0.0). Run: `python -m bench.run_anderson_slim`.

| Anderson history m | iters to residual-tol | speedup vs m=0 |
|---|---:|---:|
| 0 (plain SLIM) | 380 | 1.00× |
| 5 | 10 | 38.00× |

## Observed

- **Anderson accelerates SLIM:** plain SLIM needs **380** iterations to the residual tol; Anderson (m=5) needs **10** — a **38.00× iteration reduction** on the SAME official-SLIM map. This reproduces the edge on the HW-independent iteration axis: Anderson is a wrapper that speeds SLIM up, it does not replace it.
- The acceleration is large precisely because this instance sits on SLIM's **slowly-contracting linear tail** — where the fixed-point residual creeps down over hundreds of iterations, Anderson's history-based extrapolation collapses it in a handful. On an easy instance (SLIM already near-quadratic) there is nothing to accelerate; the edge holds where SLIM's own convergence is slow. Anderson remains a lightweight wrapper (one small least-squares per step), not a replacement solver.

_Caveat: single hard 2D instance; SLIM's soft-BC boundary drift is not re-checked here (see results/slim.md, drift 4.4e-16); official libigl SLIM grounds the base map (D3). Wall-clock is not the metric — each Anderson step adds a small least-squares over the SLIM step, and re-precompute inflates our Python wall-clock but not the iteration count._
