# AQP vs ARAP local-global vs Anderson-LG — back-solves to each method's own minimum (measured, P5.2 #6 & #5)

10×10 non-affine bend, 5 seeds. Each method does 1 prefactorization + 1 global **back-solve per iteration**, so back-solves == iterations is a per-iteration-cost-matched HW-independent proxy. Iterations to reach each method's OWN `(E-E*)/(E0-E*) < 1e-4` (AQP on symmetric-Dirichlet; local-global & Anderson-LG on ARAP). Run: `python -m bench.run_aqp_localglobal`.

| method (energy) | back-solves to tol, mean [min–max] | wall (ms) mean |
|---|---:|---:|
| AQP (symmetric-Dirichlet) | 11.4 [9–13] | 2251 |
| local-global (ARAP) | 5.6 [5–6] | 223 |
| Anderson-LG, m=5 (ARAP) | 4.0 [4–4] | 169 |

## Observed

- **#5 `anderson->aqp` (smaller cost):** Anderson-accelerated local-global reaches its ARAP minimum in **4.0** back-solves vs AQP's **11.4** to its symmetric-Dirichlet minimum — Anderson is cheaper on the shared back-solve axis, indicative support (cross-energy: 'lower final energy' is not comparable across the two objectives, only cost is).

- **Not reproduced here:** AQP needs **11.4** back-solves vs local-global's **5.6** — local-global reaches its target in fewer/equal global solves on this instance, so the faster-termination claim does not hold on the back-solve axis.
- Wall-clock (both pure NumPy, diagnostic): AQP 2251ms vs local-global 223ms — consistent with the count carries the verdict, not the millisecond number.

_Caveat: CROSS-ENERGY — AQP minimizes symmetric-Dirichlet, local-global minimizes ARAP; each is scored to its own minimum, so this is 'which reaches its target in fewer equal-cost solves,' not a same-objective race. Single mesh size, moderate shear; indicative, not a validated same-energy head-to-head._
