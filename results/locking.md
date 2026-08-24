# Round E - locking sensitivity: absolute vs clamp on standard vs crossed mesh

Re-runs the Neo-Hookean ν-sweep (stretch BC, right edge → x=2) swapping only the filter, on the **standard 2-triangle** mesh and a **crossed 4-triangle** (union-jack) mesh that partially mitigates CST volumetric locking. Cells = iterations to converge (or failure). Run: `python -m bench.run_locking`.

## Standard 2-triangle mesh

| ν | clamp | absolute | identity-shift | none |
|---|---|---|---|---|
| 0.3000 | 6 | 6 | 6 | 6 |
| 0.4500 | 9 | 12 | 13 | nondes |
| 0.4900 | 52 | 89 | 41 | nondes |
| 0.4990 | 112 | 253 | 84 | nondes |
| 0.4999 | 234 | maxite | 125 | nondes |

## Crossed 4-triangle mesh (lower locking)

| ν | clamp | absolute | identity-shift | none |
|---|---|---|---|---|
| 0.3000 | 9 | 6 | 6 | 6 |
| 0.4500 | 14 | 11 | 22 | nondes |
| 0.4900 | 230 | maxite | 38 | nondes |
| 0.4990 | 52 | 52 | 89 | nondes |
| 0.4999 | 64 | 98 | 152 | nondes |

## Observed

- At ν=0.499 the absolute−clamp iteration gap is **141** on the standard mesh and **0** on the crossed mesh — it shrinks. 
- Reducing locking with the crossed mesh shifts the filter comparison, which supports the interpretation that the standard-mesh result (absolute worse than clamp) is **partly a volumetric-locking artifact**, not a pure statement about the filters. Neither mesh is fully locking-free, so this is a *sensitivity probe*, not the definitive test.
- **Takeaway for the benchmark:** the eigenvalue-filter comparison in the near-incompressible regime is confounded by the element/discretization unless a locking-free formulation (mixed u–p / F-bar / P2) is used — exactly protocol control C1. A displacement-P1 benchmark would mis-attribute a locking effect to the solver.

_Next: a genuinely locking-free element (Taylor–Hood P2–P1 or MINI) to settle absolute-vs-clamp at high ν; this probe only shows the effect is real and mesh-dependent._
