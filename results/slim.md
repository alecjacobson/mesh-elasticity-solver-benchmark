# SLIM (official libigl) vs AQP / L-BFGS / Newton (measured)

All minimize symmetric Dirichlet; SLIM is libigl's official implementation. Fair shared criterion: iterations to reach relative energy tolerance `(E-E*)/(E0-E*) < 1e-4`. Run: `python -m bench.run_slim` (needs libigl).

E\* = 4.000000 (Newton reference), E₀ = 5.3061.

| method | iters to energy-tol |
|---|---|
| SLIM (libigl, official) | 5 |
| AQP | 19 |
| L-BFGS | 14 |
| Newton | 5 |

## Observed

- **SLIM (5 it) dramatically beats AQP (19 it)** to the same energy tolerance -- validating `slim->aqp` with the OFFICIAL libigl SLIM. SLIM's reweighted (second-order-ish) proxy converges far faster than AQP's fixed Laplacian proxy + momentum on this problem.
- SLIM is competitive with L-BFGS (14) and approaches Newton (5) in iterations here. (Unlike the aqp->l-bfgs claim, `slim->aqp` reproduces.)

_Caveat: SLIM uses soft constraints (soft_p=1e8); energy-tolerance criterion (SLIM and AQP are first-order in the gradient tail); single scenario. Official-code SLIM grounds this comparison (D3)._
