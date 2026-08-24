# SLIM (official libigl) vs AQP / L-BFGS / Newton (measured)

All minimize symmetric Dirichlet; SLIM is libigl's official implementation. Fair shared criterion: iterations to reach relative energy tolerance `(E-E*)/(E0-E*) < 1e-4`, **paired with wall-clock and a HW-independent cost (global factorizations)** per docs/metrics.md. Run: `python -m bench.run_slim` (needs libigl).

E\* = 4.000000 (hard-constrained Newton reference), E₀ = 5.3061.

| method | iters to energy-tol | wall (ms) | global factorizations |
|---|---|---|---|
| SLIM (libigl, official) | 5 | 1.8 | 5 |
| AQP | 19 | 344.5 | 1 |
| L-BFGS | 14 | 134.6 | 0 |
| Newton | 5 | 273.6 | 5 |

## Constraint-satisfaction check (soft-vs-hard confound)

SLIM pins the boundary with a **soft** penalty (`soft_p=1e8`); the other methods use **hard** pinned BCs and `E*` is the hard-constrained minimum. Measured SLIM boundary drift `||UV[b] − bc||∞ = 4.44e-16` (**negligible** — the stiff penalty effectively enforces the hard BC, so the shared elastic-energy metric and hard `E*` are fair for SLIM).

## Observed

- **On the HW-independent axis (iterations / factorizations) `slim->aqp` reproduces:** SLIM reaches the tol in **5 iterations** vs AQP's **19**, with the OFFICIAL libigl SLIM. SLIM is a **reweighted (IRLS / Gauss-Newton) second-order-ish proxy** that refactorizes a global system each iteration -- *not* a first-order method like AQP; that is why it needs far fewer iterations.
- **⚠️ Do NOT read the raw wall-clock across the SLIM row:** libigl SLIM is compiled **C++**, our AQP/L-BFGS/Newton are pure **Python/NumPy**. SLIM does the *same* 5 iterations and 5 factorizations as Newton yet reports ~155× less wall-clock -- that gap is the **compiled-vs-interpreted implementation confound**, not an algorithmic property. Wall-clock is only comparable *within* the Python group (there L-BFGS 135ms < Newton 274ms < AQP 344ms).
- **The real SLIM-vs-AQP tradeoff is factorizations vs iterations:** SLIM does **5 full factorizations**; AQP does **1** (it prefactors its fixed Laplacian once) plus 19 cheap back-solves; L-BFGS does **0**. On small meshes a factorization is cheap so SLIM's few-factorization route wins; as the mesh grows and factorization dominates, AQP's single-factorization route becomes relatively more attractive -- so the iteration-count win does not by itself settle wall-clock at scale (mesh-independence untested here, #29).

_Caveat: energy-tolerance criterion; single 8×8 scenario/seed; SLIM's scale- and mesh-independence and no-flip headlines are NOT tested here (see #29). Official-code SLIM grounds this comparison (D3), but the C++/Python wall-clock boundary means the HW-independent counts carry the verdict, not raw milliseconds._
