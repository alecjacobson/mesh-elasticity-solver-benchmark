# Faithful BCQN vs competitors — symmetric Dirichlet (measured)

Full BCQN (`bench/bcqn.py`: L=2·cotan-Laplacian proxy factored once + blend Eq.13 + cured barrier-aware direction filter + inversion-free/Armijo line search + characteristic-norm stop — reimplemented from the paper and the authors' reference code, conformance-gated). Fair shared criterion: iterations to `(E-E*)/(E0-E*)<1e-4`, `E*` an independent projected-Newton reference, over 6 mesh/seed scenarios. Run: `python -m bench.run_bcqn`.

| method | iters to energy-tol, mean [min–max] | converged / N |
|---|---:|---:|
| **BCQN (full, faithful)** | 8.0 [7–9] | 6/6 |
| SL-BFGS (BCQN ablation: no blend/cure) | 12.7 [10–17] | 6/6 |
| AQP | 26.3 [8–73] | 6/6 |
| L-BFGS (well-implemented) | 12.0 [11–13] | 6/6 |
| projected-Newton | 6.8 [5–9] | 6/6 |
| Composite Majorization | 7.0 [5–9] | 6/6 |

## Observed — edges adjudicated

- **`bcqn → aqp` — REPRODUCES on iterations:** BCQN 8.0 vs AQP 26.3 iterations. BCQN's L-BFGS blend adds a superlinear tail on top of the same Laplacian preconditioner AQP uses, so it reaches the minimum in fewer iterations — the paper's BCQN-beats-AQP direction, on the hardware-independent axis.
- **`bcqn → sobolev-lbfgs` (its own ablation) — the blend earns its keep:** full BCQN 8.0 vs the no-blend/no-cure Sobolev-L-BFGS 12.7 iterations, isolating the blend+cure contribution on the same proxy.
- **`bcqn → projected-newton` / `bcqn → composite-majorization` — NOT reproduced on iterations:** BCQN needs **more** iterations (BCQN 8.0 vs projected-Newton 6.8, CM 7.0). Expected: BCQN descends a FIXED scalar-Laplacian proxy (factored once, never refactored), whereas PN/CM refactor a coupled per-element Hessian each iteration. BCQN's headline is a **wall-clock/scale** claim — cheaper iterations and no per-iteration factorization, winning at mesh sizes where PN/CM run out of factorization memory — not a fewer-iterations claim. Same shape as PD→Newton and CM→projected-Newton: cheaper-per-step, not fewer-step.

_Faithfulness: this is the real BCQN (blend Eq.13 + cured DPJ filter + inversion-free line search + characteristic-norm stop), gated on β∈[0,1], monotone descent, and convergence to the projected-Newton minimum. The wall-clock/memory-at-scale headline is implementation- and hardware-confounded and is not adjudicated on the 2D iteration axis; the iteration-axis verdicts above are what the hardware-independent comparison supports._
