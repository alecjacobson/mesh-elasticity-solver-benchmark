# Factorization-vs-iteration cost at scale (measured counts + complexity model)

![scale cost](../figures/scale_cost.png)

_`figures/scale_cost.png`: modeled relative cost (Newton=1) vs DOF from measured iteration counts + the 2D sparse-Cholesky model. Whether AQP's single factorization beats Newton at scale hinges on AQP's iteration count staying bounded — which `results/mesh_independence.md` shows it does only to loose τ._

Answers 'does AQP's single-factorization route beat Newton's per-iteration factorizations at scale?' (review-r3 Fresh #3). We measure the **HW-independent cost structure** (factorizations + back-solves to reach τ=1e-6) — the honest axis, since raw wall-clock is C++/Python-confounded (`results/slim.md`) — and project the crossover with the standard 2D sparse-Cholesky model (factorization ~ DOF^1.5, back-solve ~ DOF, L-BFGS iter ~ m·DOF). Run: `python -m bench.run_scale_cost`.

## Measured counts (iterations to τ=1e-6)

| mesh | free dof | Newton (= factorizations) | AQP (1 fac + N back-solves) | L-BFGS (0 fac) |
|---|---|---|---|---|
| 6×6 | 70 | 4 | 49 | 28 |
| 10×10 | 198 | 4 | 97 | 44 |
| 14×14 | 390 | 4 | 120 | 71 |
| 18×18 | 646 | 4 | 206 | 95 |

## Modeled relative cost (normalized so Newton = 1 at each size)

| free dof | Newton | AQP | L-BFGS |
|---|---|---|---|
| 70 | 1.00 | 1.53 | 5.98 |
| 198 | 1.00 | 1.84 | 5.84 |
| 390 | 1.00 | 1.68 | 6.84 |
| 646 | 1.00 | 2.19 | 7.19 |

## Observed

- **The cost *structure* is the point** (measured, HW-independent): Newton does one factorization **per iteration**, AQP does **one** (its fixed Laplacian) plus cheap back-solves, L-BFGS does **none**. So the raw iteration count E2/slim rank on is not the cost — a Newton iteration is ~DOF^1.5, an AQP/L-BFGS iteration is ~DOF.
- **At this tight τ, the 'AQP wins at scale' speculation does NOT hold — it's the opposite.** Newton is **mesh-independent** (4 iters at every size → only 4 factorizations), while **AQP's iteration count blows up** (49→206 over DOF 70→646), so in the model AQP is **1.53→2.19× Newton and RISING**. The `results/slim.md` conjecture that 'as the mesh grows AQP's single-factorization route becomes relatively more attractive' is **refuted at tight τ**: AQP's growing back-solve count (N·DOF) outruns Newton's few mesh-independent factorizations (4·DOF^1.5).
- **The tradeoff is τ-dependent.** AQP's single-factorization advantage is real only where its iteration count stays flat — i.e. at LOOSE τ (`results/mesh_independence.md`: AQP p≈0 at τ=1e-3 but grows at τ=1e-6). So 'factorize-once' beats 'factorize-every-iteration' only when you don't need tight accuracy; for tight tolerances Newton's mesh-independent factorization count wins. This is the crossover the earlier verdicts asserted but never measured.
- **Counts are measured; the DOF-scaling is the standard sparse-Cholesky complexity model, not wall-clock** — raw timing awaits the sparse/compiled harness (D3), since in pure Python interpreter overhead dominates and would mislead (the C++/Python confound in `results/slim.md`).

_Caveat: 2D, single seed/stretch, single τ; the factorization/back-solve COUNTS are measured, the per-op DOF-scaling is the standard sparse-Cholesky model (not timed). L-BFGS's 0-factorization cost is offset by needing more iterations (see e2/world1_profiles)._
