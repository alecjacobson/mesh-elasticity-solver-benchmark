# `bench/` — prototype harness (phase P1)

A first, runnable slice of the component-factored harness (`docs/harness.md`). Pure
Python/NumPy so it runs anywhere; produces **measured, reproducible** numbers gated on a
conformance test. This is the start of P1, not the finished benchmark.

## Run

```bash
python -m bench.conformance   # grounding: analytic derivatives vs finite differences
python -m bench.test_smoke    # fast invariant checks (also run in CI)
python -m bench.run_e1        # E1 filter isolation (symmetric Dirichlet) -> results/e1.md
python -m bench.run_e1_nu     # E1 near-incompressible nu-sweep (Neo-Hookean)  -> results/e1_nu.md
```

## What's implemented (World-1 static distortion cell)

| slot (`harness.md`) | this slice |
|---|---|
| energy ψ | `energy.py` — 2D symmetric Dirichlet (analytic gradient; FD-of-gradient element Hessian) |
| Hessian filter | `filters.py` — `none`, `clamp`, `absolute`, `project-on-demand` (per-element; PDN-style), `identity-shift` (global Levenberg) |
| search direction | `solver.py` — projected Newton |
| line search | `solver.py` — Armijo backtracking with an inversion guard (ψ=∞ on det F ≤ 0) |
| linear solver | dense `np.linalg.solve` / Cholesky-with-shift |
| convergence criterion | `|g|inf < tol` (per-iteration log also records energy + wall-clock) |
| scenario | `mesh.py` — triangulated square, pinned boundary, perturbed interior |

Metrics are computed **from the per-iteration telemetry log**, never hand-reported
(`harness.md` invariant). The conformance test (`conformance.py`) is the admissibility gate:
for the classical analytic energy, matching finite differences *is* the reference until an
official codebase (TinyAD / libigl) is ported for regression (D3).

## Deliberately NOT yet done (tracked)

- No ν-sweep / near-incompressible energy, no locking-free element (control C1) — so this slice
  cannot yet reproduce the absolute-vs-clamp *near-incompressible* claim; it only exercises the
  machinery and shows full-Newton failing where filters succeed.
- Dense assembly + solve (small meshes only); no performance/data profiles yet; single
  scenario/seed. Sparse solve, more filters (trust-region, analytic-eigensystem, PPN), the 1b
  hyperelastic cell, and official-code regression are the next P1 steps.

See `docs/experiments.md` (E1–E5) and `docs/protocol.md` for the full plan.
