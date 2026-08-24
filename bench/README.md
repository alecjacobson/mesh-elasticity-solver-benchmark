# `bench/` — prototype component-factored harness (phase P1)

A runnable slice of the harness (`docs/harness.md`). Pure Python/NumPy/SciPy so it runs
anywhere; produces **measured, reproducible** numbers gated on a conformance test. Every
taxonomy axis (`docs/taxonomy.md`) now has ≥2 interchangeable implementations. This is P1, not
the finished benchmark.

## Run

```bash
python -m bench.conformance      # grounding: analytic derivatives vs finite differences
python -m bench.test_smoke       # fast invariant checks (also run in CI)
# experiments -> results/*.md
python -m bench.run_e1           # E1 filter isolation (symmetric Dirichlet)
python -m bench.run_e1_nu        # E1 near-incompressible ν-sweep (Neo-Hookean)
python -m bench.run_profiles     # data profiles over problem sets (~3 min)
python -m bench.run_e4           # E4 first- vs second-order (+ Adam honesty control)
python -m bench.run_e5           # E5 criterion sensitivity (log re-scoring)
python -m bench.run_1b_dynamic   # World-2 dynamic incremental potential
python -m bench.run_locking      # locking sensitivity (crossed mesh)
python -m bench.run_scaling      # mesh-independence + CG/PCG conditioning (sparse)
python -m bench.run_ls           # linear-solver axis (direct vs CG)
python -m bench.run_tr           # trust-region (Steihaug-CG) vs filtering
python -m bench.run_linesearch   # line-search axis (backtracking vs full-step)
```

## Component slots implemented (the taxonomy's six axes)

| axis (`harness.md`) | implementations |
|---|---|
| **energy ψ** | 2D symmetric Dirichlet (`energy.py`), Neo-Hookean w/ ν-parameterized volumetric term (`energy_neohookean.py`) |
| **search direction** | projected Newton (`solver.py`), trust-region Steihaug-CG (`solver.py`), L-BFGS / gradient descent / Adam (`descent.py`) |
| **Hessian filter** | `none`, `clamp`, `absolute`, `project-on-demand` (per-element PDN), `identity-shift` (Levenberg), `global-pdn` (`filters.py` + `solver.py`) |
| **line search** | `backtracking` (Armijo), `full-step` (`solver.py`) |
| **linear solver** | dense direct, sparse SuperLU, CG, Jacobi-PCG (`solver.py`: dense + `*_sparse`) |
| **convergence criterion** | \|g\|inf; and offline re-scoring under Newton-decrement / energy-gap / fixed-budget (E5) |
| **scenario** | grid + crossed (union-jack) meshes, pinned/stretched BCs, perturbation & dynamic (`mesh.py`, run scripts) |

Metrics are computed **from the per-iteration telemetry log** (energy, \|g\|, wall-clock,
assemblies, linear-solves, factorizations, mat-vecs, nnz) — never hand-reported. The
conformance test (`conformance.py`) is the admissibility gate: for the classical analytic
energy, matching finite differences *is* the reference until an official codebase (TinyAD /
libigl) is ported for regression (D3).

## Deliberately NOT yet done (tracked)

- **P2 (quadratic) element** (`p2.py`) now *settles* the ν-claim (absolute matches/beats clamp
  once locking is relieved — results/p2_nu.md); a fully locking-free Taylor–Hood P2–P1 mixed
  element is future work.
- Analytic-eigensystem and eigenvalue-blending filters; official-code regression; E2
  seed-decomposition (needs the bespoke methods ported); 2D only; small meshes.

See `docs/experiments.md`, `docs/protocol.md`, and `results/README.md`.
