# Scalability (sparse backend) - mesh-independence + CG conditioning (measured)

Projected Newton (clamp) on the symmetric-Dirichlet perturbation cell across mesh refinement, SPARSE assembly + SuperLU (direct) and CG (iterative) inner solves. Run: `python -m bench.run_scaling`.

| mesh | free dofs | Newton iters | wall (ms) | H nnz | CG mat-vecs | mat-vecs/iter |
|---|---|---|---|---|---|---|
| 8x8 | 98 | 7 | 435.0 | 1156 | 369 | 53 |
| 16x16 | 450 | 8 | 1971.2 | 5828 | 986 | 123 |
| 24x24 | 1058 | 9 | 4936.0 | 14084 | 2284 | 254 |
| 32x32 | 1922 | 11 | 10681.0 | 25924 | 4089 | 372 |
| 40x40 | 3042 | 8 | 12316.7 | 41348 | 2784 | 348 |

## Observed

- **Mesh-independence (metric #69):** Newton iteration count stays in [7,11] as DOFs grow 98->3042 (~31x) -- essentially refinement-independent, the hallmark of a well-behaved second-order outer solver. This is the HW-independent axis and it does NOT grow with mesh size.
- **CG conditioning (why preconditioning matters):** with the SAME outer method, CG mat-vecs **per Newton iteration** rise from ~53 at 98 dofs to ~348 at 3042 dofs -- the inner solve gets harder under refinement (worsening Hessian conditioning) even though the OUTER iteration count is flat. The linear-solver slot (unpreconditioned CG here) is where the cost scaling actually lives -- a preconditioner/multigrid is the fix, and this table is the quantitative motivation.
- **Wall-clock (metric #70):** sparse-direct wall ~ DOFs^1.01 for this prototype -- still Python-assembly-dominated (the per-element loop), so treat the exponent as prototype-specific, not an algorithmic complexity. The clean, portable signals are the iteration count (#69, flat) and the CG mat-vec growth.

_Caveat: Python-loop assembly dominates wall-clock; the mesh-independence and the CG-mat-vec-growth are the robust, hardware-independent findings._
