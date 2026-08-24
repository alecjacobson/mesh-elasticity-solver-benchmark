# Scalability (sparse backend) - mesh-independence + CG conditioning (measured)

Projected Newton (clamp) on the symmetric-Dirichlet perturbation cell across mesh refinement, SPARSE assembly + SuperLU (direct) and CG (iterative) inner solves. Run: `python -m bench.run_scaling`.

| mesh | free dofs | Newton iters | wall (ms) | H nnz | CG mat-vecs/iter | Jacobi-PCG mat-vecs/iter |
|---|---|---|---|---|---|---|
| 8x8 | 98 | 7 | 435.1 | 1156 | 53 | 54 |
| 16x16 | 450 | 8 | 1987.9 | 5828 | 123 | 110 |
| 24x24 | 1058 | 9 | 4920.6 | 14084 | 254 | 199 |
| 32x32 | 1922 | 11 | 10592.4 | 25924 | 372 | 283 |
| 40x40 | 3042 | 8 | 12328.6 | 41348 | 348 | 251 |

## Observed

- **Mesh-independence (metric #69):** Newton iteration count stays in [7,11] as DOFs grow 98->3042 (~31x) -- essentially refinement-independent, the hallmark of a well-behaved second-order outer solver. This is the HW-independent axis and it does NOT grow with mesh size.
- **CG conditioning (why preconditioning matters):** with the SAME outer method, unpreconditioned CG mat-vecs **per Newton iteration** rise from ~53 at 98 dofs to ~348 at 3042 dofs -- the inner solve gets harder under refinement (worsening conditioning) even though the OUTER iteration count is flat. **A Jacobi (diagonal) preconditioner already cuts this** to ~54->251 mat-vecs/iter (see the last column) -- a first, cheap fix; a stronger preconditioner/multigrid would flatten it further. The linear-solver/preconditioner slot is exactly where the inner-cost scaling lives, and the harness now measures it directly.
- **Wall-clock (metric #70):** sparse-direct wall ~ DOFs^1.01 for this prototype -- still Python-assembly-dominated (the per-element loop), so treat the exponent as prototype-specific, not an algorithmic complexity. The clean, portable signals are the iteration count (#69, flat) and the CG mat-vec growth.

_Caveat: Python-loop assembly dominates wall-clock; the mesh-independence and the CG-mat-vec-growth are the robust, hardware-independent findings._
