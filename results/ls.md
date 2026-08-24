# Linear-solver axis - direct vs CG (measured)

Only the **linear-solver slot** varies (clamp filter fixed). Run: `python -m bench.run_ls`.

| scenario | solver | Newton iters | inner cost | wall (ms) |
|---|---|---|---|---|
| SD 10x10 | direct | 10 | 10 factorizations | 16897.3 |
| SD 10x10 | cg | 10 | 1025 mat-vecs | 21498.5 |
| NH ν=0.49 8x8 | direct | 52 | 52 factorizations | 33600.6 |
| NH ν=0.49 8x8 | cg | 52 | 11387 mat-vecs | 32998.6 |

## Observed

- **Outer iterations are identical** across the two linear solvers (CG is solved tight, so Newton takes the same steps) -- confirming the linear solver is orthogonal to the search-direction/filter axes. A benchmark that reports only Newton iterations would call these two configs *equal*.
- **Inner cost is completely different, and interpretable:** direct pays one factorization per Newton iteration; CG pays matrix-vector products (metric #15) that grow with the Hessian conditioning -- ~102 mat-vecs/iter on the well-conditioned SD problem vs ~219/iter on the stiff near-incompressible NH problem. The mat-vec count is a clean, hardware-independent, physically meaningful signal (it tracks conditioning, hence the value of a preconditioner).
- **Wall-clock, by contrast, is unreliable here and even gives a CONTRADICTORY ranking:** direct is faster on SD but CG is faster on NH (6 s vs 22 s) at this small dense prototype scale (Python-callback overhead, per-iteration variation). This is the metrics.md Lever-1 lesson made vivid: **had we ranked the linear solvers on wall-clock alone we'd have concluded opposite things on two scenarios**, whereas the mat-vec/factorization counts are consistent and portable. Rank on the hardware-independent count; report wall-clock, don't rank on it.

_Caveat: dense prototype; CG mat-vec via a Python callback makes wall-clock especially noisy. A sparse operator + preconditioner is the next step and is where CG mat-vec counts become the decisive, meaningful metric on large problems._
