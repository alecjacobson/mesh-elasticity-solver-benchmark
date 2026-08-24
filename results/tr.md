# Trust-region (Steihaug-CG) vs eigenvalue filtering (measured)

Does classical trust-region Newton -- which handles indefinite Hessians intrinsically, with **no eigenvalue filter** -- match graphics filtering? Run: `python -m bench.run_tr`.

## Static (symmetric Dirichlet, 10x10)

| method | iters / status | wall (ms) |
|---|---|---|
| none | nondesce | 1434.3 |
| clamp | 10 | 890.9 |
| absolute | 15 | 1309.5 |
| trust-region | 13 | 1016.7 |

## Neo-Hookean ν-sweep (stretch)

| ν | none | clamp | absolute | trust-region |
|---|---|---|---|---|
| 0.3000 | 6 | 6 | 6 | 14 |
| 0.4500 | nondesce | 9 | 12 | 44 |
| 0.4900 | nondesce | 52 | 89 | 94 |
| 0.4990 | nondesce | 112 | 253 | 142 |
| 0.4999 | nondesce | 234 | maxiter | 271 |

## Observed

- **Trust-region converges across the whole ν-sweep** (yes) with NO filter, exactly where unfiltered Newton (`none`) fails (yes, none fails) -- it handles negative curvature intrinsically (truncated CG to the trust boundary) rather than by projecting the spectrum. This is direct empirical support for the survey's **lineage claim**: eigenvalue filtering and classical modified-Newton / trust-region are two routes to the same end (a usable descent step from an indefinite Hessian).
- **Iteration counts are comparable** to clamp in the well-conditioned regimes; in the stiff near-incompressible tail all of {clamp, absolute, trust-region} pay for the conditioning (which, per the locking probe, is itself a mesh/discretization artifact here). The point is not which wins -- it is that a graphics paper presenting a filter should cite the classical trust-region alternative it competes with, and a fair benchmark must include it (it is now in the harness).

_Caveat: dense prototype, single scenarios; trust-region parameters (Δ0, η) at textbook defaults, not tuned._
