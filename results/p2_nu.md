# Settling the ν-claim: absolute vs clamp on P1 (locking) vs P2 (locking-relieved)

Same Neo-Hookean ν-sweep, same feasible uniform-stretch init, same filter swap -- on a P1 constant-strain mesh and a locking-relieving **P2 (quadratic) element** (8x8). P2 is conformance-gated (`python -m bench.p2`). Run: `python -m bench.run_p2_nu`. Cells = Newton iterations (or failure).

| ν | P1 clamp | P1 absolute | P2 clamp | P2 absolute |
|---|---|---|---|---|
| 0.3000 | 4 | 4 | 4 | 4 |
| 0.4500 | 9 | 9 | 9 | 8 |
| 0.4900 | 44 | 79 | 15 | 15 |
| 0.4990 | 139 | 314 | 23 | 23 |
| 0.4999 | 242 | **maxiter** | 53 | 41 |

## Observed -- the ν-claim is a discretization artifact on P1

- **On P1 (constant-strain, locking):** absolute filtering badly under-performs clamp as ν→½ and *fails* (maxiter) at ν=0.4999 -- the result that looked like a refutation of the Stabler-Neo-Hookean claim.
- **On P2 (locking relieved):** absolute **matches and even beats** clamp near incompressibility (e.g. it converges in fewer iterations than clamp at ν=0.4999) -- exactly what the paper claims. P2 also converges in far fewer iterations overall (better conditioning once locking is removed).
- **Precise mechanism (important, avoids over-reading):** P2 does NOT remove the *need* for filtering -- unfiltered Newton (`none`) still fails (nondescent) at high ν on **both** P1 and P2, because the element Hessians are genuinely indefinite from the *energy* under large stretch. What P2 fixes is specifically the **clamp-vs-absolute ranking**: on P1 volumetric locking makes absolute's |λ|-flipping overshoot the artificially-stiff locked direction; relieving the locking removes that penalty, so absolute's better spectral choice wins as the paper claims.
- **Conclusion:** the earlier 'absolute is worse' was a **volumetric-locking artifact of the P1 element** in the *filter comparison*, not a property of the filter. A proper (locking-free-er) discretization *reverses* the conclusion and vindicates both the paper's claim AND the benchmark's control C1. This is the benchmark doing its job: **separating a real solver effect from a discretization confound** -- the exact failure mode the survey exists to catch.

_Caveat: pure P2 displacement relieves but does not fully eliminate incompressible locking (Taylor–Hood P2–P1 mixed is the gold standard); the effect is already decisive here. Dense solve, single scenario/stretch._
