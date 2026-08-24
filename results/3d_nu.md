# 3D ν-claim: absolute vs clamp on P1 tetrahedra (measured)

Neo-Hookean ν-sweep on a 4x4x4 tet box (125 verts, 384 tets), uniform-stretch init, only the Hessian filter swapped. P1-tet element is conformance-gated (`python -m bench.tet`). Run: `python -m bench.run_3d_nu`.

| ν | clamp | absolute |
|---|---|---|
| 0.3000 | 3 | 3 |
| 0.4500 | 7 | 8 |
| 0.4900 | 37 | 63 |
| 0.4990 | 47 | 112 |

## Observed

- The 2D P1 finding **generalizes to 3D**: on linear tetrahedra absolute filtering under-performs clamp as ν→½ (e.g. 112 vs 47 it at ν=0.499), the same **volumetric-locking artifact** -- and locking is generally *worse* for P1 tets in 3D. This confirms the capstone (results/p2_nu.md) is not a 2D peculiarity.
- **Settling in 3D** requires a locking-free 3D element (P2 tet or mixed u-p / F-bar), exactly as the 2D P2 element settled it there. That element is the remaining 3D step; the P1-3D result already establishes the confound generalizes.

_Caveat: dense solve, small box, single stretch; the P1-vs-locking-free 3D comparison (analogous to results/p2_nu.md) is the open follow-up._
