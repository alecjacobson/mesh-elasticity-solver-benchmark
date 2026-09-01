# Clamp vs absolute — locking P1 vs locking-relieved P2 tets, 3D ν-sweep (measured)

Completes the §8.1 headline in genuine 3D. Same 4×4×4 box (384 tets), Neo-Hookean, stretched 1.3×, projected-Newton to `|g|∞<1e-6`. P1 = constant-strain tet (`bench/tet_scale.py`); P2 = 10-node quadratic tet (`bench/tet_p2.py`), both analytic-tangent conformance-gated. `lam = 2·mu·nu/(1−2nu)`. Run: `python -m bench.run_tet3d_p2`.

| Poisson ν | P1 clamp | P1 absolute | P2 clamp | P2 absolute |
|---:|---:|---:|---:|---:|
| 0.300 | 4 | 4 | 5 | 5 |
| 0.450 | 7 | 7 | 7 | 7 |
| 0.490 | 18 | 27 | 11 | 9 |
| 0.499 | 267 | 300 | 19 | 17 |

## Observed — §8.1's 3D leg completed

- **P1 locks in 3D and shows the reversal:** near incompressibility (ν=0.499) the P1 iteration count climbs (clamp 4→267, absolute 4→300) and **absolute under-performs clamp** (300 vs 267) — the same P1 reversal the 2D headline isolates.
- **P2 relieves the locking, and the reversal vanishes:** on the 10-node quadratic element the counts stay low (clamp 19, absolute 17 at ν=0.499) — absolute now matches/beats clamp, so the P1 'absolute is worse' result was a **discretization (volumetric-locking) artifact of the constant-strain element, not a property of the filter** — exactly the §8.1 conclusion, now demonstrated with a genuine locking-relieved element in 3D.

_Scope: single stretch magnitude, modest mesh; P2 is a standard quadratic tet (a full Taylor–Hood / mixed u–p element is the further gold standard). The point established: the 3D P1 reversal is locking, and a locking-relieved 3D element removes it — the flagship confound-control result, now complete in 3D as well as 2D._
