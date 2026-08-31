# Clamp vs absolute filtering — genuine 3D tets at scale (10368 tets, 5577 free DOF)

P1 constant-strain tetrahedra, Neo-Hookean, projected-Newton to `|g|∞<1e-6` on the **scalable analytic-Hessian harness** (`bench/tet_scale.py`, conformance-gated). A 12×12×12 box (10368 tets) stretched 1.3×, swept toward incompressibility (`lam = 2·mu·nu/(1−2nu)`). This takes the §8.1 headline's *3D* leg off the toy scale.

| Poisson ν | Lamé λ | clamp iters | absolute iters |
|---:|---:|---:|---:|
| 0.300 | 1 | 5 | 5 |
| 0.450 | 9 | 8 | 7 |
| 0.490 | 49 | 14 | 17 |
| 0.499 | 499 | 93 | 172 |

## Observed

- **Volumetric locking is real in 3D too:** as `ν → ½` the P1 iteration count climbs steeply (clamp 5→93 over the sweep) — the constant-strain element cannot represent near-isochoric deformation, the same discretization confound the 2D headline (§8.1) isolates, now on a genuine 10368-element 3D mesh, not a 2D prototype.
- **The 2D §8.1 P1 'reversal' reproduces in 3D:** near incompressibility **absolute under-performs clamp** on the locking P1 element (172 vs 93 iterations at ν=0.499). Exactly as in 2D, taken at face value this looks like a refutation of the absolute-filtering claim, but §8.1 shows it is a **locking artifact of the P1 element**, not a filter property — the two filters are identical away from the locking limit (5 vs 5 at ν=0.30) and diverge only where the element locks, consistent with §8.5's "the filter choice is one indefinite mode" thesis.

_Scope: P1 tets only — the locking-relieved 3D control (P2 / mixed u–p tet), on which §8.1's 2D re-validation predicts absolute should *beat* clamp, is the pending next step (Thread 1). Single stretch magnitude. The point established here: the harness reaches genuine 3D scale and the P1 locking mechanism + filter reversal reproduce in 3D._
