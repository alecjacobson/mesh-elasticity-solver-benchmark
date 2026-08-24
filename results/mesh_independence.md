# AQP mesh-independence — its design regime (measured)

AQP's actual design claim is a **mesh-independent iteration count** (its Laplacian proxy is an H¹/Sobolev preconditioner), not a raw iteration win over L-BFGS at one resolution (which E2 tested and AQP lost). Fixed continuous problem (unit square, right edge stretched to x=1.5), refined; iterations to relative energy tolerance 1e-4. Run: `python -m bench.run_mesh_independence`.

| mesh | free dof | Newton | AQP | L-BFGS |
|---|---|---|---|---|
| 6×6 | 70 | 2 | 9 | 16 |
| 10×10 | 198 | 2 | 6 | 26 |
| 14×14 | 390 | 2 | 6 | 37 |

## Observed

- Over a **5.6× DOF increase**, iterations-to-tol grow by: Newton **1.0×**, AQP **0.7×**, L-BFGS **2.3×**.
- **AQP's iteration count is far flatter than L-BFGS's** across refinement (AQP 0.7× vs L-BFGS 2.3× for 5.6× more DOFs) — this is the mesh-independence AQP was built for, and it is **invisible to the single-resolution iteration comparison in E2**. So the honest reading of `aqp→l-bfgs` is: AQP loses on raw iterations at a fixed mesh, but its proxy delivers the mesh-independent *scaling* that is its real contribution. Newton is mesh-independent too (its known property) but pays a factorization per iteration (see e2).

_Caveat: 2D, single stretch/seed, dense; energy-tolerance criterion. This tests the *scaling* axis (mesh-independence), complementing E2's fixed-resolution comparison — the two together are the fair picture of AQP (#29)._
