# 3D harness scaling — off the 2D toy (measured)

Stretched Neo-Hookean bar, P1 tets, sparse analytic-Hessian projected-Newton (`bench/tet_scale.py`, conformance gate 13), `|g|∞<1e-6`. Single-threaded Python/NumPy + SciPy sparse-LU on this machine — wall-clock is implementation-bound and only indicative; the point is the **element count reached** and that Newton's iteration count stays flat (mesh-independent) as the mesh refines. Run: `python -m bench.run_tet3d_scale`.

| box n | tets | free DOF | Newton iters | wall (s) | s / iter |
|---:|---:|---:|---:|---:|---:|
| 8 | 3072 | 1701 | 4 | 0.6 | 0.16 |
| 12 | 10368 | 5577 | 4 | 2.8 | 0.70 |
| 16 | 24576 | 13005 | 4 | 10.8 | 2.71 |
| 20 | 48000 | 25137 | 4 | 36.3 | 9.06 |
| 24 | 82944 | 43125 | 4 | 110.0 | 27.50 |
| 28 | 131712 | 68121 | 5 | 834.2 | 166.84 |

**Ceiling reached: 131,712 tetrahedra / 68,121 free DOF**, converged in 5 Newton iterations — a genuine 3D mesh, three orders of magnitude past the 2D prototype's few-hundred-DOF grids. Newton's iteration count is **mesh-independent** (essentially flat across the sweep), as second-order convergence predicts; wall-clock grows with the sparse-factorization fill-in of the 3D system, an implementation cost, not an algorithmic one.

_This is the substrate for testing the scale/GPU/3D superiority claims that a 2D dense prototype could not reach, and for faithful 3D method ports (Threads 1–2)._
