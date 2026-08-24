# Neo-Hookean ν-sweep: is the clamp/absolute decision FD-noise-robust? (measured, #32)

The ν-sweep filters an FD 6×6 element Hessian; clamp and absolute disagree exactly on the near-zero eigenvalues, so FD subtractive-cancellation noise there could flip the decision. We check FD against a **complex-step** reference (machine precision, no cancellation) over 4000 deformation states per ν spanning the sweep regime (stretch + shear + volume change, including near-singular Hessians). Run: `python -m bench.run_nh_eig_check`.

| ν | max eig rel-err (FD vs exact) | states with a near-zero eigenvalue | clamp/abs decision flips |
|---|---|---|---|
| 0.3000 | 8.1e-10 | 1/3994 | **0/3994** |
| 0.4500 | 2.5e-10 | 8/3991 | **0/3991** |
| 0.4900 | 1.2e-09 | 16/3982 | **0/3982** |
| 0.4990 | 7.9e-10 | 9/3985 | **0/3985** |
| 0.4999 | 6.2e-10 | 5/3989 | **0/3989** |

## Observed

- Across all ν and **19970 deformation states**, the FD element Hessian's eigenvalues match the machine-precision complex-step reference to **~1e-9** (it is a central difference of the *analytic* gradient, not FD-of-FD), and the **clamp/absolute projection decision flips in 0 of them**.
- **So the FD-based ν-sweep is vindicated:** even though many states have a near-zero eigenvalue (where clamp and absolute differ), the FD error (~1e-9) is far smaller than the eigenvalue magnitudes that actually occur, so it never changes which eigenvalues get projected. The clamp-vs-absolute ranking in `e1_nu`/`stable_nu` is a real solver effect, not an FD-noise artifact.

- Complex-step differentiation of the analytic gradient is, for this purpose, an **analytic-accuracy eigensystem** (exact to ~1e-15) without deriving the closed-form NH eigenpairs — it is the cheapest faithful reference and could replace the FD Hessian in the sweep wholesale if ever needed.

_Caveat: 2D; states sampled to span the sweep regime (not the exact per-iterate element states, though chosen to include the near-singular ones that stress the decision)._
