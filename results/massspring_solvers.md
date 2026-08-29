# Constraint-projection solvers on one mass-spring timestep (measured, V2.2)

Mass-spring implicit-Euler step (`bench/massspring.py`, conformance-gated: ∇Φ vs FD 1e-9, PD system SPD) — the faithful home of PBD/XPBD/Projective-Dynamics/nonlinear-GS, all sharing the exact incremental potential Φ and the same inertial start x̃. Metric: iterations to cut the **incremental-potential gradient residual** to 1e-3 of its start. GPU/wall-clock speed claims are NOT adjudicated here. Run: `python -m bench.run_massspring_solvers`.

| solver | iters to 1e-3 residual |
|---|---:|
| Newton (projected) | 2 |
| local/global = Projective Dynamics / fast-mass-spring (exact) | 8 |
| nonlinear Gauss–Seidel (pbng-style) | 14 |
| **XPBD** (compliance α=1/(k h²)) | stalls @ 0.14·r₀ |
| **PBD** (no compliance) | stalls @ 4.01·r₀ |

### XPBD vs PBD: is effective stiffness iteration-count-independent?

Mean absolute constraint violation `|‖xᵢ−xⱼ‖−L|` after exactly K Gauss–Seidel sweeps (smaller ⇒ stiffer material):

| sweeps K | 2 | 5 | 20 | 80 |
|---|---|---|---|---|
| XPBD | 1.01e-01 | 1.01e-01 | 1.01e-01 | 1.01e-01 |
| PBD | 9.09e-02 | 6.77e-02 | 2.92e-02 | 3.86e-03 |

## Observed — edges adjudicated

- **`primal-xpbd → xpbd` (convergence) — REPRODUCES:** XPBD **stagnates** on the incremental-potential residual — it flat-lines at **0.14·r₀** (≈52) and never reaches the tol, because its constraint sweep omits the momentum-coupling term. Newton, local/global, and nonlinear-GS — all of which retain the full residual — drive it to 0 in 2/8/14 iterations. So a primal method that keeps the backward-Euler momentum residual converges where XPBD stalls.
- **`pbng → xpbd` (convergence) — REPRODUCES:** nonlinear Gauss–Seidel reaches the tol in **14** iterations while XPBD stagnates at 0.14·r₀ — 'reaches tolerance where XPBD stagnates', exactly as claimed.
- **`fast-mass-spring → pbd` and `projective-dynamics → pbd` (quality) — REPRODUCE:** local/global (which IS fast-mass-spring / mass-spring Projective Dynamics) converges to the exact implicit-Euler minimum (8 it, residual→0), whereas PBD (no compliance) does NOT — its residual GROWS to 4.0·r₀ as it over-constrains. PD reaches the true dynamics; PBD does not.
- **`xpbd → pbd` (quality) — REPRODUCES:** across K=2→80 sweeps XPBD's constraint violation is essentially iteration-count-INDEPENDENT (varies 1.0×, converging to the compliant equilibrium), while PBD's keeps shrinking (24× smaller by K=80) — i.e. PBD **stiffens with iteration count** (the material gets artificially stiffer the more you iterate), exactly XPBD's headline.

_Caveat: single 2D mass-spring timestep, one mesh/dt/stiffness; iteration counts and constraint-violation trends are hardware-independent. The GPU/throughput speed headlines (jgs2, vbd, pbng 6–7× Newton) are NOT adjudicated. PD here is EXACT local/global for mass-spring (not the FEM fixed-proxy stand-in of results/dynamics_solvers.md)._
