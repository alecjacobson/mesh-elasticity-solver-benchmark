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

### Quality: max position error vs the true implicit-Euler solution (after K=20 sweeps, as a fraction of the deformation scale)

| method | ‖x − x_trueIE‖∞ / scale |
|---|---:|
| XPBD | 0.7% |
| nonlinear-GS / VBD-style | 0.0% |
| PBD | 42.8% |

XPBD position error after 20 sweeps is **regime-dependent** — sweeping stiffness × timestep (it grows with both):

| k \ dt | 1/60 | 1/8 |
|---|---:|---:|
| 1e3 | 0.2% | 3.4% |
| 1e4 | 1.2% | 42.0% |
| 1e5 | 9.1% | 65.4% |

## Observed — edges adjudicated

- **`primal-xpbd → xpbd` (convergence) — REPRODUCES:** XPBD **stagnates** on the incremental-potential residual — it flat-lines at **0.14·r₀** (≈52) and never reaches the tol, because its constraint sweep omits the momentum-coupling term. Newton, local/global, and nonlinear-GS — all of which retain the full residual — drive it to 0 in 2/8/14 iterations. So a primal method that keeps the backward-Euler momentum residual converges where XPBD stalls.
- **`pbng → xpbd` (convergence) — REPRODUCES:** nonlinear Gauss–Seidel reaches the tol in **14** iterations while XPBD stagnates at 0.14·r₀ — 'reaches tolerance where XPBD stagnates', exactly as claimed.
- **`fast-mass-spring → pbd` and `projective-dynamics → pbd` (quality) — REPRODUCE:** local/global (which IS fast-mass-spring / mass-spring Projective Dynamics) converges to the exact implicit-Euler minimum (8 it, residual→0), whereas PBD (no compliance) does NOT — its residual GROWS to 4.0·r₀ as it over-constrains. PD reaches the true dynamics; PBD does not.
- **`xpbd → pbd` (quality) — REPRODUCES:** across K=2→80 sweeps XPBD's constraint violation is essentially iteration-count-INDEPENDENT (varies 1.0×, converging to the compliant equilibrium), while PBD's keeps shrinking (24× smaller by K=80) — i.e. PBD **stiffens with iteration count** (the material gets artificially stiffer the more you iterate), exactly XPBD's headline.
- **`xpbd → full-newton` (quality) — only at LOW stiffness; NOT in XPBD's stiff-cloth target regime:** at a soft operating point (k=1e3, dt=1/30) XPBD's 20-sweep positions are within **0.7%** of the true Newton solution — visually indistinguishable. But that is exactly the regime that flatters it: sweeping stiffness × timestep (table above) the same 20-sweep error climbs to **65%** by k=1e5×dt=1/8, and (measured beyond the table) ~64% at k=1e6 or dt=1. Since XPBD's whole selling point is STIFF cloth, the regime where 'visually indistinguishable' actually matters is where it FAILS at a fixed low budget. So the claim is heavily operating-point-tuned: it holds for soft materials/short steps, not for the stiff regime XPBD targets. Qualified, regime-restricted.
- **`vertex-block-descent → xpbd` (quality) — reproduces the *phenomenon*, not VBD-specifically:** a **generic** nonlinear Gauss–Seidel (plain per-vertex Newton -- NOT VBD; no vertex colouring, no block-parallel structure, no VBD acceleration; this predates Chen-2024) drives the position error to **0.0%** — it matches the true implicit-Euler solution — whereas XPBD plateaus at 0.7%. So 'a primal solver matches true implicit Euler where XPBD does not' holds, but this is a property of ANY consistent primal solver, not of VBD specifically. NB the claim's word 'diverges' is imprecise for XPBD (it *stagnates*); it is **PBD** that moves away (error grows to 43%).

_Caveat: single 2D mass-spring timestep, one mesh/dt/stiffness; iteration counts and constraint-violation trends are hardware-independent. The GPU/throughput speed headlines (jgs2, vbd, pbng 6–7× Newton) are NOT adjudicated. PD here is EXACT local/global for mass-spring (not the FEM fixed-proxy stand-in of results/dynamics_solvers.md)._
