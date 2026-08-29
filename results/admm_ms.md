# ADMM & Anderson-ADMM on one mass-spring timestep (measured, V2.3)

Faithful mass-spring incremental potential (`bench/massspring.py`, conformance-gated). ADMM-PD (Overby 2017) splits Φ with per-spring auxiliaries; the x-update reuses PD's constant global system, the z-update is a per-spring prox, the dual is a running sum. Metric: iterations to cut the incremental-potential gradient residual to 1e-3 of its start. Speed/wall-clock claims NOT adjudicated. Run: `python -m bench.run_admm_ms`.

| method | iters to 1e-3 residual |
|---|---:|
| local/global (Projective Dynamics) | 8 |
| plain ADMM (ρ=k) | 11 |
| plain ADMM (best over ρ∈{0.25,0.5,1.0,2.0,4.0}·k → ρ=1.0k) | 11 |
| **Anderson-ADMM** (m=5, ρ=k) | 6 |

## Observed — edges adjudicated

- **`aa-admm → admm` (convergence) — REPRODUCES:** Anderson acceleration of the ADMM fixed point (our map-agnostic `anderson_accelerate` core, energy-safeguarded on the ADMM residual) reaches the tol in **6** iterations versus plain ADMM's **11** at the same ρ — Anderson decreases the residual faster, as claimed.
- **`admm-pd → projective-dynamics` (convergence) — NOT reproduced on iterations:** plain ADMM converges in **11** iterations at its best penalty (ρ=1.0k) versus PD's **8**. ADMM is NOT faster than PD on the iteration axis here (PD's single fixed global solve already contracts quickly); the paper's 'faster' is at a specific weight w=½√k and may be a wall-clock statement (per-iteration cost), which is hardware-confounded and out of reach. So only the *direction* the acceleration adds (Anderson, above) reproduces.

_Caveat: single 2D mass-spring timestep, one mesh/dt/stiffness; iteration-axis (HW-independent). ADMM penalty ρ swept; the paper's w=½√k weight is one point. Anderson-ADMM uses a residual-increase safeguard (falls back to a plain ADMM step)._
