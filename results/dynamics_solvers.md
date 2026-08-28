# Inner-solver convergence on one incremental-potential timestep (measured, V2.1)

Many simulation-accelerator methods are different inner minimizers of the SAME implicit-Euler incremental potential `Phi(x)=½h⁻²(x−x̃)ᵀM(x−x̃)+E(x)` (Neo-Hookean E; `bench/incremental.py`, conformance-gated: ∇Phi vs FD 1e-9, VBD block == Hessian block 0). Hard regime (dt=1, stiff overshoot timestep) so the fixed-point methods spread. Metric: **iterations to cut the gradient residual to 1e-3 of its start** — a hardware-independent count. Speed/GPU/wall-clock claims are NOT adjudicated here (they stay hardware-confounded). Run: `python -m bench.run_dynamics_solvers`.

| method | iters to 1e-3 residual |
|---|---:|
| Newton (projected) | 3 |
| Projective Dynamics / quasi-Newton m=0 | 10 |
| Chebyshev-accelerated PD | 7 |
| quasi-Newton L-BFGS, Laplacian init (m=5) | 6 |
| quasi-Newton L-BFGS, Laplacian init (m=2) | 7 |
| plain L-BFGS, scaled-identity init (m=5) | 78 |

### Vertex Block Descent: Gauss–Seidel vs Jacobi (per-sweep rate, capped at 24 sweeps)

| VBD variant | residual after budget / start |
|---|---:|
| VBD Gauss–Seidel | 0.52 |
| VBD Jacobi | 9.75 |

## Observed — convergence edges adjudicated

- **`quasi-newton-liu2017 → l-bfgs` (convergence) — REPRODUCES (strong):** L-BFGS with the Laplacian/mass initial metric A₀=M/h²+H_rest reaches the tol in **6** iterations versus **78** for scaled-identity L-BFGS — the init is decisive, exactly the paper's claim.
- **`quasi-newton-liu2017 → projective-dynamics` (convergence) — REPRODUCES:** adding L-BFGS history (m=5, **6** it) over the m=0 fixed-metric PD step (**10** it) strictly improves per-iteration convergence on the same proxy.
- **`chebyshev-semi-iterative → projective-dynamics` (convergence) — REPRODUCES:** Chebyshev acceleration of the PD fixed point reaches the tol in **7** it versus plain PD's **10** — the ≥1-order speedup direction holds (magnitude here is modest; Chebyshev needs a spectral-radius estimate, which is the next edge).
- **`quasi-newton-liu2017 → chebyshev` (convergence) — MIXED:** even a short history (m=2, **7** it) is comparable to Chebyshev (**7** it) — and unlike Chebyshev needs no spectral-radius estimate.
- **`vertex-block-descent → jacobi` (convergence) — REPRODUCES (decisive):** on a fixed 24-sweep budget VBD **Gauss–Seidel** cuts the residual to **0.52×** its start (converging) while block **Jacobi** DIVERGES to 9.75× — Gauss-Seidel block updates converge where Jacobi does not.
- **`vertex-block-descent → l-bfgs` / `→ full-newton` (convergence/speed) — NOT reproduced for *plain* VBD:** plain VBD-GS is far slower per convergence (residual only 0.52× after 24 sweeps) than L-BFGS (**6** it) or Newton (**3** it). The paper's claim is for *accelerated* VBD (a momentum variant) and rests on GPU parallelism/wall-clock, not serial iteration count — out of this harness's hardware-independent reach.

_Caveat: single 2D timestep, one mesh/dt/stiffness; iteration counts are hardware-independent but the *speed*/GPU/throughput headlines (jgs2 8000×/step, VBD 10× XPBD, ...) are not adjudicable here and stay hardware-confounded. PD here is the fixed-proxy generalization (Liu 2017 m=0), exact local/global only for a fitting energy._
