# Inner-solver convergence on one incremental-potential timestep (measured, V2.1)

Many simulation-accelerator methods are different inner minimizers of the SAME implicit-Euler incremental potential `Phi(x)=½h⁻²(x−x̃)ᵀM(x−x̃)+E(x)` (Neo-Hookean E; `bench/incremental.py`, conformance-gated: ∇Phi vs FD 1e-9, VBD block == Hessian block 0). Hard regime (dt=1, stiff overshoot timestep) so the fixed-point methods spread. Metric: **iterations to cut the gradient residual to 1e-3 of its start** — a hardware-independent count. Speed/GPU/wall-clock claims are NOT adjudicated here (they stay hardware-confounded). Run: `python -m bench.run_dynamics_solvers`.

> Note on labels: **`fixed-proxy descent`** below is the fixed-metric step `x←x−A₀⁻¹∇Φ`, A₀=M/h²+H_rest — this IS the m=0 case of Liu-2017 quasi-Newton, and canonical Projective Dynamics is its special case for a *fitting* energy (constant global system). We do NOT claim it is bit-for-bit PD; the quasi-newton→PD edge is a genuine self-ablation (m=5 vs m=0 of the same scheme), the chebyshev→PD edge accelerates *this* proxy map (see caveat). 

| method | iters to 1e-3 residual |
|---|---:|
| Newton (projected) | 3 |
| fixed-proxy descent (Liu-2017 m=0; PD-style) | 10 |
| Chebyshev-accelerated fixed-proxy (ρ=0.9, tuned) | 7 |
| Anderson-accelerated fixed-proxy (m=5, no ρ) | 6 |
| preconditioned nonlinear-CG (same A0 proxy) | 8 |
| quasi-Newton L-BFGS, Laplacian init (m=5) | 6 |
| quasi-Newton L-BFGS, Laplacian init (m=2) | 7 |
| plain L-BFGS, scaled-identity init (m=5) | 78 |

### Vertex Block Descent: Gauss–Seidel vs UNDER-RELAXED Jacobi (residual after 24 sweeps / start; a *sweep* visits all vertices and is NOT one global-solve iteration)

| VBD variant | residual reduction after budget |
|---|---:|
| VBD Gauss–Seidel | 0.52× |
| VBD Jacobi (ω-relaxed) | 0.93× |

## Observed — convergence edges adjudicated

- **`quasi-newton-liu2017 → l-bfgs` (convergence) — REPRODUCES (strong):** L-BFGS with the Laplacian/mass initial metric A₀=M/h²+H_rest reaches the tol in **6** iterations versus **78** for scaled-identity L-BFGS — the init is decisive, exactly the paper's claim.
- **`quasi-newton-liu2017 → projective-dynamics` (convergence) — REPRODUCES:** adding L-BFGS history (m=5, **6** it) over the m=0 fixed-metric PD step (**10** it) strictly improves per-iteration convergence on the same proxy.
- **`chebyshev-semi-iterative → projective-dynamics` (convergence) — reproduces the DIRECTION (ρ-tuned, modest):** Chebyshev acceleration of the fixed-proxy map reaches the tol in **7** it versus **10** for the plain proxy step. NB the accelerated object here is the quasi-Newton proxy fixed point, not canonical PD's constant-system iteration, and the 7-vs-10 magnitude is at a HAND-TUNED ρ=0.9 (a bad ρ stalls it; the safeguard falls back to the plain step so it can never look *worse*). So the ≥1-order headline is NOT reproduced — only the 'acceleration helps' direction.
- **`quasi-newton-liu2017 → chebyshev` (convergence) — the 'no-tuning' half only:** a short history (m=2, **7** it) TIES Chebyshev (**7** it) here — so 'even m=2 slightly faster' is a TIE on this instance, NOT reproduced; what DOES hold is the substantive half — L-BFGS needs no spectral-radius estimate, Chebyshev does.
- **`vertex-block-descent → jacobi` (convergence) — REPRODUCES:** on a fixed 24-sweep budget VBD **Gauss–Seidel** cuts the residual to **0.52×** while block **Jacobi**, given the standard under-relaxation (ω=1/(1+valence)) it needs, only reaches **0.93×** — sequential (Gauss-Seidel) block updates converge faster than simultaneous (Jacobi) ones, as claimed. (Earlier draft had un-relaxed Jacobi diverge; that was a strawman — fixed.)
- **`anderson-geometry → chebyshev-semi-iterative` (convergence) — a TIE on speed + a real simplicity edge:** accelerating the SAME PD fixed point, Anderson (m=5) reaches the tol in **6** iterations vs Chebyshev's **7** — a 1-iteration gap on one instance is noise, NOT a reproduction of 'faster'. What is real: Anderson needs no spectral-radius estimate, where Chebyshev does.
- **`chebyshev-semi-iterative → nonlinear-conjugate-gradient` (convergence) — NOT supported (ordering flips with mesh):** with the SAME A0 preconditioner, at this mesh Chebyshev needs **7** and preconditioned nonlinear-CG **8** — but the ordering REVERSES on neighbouring meshes (n=6: Chebyshev 8, CG 7, i.e. CG beats Chebyshev; n=10/12 they tie). So the strict claim 'CG rate can't exceed Chebyshev's' is false on a nearby instance. Only two things are robust: both need the shared preconditioner (un-preconditioned CG ~140), and CG's recurrence costs ~2 extra inner products per iterate.
- **`vertex-block-descent → l-bfgs` / `→ full-newton` — NOT reproduced for *plain* VBD:** plain VBD-GS reduces the residual only 0.52× in 24 sweeps whereas L-BFGS and Newton fully converge in 6 and 3 *iterations* (a VBD sweep is NOT one global-solve iteration, so this is not an equal-work comparison — it only says plain serial VBD converges slowly). The paper's claim is for *accelerated* VBD and rests on GPU parallelism/wall-clock, out of this harness's hardware-independent reach.

_Caveat: single 2D timestep, one mesh/dt/stiffness; iteration counts are hardware-independent but the *speed*/GPU/throughput headlines (jgs2 8000×/step, VBD 10× XPBD, ...) are not adjudicable here and stay hardware-confounded. PD here is the fixed-proxy generalization (Liu 2017 m=0), exact local/global only for a fitting energy._
