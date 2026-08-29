# Structural solver properties: simplicity, generality, factorization count (measured, V2.6)

Architectural claims adjudicated on the conformance-gated testbeds. The 'speed' items are on the **hardware-independent factorization / back-solve count** (docs/metrics.md), NOT wall-clock. Run: `python -m bench.run_solver_properties`.

| quantity | value |
|---|---|
| PD (local/global) energy monotone with NO line search | **True** |
| PD converges (mass-spring) | 21 iters = **1 factorization + 21 back-solves** |
| Newton converges (mass-spring) | 5 iters = **5 full factorizations** (+ clamp filter + line search each) |
| fixed-proxy PD on FEM Neo-Hookean (non-mass-spring) | converges in 10 iters |
| quasi-Newton L-BFGS on FEM Neo-Hookean | converges in 6 iters |

## Observed — edges adjudicated

- **`projective-dynamics → full-newton` (simplicity) — REPRODUCES:** the local/global solver is a per-constraint projection + a single prefactored linear back-solve — **no line search, no indefinite-Hessian filter, no SVD differentiation** — and its energy decreases **monotonically** (measured: True; 2380→1975→…→1967). Newton needs a clamp/SPD filter AND a backtracking line search every iteration. The simplicity claim is architectural and holds by construction.
- **`fast-mass-spring → full-newton` and `quasi-newton-liu2017 → full-newton` (speed) — REPRODUCE on the factorization axis:** local/global and quasi-Newton **prefactor ONCE** (1 factorization, then 21 cheap back-solves) while Newton does **5 full factorizations** (one per iteration). A back-solve is far cheaper than a factorization, so a PD/quasi-Newton *iteration* is much cheaper than a Newton *iteration* — the mechanism behind 'much faster initial work-to-error' and '>10× faster than one Newton iteration'. The literal × is wall-clock/scale-dependent (hardware-confounded); the HW-independent count carries the mechanism. (Newton still needs the fewest iterations — the trade is iterations vs per-iteration cost.)
- **`projective-dynamics → fast-mass-spring` (generality) — REPRODUCES:** the same PD machinery runs on a FEM **Neo-Hookean** energy (10 iters), not only linear mass-springs — general nodal systems, exactly Bouaziz-2014's generalization of Liu-2013.
- **`quasi-newton-liu2017 → projective-dynamics` (generality) — REPRODUCES:** quasi-Newton L-BFGS minimizes the FEM Neo-Hookean incremental potential exactly (6 iters); exact local/global Projective Dynamics is restricted to quadratic-fitting energies (mass-spring/ARAP) and only *approximates* a general energy via a fixed proxy. So quasi-Newton supports arbitrary hyperelastic models (Neo-Hookean/StVK/…) that exact PD cannot — the generality claim holds.

_Caveat: simplicity/generality are structural facts demonstrated on the testbeds, not iteration races; the speed items are adjudicated on factorization/back-solve COUNTS (hardware-independent) — the wall-clock '×' figures are confounded and not claimed._
