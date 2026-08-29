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
- **`fast-mass-spring → full-newton` and `quasi-newton-liu2017 → full-newton` (speed) — MECHANISM shown, NOT a speed reproduction:** local/global and quasi-Newton **prefactor ONCE** (1 factorization, then 21 cheap back-solves) while Newton does **5 full factorizations** (one per iteration). A back-solve is asymptotically cheaper than a factorization (by inspection, not timed) — the *mechanism* behind 'faster per iteration'. This is NOT a reproduction of the speed claim: the literal × is wall-clock/scale-dependent (hardware-confounded, not claimed), 21 back-solves vs 5 factorizations could net SLOWER on a small dense system, and Newton needs the fewest iterations. Mechanism only.
- **`quasi-newton-liu2017 → projective-dynamics` (generality) — REPRODUCES:** quasi-Newton L-BFGS minimizes the FEM Neo-Hookean incremental potential exactly (6 iters) — it is just L-BFGS on an arbitrary Φ. Exact local/global Projective Dynamics is restricted to quadratic-fitting energies (mass-spring/ARAP); on a general energy it only *approximates* via a fixed proxy (our FEM 'PD', 10 iters, IS exactly that m=0 approximation, not exact PD). So quasi-Newton supports arbitrary hyperelastic models (Neo-Hookean/StVK/…) that exact PD cannot — the generality claim holds.
- **`projective-dynamics → fast-mass-spring` (generality) — NOT faithfully demonstrated, left self-claimed:** our FEM 'PD' is the fixed-proxy m=0 *approximation*, not exact Projective Dynamics with general constraint projections. Running it on Neo-Hookean shows the approximation generalizes, NOT PD's actual constraint-projection generality over Liu-2013 (strain limiting, volume, collisions). Faithfully testing that needs real PD constraint projections we did not implement; we do not claim it.

_Caveat: simplicity/generality are structural facts demonstrated on the testbeds, not iteration races; the speed items are adjudicated on factorization/back-solve COUNTS (hardware-independent) — the wall-clock '×' figures are confounded and not claimed._
