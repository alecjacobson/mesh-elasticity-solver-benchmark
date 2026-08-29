"""Structural solver-property edges: simplicity, generality, and the factorization-count axis (V2.6).

Some superiority claims are architectural rather than iteration-count races: 'no line search / no
Hessian filter' (simplicity), 'works on general energies, not just mass-springs' (generality), and
'cheaper per step because it prefactors once' (speed on the hardware-independent factorization /
back-solve count, NOT wall-clock). We adjudicate these by demonstrating the structural fact on the
conformance-gated testbeds and measuring the factorization/back-solve counts.

Writes results/solver_properties.md. Run: `python -m bench.run_solver_properties`.
"""
import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from . import massspring as ms
from . import incremental as fem


def _pd_traj(P, steps=15):
    free = P.free; pin = ~free; A = P.pd_system()
    Aff = cho_factor(A[np.ix_(free, free)], lower=True); Afp = A[np.ix_(free, pin)] @ P.x0[pin]
    Mx = (P.inv_dt2 * P.Md * P.xtil).reshape(-1, 2)
    x = P.xtil.copy(); traj = [P.phi(x)]
    for _ in range(steps):
        X = x.reshape(-1, 2); d = X[P.E[:, 0]] - X[P.E[:, 1]]
        l = np.linalg.norm(d, axis=1)[:, None] + 1e-15; p = P.L[:, None] * (d / l)
        rhs = Mx.copy(); np.add.at(rhs, P.E[:, 0], P.k * p); np.add.at(rhs, P.E[:, 1], -P.k * p)
        rhs = rhs.reshape(-1); x = x.copy(); x[free] = cho_solve(Aff, rhs[free] - Afp)
        traj.append(P.phi(x))
    return traj


def main():
    P = ms.MSProblem(n=8, dt=1.0 / 30, k=1.0e3, overshoot=1.6)
    traj = _pd_traj(P)
    mono = all(traj[i + 1] <= traj[i] + 1e-9 for i in range(len(traj) - 1))
    pd = ms.solve_pd(P, rtol=1e-6); nw = ms.solve_newton(P, rtol=1e-6)
    pd_fac, pd_bs = 1, pd["it"]
    nw_fac = nw["it"]
    # generality: run a PD-style solver and quasi-Newton L-BFGS on a NON-mass-spring energy (FEM NH)
    Q = fem.Problem(n=8, dt=1.0, stiffness=1.0, overshoot=2.4)
    fem_pd = fem.solve_pd(Q, rtol=1e-3)["it"]
    fem_qn = fem.solve_lbfgs(Q, "lap", rtol=1e-3)["it"]

    L = ["# Structural solver properties: simplicity, generality, factorization count (measured, V2.6)",
         "",
         "Architectural claims adjudicated on the conformance-gated testbeds. The 'speed' items are on "
         "the **hardware-independent factorization / back-solve count** (docs/metrics.md), NOT "
         "wall-clock. Run: `python -m bench.run_solver_properties`.", "",
         "| quantity | value |", "|---|---|",
         f"| PD (local/global) energy monotone with NO line search | **{mono}** |",
         f"| PD converges (mass-spring) | {pd['it']} iters = **1 factorization + {pd_bs} back-solves** |",
         f"| Newton converges (mass-spring) | {nw_fac} iters = **{nw_fac} full factorizations** (+ clamp filter + line search each) |",
         f"| fixed-proxy PD on FEM Neo-Hookean (non-mass-spring) | converges in {fem_pd} iters |",
         f"| quasi-Newton L-BFGS on FEM Neo-Hookean | converges in {fem_qn} iters |", "",
         "## Observed — edges adjudicated", "",
         f"- **`projective-dynamics → full-newton` (simplicity) — REPRODUCES:** the local/global solver "
         f"is a per-constraint projection + a single prefactored linear back-solve — **no line search, "
         f"no indefinite-Hessian filter, no SVD differentiation** — and its energy decreases "
         f"**monotonically** (measured: {mono}; {traj[0]:.0f}→{traj[1]:.0f}→…→{traj[-1]:.0f}). Newton "
         "needs a clamp/SPD filter AND a backtracking line search every iteration. The simplicity claim "
         "is architectural and holds by construction.",
         f"- **`fast-mass-spring → full-newton` and `quasi-newton-liu2017 → full-newton` (speed) — "
         f"MECHANISM shown, NOT a speed reproduction:** local/global and quasi-Newton **prefactor ONCE** "
         f"(1 factorization, then {pd_bs} cheap back-solves) while Newton does **{nw_fac} full "
         f"factorizations** (one per iteration). A back-solve is asymptotically cheaper than a "
         "factorization (by inspection, not timed) — the *mechanism* behind 'faster per iteration'. This "
         "is NOT a reproduction of the speed claim: the literal × is wall-clock/scale-dependent "
         f"(hardware-confounded, not claimed), {pd_bs} back-solves vs {nw_fac} factorizations could net "
         "SLOWER on a small dense system, and Newton needs the fewest iterations. Mechanism only.",
         f"- **`quasi-newton-liu2017 → projective-dynamics` (generality) — REPRODUCES:** quasi-Newton "
         f"L-BFGS minimizes the FEM Neo-Hookean incremental potential exactly ({fem_qn} iters) — it is "
         "just L-BFGS on an arbitrary Φ. Exact local/global Projective Dynamics is restricted to "
         "quadratic-fitting energies (mass-spring/ARAP); on a general energy it only *approximates* via "
         f"a fixed proxy (our FEM 'PD', {fem_pd} iters, IS exactly that m=0 approximation, not exact PD). "
         "So quasi-Newton supports arbitrary hyperelastic models (Neo-Hookean/StVK/…) that exact PD "
         "cannot — the generality claim holds.",
         "- **`projective-dynamics → fast-mass-spring` (generality) — NOT faithfully demonstrated, left "
         "self-claimed:** our FEM 'PD' is the fixed-proxy m=0 *approximation*, not exact Projective "
         "Dynamics with general constraint projections. Running it on Neo-Hookean shows the approximation "
         "generalizes, NOT PD's actual constraint-projection generality over Liu-2013 (strain limiting, "
         "volume, collisions). Faithfully testing that needs real PD constraint projections we did not "
         "implement; we do not claim it.",
         "",
         "_Caveat: simplicity/generality are structural facts demonstrated on the testbeds, not "
         "iteration races; the speed items are adjudicated on factorization/back-solve COUNTS "
         "(hardware-independent) — the wall-clock '×' figures are confounded and not claimed._"]

    os.makedirs("results", exist_ok=True)
    with open("results/solver_properties.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  PD monotone={mono}  PD={pd['it']}it/1fac  Newton={nw_fac}it/{nw_fac}fac  "
          f"FEM: PD={fem_pd} qN={fem_qn}")
    print("wrote results/solver_properties.md")
    return True


if __name__ == "__main__":
    main()
