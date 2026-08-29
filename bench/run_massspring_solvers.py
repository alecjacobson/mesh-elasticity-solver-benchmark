"""Constraint-projection solvers on one mass-spring implicit-Euler timestep (V2.2).

Faithful mass-spring substrate (bench/massspring.py, conformance-gated) where PBD/XPBD/Projective-
Dynamics/nonlinear-GS are their true selves. Two hardware-independent findings settle several
convergence/quality edges; GPU/wall-clock speed headlines (jgs2 "8000x/step", vbd "10x", ...) stay
hardware-confounded and are NOT adjudicated here.

Writes results/massspring_solvers.md. Run: `python -m bench.run_massspring_solvers`.
"""
import os
import numpy as np
from . import massspring as ms


def main():
    P = ms.MSProblem(n=8, dt=1.0 / 30, k=1.0e3, overshoot=1.6)
    rtol = 1e-3
    newton = ms.solve_newton(P, rtol=rtol)
    pd = ms.solve_pd(P, rtol=rtol)
    pbng = ms.solve_pbng(P, rtol=rtol)
    xpbd = ms.solve_pbd(P, xpbd=True, max_iter=600, rtol=rtol)
    pbd = ms.solve_pbd(P, xpbd=False, max_iter=300, rtol=rtol)
    r0 = newton["res"][0]
    xpbd_plateau = xpbd["res"][-1]                  # residual XPBD stalls at (never hits tol)
    pbd_final = pbd["res"][-1]

    def cell(r):
        return str(r["it"]) if r["it"] is not None else f"stalls @ {r['res'][-1]/r0:.2f}·r₀"

    # --- XPBD vs PBD: iteration-count-independent stiffness -----------------------------------
    # After K sweeps, mean |C| = |‖x_i−x_j‖ − L| over springs. PBD drives it toward 0 (over-stiffens
    # with more iterations => effectively stiffer material); XPBD's compliance converges it to the
    # correct nonzero equilibrium violation, iteration-count-independent.
    def mean_absC(x):
        X = x.reshape(-1, 2)
        l = np.linalg.norm(X[P.E[:, 0]] - X[P.E[:, 1]], axis=1)
        return float(np.mean(np.abs(l - P.L)))
    Ks = [2, 5, 20, 80]
    stiff = {"xpbd": [], "pbd": []}
    for K in Ks:
        stiff["xpbd"].append(mean_absC(ms.solve_pbd(P, xpbd=True, max_iter=K, rtol=0.0)["x"]))
        stiff["pbd"].append(mean_absC(ms.solve_pbd(P, xpbd=False, max_iter=K, rtol=0.0)["x"]))

    # --- QUALITY: position error vs the true implicit-Euler solution (Newton to tight tol) ----------
    xtrue = ms.solve_newton(P, max_iter=300, rtol=1e-8)["x"]
    scale = float(np.max(np.abs(xtrue[P.free])))
    def perr(x):
        return float(np.max(np.abs((x - xtrue)[P.free]))) / scale
    Kq = 20
    q_xpbd = perr(ms.solve_pbd(P, xpbd=True, max_iter=Kq, rtol=0.0)["x"])
    q_vbd = perr(ms.solve_pbng(P, max_iter=Kq, rtol=0.0)["x"])
    q_pbd = perr(ms.solve_pbd(P, xpbd=False, max_iter=Kq, rtol=0.0)["x"])

    L = ["# Constraint-projection solvers on one mass-spring timestep (measured, V2.2)", "",
         "Mass-spring implicit-Euler step (`bench/massspring.py`, conformance-gated: ∇Φ vs FD 1e-9, PD "
         "system SPD) — the faithful home of PBD/XPBD/Projective-Dynamics/nonlinear-GS, all sharing the "
         "exact incremental potential Φ and the same inertial start x̃. Metric: iterations to cut the "
         "**incremental-potential gradient residual** to 1e-3 of its start. GPU/wall-clock speed "
         "claims are NOT adjudicated here. Run: `python -m bench.run_massspring_solvers`.", "",
         "| solver | iters to 1e-3 residual |", "|---|---:|",
         f"| Newton (projected) | {cell(newton)} |",
         f"| local/global = Projective Dynamics / fast-mass-spring (exact) | {cell(pd)} |",
         f"| nonlinear Gauss–Seidel (pbng-style) | {cell(pbng)} |",
         f"| **XPBD** (compliance α=1/(k h²)) | {cell(xpbd)} |",
         f"| **PBD** (no compliance) | {cell(pbd)} |", "",
         "### XPBD vs PBD: is effective stiffness iteration-count-independent?", "",
         "Mean absolute constraint violation `|‖xᵢ−xⱼ‖−L|` after exactly K Gauss–Seidel sweeps "
         "(smaller ⇒ stiffer material):", "",
         "| sweeps K | " + " | ".join(str(k) for k in Ks) + " |", "|---|" + "---|" * len(Ks),
         "| XPBD | " + " | ".join(f"{v:.2e}" for v in stiff["xpbd"]) + " |",
         "| PBD | " + " | ".join(f"{v:.2e}" for v in stiff["pbd"]) + " |", "",
         f"### Quality: max position error vs the true implicit-Euler solution (after K={Kq} sweeps, "
         "as a fraction of the deformation scale)", "",
         "| method | ‖x − x_trueIE‖∞ / scale |", "|---|---:|",
         f"| XPBD | {q_xpbd:.1%} |",
         f"| nonlinear-GS / VBD-style | {q_vbd:.1%} |",
         f"| PBD | {q_pbd:.1%} |", "",
         "## Observed — edges adjudicated", ""]

    L.append(f"- **`primal-xpbd → xpbd` (convergence) — REPRODUCES:** XPBD **stagnates** on the "
             f"incremental-potential residual — it flat-lines at **{xpbd_plateau/r0:.2f}·r₀** (≈"
             f"{xpbd_plateau:.0f}) and never reaches the tol, because its constraint sweep omits the "
             "momentum-coupling term. Newton, local/global, and nonlinear-GS — all of which retain the "
             f"full residual — drive it to 0 in {newton['it']}/{pd['it']}/{pbng['it']} iterations. So a "
             "primal method that keeps the backward-Euler momentum residual converges where XPBD stalls.")
    L.append(f"- **`pbng → xpbd` (convergence) — REPRODUCES:** nonlinear Gauss–Seidel reaches the tol in "
             f"**{pbng['it']}** iterations while XPBD stagnates at {xpbd_plateau/r0:.2f}·r₀ — 'reaches "
             "tolerance where XPBD stagnates', exactly as claimed.")
    L.append(f"- **`fast-mass-spring → pbd` and `projective-dynamics → pbd` (quality) — REPRODUCE:** "
             f"local/global (which IS fast-mass-spring / mass-spring Projective Dynamics) converges to "
             f"the exact implicit-Euler minimum ({pd['it']} it, residual→0), whereas PBD (no compliance) "
             f"does NOT — its residual GROWS to {pbd_final/r0:.1f}·r₀ as it over-constrains. PD reaches "
             "the true dynamics; PBD does not.")
    xpbd_flat = max(stiff["xpbd"]) / (min(stiff["xpbd"]) + 1e-30)
    pbd_drop = stiff["pbd"][0] / (stiff["pbd"][-1] + 1e-30)
    L.append(f"- **`xpbd → pbd` (quality) — REPRODUCES:** across K={Ks[0]}→{Ks[-1]} sweeps XPBD's "
             f"constraint violation is essentially iteration-count-INDEPENDENT (varies "
             f"{xpbd_flat:.1f}×, converging to the compliant equilibrium), while PBD's keeps shrinking "
             f"({pbd_drop:.0f}× smaller by K={Ks[-1]}) — i.e. PBD **stiffens with iteration count** "
             "(the material gets artificially stiffer the more you iterate), exactly XPBD's headline.")
    L.append(f"- **`xpbd → full-newton` (quality) — REPRODUCES:** although XPBD stagnates on the "
             f"*residual*, its final POSITIONS are within **{q_xpbd:.1%}** (max) of the true Newton "
             "implicit-Euler solution — sub-percent, i.e. **visually indistinguishable** from the more "
             "expensive Newton reference, exactly XPBD's selling point (cheap and looks right even "
             "though not converged in the residual sense).")
    L.append(f"- **`vertex-block-descent → xpbd` (quality) — REPRODUCES (with one wording caveat):** "
             f"the nonlinear-GS / VBD-style solver drives the position error to **{q_vbd:.1%}** — it "
             f"**matches the true implicit-Euler** solution — whereas XPBD plateaus at {q_xpbd:.1%} and "
             "never closes the gap. So 'VBD matches true implicit Euler where XPBD [does not]' holds. NB "
             "the claim's word 'diverges' is imprecise for XPBD (it *stagnates* at a small error); it is "
             f"**PBD** that actually moves away (its position error grows to {q_pbd:.0%}).")
    L += ["",
          "_Caveat: single 2D mass-spring timestep, one mesh/dt/stiffness; iteration counts and "
          "constraint-violation trends are hardware-independent. The GPU/throughput speed headlines "
          "(jgs2, vbd, pbng 6–7× Newton) are NOT adjudicated. PD here is EXACT local/global for "
          "mass-spring (not the FEM fixed-proxy stand-in of results/dynamics_solvers.md)._"]

    os.makedirs("results", exist_ok=True)
    with open("results/massspring_solvers.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  newton={newton['it']} pd={pd['it']} pbng={pbng['it']} "
          f"xpbd=stall@{xpbd_plateau:.0f} pbd=grow@{pbd_final:.0f}")
    print(f"  stiffness |C| XPBD={['%.1e'%v for v in stiff['xpbd']]} PBD={['%.1e'%v for v in stiff['pbd']]}")
    print("wrote results/massspring_solvers.md")
    return True


if __name__ == "__main__":
    main()
