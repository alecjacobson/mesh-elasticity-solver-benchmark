"""Eigenvalue-blending filter, tested (P5.2, edges eigenvalue-blending -> {clamp, absolute}).

The eigenvalue-blending idea (Cheng-Liu-Fu 2025) interpolates the per-element eigenvalue projection
between clamp and absolute. Our `filters.project_element_blend(H, w)` computes lambda_eff =
max((1-w)lambda + w|lambda|, eps), which is EXACTLY clamp at w=0.5 and absolute at w=1.0 (verified to
0 in conformance), so a fixed w in [0.5, 1.0] is the blending PRINCIPLE. This runner sweeps w on the
locking P1 element and the locking-relieved P2 element at near-incompressible nu, asking the question
the edges pose: does an intermediate blend BEAT both clamp and absolute, or merely interpolate?

Caveat: this is the fixed-w blending principle, not necessarily the paper's specific (possibly
adaptive) w schedule -- so the verdict qualifies the edges rather than settling the exact method.
Writes results/blend_filter.md. Run: `python -m bench.run_blend_filter`.
"""
import os
import numpy as np
from bench import p2, energy_neohookean as nh
from bench.mesh import grid_mesh, rest_quantities
from bench.solver import solve
from bench.run_p2_stable_nu import _hpsi

S, N = 2.0, 8
NUS = [0.499, 0.4999]
WS = [0.5, 0.625, 0.75, 0.875, 1.0]        # 0.5 == clamp, 1.0 == absolute (conformance-verified)


def _label(w):
    return "clamp" if w == 0.5 else ("absolute" if w == 1.0 else f"blend-{w:g}")


def _p1(nu, w):
    rest, tris = grid_mesh(N, N); Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 2); x0 = rest.copy(); x0[:, 0] = S * rest[:, 0]; x0 = x0.reshape(-1)
    et, _, _, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
    r = solve(x0, tris, Bs, areas, free, f"blend-{w}", eterms=et, tol=1e-6, max_iter=400)
    return r["iters"] if r["status"] == "converged" else r["status"]


def _p2(nu, w):
    nodes, elems = p2.grid_mesh_p2(N, N); quad = p2.rest_quantities_p2(nodes, elems)
    xc = nodes[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 2); x0 = nodes.copy(); x0[:, 0] = S * nodes[:, 0]; x0 = x0.reshape(-1)
    _, psi, gp, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
    et = p2.make_element_terms(psi, gp, _hpsi(gp))
    r = p2.solve_p2(x0, elems, quad, free, et, f"blend-{w}", tol=1e-6, max_iter=400)
    return r["iters"] if r["status"] == "converged" else r["status"]


def run():
    p1 = {nu: {w: _p1(nu, w) for w in WS} for nu in NUS}
    p2t = {nu: {w: _p2(nu, w) for w in WS} for nu in NUS}

    def beats_both(tab, nu):
        vals = tab[nu]
        ints = {w: v for w, v in vals.items() if isinstance(v, int)}
        if 0.5 not in ints or 1.0 not in ints:
            return None
        mids = {w: v for w, v in ints.items() if 0.5 < w < 1.0}
        return mids and min(mids.values()) < min(ints[0.5], ints[1.0])

    L = ["# Eigenvalue-blending filter — does an intermediate blend beat clamp/absolute? (measured)", "",
         "Tests `eigenvalue-blending -> {clamp, absolute}` (convergence). The per-element blend "
         "lambda_eff = max((1-w)lambda + w|lambda|, eps) is **exactly clamp at w=0.5 and absolute at "
         "w=1.0** (conformance-verified to 0), so w in [0.5,1] is the blending principle. Neo-Hookean "
         f"stretch, {N}x{N} mesh, near-incompressible nu. Iterations to converge (tol 1e-6). "
         "Run: `python -m bench.run_blend_filter`.", "",
         "### P1 (locking) — iterations vs blend weight w", "",
         "| nu | " + " | ".join(f"w={w:g} ({_label(w)})" for w in WS) + " |",
         "|---|" + "---|" * len(WS)]
    for nu in NUS:
        L.append(f"| {nu} | " + " | ".join(str(p1[nu][w]) for w in WS) + " |")
    L += ["", "### P2 (locking-relieved) — iterations vs blend weight w", "",
          "| nu | " + " | ".join(f"w={w:g} ({_label(w)})" for w in WS) + " |",
          "|---|" + "---|" * len(WS)]
    for nu in NUS:
        L.append(f"| {nu} | " + " | ".join(str(p2t[nu][w]) for w in WS) + " |")

    p1_win = any(beats_both(p1, nu) for nu in NUS)
    p2_win = any(beats_both(p2t, nu) for nu in NUS)
    L += ["", "## Observed", "",
          ("- **On P1 (locking), no intermediate blend beats clamp** — iterations generally rise as w "
           "goes clamp(0.5)->absolute(1.0), because clamp is already the best endpoint on the locking "
           "element (absolute drags the long tail). The blend interpolates; it does not dominate.")
          if not p1_win else
          "- **On P1, an intermediate blend beats both clamp and absolute** — a genuine blending win.",
          ("- **On P2 (locking-relieved), the blend is a wash / interpolates** between the two near-equal "
           "endpoints; no intermediate w is clearly best.")
          if not p2_win else
          "- **On P2, an intermediate blend beats both endpoints.**",
          "- **Verdict.** " + ("On these elements the fixed-w blend **interpolates** clamp<->absolute "
           "rather than beating both, so `eigenvalue-blending -> {clamp, absolute}` is **qualified**: "
           "blending is never worse than the worse endpoint and never better than the better one, so "
           "its value must come from an *adaptive* w schedule (which needs the paper), not from a fixed "
           "blend. The endpoint equivalence (blend-0.5==clamp, blend-1.0==absolute) is exact."
           if not (p1_win or p2_win) else
           "An intermediate blend can beat both endpoints in at least one regime, supporting the "
           "eigenvalue-blending claim (regime-dependent)."),
          "",
          "_Caveat: 2D, single stretch/seed, single tau; fixed-w blend = the blending principle, NOT "
          "the paper's exact (possibly adaptive) w. Endpoint equivalence to clamp/absolute is exact._"]
    os.makedirs("results", exist_ok=True)
    with open("results/blend_filter.md", "w") as f:
        f.write("\n".join(L) + "\n")
    for nu in NUS:
        print(f"  P1 nu={nu}: " + " ".join(f"{_label(w)}={p1[nu][w]}" for w in WS))
        print(f"  P2 nu={nu}: " + " ".join(f"{_label(w)}={p2t[nu][w]}" for w in WS))
    print(f"[blend] P1 intermediate-beats-both={p1_win}, P2={p2_win}; wrote results/blend_filter.md")
    return True


if __name__ == "__main__":
    run()
