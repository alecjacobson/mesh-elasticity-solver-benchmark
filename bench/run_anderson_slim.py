"""Anderson acceleration OF SLIM — P5.2 edge #9 (anderson-geometry -> slim, convergence).

The claim (Peng et al. 2018, Fig.10) is NOT "Anderson replaces SLIM" but "Anderson *accelerates*
SLIM's optimization" — it wraps SLIM's fixed-point map and reaches the same minimum in fewer
iterations. We test exactly that, with the OFFICIAL libigl SLIM as the base map:

  G(UV) = one official `igl.slim_solve` step, made a PURE fixed-point map by re-`slim_precompute`-ing
  from the current iterate (verified to reproduce continuous SLIM to 0.0). We then run the SAME
  map-agnostic `anderson_accelerate` core (Peng et al.: m-history min-norm least squares + energy
  safeguard) at m=0 (== plain SLIM, a fair same-map baseline) and m>0 (accelerated), and count
  iterations to shrink the fixed-point residual (SD gradient norm) by a fixed factor.

To leave room for acceleration we use a HARD instance (large shear + perturbed init) where plain
SLIM needs many iterations — on an easy instance SLIM is already near-quadratic and nothing can help.
Writes results/anderson_slim.md. Run: `python -m bench.run_anderson_slim` (needs libigl).
"""
import os
import numpy as np


def _hard_scenario(nx=12, k=0.8):
    """Boundary pinned to a NON-AFFINE bend (rotate each vertex by angle k·(x−½) about the centre);
    whole mesh initialised to the same bend so the start is injective (no flips) but far from the
    symmetric-Dirichlet minimum. An affine boundary would make the affine interior the exact
    minimiser (1 SLIM step, no headroom); the bend's spatially-varying stretch gives a slowly-
    contracting fixed point where plain SLIM needs ~380 iterations to 1e-3 residual — real room to accelerate."""
    from .mesh import grid_mesh, boundary_mask
    rest, tris = grid_mesh(nx, nx)
    bmask = boundary_mask(rest)
    xr = rest[:, 0] - 0.5; yr = rest[:, 1] - 0.5; th = k * xr
    x = np.stack([np.cos(th) * xr - np.sin(th) * yr, np.sin(th) * xr + np.cos(th) * yr], 1)
    free = ~np.repeat(bmask, 2)
    return rest, tris, x.reshape(-1), free, bmask


def _iters_to(energies, E0, Estar, rtol=1e-4):
    span = (E0 - Estar) + 1e-30
    for k, E in enumerate(energies):
        if (E - Estar) / span < rtol:
            return k
    return None


def main():
    try:
        import igl
    except ImportError:
        print("[anderson-slim] SKIP: libigl not installed"); return True
    from .solver import solve, assemble, energy_only
    from .mesh import rest_quantities
    from .energy import element_terms as sd
    from .world1 import anderson_accelerate

    rest, tris, x0, free, bmask = _hard_scenario()
    Bs, areas = rest_quantities(rest, tris)
    bidx = np.where(bmask)[0].astype(np.int32)
    bc = x0.reshape(-1, 2)[bidx].astype(np.float64)   # pin to the BENT boundary (matches x0 / Newton ref)
    V3 = np.hstack([rest, np.zeros((rest.shape[0], 1))]).astype(np.float64)
    F = tris.astype(np.int32)

    def sd_e(xflat):
        return energy_only(xflat, tris, Bs, areas, sd)

    def resid(xflat):
        _, g, _ = assemble(xflat, tris, Bs, areas, "none", sd)
        if g is None:                                # inverted element (SD barrier +inf): not converged
            return np.inf
        return float(np.max(np.abs(g[free])))

    # pure SLIM fixed-point map on the flat coordinate vector
    def G(xflat):
        uv = xflat.reshape(-1, 2).astype(np.float64)
        d = igl.slim_precompute(V3, F, uv, igl.SYMMETRIC_DIRICHLET, bidx, bc, 1e8)
        return igl.slim_solve(d, 1).reshape(-1)

    # VERIFY the re-precompute map reproduces CONTINUOUS SLIM (the load-bearing claim): run one
    # slim_precompute + N continuous slim_solve steps, and the re-precompute-per-step G path, from
    # the same start; the max coordinate discrepancy must be ~0 or the 'plain SLIM' baseline (and
    # thus the speedup) is not faithful (review-P5.2).
    d_cont = igl.slim_precompute(V3, F, x0.reshape(-1, 2).astype(np.float64),
                                 igl.SYMMETRIC_DIRICHLET, bidx, bc, 1e8)
    uvc = x0.reshape(-1, 2).astype(np.float64); xr = x0.copy(); reprecompute_diff = 0.0
    for _ in range(20):
        uvc = igl.slim_solve(d_cont, 1)
        xr = G(xr)
        reprecompute_diff = max(reprecompute_diff, float(np.max(np.abs(xr - uvc.reshape(-1)))))

    # independent high-accuracy reference minimum (hard-constrained Newton)
    rn = solve(x0, tris, Bs, areas, free, "clamp", eterms=sd, tol=1e-9, max_iter=300)
    Estar = rn["final_energy"]; E0 = sd_e(x0)

    # convergence is measured on the FIXED-POINT RESIDUAL (gradient inf-norm), NOT energy: the
    # symmetric-Dirichlet energy saturates in ~1 SLIM step here, but the residual has the long
    # slowly-contracting tail (~380 it) that acceleration actually acts on.
    def _iters_to_grad(ghist, rtol=1e-3):
        g0 = ghist[0]
        for k, g in enumerate(ghist):
            if g <= rtol * g0:
                return k
        return None

    def _iters_abs(ghist, atol=1e-4):               # absolute gradient tol (like the anderson->LG edge)
        for k, g in enumerate(ghist):
            if g < atol:
                return k
        return None

    MAXIT = 400
    runs = {}
    for m in (0, 5):
        r = anderson_accelerate(G, sd_e, resid, x0, free, m=m, max_iter=MAXIT, tol=1e-9)
        ghist = [e["grad_inf"] for e in r["log"]]
        runs[m] = {"it": _iters_to_grad(ghist), "it_abs": _iters_abs(ghist), "status": r["status"],
                   "final": r["final_energy"], "nlog": len(ghist), "g0": ghist[0]}

    base = runs[0]["it"]
    L = ["# Anderson acceleration OF SLIM — does it cut iterations? (measured, P5.2 #9)", "",
         "Wraps the **official libigl SLIM** fixed-point map in the map-agnostic "
         "`anderson_accelerate` core (Peng et al. 2018). `m=0` is plain SLIM (same-map baseline); "
         "`m>0` is Anderson-accelerated. Hard instance (12×12, non-affine bend k=0.8, injective "
         "start) so plain SLIM's fixed point contracts slowly. We report iterations to cut the "
         "**fixed-point residual** (symmetric-Dirichlet gradient ‖·‖∞) to 1e-3 of its start AND to an "
         "**absolute** tol ‖·‖∞<1e-4 (the criterion the validated anderson→local-global edge uses, "
         "for consistency). The SD *energy* saturates in ~1 SLIM step here, so an energy criterion "
         "shows nothing — the residual carries the tail acceleration acts on; we report both so the "
         "metric choice is transparent, not selected. The SLIM map is made pure by "
         f"re-`slim_precompute`-ing each step; we VERIFY this equals continuous SLIM: max coordinate "
         f"discrepancy over 20 steps = **{reprecompute_diff:.1e}** (so the plain-SLIM baseline is "
         "faithful, not an inflated re-precompute artifact). "
         "Run: `python -m bench.run_anderson_slim`.", "",
         "| Anderson history m | iters to residual-tol (1e-3 rel) | iters to ‖g‖<1e-4 (abs) | speedup (rel/abs) |",
         "|---|---:|---:|---:|"]
    for m in (0, 5):
        it = runs[m]["it"]; ita = runs[m]["it_abs"]
        base_a = runs[0]["it_abs"]
        sp_r = (f"{base / it:.1f}×" if (it and base) else "—")
        sp_a = (f"{base_a / ita:.1f}×" if (ita and base_a) else "—")
        tag = " (plain SLIM)" if m == 0 else ""
        L.append(f"| {m}{tag} | {it if it is not None else runs[m]['status']} | "
                 f"{ita if ita is not None else 'did-not-reach'} | {sp_r} / {sp_a} |")

    best_m = min((m for m in (5,) if runs[m]["it"]),
                 key=lambda m: runs[m]["it"], default=None)
    L += ["", "## Observed", ""]
    if base and best_m and runs[best_m]["it"] < base:
        L.append(f"- **Anderson accelerates SLIM (instance-dependent):** on this deliberately "
                 f"slow-contracting instance plain SLIM needs **{base}** iterations to the residual "
                 f"tol; Anderson (m={best_m}) needs **{runs[best_m]['it']}** — a "
                 f"**{base / runs[best_m]['it']:.0f}× reduction** on the SAME official-SLIM map, "
                 f"confirmed on the absolute-tol criterion too "
                 f"({runs[0]['it_abs']}→{runs[best_m]['it_abs']}). The DIRECTION (Anderson is a "
                 "wrapper that speeds SLIM up, not a replacement) is the result; the MAGNITUDE is "
                 "instance-selected — we chose a bend that makes plain SLIM's tail long, so this is "
                 "an *up-to* figure, not a typical speedup.")
        L.append("- Why the effect exists here and vanishes elsewhere: this instance sits on SLIM's "
                 "**slowly-contracting linear tail** (residual creeps down over hundreds of "
                 "iterations), which is exactly what Anderson's history extrapolation collapses; on "
                 "an easy instance where SLIM is already near-quadratic there is nothing to "
                 "accelerate. So the edge is regime-dependent: `qualified/indicative` on a single "
                 "hand-picked instance (m∈{0,5} only, no multi-seed/mesh sweep, no m-profile) — NOT "
                 "the multi-condition evidence the repo reserves for `validated`.")
    elif base:
        L.append(f"- **No acceleration observed here:** plain SLIM already reaches the residual tol in "
                 f"**{base}** iterations and no history m beats it on this instance — SLIM's "
                 "Gauss-Newton-like steps are already near-optimal, leaving little for Anderson to "
                 "extrapolate. The edge is **not distinguished** on this instance (would need a "
                 "regime where SLIM's fixed point is more slowly contracting).")
    else:
        L.append("- Inconclusive: plain SLIM did not reach the tol within the iteration budget.")
    L += ["",
          "_Caveat: single hard 2D instance; SLIM's soft-BC boundary drift is not re-checked here "
          "(see results/slim.md, drift 4.4e-16); official libigl SLIM grounds the base map (D3). "
          "Wall-clock is not the metric — each Anderson step adds a small least-squares over the "
          "SLIM step, and re-precompute inflates our Python wall-clock but not the iteration count._"]

    os.makedirs("results", exist_ok=True)
    with open("results/anderson_slim.md", "w") as f:
        f.write("\n".join(L) + "\n")
    for m in (0, 5):
        print(f"  m={m:2d}: it={runs[m]['it']} status={runs[m]['status']} final={runs[m]['final']:.6f}")
    print("wrote results/anderson_slim.md")
    return True


if __name__ == "__main__":
    main()
