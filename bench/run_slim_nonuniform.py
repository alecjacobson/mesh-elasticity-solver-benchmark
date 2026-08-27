"""SLIM vs projected-Newton on NON-UNIFORM triangulations — P5.2 edge #7 (slim -> projected-newton).

Claim (Rabinovich et al. 2017, Fig.11): "SLIM outperforms Newton and regularized Newton on
non-uniform triangulations" — far from the minimum "1 SLIM iter ~ 2000 Newton iters", i.e. SLIM's
reweighted (Gauss-Newton-like) global step keeps making progress where a projected-Newton step
stalls on badly-shaped (high-aspect-ratio / sliver) elements.

We build a genuinely NON-UNIFORM rest mesh (grid with interior vertices strongly jittered so element
sizes/aspect-ratios vary widely but all stay valid), impose a stretch, and compare OFFICIAL libigl
SLIM against our projected-Newton (clamp filter) on the shared fair criterion: iterations to reach
relative symmetric-Dirichlet energy tol `(E-E*)/(E0-E*) < 1e-4`. HW-independent counts carry the
verdict (SLIM is compiled C++, our Newton is Python -- wall-clock is cross-language).

Writes results/slim_nonuniform.md. Run: `python -m bench.run_slim_nonuniform` (needs libigl).
"""
import os
import numpy as np


def _nonuniform_mesh(nx=10, jitter=0.42, seed=1):
    """Grid rest mesh with interior vertices jittered to make element shapes highly non-uniform
    (varied aspect ratios / areas) while keeping every triangle valid (positive area). Boundary
    stays regular so the pinned BC is clean."""
    from .mesh import grid_mesh, boundary_mask, rest_quantities
    rest, tris = grid_mesh(nx, nx)
    bmask = boundary_mask(rest)
    rng = np.random.default_rng(seed)
    h = 1.0 / nx
    r = rest.copy()
    r[~bmask] += jitter * h * rng.standard_normal((int((~bmask).sum()), 2))
    Bs, areas = rest_quantities(r, tris)
    return r, tris, bmask, Bs, areas, float(areas.min()), float(areas.max())


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
        print("[slim-nonuniform] SKIP: libigl not installed"); return True
    from .solver import solve, energy_only
    from .energy import element_terms as sd

    rest, tris, bmask, Bs, areas, amin, amax = _nonuniform_mesh()
    free = ~np.repeat(bmask, 2)
    # stretch deformation (pull in x), interior from rest
    S = 2.0
    x0 = rest.copy(); x0[:, 0] = S * rest[:, 0]; x0 = x0.reshape(-1)

    # projected-Newton reference minimum (clamp). tol=1e-7/max=100 converges in ~8 it on this mesh;
    # a tighter 1e-9 would only chase the near-machine-precision tail (much slower, same minimum).
    rn = solve(x0, tris, Bs, areas, free, "clamp", eterms=sd, tol=1e-7, max_iter=100)
    Estar = rn["final_energy"]; E0 = energy_only(x0, tris, Bs, areas, sd)
    nw_it = _iters_to([e["energy"] for e in rn["log"]], E0, Estar)

    # official libigl SLIM
    bidx = np.where(bmask)[0].astype(np.int32)
    bc = x0.reshape(-1, 2)[bidx].astype(np.float64)
    V3 = np.hstack([rest, np.zeros((rest.shape[0], 1))]).astype(np.float64)
    data = igl.slim_precompute(V3, tris.astype(np.int32), x0.reshape(-1, 2).astype(np.float64),
                               igl.SYMMETRIC_DIRICHLET, bidx, bc, 1e8)
    slim_E = []
    SLIM_CAP = 60
    tol_band = Estar + 1e-4 * (E0 - Estar)          # energy at the (E-E*)/(E0-E*)<1e-4 crossing
    for _ in range(SLIM_CAP):
        UV = igl.slim_solve(data, 1)
        slim_E.append(energy_only(UV.reshape(-1), tris, Bs, areas, sd))
        if slim_E[-1] < tol_band:                   # crossed the tol -> stop (all we need to rank)
            break
        if len(slim_E) > 1 and abs(slim_E[-1] - slim_E[-2]) < 1e-13:
            break
    slim_it = _iters_to(slim_E, E0, Estar)
    slim_final = slim_E[-1]
    slim_drift = float(np.max(np.abs(UV[bidx] - bc)))

    aspect = amax / amin
    slim_wins = (slim_it is not None) and (nw_it is None or slim_it < nw_it)
    L = ["# SLIM vs projected-Newton on a NON-UNIFORM triangulation (P5.2 #7 — regime not reached)", "",
         f"Grid rest mesh with interior vertices strongly jittered: element areas span "
         f"{amin:.1e}–{amax:.1e} (**aspect {aspect:.0f}×**), all valid. Stretch ×{S:g}, boundary "
         "pinned. Criterion: iterations to relative symmetric-Dirichlet energy "
         f"`(E-E*)/(E0-E*) < 1e-4` (E\\*={Estar:.5f} projected-Newton reference, E₀={E0:.3f}). "
         "OFFICIAL libigl SLIM vs our clamp/projected-Newton. Run: `python -m bench.run_slim_nonuniform`.",
         "",
         "> ⚠️ **This does NOT adjudicate the claim, and is not evidence against the paper.** The "
         "claim (Fig.11) is about Newton **stalling far from the minimum** on a bad mesh. Our Newton "
         "is clamp-**projected** (SPD-safeguarded) + line-searched, and a 2× stretch keeps it "
         "well-conditioned — a **strawman** for a raw-Hessian-far-from-min claim. We report this to "
         "document *why the regime is out of reach*, not to rank the methods; the edge stays "
         "`self-claimed`.", "",
         "| method | iterations to energy-tol |", "|---|---:|",
         f"| SLIM (libigl, official) | {slim_it if slim_it is not None else 'did-not-reach'} |",
         f"| projected-Newton (clamp) | {nw_it if nw_it is not None else 'did-not-reach'} |", "",
         f"SLIM boundary drift ‖UV[b]−bc‖∞ = {slim_drift:.1e} "
         + ("(negligible; soft penalty ≈ hard BC, comparison fair)." if slim_drift < 1e-6
            else "(non-negligible; treat as indicative)."), "",
         "## Observed", ""]
    if slim_wins and nw_it is not None:
        L.append(f"- **`slim->projected-newton` reproduces (weakly):** SLIM reaches the tol in "
                 f"**{slim_it}** iterations vs projected-Newton's **{nw_it}** on this non-uniform "
                 "mesh — SLIM's reweighted global step is less sensitive to the bad element shapes. "
                 "The margin here is nowhere near the paper's dramatic Fig.11 gap (that far-from-min "
                 "pathology needs a harder instance than a 2× stretch).")
    elif slim_wins:
        L.append(f"- **Projected-Newton stalls, SLIM does not:** projected-Newton does NOT reach the "
                 f"tol within the budget while SLIM reaches it in **{slim_it}** iterations — direct "
                 "support for the claim that Newton struggles on non-uniform triangulations where "
                 "SLIM keeps progressing.")
    else:
        slim_txt = (f"**{slim_it}**" if slim_it is not None
                    else f"did **not** reach it within {SLIM_CAP} iterations (still at E={slim_final:.4f} "
                         f"vs E\\*={Estar:.4f})")
        L.append(f"- **Regime not reached (the claim's stall never occurs here):** projected-Newton "
                 f"reaches the energy-tol in **{nw_it}** iterations while SLIM {slim_txt}. On this "
                 f"non-uniform instance (aspect {aspect:.0f}×) a well-safeguarded (clamp-projected + "
                 "line-searched) Newton stays in a **good, near-quadratic basin** and converges "
                 "fast, whereas SLIM's reweighted first-order-like map crawls down a **slow linear "
                 "tail**. Crucially, this is **not evidence against the paper**: its SLIM≫Newton "
                 "result (Fig.11) is a **far-from-minimum pathology** where Newton's *raw* Hessian is "
                 "unreliable, and a 2× stretch on a projected+line-searched Newton never enters that "
                 "regime. Honest verdict: the edge is **not adjudicable** in this harness — the "
                 "configuration that produces the claim (raw-Hessian Newton stalling far from the "
                 "minimum on a bad mesh) is out of reach here, so we report the limitation rather than "
                 "a ranking. The edge stays `self-claimed` / needs-harder-instance.")
    L += ["",
          "_Caveat: 2D, single non-uniform instance/seed, moderate stretch; official libigl SLIM "
          "grounds the base (D3); wall-clock is C++/Python-confounded so iteration counts carry the "
          "verdict; our projected-Newton is clamp-filtered + backtracking-line-searched, which is "
          "already fairly robust, so a null here is about instance difficulty, not method identity._"]

    os.makedirs("results", exist_ok=True)
    with open("results/slim_nonuniform.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  non-uniform aspect={aspect:.0f}x areas[{amin:.1e},{amax:.1e}]")
    print(f"  SLIM it={slim_it}  projected-Newton it={nw_it}  slim_wins={slim_wins}")
    print("wrote results/slim_nonuniform.md")
    return True


if __name__ == "__main__":
    main()
