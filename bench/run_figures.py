"""Generate all benchmark figures into figures/ (viz phase). `python -m bench.run_figures [name...]`.

Deterministic; each fig_* function writes one or more PNGs. Kept modular so figures can be added
incrementally and regenerated individually.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from . import viz


# ---------------------------------------------------------------- World-2: the locking figure
def fig_locking():
    """The headline confound, made visual: P1 vs P2 vs SRI-P2 deformed mesh coloured by J=det F at
    the near-incompressible stretch. P1 shows volumetric-locking stress; P2/SRI relieve it."""
    from .mesh import grid_mesh, rest_quantities
    from .solver import solve
    from . import p2, p2_sri, energy_neohookean as nh
    from .run_world2_filters import _hpsi
    N, S, nu = 10, 2.0, 0.499
    lam = nh.lam_from_nu(nu)

    # P1
    rest1, tris1 = grid_mesh(N, N); Bs, areas = rest_quantities(rest1, tris1)
    xc = rest1[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free1 = ~np.repeat(pin, 2); x01 = rest1.copy(); x01[:, 0] = S * rest1[:, 0]
    et1, _, _, _ = nh.make(1.0, lam)
    r1 = solve(x01.reshape(-1), tris1, Bs, areas, free1, "clamp", eterms=et1, tol=1e-6, max_iter=400)
    V1 = r1["x"].reshape(-1, 2); J1 = viz.face_detF_2d(V1, tris1, rest1)

    # P2 and SRI-P2 (render corner triangles coloured by centroid J)
    nodes, elems = p2.grid_mesh_p2(N, N); quad = p2.rest_quantities_p2(nodes, elems)
    xc2 = nodes[:, 0]; pin2 = (np.abs(xc2) < 1e-9) | (np.abs(xc2 - 1) < 1e-9)
    free2 = ~np.repeat(pin2, 2); x02 = nodes.copy(); x02[:, 0] = S * nodes[:, 0]
    _, psi, gp, _ = nh.make(1.0, lam)
    r2 = p2.solve_p2(x02.reshape(-1), elems, quad, free2, p2.make_element_terms(psi, gp, _hpsi(gp)),
                     "clamp", tol=1e-6, max_iter=400)
    quad_s = p2_sri.rest_quantities_sri(nodes, elems)
    rs = p2.solve_p2(x02.reshape(-1), elems, quad_s, free2, p2_sri.make_sri_terms(1.0, lam),
                     "clamp", tol=1e-6, max_iter=400)

    def p2_corner(V_flat):
        V = V_flat.reshape(-1, 2)
        cf = elems[:, :3]                       # corner-node triangles
        Jc = viz.face_detF_2d(V, cf, nodes[:, :2] if False else nodes)
        return V, cf, Jc
    V2, cf2, J2 = p2_corner(r2["x"]); Vs, cfs, Js = p2_corner(rs["x"])

    import matplotlib.colors as mcolors
    allJ = np.concatenate([J1, J2, Js])
    # TRUE range, colour centred at the incompressible target J=1 — do NOT clip: locking's signal is
    # exactly the J-tail a percentile clip would saturate (review-r4 viz #2).
    lo, hi = float(allJ.min()), float(allJ.max()); half = max(1 - lo, hi - 1)
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=1 - half, vmax=1 + half)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    tpc = viz.trimesh(axes[0], V1, tris1, values=J1, cmap="RdBu_r", norm=norm,
                      title=f"P1 (locking): {r1['iters']} it\nJ∈[{J1.min():.2f},{J1.max():.2f}]")
    viz.trimesh(axes[1], V2, cf2, values=J2, cmap="RdBu_r", norm=norm,
                title=f"P2 (relieved): {r2['iters']} it\nJ∈[{J2.min():.2f},{J2.max():.2f}]")
    viz.trimesh(axes[2], Vs, cfs, values=Js, cmap="RdBu_r", norm=norm,
                title=f"SRI-P2 (relieved): {rs['iters']} it\nJ∈[{Js.min():.2f},{Js.max():.2f}]")
    fig.suptitle(f"Volumetric locking made visible — Neo-Hookean stretch, ν={nu} (colour = J = det F, "
                 "centred at J=1; true range, unclipped)",
                 y=1.04, fontsize=11, fontweight="bold")
    cb = fig.colorbar(tpc, ax=axes, fraction=0.025, pad=0.02); cb.set_label("J = det F")
    viz.save(fig, "locking_p1_p2_sri")


def fig_filter_convergence():
    """Per-iteration energy-gap (log-y) for clamp / absolute / trust-region on P1 vs P2."""
    from .mesh import grid_mesh, rest_quantities
    from .solver import solve
    from . import p2, energy_neohookean as nh
    from .run_world2_filters import _hpsi
    N, S, nu = 8, 2.0, 0.499
    lam = nh.lam_from_nu(nu)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # P1
    rest, tris = grid_mesh(N, N); Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 2); x0 = (rest.copy()); x0[:, 0] = S * rest[:, 0]; x0 = x0.reshape(-1)
    et, _, _, _ = nh.make(1.0, lam)
    # shared E* = tightest fixed point across ALL filters (review: per-filter E* floors curves / draws
    # a fake vertical cliff when a run undershoots clamp's own E*)
    runs1 = {f: solve(x0, tris, Bs, areas, free, f, eterms=et, tol=1e-9, max_iter=400)
             for f in ("clamp", "absolute", "trust-region")}
    Estar = min(runs1[f]["final_energy"] for f in runs1)
    for filt in ("clamp", "absolute", "trust-region"):
        r = solve(x0, tris, Bs, areas, free, filt, eterms=et, tol=1e-6, max_iter=400)
        g = np.array([e["energy"] - Estar for e in r["log"]]); g = np.maximum(g, 1e-12)
        axes[0].semilogy(range(len(g)), g, color=viz.COL[filt], ls=viz.LS[filt],
                         label=f"{filt} ({r['iters']})", lw=1.8)
    axes[0].set_title(f"P1 (locking), ν={nu}"); axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("E − E*  (shared E*, floored at 1e-12)"); axes[0].legend()

    # P2
    nodes, elems = p2.grid_mesh_p2(N, N); quad = p2.rest_quantities_p2(nodes, elems)
    xc2 = nodes[:, 0]; pin2 = (np.abs(xc2) < 1e-9) | (np.abs(xc2 - 1) < 1e-9)
    free2 = ~np.repeat(pin2, 2); x02 = nodes.copy(); x02[:, 0] = S * nodes[:, 0]; x02 = x02.reshape(-1)
    _, psi, gp, _ = nh.make(1.0, lam); etp = p2.make_element_terms(psi, gp, _hpsi(gp))
    runs2 = {f: p2.solve_p2(x02, elems, quad, free2, etp, f, tol=1e-9, max_iter=400)
             for f in ("clamp", "absolute", "trust-region")}
    Estar2 = min(runs2[f]["final_energy"] for f in runs2)
    for filt in ("clamp", "absolute", "trust-region"):
        r = p2.solve_p2(x02, elems, quad, free2, etp, filt, tol=1e-6, max_iter=400)
        g = np.array([e["energy"] - Estar2 for e in r["log"]]); g = np.maximum(g, 1e-12)
        axes[1].semilogy(range(len(g)), g, color=viz.COL[filt], ls=viz.LS[filt],
                         label=f"{filt} ({r['iters']})", lw=1.8)
    axes[1].set_title(f"P2 (relieved), ν={nu}"); axes[1].set_xlabel("iteration"); axes[1].legend()
    fig.suptitle("Filter convergence: absolute's slow tail on locking P1 (left) vs its win on relieved P2 (right)",
                 y=1.02, fontweight="bold")
    viz.save(fig, "filter_convergence_p1_p2")


# ---------------------------------------------------------------- Survey: corpus & claims graph
WORLD_NAME = {0: "classical/baseline", 1: "World-1 distortion", 2: "World-2 sim/filtering", 3: "World-3"}


def _load_claims():
    import yaml
    with open("claims/claims.yaml") as f:
        d = yaml.safe_load(f)
    return d.get("nodes", []), d.get("edges", [])


def fig_corpus_breadth():
    """Breadth guard: papers-per-year, stacked by world — shows the survey spans a decade and both
    worlds, not one corner. Baseline/classical (year 0 / world 0) excluded from the timeline."""
    nodes, _ = _load_claims()
    years = [n for n in nodes if n.get("year", 0) and n["world"] in (1, 2, 3)]
    yr = sorted({n["year"] for n in years})
    worlds = [1, 2, 3]
    counts = {w: [sum(1 for n in years if n["year"] == y and n["world"] == w) for y in yr] for w in worlds}
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    bottom = np.zeros(len(yr))
    for w in worlds:
        c = np.array(counts[w])
        axb.bar(yr, c, bottom=bottom, color=viz.WORLD_COL[w], label=WORLD_NAME[w], width=0.8)
        bottom += c
    axb.set_title(f"Corpus: {len(years)} papers, {yr[0]}–{yr[-1]}, both worlds")
    axb.set_xlabel("year"); axb.set_ylabel("papers"); axb.legend(loc="upper left")
    axb.grid(axis="x", alpha=0)
    # right: world totals (incl. baselines) as an honest coverage bar
    alln = nodes
    tot = {w: sum(1 for n in alln if n["world"] == w) for w in (0, 1, 2, 3)}
    ws = [w for w in (1, 2, 0, 3) if tot[w]]
    axt.barh([WORLD_NAME[w] for w in ws], [tot[w] for w in ws],
             color=[viz.WORLD_COL[w] for w in ws])
    for i, w in enumerate(ws):
        axt.text(tot[w] + 0.3, i, str(tot[w]), va="center", fontsize=9)
    axt.set_title(f"All {len(alln)} graph nodes by world"); axt.set_xlabel("nodes"); axt.grid(axis="y", alpha=0)
    axt.invert_yaxis()
    fig.text(0.5, -0.04, f"left panel: {len(years)} dated papers (worlds 1–3); "
             f"right panel: all {len(alln)} graph nodes incl. undated classical baselines "
             "(node ≠ paper)", ha="center", fontsize=8, color="#666")
    fig.suptitle("Survey breadth — spans ~2 decades and all three worlds (World-1-heavy: "
                 f"{tot[1]} vs {tot[2]} vs {tot[3]} nodes)", y=1.03, fontweight="bold")
    viz.save(fig, "corpus_breadth")


def fig_claims_ledger():
    """The epistemic scoreboard: how many superiority-edges are self-claimed vs qualified vs
    independently validated/refuted by THIS benchmark. The honest core of the survey."""
    _, edges = _load_claims()
    order = ["self-claimed", "unmeasured", "qualified", "validated"]
    cnt = {s: sum(1 for e in edges if e.get("status") == s) for s in order}
    other = [e for e in edges if e.get("status") not in order]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.barh(order[::-1], [cnt[s] for s in order[::-1]],
                   color=[viz.STATUS_COL[s] for s in order[::-1]])
    for b, s in zip(bars, order[::-1]):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, str(cnt[s]),
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("superiority-claim edges")
    ax.set_title(f"Claims ledger: {len(edges)} extracted edges by evidentiary status"
                 + (f" (+{len(other)} other)" if other else ""))
    ax.text(0.98, 0.06, "self-claimed = paper's word · unmeasured = extracted, not yet tested\n"
            "qualified = regime-limited / benchmark-pending · validated = independently confirmed here\n"
            "(0 refuted — this benchmark qualifies rather than overturns)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555")
    viz.save(fig, "claims_ledger")


def fig_claims_network():
    """The superiority-claims graph rendered: nodes = methods (coloured by world), directed edges =
    'A claims to beat B' (coloured by evidentiary status). Shows the literature's claim topology."""
    import networkx as nx
    nodes, edges = _load_claims()
    G = nx.DiGraph()
    wof = {}
    for n in nodes:
        G.add_node(n["id"], world=n["world"]); wof[n["id"]] = n["world"]
    for e in edges:
        if e.get("from") in G and e.get("to") in G:
            G.add_edge(e["from"], e["to"], status=e.get("status", "self-claimed"))
    # keep the connected claim-graph (drop isolated nodes with no superiority edges)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    # LAYERED layout: x-position ENCODES world (meaningful), unlike a spring layout whose positions
    # are an arbitrary artifact of the seed (review-r4 viz #4). Vertical order within a column is
    # deterministic but not itself meaningful — only x-column, node colour and edges encode data.
    pos = nx.multipartite_layout(G, subset_key="world", align="vertical", scale=2.4)
    nval = sum(1 for _, _, d in G.edges(data=True) if d["status"] == "validated")
    fig, ax = plt.subplots(figsize=(13, 9))
    for st in ["self-claimed", "unmeasured", "qualified", "validated"]:
        el = [(u, v) for u, v, d in G.edges(data=True) if d["status"] == st]
        if el:
            nx.draw_networkx_edges(G, pos, edgelist=el, ax=ax, edge_color=viz.STATUS_COL[st],
                                   width=2.2 if st == "validated" else 1.0,
                                   alpha=0.85, arrowsize=11, connectionstyle="arc3,rad=0.08",
                                   node_size=[300 + 120 * G.degree(n) for n in G.nodes])
    sizes = [300 + 120 * G.degree(n) for n in G.nodes]
    cols = [viz.WORLD_COL[wof[n]] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=cols,
                           edgecolors="#333", linewidths=0.6)
    # label only the hubs (degree >= 3) to keep it legible
    lab = {n: n for n in G.nodes if G.degree(n) >= 3}
    nx.draw_networkx_labels(G, pos, labels=lab, ax=ax, font_size=7.2)
    ax.axis("off")
    from matplotlib.lines import Line2D
    wl = [Line2D([0], [0], marker="o", color="w", markerfacecolor=viz.WORLD_COL[w], markersize=9,
                 label=WORLD_NAME[w]) for w in (1, 2, 0, 3)]
    sl = [Line2D([0], [0], color=viz.STATUS_COL[s], lw=2.4, label=s)
          for s in ["self-claimed", "unmeasured", "qualified", "validated"]]
    l1 = ax.legend(handles=wl, title="node = method (column = world)", loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=sl, title="edge = ‘beats’ (status)", loc="lower left", fontsize=8)
    ax.set_title(f"Superiority-claims graph — {G.number_of_nodes()} methods, "
                 f"{G.number_of_edges()} claimed wins, only {nval} independently validated "
                 "(hubs labelled)", fontweight="bold")
    ax.text(0.5, -0.02, "x-position encodes world; vertical order within a column is not meaningful. "
            "The honest headline is the near-absence of validated (green) edges — see claims_ledger.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#666")
    viz.save(fig, "claims_network")


# ---------------------------------------------------------------- World-1: accelerator convergence
def fig_accelerator_convergence():
    """World-1 distortion: normalized energy-gap vs iteration for the accelerator cohort (Newton,
    L-BFGS, Sobolev-L-BFGS, AQP) on a perturbed grid — the shape of each method's descent."""
    from .solver import solve
    from .energy import element_terms as sd, element_eg
    from .descent import solve_lbfgs
    from . import world1
    from .run_e1 import build_scenario
    SEED = 7; TAU = 1e-6
    sc = build_scenario(nx=12, ny=12, seed=SEED)
    a = (sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"])
    ref = solve(*a, "clamp", eterms=sd, tol=1e-10, max_iter=120)
    # shared, independent E* = tightest final energy across all methods (not Newton's alone)
    res = {
        "newton": ref,
        "l-bfgs": solve_lbfgs(*a, element_eg, max_iter=300, tol=1e-9),
        "sobolev-lbfgs": world1.solve_sobolev_lbfgs(sc["x0"], sc["tris"], sc["rest"], sc["free"], max_iter=300, tol=1e-9),
        "aqp": world1.solve_aqp(sc["x0"], sc["tris"], sc["rest"], sc["free"], max_iter=400, tol=1e-9),
    }
    Estar = min(r["final_energy"] for r in res.values()); span = (sc["E0"] - Estar) + 1e-30
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for m, r in res.items():
        g = np.array([max((e["energy"] - Estar) / span, 1e-13) for e in r["log"]])
        # first iteration that reaches τ (the benchmarked quantity), and whether it converged at all
        hit = next((i for i, v in enumerate(g) if v <= TAU), None)
        reached = r.get("status") == "converged"
        tag = (f"reaches τ @ it {hit}" if hit is not None else "never reaches τ")
        suff = "" if reached else " · max_iter"
        ax.semilogy(range(len(g)), g, color=viz.COL[m], ls=viz.LS[m], lw=1.9,
                    label=f"{m}: {tag}{suff}")
        if hit is not None:
            ax.plot([hit], [g[hit]], "o", color=viz.COL[m], ms=6, mec="white", mew=0.8, zorder=5)
    ax.axhline(TAU, color="#aaa", ls="--", lw=0.8); ax.text(1, TAU * 1.4, "τ=1e-6", fontsize=8, color="#777")
    ax.set_xlim(0, 120); ax.set_xlabel("iteration"); ax.set_ylabel("(E − E*) / (E₀ − E*)  (shared E*)")
    ax.set_title(f"World-1 accelerators, symmetric-Dirichlet — SINGLE instance (12×12, seed={SEED}); "
                 "shapes are illustrative")
    ax.legend(title="dot = first τ-crossing")
    fig.suptitle("Descent shape: Newton's quadratic tail vs first-order tails — AQP crosses τ early "
                 "then STALLS (multi-seed picture: profiles / mesh_independence)",
                 y=1.0, fontweight="bold", fontsize=9.6)
    viz.save(fig, "accelerator_convergence")


# ---------------------------------------------------------------- Metrics: histograms
def fig_histograms():
    """Two distributions the point-estimates hide: (a) unfiltered Newton's failure RATE vs clamp
    across random inits (why eigenvalue filtering exists at all), and (b) the per-seed iteration
    SPREAD (a single-seed number is not a benchmark)."""
    from .solver import solve
    from .energy import element_terms as sd
    from .run_e1 import build_scenario
    seeds = list(range(40))
    status = {"none": [], "clamp": []}
    clamp_iters = []
    for s in seeds:
        sc = build_scenario(nx=10, ny=10, amp_frac=0.55, seed=s)   # stronger perturbation → some inits break raw Newton
        a = (sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"])
        for filt in ("none", "clamp"):
            r = solve(*a, filt, eterms=sd, tol=1e-6, max_iter=200)
            status[filt].append(r["status"])
            if filt == "clamp" and r["status"] == "converged":
                clamp_iters.append(r["iters"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cats = ["converged", "nondescent", "linesearch", "maxiter", "infeasible"]
    x = np.arange(len(cats)); w = 0.38
    for j, filt in enumerate(("none", "clamp")):
        h = [status[filt].count(c) for c in cats]
        axa.bar(x + (j - 0.5) * w, h, w, color=viz.COL["newton"] if filt == "none" else viz.COL["clamp"],
                label=f"{'unfiltered Newton' if filt == 'none' else 'clamp-filtered'}")
    axa.set_xticks(x); axa.set_xticklabels(cats, rotation=20, ha="right", fontsize=8.5)
    conv_none = status["none"].count("converged"); conv_clamp = status["clamp"].count("converged")
    axa.set_ylabel(f"count (of {len(seeds)} random inits)")
    axa.set_title(f"Convergence outcome\n(unfiltered {conv_none}/{len(seeds)} vs clamp {conv_clamp}/{len(seeds)} converged)",
                  fontsize=10)
    axa.legend(fontsize=9, loc="upper right")
    axb.hist(clamp_iters, bins=range(min(clamp_iters), max(clamp_iters) + 2), color=viz.COL["clamp"],
             edgecolor="white")
    axb.axvline(np.median(clamp_iters), color="#111", ls="--", lw=1.2,
                label=f"median {int(np.median(clamp_iters))} (range {min(clamp_iters)}–{max(clamp_iters)})")
    axb.set_xlabel("iterations to converge (clamp)"); axb.set_ylabel("# seeds")
    axb.set_title("Iteration spread (clamp)\na single seed is not a benchmark", fontsize=10)
    axb.legend(fontsize=9)
    fig.suptitle("What point estimates hide: filtering changes the SUCCESS RATE, and iteration counts "
                 "SPREAD across seeds", y=1.02, fontweight="bold", fontsize=10.5)
    viz.save(fig, "histograms")
    print(f"  unfiltered converged {conv_none}/{len(seeds)}, clamp {conv_clamp}/{len(seeds)}; "
          f"clamp iters median {np.median(clamp_iters):.0f}")


# ---------------------------------------------------------------- Metrics: performance/data profiles
def fig_profiles():
    """Dolan–Moré performance profile + Moré–Wild-style data profile over the World-1 instance set
    (multi-seed, multi-mesh) at τ=1e-6 — the standard, cutoff-robust way to compare solvers."""
    from . import run_world1_profiles as wp
    insts = [wp.run_instance(nx, s) for nx in wp.MESHES for s in wp.SEEDS]
    tau = 1e-6; N = len(insts); methods = wp.METHODS
    BIG = 1e9
    iters = {m: np.array([i[m][tau] if i[m][tau] is not None else np.nan for i in insts]) for m in methods}
    best = np.nanmin(np.vstack([np.where(np.isnan(iters[m]), BIG, iters[m]) for m in methods]), axis=0)

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # performance profile: ρ_m(α) = frac problems with iters ≤ α·best
    alphas = np.logspace(0, np.log10(20), 100)
    for m in methods:
        rat = np.where(np.isnan(iters[m]), np.inf, iters[m] / best)
        rho = [np.mean(rat <= a) for a in alphas]
        axp.step(alphas, rho, where="post", color=viz.COL[m], lw=1.9, label=m)
    axp.set_xscale("log"); axp.set_xlabel("α  (within α× the best solver)"); axp.set_ylabel("fraction of problems")
    axp.set_ylim(-0.02, 1.02); axp.set_title("Performance profile (Dolan–Moré)"); axp.legend(fontsize=8.5, loc="lower right")
    # data profile: κ_m(b) = frac problems solved within b iterations
    budgets = np.arange(1, 260)
    for m in methods:
        kappa = [np.mean(np.where(np.isnan(iters[m]), np.inf, iters[m]) <= b) for b in budgets]
        axd.step(budgets, kappa, where="post", color=viz.COL[m], lw=1.9, label=m)
    axd.set_xlabel("iteration budget"); axd.set_ylabel("fraction of problems solved")
    axd.set_ylim(-0.02, 1.02); axd.set_title("Data profile (fraction solved vs budget)"); axd.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"Solver profiles over {N} World-1 symmetric-Dirichlet instances (τ={tau:g})",
                 y=1.02, fontweight="bold")
    viz.save(fig, "profiles")


# ---------------------------------------------------------------- Metrics: mesh-independence log-log
def fig_mesh_independence():
    """AQP's mesh-independence is TOLERANCE-DEPENDENT: iters-vs-DOF on log-log for AQP & L-BFGS at a
    loose vs a tight τ, with min–max bands and the CI-gated growth exponent. The τ-flip, visual."""
    from . import run_mesh_independence as mi
    data = {n: [mi.run_instance(n, s) for s in mi.SEEDS] for n in mi.SIZES}
    dofs = np.array([data[n][0]["ndof"] for n in mi.SIZES], float)

    def series(m, tau):
        med, lo, hi = [], [], []
        for n in mi.SIZES:
            vals = [r[m][tau] for r in data[n] if r[m][tau]]
            if len(vals) < len(mi.SEEDS):     # censored at this τ/size — drop from the curve
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            vs = sorted(vals); md = vs[len(vs) // 2] if len(vs) % 2 else 0.5 * (vs[len(vs) // 2 - 1] + vs[len(vs) // 2])
            med.append(md); lo.append(min(vals)); hi.append(max(vals))
        return np.array(med), np.array(lo), np.array(hi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tau in zip(axes, [mi.TAUS[0], mi.TAUS[-1]]):
        for m in ("aqp", "l-bfgs"):
            med, lo, hi = series(m, tau)
            ok = ~np.isnan(med)
            ax.fill_between(dofs[ok], lo[ok], hi[ok], color=viz.COL[m], alpha=0.15)
            ax.plot(dofs[ok], med[ok], "o-", color=viz.COL[m], lw=1.9, label=None)
            e = mi._fit_exponent(list(dofs[ok]), list(med[ok]))
            lab = f"{m}: p={e[0]:+.2f}±{2 * e[1]:.2f}" if e else f"{m}: —"
            ax.plot([], [], "o-", color=viz.COL[m], label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"τ = {tau:g}  ({'loose' if tau >= 1e-3 else 'tight'})")
        ax.set_xlabel("free DOF"); ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("iterations to reach τ")
    fig.suptitle("AQP mesh-independence is tolerance-dependent — flat at loose τ, grows at tight τ "
                 "(p≈0 ⇔ mesh-independent)", y=1.02, fontweight="bold")
    viz.save(fig, "mesh_independence")


# ---------------------------------------------------------------- Metrics: scale-cost crossover
def fig_scale_cost():
    """Factorization-vs-iteration cost at scale: modeled relative cost (Newton=1) vs DOF for
    Newton / AQP / L-BFGS, from measured iteration counts + the 2D sparse-Cholesky complexity model."""
    from . import run_scale_cost as sc
    rows = [sc.run_size(n) for n in sc.SIZES]
    dofs = np.array([r["dof"] for r in rows], float)

    def costs(r):
        dof = r["dof"]; cf, cb = dof ** 1.5, dof
        nw, aq, lb = r["newton"], r["aqp"], r["l-bfgs"]
        return (nw * cf + nw * cb if nw else np.nan,
                1 * cf + aq * cb if aq else np.nan,
                lb * sc.M_LBFGS * dof if lb else np.nan)
    C = np.array([costs(r) for r in rows])            # columns: newton, aqp, lbfgs
    base = C[:, 0]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for j, m in enumerate(["newton", "aqp", "l-bfgs"]):
        ax.plot(dofs, C[:, j] / base, "o-", color=viz.COL[m], lw=1.9, label=m)
    ax.axhline(1.0, color="#bbb", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("free DOF")
    ax.set_ylabel("modeled cost ÷ Newton cost (τ=1e-6)")
    ax.set_title("Factorization-vs-iteration cost at scale (HW-independent model)")
    for j, m in enumerate(["aqp", "l-bfgs"], start=1):
        ax.annotate(m, (dofs[-1], C[-1, j] / base[-1]), fontsize=8.5, color=viz.COL[m],
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.legend(loc="best", fontsize=9)
    fig.suptitle("Does AQP's single factorization beat Newton at scale? Only if its iteration count "
                 "stays bounded", y=1.0, fontweight="bold", fontsize=10.5)
    viz.save(fig, "scale_cost")
    print("  DOF:", [int(d) for d in dofs], " AQP/Newton:", [round(v, 2) for v in (C[:, 1] / base)])


# ---------------------------------------------------------------- 3D: polyscope headless tet render
def fig_tet3d():
    """Genuine-3D example via polyscope headless (EGL): a P1-tet box stretched at near-incompressible
    ν, coloured by per-tet J=det F. Confirms the 2D locking story is not a 2D peculiarity."""
    from . import tet
    from .run_3d_nu import lam_of
    n, S, nu = 6, 1.5, 0.49
    verts, tets = tet.box_tet_mesh(n, n, n)
    Bs, vols = tet.rest_quantities(verts, tets)
    quad = list(zip(Bs, vols))
    xc = verts[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 3)
    x0 = verts.copy(); x0[:, 0] = S * verts[:, 0]; x0 = x0.reshape(-1)
    et, _, _ = tet.make(mu=1.0, lam=lam_of(nu))
    r = tet.solve(x0, tets, quad, free, et, "clamp", tol=1e-6, max_iter=400)
    V = r["x"].reshape(-1, 3)
    Js = np.array([np.linalg.det((B @ V.reshape(-1)[tet._edofs(t)]).reshape(3, 3))
                   for B, t in zip(Bs, tets)])

    # true range centred at the incompressible target J=1 (no percentile clip — the tail is the signal)
    half = max(1 - float(Js.min()), float(Js.max()) - 1)
    vlo, vhi = 1 - half, 1 + half
    ps = viz.ps_headless()
    ps.remove_all_structures()
    vm = ps.register_volume_mesh("stretched tets", V, tets=np.asarray(tets))
    vm.add_scalar_quantity("J = det F", Js, defined_on="cells", enabled=True, cmap="coolwarm",
                           vminmax=(vlo, vhi))
    vm.set_edge_width(1.0)
    ps.set_up_dir("y_up"); ps.set_front_dir("z_front")
    ps.look_at((1.55, 1.05, 1.95), (S * 0.5, 0.5, 0.5))
    shot = viz.ps_shot("tet3d_stretch_J")

    # wrap the (scale-free) polyscope screenshot in a matplotlib frame that carries a real colorbar
    import matplotlib.colors as mcolors
    img = plt.imread(shot)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(img); ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=mcolors.TwoSlopeNorm(1.0, vlo, vhi)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02); cb.set_label("J = det F (centred at 1)")
    ax.set_title(f"3D P1-tet box, Neo-Hookean stretch ν={nu} (polyscope headless / EGL)\n"
                 f"clamp {r['iters']} it · J∈[{Js.min():.2f}, {Js.max():.2f}] · Poisson necking visible",
                 fontsize=10)
    viz.save(fig, "tet3d_stretch_J")
    print(f"  3D tet: ν={nu} clamp {r['iters']} it, J∈[{Js.min():.3f},{Js.max():.3f}]")


FIGS = {"locking": fig_locking, "filter_convergence": fig_filter_convergence,
        "corpus_breadth": fig_corpus_breadth, "claims_ledger": fig_claims_ledger,
        "claims_network": fig_claims_network, "tet3d": fig_tet3d,
        "mesh_independence": fig_mesh_independence,
        "accelerator_convergence": fig_accelerator_convergence,
        "scale_cost": fig_scale_cost, "profiles": fig_profiles,
        "histograms": fig_histograms}


def main(names=None):
    names = names or list(FIGS)
    for n in names:
        if n not in FIGS:
            print(f"  skip unknown fig '{n}'"); continue
        print(f"[fig] {n}"); FIGS[n]()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
