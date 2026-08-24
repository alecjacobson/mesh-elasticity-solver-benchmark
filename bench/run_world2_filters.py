"""World-2 filter head-to-head (#23): clamp / absolute / trust-region on P1 vs P2.

Definitive filter comparison in the near-incompressible regime, on the locking (P1) and
locking-relieved (P2) element. Key question: does the adaptive trust-region filter track the
GOOD choice on a proper element? Writes results/world2_filters.md.
"""
import os
import numpy as np
from bench import p2, energy_neohookean as nh
from bench.mesh import grid_mesh, rest_quantities
from bench.solver import solve

NUS = [0.30, 0.45, 0.49, 0.499, 0.4999]
FILTERS = ["clamp", "absolute", "trust-region"]
S, N = 2.0, 8


def _hpsi(gp):
    def h(F, hh=1e-6):
        Ff = F.reshape(4); H = np.zeros((4, 4))
        for k in range(4):
            fp = Ff.copy(); fp[k] += hh; fm = Ff.copy(); fm[k] -= hh
            H[:, k] = (gp(fp.reshape(2, 2)).reshape(4) - gp(fm.reshape(2, 2)).reshape(4)) / (2 * hh)
        return 0.5 * (H + H.T)
    return h


def run_p1():
    rest, tris = grid_mesh(N, N); Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 2); x0 = rest.copy(); x0[:, 0] = S * rest[:, 0]; x0 = x0.reshape(-1)
    tab = {}
    for nu in NUS:
        et, _, _, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu)); tab[nu] = {}
        for f in FILTERS:
            r = solve(x0, tris, Bs, areas, free, f, eterms=et, tol=1e-6, max_iter=400)
            tab[nu][f] = _cell(r)
    return tab


def _cell(r):
    return {"it": (r["iters"] if r["status"] == "converged" else r["status"]),
            "status": r["status"], "wall": r.get("wall_s"),
            "fac": r.get("counts", {}).get("factorizations")}


def run_p2():
    nodes, elems = p2.grid_mesh_p2(N, N); quad = p2.rest_quantities_p2(nodes, elems)
    xc = nodes[:, 0]; pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    free = ~np.repeat(pin, 2); x0 = nodes.copy(); x0[:, 0] = S * nodes[:, 0]; x0 = x0.reshape(-1)
    tab = {}
    for nu in NUS:
        _, psi, gp, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
        et = p2.make_element_terms(psi, gp, _hpsi(gp)); tab[nu] = {}
        for f in FILTERS:
            r = p2.solve_p2(x0, elems, quad, free, et, f, tol=1e-6, max_iter=400)
            tab[nu][f] = _cell(r)
    return tab


def _fmt(cell, key):
    v = cell[key]
    if v is None:
        return "—"
    if key == "wall":
        return f"{v*1e3:.0f}"
    return str(v)


def _iters_val(cell):
    return cell["it"] if isinstance(cell["it"], int) else None


def main():
    print("== World-2 filter head-to-head: P1 vs P2 ==\n")
    p1, pp2 = run_p1(), run_p2()
    for name, tab in (("P1 (locking)", p1), ("P2 (locking-relieved)", pp2)):
        print(name)
        print("  nu      " + " ".join(f"{f:>13}" for f in FILTERS))
        for nu in NUS:
            print(f"  {nu:.4f} " + " ".join(f"{str(tab[nu][f]['it']):>13}" for f in FILTERS))
        print()

    def tbl(tab, key):
        out = ["| ν | " + " | ".join(FILTERS) + " |", "|" + "---|" * (len(FILTERS) + 1)]
        for nu in NUS:
            out.append(f"| {nu:.4f} | " + " | ".join(_fmt(tab[nu][f], key) for f in FILTERS) + " |")
        return "\n".join(out)

    # de-hardcoded comparison values at the most-incompressible ν
    nu_hi = NUS[-1]
    def cmp_line(tab, elem):
        tr, cl, ab = tab[nu_hi]["trust-region"], tab[nu_hi]["clamp"], tab[nu_hi]["absolute"]
        it_tr, it_cl = _iters_val(tr), _iters_val(cl)
        w_tr = tr["wall"] * 1e3 if tr["wall"] else None
        w_cl = cl["wall"] * 1e3 if cl["wall"] else None
        wall_ratio = (w_tr / w_cl) if (w_tr and w_cl) else None
        seg = f"{elem} at ν={nu_hi}: trust-region {tr['it']} it"
        if w_tr:
            seg += f" / {w_tr:.0f} ms" + (f" ({tr['fac']} fac)" if tr["fac"] is not None else "")
        seg += f" vs clamp {cl['it']} it"
        if w_cl:
            seg += f" / {w_cl:.0f} ms" + (f" ({cl['fac']} fac)" if cl["fac"] is not None else "")
        seg += f" vs absolute {ab['it']} it."
        if wall_ratio:
            seg += (f" So TR uses ~{it_cl/it_tr:.1f}× FEWER iterations but ~{wall_ratio:.1f}× "
                    f"MORE wall-clock than clamp." if (it_tr and it_cl) else
                    f" TR wall-clock is ~{wall_ratio:.1f}× clamp's.")
        return seg

    lines = ["# World-2 filter head-to-head: clamp / absolute / trust-region, P1 vs P2 (measured)",
             "", "Neo-Hookean ν-sweep (stretch init), only the filter swapped. **Three axes** per "
             "cell (docs/metrics.md): iterations, wall-clock, and — where available — global "
             "factorizations. Run: `python -m bench.run_world2_filters`.", "",
             "### Iterations to converge", "", tbl(p1, "it"),
             "", "_(P2, locking-relieved)_", "", tbl(pp2, "it"),
             "", "### Wall-clock (ms)", "", tbl(p1, "wall"),
             "", "_(P2)_", "", tbl(pp2, "wall"),
             "", "### Global factorizations (P1 solver; P2 solver does not expose counts)", "",
             tbl(p1, "fac"), "",
             "## Observed (corrected, review-r2 #42/#43/#44/#45)", "",
             "Trust-region is the three-state blend λ_eff=(1−w)λ+w|λ|, w∈{0,0.5,1} → {full Newton, "
             "clamp, absolute} driven by the model-fit ratio ρ. **Two honest corrections to the "
             "round-1 write-up:**",
             "- **Fewer iterations is NOT cheaper here.** " + cmp_line(p1, "P1") + " On P1 the "
             "trust-region step does a **full eigendecomposition of the assembled Hessian** every "
             "iteration (plus an extra one on each non-descent escalation), whereas clamp/absolute do "
             "cheap per-element 6×6 projections — so TR's few iterations cost more wall-clock than "
             "clamp's many. The earlier 'trust-region dominates / validated on both' verdict was drawn "
             "on **iteration count alone** and is withdrawn; on the paired (iterations, wall-clock, "
             "factorizations) view TR trades iterations for per-step cost.",
             "- **P1 and P2 use different trust-region implementations.** P1 (this solver) uses the "
             "assembled three-state eigen-blend; P2 (`bench/p2.py`) still uses a *per-element two-state* "
             "clamp/absolute switch. So the P1 and P2 trust-region columns are **not the same operator**, "
             "and the cross-element comparison is apples-to-oranges — flagged rather than hidden. A "
             "per-element three-state blend (same cost as clamp/absolute) is the right unification and "
             "is future work.",
             "- What DOES hold on iteration count: on the locking-free P2 the adaptive rule is ≤ both "
             "standalone filters at every ν, and on P1 it uses fewer iterations than both at the most "
             "incompressible ν — but **not uniformly** (e.g. P1 ν=0.49 it slightly trails clamp), so "
             "'dominates everywhere' was an overclaim. The eps floor is now 1e-9, matching the "
             "standalone filters (so the w=0.5/w=1 states ARE those filters — conformance-gated against "
             "`filters.project_element`; note this changed the P1 counts vs the round-1 run, which "
             "used a 0.01 floor). The absolute-vs-clamp gap on P1 stays locking-confounded (control C1).",
             "",
             "_Caveat: dense solve, single stretch/mesh/seed, single τ=1e-6 — an **indicative** "
             "head-to-head, not a validated verdict (review-r2). ρ→w thresholds ρ≥0.75→Newton, "
             "≤0→absolute, else clamp. No official-code regression (code unavailable); the operator is "
             "conformance-gated to reproduce full-Newton and the real clamp/absolute filters at eps=1e-9._"]
    os.makedirs("results", exist_ok=True)
    with open("results/world2_filters.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/world2_filters.md")


if __name__ == "__main__":
    main()
