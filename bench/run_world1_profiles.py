"""World-1 accelerator comparison, RIGOROUS (metrics-rigor phase; closes review-r2 #48/#49/#51).

Upgrades the old single-run data profile to the rigor template established in run_mesh_independence:
  - **multiple seeds** with reported spread (mean [min-max] over instances), not one line (#48);
  - **independent high-accuracy E\***: Newton to |g|<1e-9 per instance, NOT best-final-among-the-
    compared-methods (#51 -- removes the bias toward the strongest solver);
  - **tau-sweep** (tau in {1e-3, 1e-6}) so orderings aren't cutoff artifacts (#50);
  - a data profile reported as PAIRWISE fractions, not an N-solver total order (#49, Gould-Scott).

Methods (all minimize the same symmetric-Dirichlet energy): Newton (clamp), L-BFGS, Sobolev-L-BFGS
(the isolated BCQN proxy), AQP. Metric: iterations to (E-E*)/(E0-E*) < tau. Writes
results/world1_profiles.md. Run: `python -m bench.run_world1_profiles`.
"""
import os
import numpy as np
from .solver import solve
from .energy import element_terms as sd, element_eg
from .descent import solve_lbfgs
from . import world1
from .run_e1 import build_scenario

MESHES = [5, 6]
SEEDS = [0, 1, 2, 3, 4]
TAUS = [1e-3, 1e-6]
BUDGETS = [5, 10, 20, 40, 80, 160]
METHODS = ["newton", "l-bfgs", "sobolev-lbfgs", "aqp"]


def iters_to(log, E0, Estar, tau):
    span = (E0 - Estar) + 1e-30
    for e in log:
        if (e["energy"] - Estar) / span < tau:
            return e["iter"]
    return None


def run_instance(nx, seed):
    sc = build_scenario(nx=nx, ny=nx, seed=seed)
    a = (sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"])
    ref = solve(*a, "clamp", eterms=sd, tol=1e-9, max_iter=80)   # independent high-accuracy E*
    Estar = ref["final_energy"]; E0 = sc["E0"]
    logs = {
        "newton": ref["log"],
        "l-bfgs": solve_lbfgs(*a, element_eg, max_iter=300, tol=1e-8)["log"],
        "sobolev-lbfgs": world1.solve_sobolev_lbfgs(sc["x0"], sc["tris"], sc["rest"], sc["free"],
                                                    max_iter=300, tol=1e-8)["log"],
        "aqp": world1.solve_aqp(sc["x0"], sc["tris"], sc["rest"], sc["free"], max_iter=400, tol=1e-8)["log"],
    }
    return {m: {tau: iters_to(logs[m], E0, Estar, tau) for tau in TAUS} for m in METHODS}


def main():
    print("== World-1 accelerators (rigorous: multi-seed, independent E*, tau-sweep) ==\n")
    insts = [run_instance(nx, s) for nx in MESHES for s in SEEDS]
    N = len(insts)

    def stats(m, tau):
        v = [i[m][tau] for i in insts if i[m][tau] is not None]
        return (np.mean(v), min(v), max(v), len(v)) if v else (None, None, None, 0)

    def frac(m, tau, b):
        return sum(1 for i in insts if i[m][tau] is not None and i[m][tau] <= b) / N

    for tau in TAUS:
        print(f"tau={tau:g}: iters mean[min-max] (n solved/{N})")
        for m in METHODS:
            mn, lo, hi, k = stats(m, tau)
            print(f"  {m:14s} " + (f"{mn:5.1f} [{lo}-{hi}]  ({k}/{N})" if mn is not None else "—"))
        print()

    L = ["# World-1 accelerators — rigorous data profile (measured)", "",
         f"{N} symmetric-Dirichlet instances (meshes {MESHES} × seeds {SEEDS[0]}–{SEEDS[-1]}). "
         "Rigor template (review-r2 #48/#49/#50/#51): **multi-seed spread**, **independent E\\*** "
         "(Newton to |g|<1e-9, not best-of-compared), **τ-sweep**, and **pairwise** (not total-order) "
         "reading. Metric: iterations to `(E−E*)/(E0−E*)<τ`. Run: `python -m bench.run_world1_profiles`."]
    for tau in TAUS:
        L += ["", f"### τ = {tau:g}", "",
              "| method | iters mean [min–max] | solved | " + " | ".join(f"≤{b}" for b in BUDGETS) + " |",
              "|---|---|---|" + "---|" * len(BUDGETS)]
        for m in METHODS:
            mn, lo, hi, k = stats(m, tau)
            cells = " | ".join(f"{frac(m,tau,b):.2f}" for b in BUDGETS)
            L.append(f"| {m} | {f'{mn:.1f} [{lo}–{hi}]' if mn is not None else '—'} | {k}/{N} | {cells} |")

    def med(m, tau):
        mn, *_ = stats(m, tau); return mn
    L += ["", "## Observed (pairwise, per τ)", ""]
    for tau in TAUS:
        nw, lb, so, aq = (med(m, tau) for m in METHODS)
        pair = []
        if so and lb:
            pair.append(f"Sobolev-L-BFGS {'<' if so < lb else '≈' if abs(so-lb) < 1 else '>'} L-BFGS "
                        f"({so:.0f} vs {lb:.0f} it)")
        if aq and lb:
            pair.append(f"AQP {'<' if aq < lb else '>'} L-BFGS ({aq:.0f} vs {lb:.0f})")
        if nw:
            pair.append(f"Newton fewest iters ({nw:.0f}) but 1 factorization/iter (see e2)")
        L.append(f"- **τ={tau:g}:** " + "; ".join(pair) + ".")
    so3, so6 = med("sobolev-lbfgs", 1e-3), med("sobolev-lbfgs", 1e-6)
    aq3, aq6 = med("aqp", 1e-3), med("aqp", 1e-6)
    lb3, lb6 = med("l-bfgs", 1e-3), med("l-bfgs", 1e-6)
    stable = (so3 and so6 and lb3 and lb6 and (so3 < lb3) == (so6 < lb6))
    L += ["",
          "- **τ-stability:** the Sobolev-vs-L-BFGS ordering "
          + ("holds at both τ" if stable else "changes with τ — a cutoff-sensitive comparison") +
          (f". AQP's iteration count grows from τ=1e-3 to 1e-6 ({aq3:.0f}→{aq6:.0f}), the same "
           "loose-vs-tight first-order-tail effect quantified in `results/mesh_independence.md`."
           if (aq3 and aq6) else ".") +
          " Rankings are stated **pairwise**, not as an N-solver total order (Gould–Scott); read the "
          "budget columns as *iterations*, not cost (a Newton iteration is a factorization, see e2).",
          "",
          "_Caveat: 2D, dense, small meshes; independent E\\* is our Newton to |g|<1e-9 (energy to "
          "~machine precision), not a third-party oracle. Spread is min–max over instances._"]
    os.makedirs("results", exist_ok=True)
    with open("results/world1_profiles.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"{N} instances; wrote results/world1_profiles.md")


if __name__ == "__main__":
    main()
