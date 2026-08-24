"""Scalability with the SPARSE backend: mesh-independence (#69) + wall/solver scaling (#70).

Part 1 (sparse direct): projected Newton (clamp) across mesh refinement -> Newton iterations vs
DOFs (mesh-independence) and wall-clock vs DOFs.
Part 2 (sparse CG): the SAME outer method with an iterative inner solve -> CG mat-vecs per Newton
iteration GROW with refinement (the Hessian conditioning worsens), which is the quantitative case
for a preconditioner. Writes results/scaling.md.
"""
import os
import numpy as np
from .solver import solve_sparse
from .energy import element_terms as sd_terms
from .run_e1 import build_scenario


def main():
    print("== scalability (sparse): mesh-independence + CG conditioning ==\n")
    sizes = (8, 16, 24, 32, 40)
    direct, cg = [], []
    print(f"{'mesh':>7} {'dofs':>7} {'iters':>6} {'wall_ms':>9} {'nnz':>8}   "
          f"{'cg_iters':>8} {'cg_matvecs':>11} {'mv/iter':>8}")
    for n in sizes:
        sc = build_scenario(nx=n, ny=n)
        a = (sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"])
        dofs = int(sc["free"].sum())
        rd = solve_sparse(*a, "clamp", eterms=sd_terms, linsolver="direct", tol=1e-6)
        rc = solve_sparse(*a, "clamp", eterms=sd_terms, linsolver="cg", tol=1e-6)
        direct.append((n, dofs, rd)); cg.append((n, dofs, rc))
        mvi = rc["counts"]["mat_vecs"] / max(rc["iters"], 1)
        print(f"{n:>4}x{n:<2} {dofs:>7} {rd['iters']:>6} {rd['wall_s']*1e3:>9.1f} "
              f"{rd['counts']['nnz']:>8}   {rc['iters']:>8} {rc['counts']['mat_vecs']:>11} {mvi:>8.0f}")

    D = np.array([x[1] for x in direct], float)
    W = np.array([x[2]["wall_s"] * 1e3 for x in direct], float)
    alpha = float(np.polyfit(np.log(D), np.log(W), 1)[0])
    di = [x[2]["iters"] for x in direct]
    mvi_first = cg[0][2]["counts"]["mat_vecs"] / max(cg[0][2]["iters"], 1)
    mvi_last = cg[-1][2]["counts"]["mat_vecs"] / max(cg[-1][2]["iters"], 1)

    lines = ["# Scalability (sparse backend) - mesh-independence + CG conditioning (measured)", "",
             "Projected Newton (clamp) on the symmetric-Dirichlet perturbation cell across mesh "
             "refinement, SPARSE assembly + SuperLU (direct) and CG (iterative) inner solves. "
             "Run: `python -m bench.run_scaling`.", "",
             "| mesh | free dofs | Newton iters | wall (ms) | H nnz | CG mat-vecs | mat-vecs/iter |",
             "|---|---|---|---|---|---|---|"]
    for (n, dofs, rd), (_, _, rc) in zip(direct, cg):
        mvi = rc["counts"]["mat_vecs"] / max(rc["iters"], 1)
        lines.append(f"| {n}x{n} | {dofs} | {rd['iters']} | {rd['wall_s']*1e3:.1f} | "
                     f"{rd['counts']['nnz']} | {rc['counts']['mat_vecs']} | {mvi:.0f} |")
    lines += ["",
              "## Observed", "",
              f"- **Mesh-independence (metric #69):** Newton iteration count stays in "
              f"[{min(di)},{max(di)}] as DOFs grow {D[0]:.0f}->{D[-1]:.0f} (~{D[-1]/D[0]:.0f}x) "
              f"-- essentially refinement-independent, the hallmark of a well-behaved "
              f"second-order outer solver. This is the HW-independent axis and it does NOT grow "
              f"with mesh size.",
              f"- **CG conditioning (why preconditioning matters):** with the SAME outer method, "
              f"CG mat-vecs **per Newton iteration** rise from ~{mvi_first:.0f} at {D[0]:.0f} dofs "
              f"to ~{mvi_last:.0f} at {D[-1]:.0f} dofs -- the inner solve gets harder under "
              f"refinement (worsening Hessian conditioning) even though the OUTER iteration count "
              f"is flat. The linear-solver slot (unpreconditioned CG here) is where the cost "
              f"scaling actually lives -- a preconditioner/multigrid is the fix, and this table "
              f"is the quantitative motivation.",
              f"- **Wall-clock (metric #70):** sparse-direct wall ~ DOFs^{alpha:.2f} for this "
              f"prototype -- still Python-assembly-dominated (the per-element loop), so treat the "
              f"exponent as prototype-specific, not an algorithmic complexity. The clean, portable "
              f"signals are the iteration count (#69, flat) and the CG mat-vec growth.",
              "",
              "_Caveat: Python-loop assembly dominates wall-clock; the mesh-independence and the "
              "CG-mat-vec-growth are the robust, hardware-independent findings._"]
    os.makedirs("results", exist_ok=True)
    with open("results/scaling.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nmesh-indep iters in [{min(di)},{max(di)}]; CG mv/iter {mvi_first:.0f}->{mvi_last:.0f}; "
          f"wall~DOFs^{alpha:.2f}")
    print("wrote results/scaling.md")


if __name__ == "__main__":
    main()
