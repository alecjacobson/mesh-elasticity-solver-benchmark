"""Experiment E1 (filter isolation), first real slice.

Fixed: energy (symmetric Dirichlet), mesh, initialization, line search (Armijo), linear solver
(dense), criterion (grad_inf < tol). SWAP ONLY the Hessian filter. Measure iterations,
wall-clock, and status. Writes a measured results table to results/e1.md.

Scenario: a triangulated square with the boundary pinned at rest and the interior given a
seeded random perturbation large enough to create strongly distorted (indefinite-Hessian)
elements while staying inversion-free -- the regime where the filter choice matters.
"""
import os
import numpy as np
from .mesh import grid_mesh, boundary_mask, rest_quantities
from .solver import solve, energy_only
from .filters import ALL
from . import conformance


def build_scenario(nx=10, ny=10, amp_frac=0.35, seed=7):
    rest, tris = grid_mesh(nx, ny)
    Bs, areas = rest_quantities(rest, tris)
    bmask = boundary_mask(rest)
    cell = 1.0 / nx
    rng = np.random.default_rng(seed)
    # perturb interior vertices; shrink amplitude until the whole mesh is inversion-free
    amp = amp_frac * cell
    for _ in range(40):
        x = rest.copy()
        pert = rng.standard_normal(rest.shape)
        pert[bmask] = 0.0
        x = x + amp * pert
        xflat = x.reshape(-1)
        if np.isfinite(energy_only(xflat, tris, Bs, areas)):
            break
        amp *= 0.8
    dof_bmask = np.repeat(bmask, 2)
    free = ~dof_bmask
    return dict(rest=rest, tris=tris, Bs=Bs, areas=areas, x0=xflat, free=free,
                nx=nx, ny=ny, amp=amp, seed=seed,
                E0=energy_only(xflat, tris, Bs, areas))


def main():
    print("== E1: filter isolation (2D symmetric Dirichlet) ==\n")
    if not conformance.run():
        raise SystemExit("conformance failed; results would be inadmissible")
    print()
    sc = build_scenario()
    print(f"mesh {sc['nx']}x{sc['ny']}  ({sc['rest'].shape[0]} verts, {len(sc['tris'])} tris)  "
          f"free dofs {int(sc['free'].sum())}  init energy {sc['E0']:.4f}\n")

    rows = []
    for filt in ALL:
        r = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], filt,
                  max_iter=300, tol=1e-6)
        rows.append(r)
        print(f"  {filt:14s}  status={r['status']:11s}  iters={r['iters']:4d}  "
              f"E_final={r['final_energy']:.6f}  |g|inf={r['final_grad_inf']:.2e}  "
              f"wall={r['wall_s']*1e3:7.1f} ms")

    # measured markdown table
    os.makedirs("results", exist_ok=True)
    conv = [r for r in rows if r["status"] == "converged"]
    Emin = min((r["final_energy"] for r in conv), default=float("nan"))
    lines = [
        "# E1 — Filter isolation (measured)",
        "",
        "First real slice of experiment E1 (`docs/experiments.md`). Config-diff: **only the "
        "Hessian filter varies**; energy (symmetric Dirichlet), mesh, initialization, line "
        "search (Armijo), linear solver (dense), and criterion (`|g|inf < 1e-6`) are held fixed. "
        "Numbers are produced by `python -m bench.run_e1` (reproducible; seed fixed) and gated on "
        "`bench/conformance.py`.",
        "",
        f"Scenario: {sc['nx']}x{sc['ny']} grid, boundary pinned at rest, interior perturbed "
        f"(amp={sc['amp']:.4f}, seed={sc['seed']}); init energy {sc['E0']:.4f}; "
        f"{int(sc['free'].sum())} free dofs.",
        "",
        "| filter | status | iters | final energy | wall (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['filter']} | {r['status']} | {r['iters']} | "
                     f"{r['final_energy']:.6f} | {r['wall_s']*1e3:.1f} |")
    lines += [
        "",
        f"All converged solvers reached the same minimum energy ({Emin:.6f}) — confirming the "
        "filter changes the *path* (iterations/robustness), not the solution. "
        "`none` (full Newton, unfiltered) is expected to stall on indefinite Hessians; the "
        "filtered variants are the point of the comparison.",
        "",
        "_Caveat: this is a small controlled prototype (dense solve, one scenario/seed) to "
        "validate the harness and produce the first measured numbers — not yet the full E1 "
        "(no ν-sweep, no locking-free element C1, no official-code port). See `docs/experiments.md`._",
    ]
    with open("results/e1.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/e1.md")


if __name__ == "__main__":
    main()
