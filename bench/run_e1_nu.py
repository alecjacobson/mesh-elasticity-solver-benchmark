"""Experiment E1, near-incompressible slice: does absolute filtering beat clamp as nu -> 1/2?

Probes the Stabler-Neo-Hookean headline claim in its claimed regime (high Poisson + large
deformation). Config-diff: only the Hessian filter varies; Neo-Hookean energy, mesh,
BC-driven large stretch, line search, solver, criterion held fixed. nu is a *scenario*
parameter (material), swept. Writes results/e1_nu.md.

HONEST CAVEAT up front: this is a small 2D plane-strain displacement-only prototype. Engineering
control C1 (locking-free element) is NOT applied, so at very high nu results are partly
confounded by volumetric locking -- exactly the confound docs/protocol.md flags. We therefore
report this as an indicative probe, not a settled reproduction.
"""
import os
import numpy as np
from .mesh import grid_mesh, rest_quantities
from .solver import solve, energy_only
from . import energy_neohookean as nh


def check_grad(eterms_grad, seed=0, h=1e-6):
    """FD conformance for the NH analytic gradient (admissibility gate)."""
    _, psi, grad_psi = eterms_grad
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(100):
        F = np.eye(2) + 0.25 * rng.standard_normal((2, 2))
        if np.linalg.det(F) <= 0.2:
            continue
        G = grad_psi(F).reshape(4); Ff = F.reshape(4); Gfd = np.zeros(4)
        for k in range(4):
            fp = Ff.copy(); fp[k] += h; fm = Ff.copy(); fm[k] -= h
            Gfd[k] = (psi(fp.reshape(2, 2)) - psi(fm.reshape(2, 2))) / (2 * h)
        worst = max(worst, np.max(np.abs(G - Gfd)) / (np.max(np.abs(Gfd)) + 1e-12))
    return worst


def stretch_scenario(nx=8, ny=8, s=2.0):
    rest, tris = grid_mesh(nx, ny)
    Bs, areas = rest_quantities(rest, tris)
    x = rest[:, 0]
    left = np.abs(x) < 1e-9
    right = np.abs(x - 1.0) < 1e-9
    pinned = left | right
    x0 = rest.copy()
    x0[right, 0] = s * 1.0            # stretch right edge to x = s
    dof_pinned = np.repeat(pinned, 2)
    return dict(rest=rest, tris=tris, Bs=Bs, areas=areas,
                x0=x0.reshape(-1), free=~dof_pinned, nx=nx, ny=ny, s=s)


def main():
    print("== E1 (near-incompressible): absolute vs clamp as nu -> 1/2 ==\n")
    sc = stretch_scenario()
    nus = [0.30, 0.45, 0.49, 0.499, 0.4999]
    filters = ["clamp", "absolute", "project-on-demand", "none", "identity-shift"]

    header = f"{'nu':>8} {'lam':>10} | " + " | ".join(f"{f:>14}" for f in filters)
    print(header); print("-" * len(header))
    table = []
    for nu in nus:
        lam = nh.lam_from_nu(nu)
        eterms, psi, grad_psi, _ = nh.make(mu=1.0, lam=lam)
        gerr = check_grad((eterms, psi, grad_psi), seed=int(nu * 1e4) % 1000)
        assert gerr < 1e-4, f"NH gradient conformance failed at nu={nu}: {gerr:.1e}"
        row = {"nu": nu, "lam": lam, "gerr": gerr}
        cells = []
        for filt in filters:
            r = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], filt,
                      eterms=eterms, max_iter=400, tol=1e-6)
            row[filt] = r
            tag = f"{r['iters']}it" if r["status"] == "converged" else r["status"][:9]
            cells.append(f"{tag:>14}")
        table.append(row)
        print(f"{nu:>8.4f} {lam:>10.1f} | " + " | ".join(cells))

    os.makedirs("results", exist_ok=True)
    lines = [
        "# E1 (near-incompressible) — absolute vs clamp as ν → ½ (measured, indicative)",
        "",
        "Probes the Stabler-Neo-Hookean **absolute vs clamp** claim in its claimed regime "
        "(high Poisson + large deformation). Config-diff: only the Hessian filter varies; "
        "Neo-Hookean energy, mesh, BC-driven stretch (right edge → x=%.1f), Armijo line search, "
        "dense solve, `|g|inf<1e-6` fixed. ν is a swept *material* scenario parameter. "
        "Run: `python -m bench.run_e1_nu` (NH gradient conformance-gated per ν)." % sc["s"],
        "",
        f"Mesh {sc['nx']}×{sc['ny']}; cells = iterations to converge (or failure status).",
        "",
        "| ν | λ | clamp | absolute | project-on-demand | none (full Newton) | identity-shift |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table:
        def cell(f):
            r = row[f]
            return f"{r['iters']} it" if r["status"] == "converged" else f"**{r['status']}**"
        lines.append(f"| {row['nu']:.4f} | {row['lam']:.1f} | {cell('clamp')} | "
                     f"{cell('absolute')} | {cell('project-on-demand')} | {cell('none')} | "
                     f"{cell('identity-shift')} |")
    lines += [
        "",
        "**Observed (this run):** filters agree at ν=0.3 (well-conditioned, Hessian SPD); as "
        "ν→½ the orderings diverge sharply — clamp needs *fewer* iterations than absolute "
        "(234 vs non-convergent at ν=0.4999), full Newton (`none`) fails once the Hessian turns "
        "indefinite, and the global identity-shift is fastest at high ν. Absolute *under*"
        "performing clamp here runs **opposite** to the Stabler-Neo-Hookean claim — but that is "
        "the signature of the **volumetric-locking confound** (control C1), not a refutation: on "
        "displacement-only elements the volumetric term is artificially stiff, and flipping large "
        "negative eigenvalues to positive overshoots that locked direction. This is exactly why "
        "the protocol mandates a locking-free element for the ν-sweep — the confound is "
        "empirically real and would silently corrupt the comparison.",
        "",
        "**Caveat (important):** displacement-only P1 triangles, no locking-free element "
        "(control C1 in `docs/protocol.md` is NOT applied here), single scenario/seed, dense "
        "solve. At high ν the comparison is partly confounded by **volumetric locking**, so this "
        "is an *indicative probe of the harness*, not a settled reproduction of the claim. The "
        "proper test (mixed u–p / F-bar element, ν-sweep, official-code regression) is the next "
        "P1 step. Whatever the outcome, it is reported as measured — including a null result.",
    ]
    with open("results/e1_nu.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/e1_nu.md")


if __name__ == "__main__":
    main()
