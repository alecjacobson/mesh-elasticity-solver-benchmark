"""E2 - World-1 accelerator decomposition across REGIMES (docs/experiments.md).

All methods minimize the SAME symmetric-Dirichlet energy (so they are comparable). We run them
on a WELL-conditioned (perturbation-recovery) and an ILL-conditioned (large-stretch) scenario,
because the World-1 superiority claims (e.g. AQP faster than L-BFGS) are regime-specific -- the
confound the benchmark exists to expose. Writes results/e2.md; hardens the aqp->l-bfgs and
aqp->gradient-descent edges with a regime caveat.
"""
import os
import numpy as np
from .solver import solve
from .energy import element_terms as sd, element_eg
from .descent import solve_lbfgs, solve_gd, solve_adam
from . import world1
from .mesh import grid_mesh, rest_quantities
from .run_e1 import build_scenario

TOL = 1e-6


def ill_scenario(n=8, s=3.0, seed=0):
    """Ill-conditioned symmetric-Dirichlet UV: recover a high-stretch (s x) equilibrium from a
    perturbed init. Near a large-stretch state the Hessian is badly conditioned -- AQP's regime."""
    from .solver import energy_only
    rest, tris = grid_mesh(n, n)
    Bs, areas = rest_quantities(rest, tris)
    xc = rest[:, 0]
    pin = (np.abs(xc) < 1e-9) | (np.abs(xc - 1) < 1e-9)
    base = rest.copy(); base[:, 0] = s * rest[:, 0]        # high-stretch state
    rng = np.random.default_rng(seed)
    amp = 0.15 / n
    for _ in range(40):
        pert = rng.standard_normal(rest.shape); pert[pin] = 0.0
        x0 = (base + amp * pert).reshape(-1)
        if np.isfinite(energy_only(x0, tris, Bs, areas, sd)):
            break
        amp *= 0.7
    return dict(x0=x0, tris=tris, Bs=Bs, areas=areas, rest=rest, free=~np.repeat(pin, 2))


def run_all(sc):
    a = (sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"])
    return {
        "newton": solve(*a, "clamp", eterms=sd, tol=TOL),
        "l-bfgs": solve_lbfgs(*a, element_eg, max_iter=6000, tol=TOL),
        "sobolev-lbfgs": world1.solve_sobolev_lbfgs(sc["x0"], sc["tris"], sc["rest"], sc["free"],
                                                    max_iter=6000, tol=TOL),
        "aqp": world1.solve_aqp(sc["x0"], sc["tris"], sc["rest"], sc["free"], max_iter=8000, tol=TOL),
        "gradient-descent": solve_gd(*a, element_eg, max_iter=8000, tol=TOL),
        "adam(lr=.01)": solve_adam(*a, element_eg, lr=0.01, max_iter=4000, tol=TOL),
    }


def tag(r):
    return f"{r['iters']} it" if r["status"] == "converged" else r["status"]


# HW-independent per-iteration cost (docs/metrics.md Lever 1): global factorizations to converge.
# Newton (dense-direct) factorizes the Hessian EACH iteration; Sobolev-L-BFGS and AQP prefactor a
# fixed operator ONCE; L-BFGS/GD/Adam never factorize. So a low iteration count that costs a
# factorization each is not cheaper than a higher count of back-solve-only iterations.
def facs(m, r):
    if m == "newton":
        return r["iters"] if r["status"] == "converged" else "—"
    if m in ("sobolev-lbfgs", "aqp"):
        return 1
    return 0


def main():
    print("== E2: World-1 accelerators across regimes ==\n")
    scens = [("well-conditioned (perturbation)", build_scenario(nx=8, ny=8)),
             ("ill-conditioned (3x stretch)", ill_scenario(n=8, s=3.0))]
    data = {}
    order = ["newton", "l-bfgs", "sobolev-lbfgs", "aqp", "gradient-descent", "adam(lr=.01)"]
    for name, sc in scens:
        res = run_all(sc); data[name] = res
        print(name)
        for m in order:
            r = res[m]
            print(f"  {m:16s} {tag(r):>12}  wall={r['wall_s']*1e3:8.1f}ms  |g|={r['final_grad_inf']:.1e}")
        print()

    lines = ["# E2 - World-1 accelerators across regimes (measured)", "",
             "All minimize the SAME symmetric-Dirichlet energy. Same criterion (|g|inf<1e-6). "
             "Two regimes, because World-1 superiority claims are regime-specific. "
             "Run: `python -m bench.run_e2`.", ""]
    for name, _ in scens:
        lines += [f"## {name}", "",
                  "| method | iters / status | global factorizations | wall (ms) |",
                  "|---|---|---|---|"]
        for m in order:
            r = data[name][m]
            lines.append(f"| {m} | {tag(r)} | {facs(m, r)} | {r['wall_s']*1e3:.1f} |")
        lines.append("")
    well = data[scens[0][0]]; ill = data[scens[1][0]]

    def it(res, m):
        return res[m]["iters"] if res[m]["status"] == "converged" else None
    lines += ["## Observed -- two claims decomposed", "",
              f"- **Sobolev preconditioning is a REAL, regime-specific win (validates a BCQN "
              f"component).** In the ill-conditioned regime Sobolev-L-BFGS ({it(ill,'sobolev-lbfgs')} "
              f"it) beats plain L-BFGS ({it(ill,'l-bfgs')} it); in the well-conditioned regime it "
              f"does not ({it(well,'sobolev-lbfgs')} vs {it(well,'l-bfgs')}). So the D0=L^-1 "
              f"Sobolev-init component helps *exactly* where it's designed to (ill-conditioning) "
              f"-- hardens the Sobolev-init claim as **`qualified`** (regime-dependent).",
              f"- **The AQP-beats-L-BFGS claim does NOT reproduce here.** AQP loses to L-BFGS in "
              f"BOTH regimes ({it(well,'aqp')} vs {it(well,'l-bfgs')} well; {it(ill,'aqp')} vs "
              f"{it(ill,'l-bfgs')} ill). The paper's ~200x AQP-over-L-BFGS number was against a "
              f"MATLAB L-BFGS baseline; against a well-implemented L-BFGS the advantage vanishes. "
              f"This is the classic **baseline-quality confound** ('did the method beat a *tuned* "
              f"generic optimizer?' -- here, no) -- flag `aqp->l-bfgs` as **confound-borne / "
              f"unreproduced** in this harness. (AQP's genuine claim is mesh-independent iteration "
              f"count + cheap per-iter, not raw iterations vs L-BFGS.)",
              f"- **Second order still wins iterations -- but iterations are not free:** Newton takes "
              f"only {it(well,'newton')}/{it(ill,'newton')} iters, yet **each is a full Hessian "
              f"factorization** ({it(well,'newton')}/{it(ill,'newton')} of them), whereas AQP and "
              f"Sobolev-L-BFGS prefactor a fixed operator **once** and L-BFGS never factorizes. So "
              f"the low Newton iteration count is the expensive kind; the factorizations column is "
              f"the honest HW-independent cost that iteration-count alone hides, and it is why "
              f"wall-clock does not track iterations (E4).",
              "",
              "_Caveat: dense Newton; single mesh/seed per regime. We report THREE axes per method -- "
              "iterations, global factorizations (HW-independent cost), and wall-clock (HW-dependent) "
              "-- because no single one settles a cross-method verdict (docs/metrics.md). AQP's "
              "mesh-independence claim (vs its L-BFGS-speed claim) is a separate, still-open test (#29)._"]
    os.makedirs("results", exist_ok=True)
    with open("results/e2.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/e2.md")


if __name__ == "__main__":
    main()
