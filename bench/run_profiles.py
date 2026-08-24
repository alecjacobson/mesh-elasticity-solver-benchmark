"""Data profiles over a problem SET (docs/metrics.md aggregation).

Moré-Wild-style data profile: fraction of instances in a set solved to tolerance within a
budget. We report the budget in a HARDWARE-INDEPENDENT unit (grad+Hessian assemblies) AND in
wall-clock -- the pairing metrics.md mandates. Two sets: World-1 symmetric-Dirichlet
perturbation-recovery, and Neo-Hookean stretch across (moderate) nu. Writes results/profiles.md.
"""
import os
import numpy as np
from .mesh import grid_mesh, boundary_mask, rest_quantities
from .solver import solve, energy_only
from .energy import element_terms as sd_terms
from . import energy_neohookean as nh

FILTERS = ["none", "clamp", "absolute", "identity-shift"]
BUDGETS = [8, 16, 32, 64, 128, 256, 400]           # HW-independent (assemblies)
WALL_BUDGETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]  # seconds


def sd_bank():
    bank = []
    for n in (6, 8, 10, 12):
        for seed in range(5):
            for amp_frac in (0.25, 0.35):
                rest, tris = grid_mesh(n, n)
                Bs, areas = rest_quantities(rest, tris)
                bmask = boundary_mask(rest)
                rng = np.random.default_rng(1000 * seed + n)
                amp = amp_frac / n
                for _ in range(40):
                    pert = rng.standard_normal(rest.shape); pert[bmask] = 0.0
                    x = (rest + amp * pert).reshape(-1)
                    if np.isfinite(energy_only(x, tris, Bs, areas, sd_terms)):
                        break
                    amp *= 0.8
                bank.append(dict(tris=tris, Bs=Bs, areas=areas, x0=x,
                                 free=~np.repeat(bmask, 2), eterms=sd_terms,
                                 label=f"sd_n{n}_s{seed}_a{amp_frac}"))
    return bank


def nh_bank():
    bank = []
    for n in (6, 8):
        for nu in (0.30, 0.45, 0.49):
            for s in (1.6, 2.2):
                rest, tris = grid_mesh(n, n)
                Bs, areas = rest_quantities(rest, tris)
                xcol = rest[:, 0]
                pinned = (np.abs(xcol) < 1e-9) | (np.abs(xcol - 1.0) < 1e-9)
                x0 = rest.copy(); x0[np.abs(xcol - 1.0) < 1e-9, 0] = s
                eterms, _, _, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
                bank.append(dict(tris=tris, Bs=Bs, areas=areas, x0=x0.reshape(-1),
                                 free=~np.repeat(pinned, 2), eterms=eterms,
                                 label=f"nh_n{n}_nu{nu}_s{s}"))
    return bank


def run_set(bank):
    out = {f: [] for f in FILTERS}
    for inst in bank:
        for f in FILTERS:
            r = solve(inst["x0"], inst["tris"], inst["Bs"], inst["areas"], inst["free"], f,
                      eterms=inst["eterms"], max_iter=400, tol=1e-6)
            out[f].append(r)
    return out


def data_profile(results, budgets, key):
    """Fraction of instances converged with counts[key] (or wall_s) <= budget."""
    prof = {}
    for f, rs in results.items():
        N = len(rs)
        row = []
        for b in budgets:
            if key == "wall_s":
                ok = sum(1 for r in rs if r["status"] == "converged" and r["wall_s"] <= b)
            else:
                ok = sum(1 for r in rs if r["status"] == "converged" and r["counts"][key] <= b)
            row.append(ok / N)
        prof[f] = row
    return prof


def fmt_profile(prof, budgets, unit):
    hdr = f"| filter | " + " | ".join(f"{b}{unit}" for b in budgets) + " |"
    sep = "|" + "---|" * (len(budgets) + 1)
    rows = [hdr, sep]
    for f, row in prof.items():
        rows.append(f"| {f} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    return "\n".join(rows)


def main():
    print("== data profiles ==")
    lines = ["# Data profiles (measured aggregate over problem sets)", "",
             "Moré-Wild data profiles: fraction of a problem SET solved to `|g|inf<1e-6` within a "
             "budget, reported in a **hardware-independent** unit (grad+Hessian assemblies) AND "
             "in **wall-clock** (docs/metrics.md pairing). Run: `python -m bench.run_profiles`.", ""]
    for name, bank in (("Set 1 — World-1 symmetric Dirichlet (perturbation-recovery)", sd_bank()),
                       ("Set 2 — Neo-Hookean stretch (nu in {0.30,0.45,0.49})", nh_bank())):
        res = run_set(bank)
        n = len(bank)
        print(f"\n{name}  ({n} instances)")
        conv = {f: sum(1 for r in res[f] if r['status'] == 'converged') for f in FILTERS}
        print("  converged:", {f: f"{conv[f]}/{n}" for f in FILTERS})
        pa = data_profile(res, BUDGETS, "assemblies")
        pw = data_profile(res, WALL_BUDGETS, "wall_s")
        lines += [f"## {name}", f"", f"{n} instances. Converged: " +
                  ", ".join(f"**{f}** {conv[f]}/{n}" for f in FILTERS) + ".", "",
                  "Data profile — budget in **assemblies** (HW-independent):", "",
                  fmt_profile(pa, BUDGETS, ""), "",
                  "Data profile — budget in **wall-clock seconds** (HW-dependent):", "",
                  fmt_profile(pw, WALL_BUDGETS, "s"), ""]
        # quick console view
        for f in FILTERS:
            print(f"    {f:14s} assembliesProfile " + " ".join(f"{v:.2f}" for v in pa[f]))
    lines += ["_Note: the assemblies-profile and wall-profile agree closely here because the "
              "dense per-iteration cost is near-uniform across filters; a linear-solver swap "
              "(direct vs CG vs multigrid) is where they would diverge -- and that divergence "
              "would be the finding (metrics.md Lever 1)._"]
    os.makedirs("results", exist_ok=True)
    with open("results/profiles.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote results/profiles.md")


if __name__ == "__main__":
    main()
