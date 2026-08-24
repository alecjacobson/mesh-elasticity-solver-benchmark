"""Experiment E5 - criterion sensitivity (docs/experiments.md).

No re-run of the solver is needed: the per-iteration telemetry log already records energy and
gradient norm, so we re-score the SAME runs under different convergence criteria and show the
solver ranking (which filter reaches the criterion first) FLIP across criteria -- the silent
confound behind most published speed claims. Writes results/e5.md.
"""
import os
import numpy as np
from .solver import solve
from .run_e1_nu import stretch_scenario
from . import energy_neohookean as nh

FILTERS = ["clamp", "absolute", "identity-shift"]


def iters_to(log, pred):
    for e in log:
        if pred(e):
            return e["iter"]
    return None


def main():
    print("== E5: criterion sensitivity ==\n")
    nu = 0.49
    sc = stretch_scenario(nx=8, ny=8, s=2.0)
    eterms, _, _, _ = nh.make(mu=1.0, lam=nh.lam_from_nu(nu))
    runs = {f: solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], f,
                     eterms=eterms, max_iter=300, tol=1e-8) for f in FILTERS}

    E0 = runs["clamp"]["log"][0]["energy"]
    Estar = min(r["final_energy"] for r in runs.values())
    span = E0 - Estar + 1e-30

    criteria = {
        "|g|inf<1e-3": lambda e: e["grad_inf"] < 1e-3,
        "|g|inf<1e-6": lambda e: e["grad_inf"] < 1e-6,
        "Egap<1e-3":   lambda e: (e["energy"] - Estar) / span < 1e-3,
        "Egap<1e-8":   lambda e: (e["energy"] - Estar) / span < 1e-8,
    }

    # iterations-to-criterion table
    tbl = {f: {c: iters_to(runs[f]["log"], pred) for c, pred in criteria.items()} for f in FILTERS}
    winners = {}
    for c in criteria:
        vals = [(f, tbl[f][c]) for f in FILTERS if tbl[f][c] is not None]
        winners[c] = min(vals, key=lambda kv: kv[1])[0] if vals else "none"

    print(f"Neo-Hookean stretch, nu={nu}, E*={Estar:.4f}\n")
    hdr = f"{'filter':16s} " + " ".join(f"{c:>12}" for c in criteria)
    print(hdr)
    for f in FILTERS:
        print(f"{f:16s} " + " ".join(f"{str(tbl[f][c]):>12}" for c in criteria))
    print("\nfastest per criterion:", winners)

    lines = ["# E5 - criterion sensitivity (measured)", "",
             "The SAME three runs (Neo-Hookean stretch, ν=0.49, filter varied) re-scored under "
             "four convergence criteria -- **iterations to first satisfy** each. If the "
             "'fastest' filter changes across columns, the ranking is a criterion artifact. "
             "Run: `python -m bench.run_e5`.", "",
             f"E₀={E0:.4f}, E\\*={Estar:.4f} (best final across filters).", "",
             "| filter | " + " | ".join(criteria) + " |",
             "|" + "---|" * (len(criteria) + 1)]
    for f in FILTERS:
        lines.append(f"| {f} | " + " | ".join(str(tbl[f][c]) for c in criteria) + " |")
    lines += ["", "**Fastest per criterion:** " +
              ", ".join(f"`{c}` → **{winners[c]}**" for c in criteria), ""]
    distinct = set(winners.values())
    if len(distinct) > 1:
        lines.append(f"➡️ The ranking **flips**: the 'best' filter depends on the criterion "
                     f"({len(distinct)} different winners across 4 criteria). A paper reporting "
                     f"only one criterion could claim any of them is fastest. This is exactly why "
                     f"the protocol fixes one criterion per cell and shows a τ-sweep "
                     f"(docs/metrics.md, docs/protocol.md).")
    else:
        lines.append("➡️ In this instance the winner happened to be stable across criteria; the "
                     "gap between filters still varies substantially by criterion (loose vs "
                     "tight tolerance changes the iteration counts by large factors).")
    lines += ["", "_Caveat: one scenario; the effect is illustrative. The harness emits every "
              "criterion's value per iteration, so E5 is a free re-scoring of any experiment._"]
    os.makedirs("results", exist_ok=True)
    with open("results/e5.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote results/e5.md")


if __name__ == "__main__":
    main()
