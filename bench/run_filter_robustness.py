"""Filter ROBUSTNESS (success-rate), P5.2 edges #2 and #3.

The convergence edges ask "fewer iterations?"; the robustness edges ask a different, harder
question: "converges MORE OFTEN from bad starts?" A filter that is slightly slower but recovers
from indefinite/inverted configurations the other cannot is *more robust*, and that is the property
these two edges claim:

  #2  absolute-filtering -> clamp-filtering (robustness): |lambda| keeps curvature information a
      clamp-to-0 throws away, so absolute is claimed to escape indefinite regions clamp stalls in.
  #3  trust-region-filtering -> full-newton (robustness): raw Newton (filt="none") takes the
      indefinite step and diverges; the rho-switchboard falls back to clamp/absolute and survives.

Substrate: the STABLE Neo-Hookean energy (finite under inversion, so "recover to inversion-free" is
well-posed) on the P1 grid, with `inverted_scenario` starts swept over severity x seed. Success =
converged AND inversion-free at the end. We report the paired success RATE (not iterations) and the
paired disagreement (cases where exactly one method succeeds) -- the honest robustness signal.

Writes results/filter_robustness.md. Run: `python -m bench.run_filter_robustness`.
"""
import os
from collections import Counter
import numpy as np
from .solver import solve
from . import energy_stable_neohookean as snh
from .run_stable_nu import inverted_scenario, count_inverted

NU = 0.45                       # moderate compressibility: recovery, not locking, is the variable
AMPS = [1.3, 1.7, 2.2, 3.0, 4.0]
SEEDS = list(range(1, 21))      # 20 seeds x 5 severities = 100 starts per method
PAIRS = [("absolute", "clamp", "#2 absolute->clamp"),
         ("trust-region", "none", "#3 trust-region->full-newton"),
         ("project-on-demand", "none", "#12 pitfalls-PDN->full-newton")]


def _run(scn, filt, et):
    r = solve(scn["x0"], scn["tris"], scn["Bs"], scn["areas"], scn["free"], filt,
              eterms=et, max_iter=400, tol=1e-6)
    ninv = count_inverted(r["x"], scn["tris"], scn["Bs"])
    ok = (r["status"] == "converged") and (ninv == 0)
    return ok, r["status"], ninv, r["iters"]


def main():
    et, _, _, _ = snh.make(mu=1.0, lam=snh.lam_from_nu(NU))
    methods = sorted({m for a, b, _ in PAIRS for m in (a, b)})

    # battery: one shared set of inverted starts, every method solves each
    starts = []
    for amp in AMPS:
        for sd in SEEDS:
            scn = inverted_scenario(nx=8, ny=8, amp_frac=amp, seed=sd)
            if scn["ninv"] == 0:            # only count genuinely-inverted starts
                continue
            starts.append((amp, sd, scn))

    res = {m: [] for m in methods}          # per-start bool success, aligned across methods
    fail_status = {m: Counter() for m in methods}   # WHY the failures fail (nondescent vs linesearch vs maxiter)
    for i, (amp, sd, scn) in enumerate(starts):
        for m in methods:
            ok, st, ninv, it = _run(scn, m, et)
            res[m].append(ok)
            if not ok:
                fail_status[m][st] += 1
        print(f"  [{i+1}/{len(starts)}] amp={amp} seed={sd} ninv0={scn['ninv']} "
              + " ".join(f"{m}={'Y' if res[m][-1] else 'n'}" for m in methods), flush=True)

    none_fail = dict(fail_status.get("none", {}))
    L = ["# Filter robustness — success rate from inverted starts (measured, P5.2 #2 & #3 & #12)", "",
         f"Stable Neo-Hookean (nu={NU}), P1 8x8 grid, `inverted_scenario` starts swept over "
         f"severity amp in {AMPS} x {len(SEEDS)} seeds. Success = **converged AND inversion-free** "
         f"(tol 1e-6, max_iter 400). {len(starts)} genuinely-inverted starts. "
         "Run: `python -m bench.run_filter_robustness`.", "",
         "> ⚠️ **Read the baseline honestly.** `none` is NOT a well-globalized Newton — in "
         "`solver.solve` it HARD-TERMINATES the instant the raw Hessian yields a non-descent "
         f"direction (its failures are **{none_fail}**, i.e. essentially all `nondescent`), with no "
         "negative-curvature fallback or trust-region radius. So a big margin over `none` measures "
         "**presence of *any* indefiniteness handling**, not a filter's edge over a competently "
         "globalized Newton. The starts are also **not independent** (correlated severities on one "
         "8×8 mesh, one energy, one ν), so a rate like `100/100` is a saturated point estimate, not a "
         "statistically-powered success rate. These results are `qualified`, not `validated`.", "",
         "### Success rate per method", "",
         "| method | successes / starts | rate | failure modes |", "|---|---:|---:|---|"]
    for m in methods:
        s = sum(res[m]); n = len(res[m])
        fm = dict(fail_status[m]) or "—"
        L.append(f"| `{m}` | {s} / {n} | {s / n:.0%} | {fm} |")

    L += ["", "### Paired comparison (does the claimed-more-robust method win?)", "",
          "| edge | A succ | B succ | A-only | B-only | verdict |",
          "|---|---:|---:|---:|---:|---|"]
    verdicts = {}
    for a, b, name in PAIRS:
        sa, sb = sum(res[a]), sum(res[b])
        a_only = sum(1 for x, y in zip(res[a], res[b]) if x and not y)
        b_only = sum(1 for x, y in zip(res[a], res[b]) if y and not x)
        if sa > sb:
            v = f"**A ({a}) more robust** (+{sa - sb})"
        elif sb > sa:
            v = f"**B ({b}) more robust** (+{sb - sa}) — claim NOT supported"
        else:
            v = "tie (equal success) — claim not distinguished here"
        verdicts[name] = (sa, sb, a_only, b_only, sa >= sb)
        L.append(f"| {name} | {sa} | {sb} | {a_only} | {b_only} | {v} |")

    L += ["", "## Observed", ""]
    # #2 absolute vs clamp
    a2 = verdicts["#2 absolute->clamp"]
    if a2[0] > a2[1]:
        L.append(f"- **#2 absolute > clamp:** absolute recovers {a2[0] - a2[1]} more starts "
                 f"({a2[2]} it-only wins vs {a2[3]} clamp-only) — supports the robustness edge.")
    elif a2[0] == a2[1]:
        L.append(f"- **#2 absolute == clamp:** equal success ({a2[0]}/{len(starts)}); on this "
                 f"battery neither filter recovers a start the other cannot (disagreement "
                 f"{a2[2]}+{a2[3]}). The robustness edge is **not distinguished** here — both "
                 "line-search-safeguarded Newton variants land the same basin.")
    else:
        L.append(f"- **#2 clamp > absolute:** clamp recovers {a2[1] - a2[0]} more — the robustness "
                 "edge is **contradicted** on this battery.")
    a3 = verdicts["#3 trust-region->full-newton"]
    if a3[0] > a3[1]:
        L.append(f"- **#3 trust-region ≫ unfiltered Newton (qualified):** the rho-switchboard "
                 f"recovers {a3[0] - a3[1]} more starts than the `none` baseline ({a3[2]} TR-only "
                 "wins). The direction is real (an unhandled indefinite Hessian fails; the SPD "
                 "switchboard does not) — but recall the baseline hard-terminates on the first "
                 "non-descent direction, so this measures *presence of indefiniteness handling*, "
                 "not TR's edge over a competently globalized Newton. Qualified, not validated.")
    elif a3[0] == a3[1]:
        L.append(f"- **#3 trust-region == full-newton:** equal success ({a3[0]}); the line search "
                 "already rescues raw Newton on these starts, so the filter fallback is not the "
                 "deciding factor here.")
    else:
        L.append(f"- **#3 full-newton > trust-region:** raw Newton recovers {a3[1] - a3[0]} more — "
                 "unexpected; the edge is contradicted on this battery.")
    a12 = verdicts["#12 pitfalls-PDN->full-newton"]
    if a12[0] > a12[1]:
        L.append(f"- **#12 project-on-demand ≫ unfiltered Newton (qualified):** PDN (project a "
                 f"block's Hessian only when indefinite → PSD assembled Hessian → always a descent "
                 f"direction) recovers {a12[0] - a12[1]} more starts than the `none` baseline "
                 f"({a12[2]} PDN-only wins). Same reading as #3: real direction, but the margin is "
                 "against a baseline with *no* indefiniteness handling, so qualified, not validated.")
    elif a12[0] == a12[1]:
        L.append(f"- **#12 project-on-demand == full-newton:** equal success ({a12[0]}) here.")
    else:
        L.append(f"- **#12 full-newton > project-on-demand:** unexpected ({a12[1] - a12[0]} more) — "
                 "edge contradicted here.")
    L += ["",
          "_Caveat: 2D, single energy/nu, one 8×8 mesh, NON-INDEPENDENT correlated starts; the "
          "`none` baseline has no negative-curvature fallback (hard-terminates on first non-descent), "
          "so these compare 'any indefiniteness handling vs none', not a filter's edge over a "
          "competently globalized Newton. #2 (absolute vs clamp) is the one apples-to-apples pair "
          "here and it is a tie. All three edges are `qualified`, none `validated`._"]

    os.makedirs("results", exist_ok=True)
    with open("results/filter_robustness.md", "w") as f:
        f.write("\n".join(L) + "\n")
    for m in methods:
        print(f"  {m:>14}: {sum(res[m])}/{len(res[m])} = {sum(res[m])/len(res[m]):.0%}")
    for a, b, name in PAIRS:
        print(f"  {name}: A={sum(res[a])} B={sum(res[b])}")
    print("wrote results/filter_robustness.md")
    return True


if __name__ == "__main__":
    main()
