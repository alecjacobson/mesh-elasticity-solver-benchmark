# The Living Benchmark — divisions, admissibility, reproducibility

This document is the operational seed of the living benchmark described in the paper (§10). It says
how to **reproduce** every result, how the **divisions** are structured, and what counts as a
**contribution** (and how it is admitted). It complements [`CONTRIBUTING.md`](../CONTRIBUTING.md)
(curation conventions) — this file governs *measured* contributions.

## 1. Reproducibility

Everything measured is regenerable from `bench/` (Python/NumPy; SciPy; matplotlib; polyscope for the
one 3D figure). Nothing is hand-entered.

```bash
# 1. admissibility gate — must pass before any number is trusted (8 gates)
python -m bench.conformance

# 2. regenerate a result (each results/<name>.md is written by one runner)
python -m bench.run_e1            # E1 filter isolation            -> results/e1.md
python -m bench.run_e3            # E3 BCQN 2^3 factorial          -> results/e3.md
python -m bench.run_1a_profiles   # full 1a performance profiles   -> results/1a_profiles.md
python -m bench.run_injectivity   # feasibility / untangling       -> results/injectivity.md
python -m bench.run_twist_analysis# twist-eigenvalue analysis      -> results/twist_analysis.md
python -m bench.run_world2_filters# clamp/absolute/TR, P1 vs P2    -> results/world2_filters.md
python -m bench.run_p2_stable_nu  # the nu-claim, both confounds   -> results/p2_stable_nu.md
# ... one runner per results/*.md; the module name mirrors the file.

# 3. regenerate all figures (deterministic), or one by name
python -m bench.run_figures                     # all 20
python -m bench.run_figures locking twist_phase # specific

# 4. assemble the paper draft
python paper/assemble.py                          # -> paper/paper.md
```

Two performance runners (`run_1a_profiles`, the near-incompressible sweeps) do dense solves and take
several minutes; `run_1a_profiles` disk-caches its solve so the doc and figure share one computation.
Every quantitative claim in `paper/paper.md` cites the `results/*.md` it comes from; the
reference-integrity check (all cited files exist, headline numbers match) is part of the paper build.

## 2. Divisions

A leaderboard rewards the wrong thing (raw speed, often confounded). The living benchmark is instead
organized so that a contribution is *a controlled, reproducible attribution*:

- **Closed division** — components are fixed and a single axis is raced (the decomposition-experiment
  discipline). This is where superiority claims are promoted/qualified: change one slot
  (energy / filter / direction / line search / linear solver / criterion), hold the rest, report the
  hardware-independent count. Results here move edges of the claims graph.
- **Open division** — any method on a shared problem set and an independent reference `E*`. Useful for
  end-to-end comparison, but a win here is attributed only to "the whole method," not a component.
- **Hidden tier** — a rotating, held-out set of problem instances (meshes, boundary conditions, seeds)
  never published, used to check that a submission did not overfit the public strata. Rankings are
  reported on the hidden tier; the public set is for development.

All three race on **hardware-independent verdicts** (iterations, factorizations, back-solves,
matrix-vector products), with wall-clock reported only where the comparison is implementation-fair.
Robustness is a Dolan–Moré performance profile + a Moré–Wild data profile with pairwise (Gould–Scott)
win-fractions — never a single speed number.

## 3. What counts as a contribution, and how it is admitted

A contribution is **an edge of the claims graph moving from `self-claimed` toward
`validated`/`qualified`, by a single-axis experiment, reproducibly** — not "my method is fastest."
Concretely, to add a measured result:

1. **Implement the component as a slot** in `bench/` (energy / filter / direction / line search /
   solver / criterion), grounded in official code where it exists.
2. **Pass the conformance gate.** Add a gate to `bench/conformance.py`: analytic derivatives vs finite
   differences, and — for a ported method — an **official-code regression** reproducing the source
   implementation's energy/step on a shared instance. *A component with no passing gate is not
   admissible into any comparison* (the invariant from `CONTRIBUTING.md` §1).
3. **Run the single-axis experiment**, writing a `results/<name>.md` from a `bench.run_<name>` runner,
   using an independent reference `E*` (not best-of-compared) and reporting hardware-independent counts
   with the regime caveat.
4. **Update the claims edge** in `claims/claims.yaml` with the measured status and a note citing the
   result; regenerate `figures/claims_ledger.png` / `claims_network.png`.
5. **Survive adversarial review.** Per the paper's §9 discipline, a submitted result must survive an
   attempt to find its confound (element, energy, tolerance, baseline quality, hardware, pooled
   interaction). The review is part of acceptance, not an afterthought — the benchmark audits itself as
   adversarially as it audits the literature.

Deferred contributions that would be especially valuable: faithful ports of **Composite Majorization**
(the twist-mode convex majorizer, §8.5) and the individual **injectivity-cohort** methods (TLC's lifted
content, foldover-free) — both need their source paper's specific construction, which is why they are
open invitations rather than substituted look-alikes.
