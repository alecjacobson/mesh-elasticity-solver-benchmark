# Paper Outline (STAR-style survey + benchmark)

Target: a State-of-the-Art Report (Eurographics/CGF norm, ~20pp) that is *also* a benchmark —
a citable v1 snapshot whose harness seeds a living benchmark. The punchline is **honest
attribution**, not a leaderboard. Built on `taxonomy.md`, `corpus.md`, `metrics.md`,
`harness.md`, `protocol.md`, `experiments.md`, and the `claims/` graph.

## Thesis (one sentence)

A decade of "faster/robuster elastic solvers" is largely **component swaps inside one shared
metric-descent iteration**; papers change several components at once and credit one, so we
(i) give a taxonomy + a unifying view that makes the components explicit, (ii) map each
"innovation" to its classical ancestor, and (iii) benchmark component-by-component to report
which published superiority claims survive confound control.

## Structure

1. **Introduction.** The proliferation of elastic-solver papers; the entanglement problem
   (2–3 changes, one credit); the two failure poles (insular/narrow vs unfair/broad). Contributions:
   taxonomy, unifying view, lineage map, superiority-claims graph, component-factored benchmark
   with honest-attribution findings.

2. **The unifying view.** Metric descent `x' = x − α M⁻¹∇E`; the six component axes as choices
   of/modifications to `M` and the step. Table: `M` = I / ∇²E / Fisher / Laplacian-H¹ / Killing
   / reweighted-energy ⇒ GD / Newton / natural-gradient / Sobolev(AQP,BCQN) / AKVF / SLIM.
   (Source: `design.md` §12.1.) Payoff: "graphics Sobolev preconditioning = ML natural gradient
   under a chosen metric," with the honest caveat (Fisher undefined here; PINN energy-NGD
   collapses to Gauss-Newton for positions-as-unknowns).

3. **Taxonomy.** Six method axes × problem-class/capability cells; the three "worlds"
   (static distortion / hyperelastic / contact) that share machinery but not metrics;
   orthogonality evaluation; the fairness gate. (Source: `taxonomy.md`.)

4. **Survey by axis** (the annotated corpus, organized by component, not chronology):
   energy · search direction · Hessian filter · line search · linear solver · convergence
   criterion — each subsection covers the graphics methods **and** their World-0 baselines.
   (Source: `corpus.md`.)

5. **Lineage map.** Graphics "innovations" ⇐ named classical ancestors (eigenvalue filtering ⇐
   modified Cholesky / Gill–Murray + Abaqus viscous stabilization; AQP ⇐ Nesterov; Anderson-PD ⇐
   Anderson 1965; PD ⇐ ADMM + Gauss-Newton; IPC ⇐ interior-point + augmented-Lagrangian contact;
   near-incompressible handling ⇐ F-bar/mixed-u–p). Cite as adaptations, not inventions — arguably
   the paper's sharpest contribution. (Source: `design.md` §12.2.)

6. **The superiority-claims graph.** Who claims to beat whom, on what dimension, with what
   status. The honesty patterns: fixed-budget vs converged; GPU-vs-CPU confounds; entanglement;
   author disclaimers. A figure (the Mermaid clusters) + the qualified-edge table.
   (Source: `claims/`.)

7. **Benchmark design.** The component-factored harness (config = point in component space;
   conformance-suite admissibility; official-code-first + regression tested), the metric
   deliberation (per-cell orthogonal core; HW-dependent paired with HW-independent; data
   profiles), and the frozen protocol (strata, controls C1/C2, reference protocol, equal budget,
   no per-problem tuning). (Sources: `harness.md`, `metrics.md`, `protocol.md`.)

8. **Results — the decomposition experiments (E1–E5).** Filter isolation; seed-claim
   decomposition; BCQN triple-split; first-vs-second-order wall-clock inversion; criterion
   sensitivity. Each reported as attributed-vs-confound-borne, with the claims-graph edges it
   hardened. (Source: `experiments.md`.) **This section is the paper's reason to exist.**

9. **What survived.** The hardened claims graph as the summary contribution: which decade-old
   superiority claims are `validated`, which are `qualified` (and by what regime), which are
   `refuted`. Reflection on how much published advantage was confound-borne.

10. **Open problems & the living benchmark.** Contact track (v2); learned-accelerator companion
    track; the hidden-tier governance; what the community should standardize.

## Figures (planned)

- F1: the metric-descent table (unifying view).
- F2: taxonomy grid (axes × cells) with entanglement flags.
- F3: lineage map (graphics ⇐ classical), as a two-column diagram.
- F4: claims-graph clusters (Mermaid → vector), colored by status.
- F5–F9: one per decomposition experiment (data profiles + the attribution bar).
- F10: the hardened claims table (before/after status).

## Scoping discipline (so it's a STAR, not a bibliography)

Organize by **axis and claim**, not by paper; every method appears where its *contribution*
lives; the benchmark results carry the narrative. Cross-cutting inclusion boundary (why neural
surrogates/PINNs are excluded from the core) stated once, tied to the scope ledger
(`design.md` §12.3).

## Status / mapping to issues

Each section maps to a doc already in the repo; drafting = lifting + tightening those into prose.
Blocked only on: benchmark *results* (E1–E5 need the harness, P1) for §8–9. §2–7 are writable now.
