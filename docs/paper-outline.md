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

8. **Results — the decomposition experiments.** **This section is the paper's reason to exist.**
   Harness (`bench/`, `results/`, 26 measured experiments, conformance-gated) measurements in hand.
   Each headline was hardened through **three rounds of adversarial peer review** (one reviewer per
   referenced paper, protective of its own claims), which repeatedly caught over-reach in *our own*
   prior conclusions — the benchmark's confound-untangling applied reflexively to itself:
   - **The ν-claim (the headline), settled across FOUR independent locking treatments.** The
     absolute-filtering paper claims absolute > clamp near incompressibility; on P1 constant-strain
     elements we find the *opposite* (absolute fails at ν=0.4999) — but that is a **volumetric-locking
     artifact**. Round 3 caught that the "P2 fixes it" result was on the *wrong* (classical barrier)
     energy; the definitive test needs **both** confounds removed. With the locking-relieved element
     **and** the correct Stable-Neo-Hookean energy, absolute **beats** clamp and its advantage
     *grows* toward the incompressible limit (48/38 at ν=0.4999 → 113/71 at 0.49999, `p2_stable_nu`).
     A locking artifact would *collapse* at the limit; instead it strengthens. **Four independent
     locking treatments now concur** — P1 crossed-mesh, standard P2, stable-NH P2, and a validated
     **selective-reduced-integration P2** (`sri_nu.md`, on which absolute crushes clamp 23 vs 250) —
     so the P1 "refutation" is robustly a discretization confound, not a filter property. *This is
     the paper's headline: a decade-old superiority claim that reverses, then re-validates, only once
     two entangled confounds (element + energy) are separately controlled.*
   - **"Innovations" that don't survive fair, faithful re-measurement.** (i) *Trust-region filtering*
     "beats both clamp and absolute": our round-1 win was an artifact of an expensive global-`eigh`
     operator; the faithful **per-element** blend (+ an SPD-probe schedule) reverses it — TR wins on
     the locking element (where plain filters struggle) but is a wash on the locking-relieved one — the operative axis is volumetric locking, not Hessian conditioning (`world2_filters`).
     (ii) *AQP mesh-independence*: real only to **loose** tolerance — a τ-sweep + CI-gated growth
     exponent shows it *grows* at tight τ (and CI-gating **retracted** our own "AQP scales worse than
     L-BFGS" as not statistically supported) (`mesh_independence`). (iii) *AQP's single-factorization
     wins at scale*: **refuted at tight τ** — measured factorization/back-solve counts + a
     sparse-Cholesky model show AQP's iteration blow-up makes it 1.5–2.2× Newton and rising
     (`scale_cost`). (iv) *AQP > L-BFGS ×200*: a MATLAB-baseline confound (`e2`).
   - **Confounds the benchmark quantifies.** First-vs-second-order (Newton wins iterations,
     **L-BFGS wins wall-clock**; Adam plateaus — honesty control, `e4`); criterion sensitivity (3
     "fastest" filters across 4 criteria, `e5`); C++/Python wall-clock confound (SLIM's compiled
     speed is not algorithmic — HW-independent *counts* carry every verdict, `slim`); Pitfalls of
     Projection (projection breaks **affine invariance** — Newton covariant to 3e-13, every filter
     O(1), `pitfalls`); filtering ≈ trust-region / modified-Newton lineage (`tr`); regime interaction
     (inertia makes filtering optional, `1b_dynamic`; a strong filter makes line-search inert,
     `linesearch`).

9. **What survived — and the review loop as method.** The hardened claims graph
   (`claims/hardening.md`): **2 validated (2D)**, **21 qualified**, **115 self-claimed**, **22
   unmeasured** (contact, deferred to v2). The summary contribution is not just *which* decade-old
   claims survive, but a demonstrated **methodology for honest attribution**: three adversarial
   review rounds (52 issues, 51 resolved) each caught real over-reach in the *previous* round's
   measurements — an operator-cost artifact, a loose-tolerance artifact, a wrong-energy flagship, an
   statistically-unsupported ordering — and forced retractions. The lesson for the field: superiority claims are
   entangled with element choice, energy, tolerance, baseline quality, and hardware, and a single
   confound rarely acts alone (the ν-claim needed *two* removed at once). Reflection on how much
   published advantage is confound-borne, and a call to report claims with the honest status ladder.

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
§2–7 writable now; **§8–9 now have prototype data** (`bench/` + `results/`, 26 experiments incl.
the ν-claim disentangled (indicative, 2D) via the P2 element). Remaining for a full paper: official-code-regression
ports, 3D, E2/E3 seed-method splits, and larger-scale reruns.
