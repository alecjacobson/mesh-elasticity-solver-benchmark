# v1 Benchmark Protocol (frozen)

The v1 contract: **what problems, what reference, what budget, what is measured.** Freezing this
before implementation is what makes the results defensible (Beiranvand–Hare–Lucet topic 1: state
the goal and fix the setup up front). v1 is contact-free (Worlds 1–2); Track 2 (contact) is v2.
Metrics are defined in [`metrics.md`](metrics.md); the harness that runs this is
[`harness.md`](harness.md).

## 0. Goal statement (what v1 is licensed to conclude)

"For contact-free elastic-energy minimization, attribute each component's contribution to
convergence / cost / robustness by swapping **one** component at a time on a fixed problem set,
and report where published head-to-head claims survive confound control." Not a leaderboard —
an attribution.

## 1. Suite 1a — static distortion / parametrization (World 1)

- **Energy (fixed):** symmetric Dirichlet on free-boundary UV. (MIPS as a declared secondary
  energy for a generality check, not the primary axis.)
- **Task:** map each surface patch to the plane minimizing distortion; initialization = Tutte
  (or, for the adversarial stratum, a deliberately folded init).
- **Instance strata** (fixed, versioned):
  - *easy:* smooth, well-conditioned disk-topology patches.
  - *typical:* Thingi10K-sourced patches (real, irregular) + Shay–Solomon–Stein 2022 parameterization set.
  - *adversarial:* near-degenerate/sliver meshes; folded/inverted initializations.
  - *feasibility sub-suite:* the Du-2020 locally-injective-mappings benchmark (11,647 tri/tet)
    for the injectivity/success-rate cell — scored on success, **not** speed (never co-mingled).
- **Convergence criterion (fixed):** BCQN characteristic gradient norm below τ.

## 2. Suite 1b — quasistatic / dynamic hyperelastic (World 2)

- **Energy (fixed):** Stable Neo-Hookean. (Corotated + ARAP as declared secondary energies.)
- **Task:** quasistatic equilibrium and a short implicit-Euler dynamic sequence under prescribed
  boundary displacement / body load.
- **Instance strata** (fixed, versioned): stress/twist/compression scenes over a grid of
  **resolution × element order (P1/P2) × stiffness**, plus a **near-incompressible ν sweep**
  ν ∈ {0.3, 0.45, 0.49, 0.499, 0.4999} — the regime where absolute/trust-region filtering claim
  their edge.
- **Convergence criterion (fixed):** Newton decrement λ² below τ.

## 3. Controls (mandatory)

- **C1 — locking-free element in the ν sweep.** The ν-sweep runs a locking-free formulation
  (mixed u–p / F-bar / Simo three-field; deal.II step-44 as the oracle). Displacement-only P1
  tets would confound solver robustness with volumetric locking and wrongly credit the filter.
  *Report both P1 and locking-free to separate the effects.*
- **C2 — load-parametrization policy.** Declared once and applied uniformly: quasistatic scenes
  may use monotone load stepping; scenes with limit points use arc-length. **Divergence at a
  snap-through / limit point is NOT scored as a solver failure** (it's a parametrization
  artifact). Dynamic scenes use a fixed Δt per stratum.
- **Fixed discretization per comparison.** When swapping a solver component, mesh + element +
  energy + tolerance are held identical — the fairness gate from `taxonomy.md`.

## 4. Reference-solution protocol (the E\*/trajectory question)

- **1a energy gap (metric #4):** the reference minimizer per instance is the **best converged
  result across all closed-division solvers** at the tightest tolerance (Moré–Wild convention).
  Bias risk (favors the strongest solver) is mitigated by (a) also reporting the raw geometric
  distortion (#37–#41), which needs no E\*, and (b) cross-checking a random 5% of instances
  against an independent high-accuracy solve (TinyAD-backed projected Newton to τ=1e-12).
- **1b trajectory error (metric #47):** the reference trajectory is a high-accuracy run —
  **Δt/8, Newton decrement τ=1e-12, validated against a PETSc SNES / FEniCS oracle** on a fixed
  subset. Errors reported over **short horizons** (dynamics can be chaotic); statistics, not a
  single endpoint.
- **Oracle gate:** before any component swap is trusted, the harness's Newton-LS + Cholesky base
  config must reproduce the PETSc SNES / FEniCS oracle on a fixed BVP (harness.md §4).

## 5. Budget & tuning rules (fairness)

- **Equal budget per cell:** a fixed compute budget (hardware-independent: max evals / mat-vecs;
  plus a wall-clock cap) applied identically to every solver in a head-to-head.
- **No per-problem tuning.** Each component carries a **single hyperparameter block fixed across
  the entire suite** (Beiranvand–Hare–Lucet / COCO rule). Any per-instance adaptivity must be
  part of the algorithm, not hand-set. Tuning fragility (metric #62) is itself reported.
- **Warm-starting** is allowed only if uniform across compared solvers and declared.

## 6. Metric protocol (per cell → see `metrics.md`)

- Report the **per-cell orthogonal core** (metrics.md Part 2), always pairing a hardware-
  dependent cost (wall-clock) with a hardware-independent proxy (iterations / linear-solves /
  mat-vecs).
- Aggregate each cell with a **hardware-independent data profile** + a paired **wall-clock data
  profile**; performance profiles only for isolated two-solver views (Gould–Scott caveats).
- Fix τ per cell; show a τ ∈ {1e-3, 1e-6} sweep to prove orderings aren't cutoff artifacts.
- Report run-to-run variance as error bars; record hardware/threads/precision/library versions
  and the full solver set with every profile.

## 7. Tiers & governance

- **Closed division:** conformance-passing components, one-slot swaps — the fair convergence numbers.
- **Open division:** experimental / non-conformant components — breadth only, never mixed into
  closed rankings.
- **Hidden/rotating tier:** ~15% of each stratum withheld to detect overfitting; rotated per release.

## 8. Frozen vs still-open

**Frozen (this doc):** energies, criteria, strata definitions, controls C1/C2, reference
protocol, budget/tuning rules, aggregation. **Still open (tracked):** exact mesh lists per
stratum (curation task under #2); the P2 tangent for mixed u–p (implementation under #3); the
precise ν-sweep element for C1 (deal.II step-44 port).
