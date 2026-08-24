# Untangling a Decade of Mesh-Elasticity Solver Claims — Design Notes (v0.1)

Status: pre-execution design. Built from a 5-way literature fan-out (~130 papers) + a
survey/benchmark methodology pass. See `corpus.md` for the annotated records.

---

## 1. Goal

Untangle the *claimed* numerical-method improvements for mesh elasticity problems (mostly
SIGGRAPH, ~2010–2025) by (a) building a defensible taxonomy and (b) narrowing to a subset
that can be **cleanly, fairly** compared in a common benchmark. The scientific product is
not "which paper wins" but **honest attribution**: most papers change 2–3 things at once
and credit one.

## 2. The field has 6 component axes and 3 "worlds"

Every method is a component swap inside one shared elastic-energy minimization
`min_x Σ_e V_e ψ(F_e(x)) (+ inertia, + barriers)`, solved by a descent iteration. The
axes:

1. **Energy ψ** — ARAP, symmetric Dirichlet/MIPS, (Stable) Neo-Hookean, corotated, StVK.
2. **Search direction** — full Newton, Gauss-Newton, projected Newton, L-BFGS, AQP proxy,
   AKVF/Sobolev, local-global/PD, ADMM, coordinate descent (VBD), Anderson.
3. **Hessian eigenvalue filtering / SPD projection** — none, clamp-to-ε, absolute value,
   trust-region adaptive, project-on-demand, progressive/selective, analytic eigensystem,
   spectral shift, eigenvalue blending.
4. **Line search / feasibility** — backtracking, exact, injectivity-barrier-aware, CCD-filtered.
5. **Linear solver / preconditioner** — direct Cholesky, PCG, multigrid, additive Schwarz, subspace.
6. **Convergence criterion + tolerance** — Newton decrement, characteristic gradient norm,
   backward-Euler residual, fixed iteration budget.

But the field does not partition cleanly by axis — it partitions by **problem class**, and
this is what governs comparability. Three worlds share solver machinery but **not metrics**:

- **World 1 — Static geometry/distortion optimization** (no inertia, no contact). UV
  parametrization & deformation. AQP, SLIM, Composite Majorization, BCQN, AKVF, ABCD,
  Splitting, TLC, GOSS. Metric: wall-clock-to-tolerance + final distortion; or success-rate
  from bad init.
- **World 2 — Quasistatic/dynamic hyperelasticity** (inertia / incremental potential, no
  contact). The projected-Newton / eigenvalue-filtering world. Stable Neo-Hookean, Analytic
  Eigensystems, Absolute / Trust-Region / Progressive filtering, Pitfalls of Projection,
  L-BFGS QN, ADMM, VBD, JGS2. Metric: residual-to-tolerance + iterations, OR fixed-budget error.
- **World 3 — Contact-coupled dynamics** (barriers, CCD, friction). The IPC world. IPC,
  C-IPC, ABD, GIPC, OGC, cubic-barrier, barrier-free, complementarity friction. Metric:
  **binary non-penetration guarantee** + timing + robustness stress tests.

Worlds 1 and 2 share the *entire* projected-Newton skeleton (only inertia differs). World 3
adds four parameters that belong to no solver: barrier stiffness / activation distance `d̂`,
CCD tolerance, friction `ε_v`, time-step `Δt`.

## 3. Taxonomy (meta-characteristic + capability cells)

Following Nickerson et al. 2013 (non-arbitrary taxonomy construction): pick ONE
meta-characteristic — **"what does this method change to improve elastic-energy
minimization, and under what problem class is the claim measured?"** — and derive all axes
from it. The 6 axes above are the dimensions; **problem class (World 1/2/3) × capability
(contact? inversion-safe? codimensional? general-energy vs restricted-energy?)** defines
the **capability cells**.

The taxonomy's job in the benchmark is a **fairness gate**: two methods get a head-to-head
*number* only if they occupy the same capability cell. Across cells we report *coverage /
robustness* (performance profiles), never a single speed number. This is the structural
resolution to the fairness problem (Unterkalmsteiner 2023: dimensions must be orthogonal &
mutually exclusive).

## 4. Central tension and its resolution

- **Too narrow** → benchmark overfitting / Goodhart; findings feel artificial & don't
  transfer (Lipton–Steinhardt 2018; benchmark-leakage literature).
- **Too wide** → confounds dominate (barrier/CCD/friction/Δt, energy mismatch,
  convergence-criterion drift, implementation/hardware, GPU non-determinism).

**Resolution (all five research threads converged on this independently):** don't pick
narrow *or* wide — build a **tiered benchmark**:

    taxonomy → capability cells → tiered problem classes
      → within-class FAIR head-to-head (fixed energy, fixed criterion, single-axis ablation)
      → cross-class performance profiles (Dolan–Moré; Gould–Scott caveats) for robustness
      → equal tuning budget, standardized harness, hidden/rotating instance tier

This is the recurring resolution across the optimization-benchmarking (Beiranvand/Hare/Lucet
2017; Dolan–Moré 2002; CUTEst; COCO/BBOB), ML-rigor, and taxonomy literatures.

## 5. Scope options — arguments (this is the decision the evidence should inform)

### Option A — Contact-free (Worlds 1 + 2 only)
**For:** Cleanest fairness — Worlds 1&2 share one skeleton, so single-axis ablation is
honest. The eigenvalue-filtering cluster is *near-prebuilt*: Trust-Region filtering (2024)
already unifies full-Newton / clamp / absolute as one adaptive rule — a ready "switchboard."
Most seed papers live here. Shared energy is achievable (Stable Neo-Hookean; symmetric
Dirichlet). Reusable datasets exist (Du 2020 injectivity, 11,647 meshes; Shay–Solomon–Stein
2022 parameterization). Code released for most World-1 solvers.
**Against:** Risks "artificial / boring." Since 2020 nearly every high-impact solver is
*defined by* its contact handling; a contact-free benchmark measures a regime the community
partly considers solved, and may under-sell impact.

### Option B — Add contact at FIXED shared barrier settings (Worlds 1+2+3, contact = constant)
**For:** Relevance to modern practice without losing fairness. When the barrier is a shared
constant and only the *inner solve* varies, comparison stays fair — this is exactly what
GIPC / StiffGIPC / Barrier-Augmented-Lagrangian do (same IPC barrier, different
preconditioner/Hessian). Pitfalls-of-Projection and Progressively-Projected-Newton already
run controlled head-to-heads *with* contact by holding the incremental-potential formulation
fixed and varying only the projection strategy.
**Against:** `d̂` / CCD / `ε_v` / `Δt` become protocol constants that structurally advantage
whichever family the harness natively matches; convergence rate becomes barrier-dependent;
and you must exclude methods that *are* the contact model.

### Option C — Contact-model-swap dynamics (World 3, contact model varies)
**For:** Highest impact; captures the actual 2024–2025 frontier (OGC, cubic barrier,
barrier-free aug-Lagrangian, complementarity friction).
**Against:** Methods are not drop-in; the community itself **abandons metric-level fairness**
here, reporting binary non-penetration + timing + robustness demos instead of shared
convergence curves. Cannot produce fair convergence numbers. This is a **capability track**,
not a solver track.

**Reading:** A/B/C are not mutually exclusive — they map onto benchmark **tracks**, not a
single choice. Recommendation below.

## 6. Recommended design

**Track 1 — Solver track (fair convergence).** The spine. Fix energy + convergence criterion
+ line search; vary ONE axis. Two suites sharing the harness:
- **1a — Static distortion** (World 1): symmetric Dirichlet, free-boundary UV, shared mesh
  set. Ablate the *accelerator/direction* axis: AQP / SLIM / Composite Majorization / BCQN /
  Anderson / ABCD / Splitting / analytic-Hessian projected Newton. Metric:
  wall-clock-to-tolerance (BCQN characteristic gradient norm) + final distortion + failures.
  Separately the *feasibility* sub-metric on Du-2020 (TLC / Progressive Embedding /
  Foldover-free / GOSS — success rate from folded init). **Never co-mingle the two metrics.**
- **1b — Quasistatic/dynamic hyperelastic** (World 2): Stable Neo-Hookean (+ near-incompressible
  regime), shared mesh set. Ablate the *eigenvalue-filtering* axis on ONE projected-Newton
  skeleton: none/full-Newton, clamp-to-ε, absolute, trust-region, project-on-demand, kinetic,
  progressive/selective, analytic eigensystem, spectral shift, eigenvalue blending. Metric:
  iterations & wall-clock to fixed decrement, across resolution / element order / stiffness.

**Track 2 — Capability track (contact).** World 3. Score = guaranteed non-penetration
(binary) + timing + robustness under a fixed stress-test scene set, with `d̂` / CCD / `ε_v` /
`Δt` **declared as scenario constants** (part of the problem, not the solver). Optionally a
"fixed-barrier inner-solve" sub-track (Option B) for the GPU-scaling cluster where fairness
is recoverable.

**The v1 "meaningful, cleanly-comparable subset" = Track 1b (eigenvalue-filtering ablation),
with Track 1a as the second clean ablation.** This satisfies both poles: narrow enough to be
fair (one skeleton, one energy, one criterion), yet it directly adjudicates *live* 2024–2025
claims — three seed papers + Eigenvalue Blending (2025) + Progressively-Projected-Newton
(2025) + Pitfalls-of-Projection — so it is neither artificial nor boring. It is also
near-prebuilt: Trust-Region filtering is the natural switchboard, and Pitfalls / PPN already
supply controlled-ablation harnesses to borrow.

## 7. Fair-comparison protocol (confounds to pin)

- **Fix the energy** per suite (mismatch is the #1 hazard: mass-spring / vanilla PD solve a
  *restricted* energy, so their "N× vs Newton" is partly an artifact).
- **Fix ONE convergence criterion** and report BOTH converged wall-clock AND fixed-budget
  error (separates genuine algorithmic advances from unconverged fixed-budget FPS reporting —
  PBD/XPBD do not converge to backward-Euler; Primal-XPBD / position-based nonlinear GS exist
  *because* XPBD stagnates).
- **Iterations are NOT cross-comparable** across first- vs second-order methods; **wall-clock
  to tolerance** is the only cross-cluster-honest metric. Report via **performance / data
  profiles** (Dolan–Moré), with Gould–Scott caveats (no valid ranking for >2 solvers via a
  single profile; iterate drop-the-winner; report multiple metrics).
- **Equal tuning budget** for every method incl. baselines (tuning asymmetry is the most
  common way benchmarks lie).
- **Control implementation/hardware**; report precision, threading, CPU/GPU; GPU numerics are
  not even bit-reproducible.
- **Report statistics** (multiple seeds, quantiles, failure counts), release code/data/configs,
  keep a **hidden/rotating instance tier**.

## 8. Author-proxy agents — where they fit

NOT for generating numerical implementations (a wrong Hessian silently poisons everything).
Two strong roles:
1. **Curation / breadth** — the 5-way fan-out that produced `corpus.md` is a working
   prototype; author proxies surface papers & claims a single curator misses.
2. **Adversarial fairness review** — a per-method proxy reads the Track-1 protocol and files
   objections; consensus = resolving objections, not voting. The corpus already implies
   concrete objections, e.g.: *abs-filtering proxy:* "converging at 1e-5 decrement flatters
   clamping in the near-incompressible regime — sweep ν"; *PBD proxy:* "fixed-budget; do not
   compare my FPS to a converged Newton solve"; *BCQN proxy:* "normalize by characteristic
   gradient norm or my convergence claim is unfair"; *IPC proxy:* "report `d̂`, CCD tol, `ε_v`,
   `Δt` as scenario constants or the comparison is rigged."

## 9. Reusable assets

Datasets: Du et al. 2020 locally-injective-mappings benchmark (11,647 tri/tet); Shay–Solomon–
Stein 2022 parameterization benchmark; Thingi10K (adversarial real meshes); SimJEB (FEM
ground-truth). Released solver code (fair reimplementation targets / validation oracles):
AQP, SLIM, Composite Majorization, BCQN, Anderson (AASolver), ABCD, Splitting, TinyAD,
ADMM-elastic, AA-ADMM, abs-psd, trust-region-newton, TLC, IPC / C-IPC / Rigid-IPC, GIPC,
HOBAK (Dynamic Deformables). Governance template: MLPerf closed/open divisions.

## 10. Decisions

- **D1 — Deliverable form: LOCKED → both, phased.** Ship a STAR-style survey+benchmark
  *paper* as v1, architected so the harness becomes a *living benchmark* (MLPerf-style
  closed/open divisions + hidden tier) afterward.
- **D2 — v1 anchor: LOCKED → full solver spine (Track 1a + 1b).** Both clean ablations —
  distortion accelerators AND eigenvalue filtering — for the broader survey story. Two metric
  regimes to standardize.
- **D3 — Implementation strategy: LOCKED → common-harness reimplementation, official-code-first,
  regression-tested.** One shared framework of hot-swappable components. Each component is
  reimplemented in the harness but **uses / references the official codebase wherever one
  exists**, and is **regression-tested against that official reference (or an independent
  oracle: PETSc SNES/TAO, FEniCS, deal.II step-44)** before it may enter a comparison.
  Trajectory: components become increasingly **agent-generated** under the same
  grounding+regression rule — a component with no passing regression test is inadmissible, and
  benchmark numbers are always *measured against a validated implementation*, never asserted.
  This neutralizes the implementation-quality confound (a reimplementation that matches official
  code isolates the *algorithm*, not its C++), while keeping official code as both the port
  target and the correctness oracle. **First instance DEMONSTRATED (bench/check_libigl.py):** our
  symmetric-Dirichlet solve reaches the same minimizer as libigl's SLIM to ~1e-9.
- **D4 — Contact: LOCKED → defer to v2.** v1 is contact-free (Worlds 1–2). Track 2 is a v2
  extension; v1 harness must be *architected* to admit it (contact as a pluggable scenario
  layer + capability-track metrics) without a rewrite.

---

## 11. v1 execution plan (contact-free, Tracks 1a + 1b, paper → living benchmark)

**Deliverable:** a survey+benchmark paper whose harness/dataset/protocol are released as the
seed of a living benchmark. The paper's *punchline* is honest attribution, not a leaderboard.

### Workstreams
- **W1 — Taxonomy, finalized & evaluated.** Classify all ~130 corpus papers on the 6 axes ×
  problem-class/capability cells; demonstrate orthogonality & mutual-exclusiveness
  (Unterkalmsteiner rubric); this is the survey backbone and the fairness gate.
- **W2 — Harness architecture (component-factored).** One skeleton with pluggable slots:
  {energy ψ, search direction, eigenvalue filter, line search, linear solver, convergence
  criterion}. A benchmark *config* = a point in this component space; the "closed division" =
  same skeleton, swap one slot. Must expose a scenario layer (mesh/BCs/energy/tolerance) so
  v2 contact drops in as a scenario, not a fork. Per D3: each slot implementation is
  **official-code-first + regression-tested** against the official reference / oracle; a
  component ships with its regression test (a **conformance suite**: match the official code's
  iterates/energy to tolerance on canonical inputs) so agent-generated components are admissible
  only when they pass. Metric layer per `docs/metrics.md`.
- **W3 — Problem sets & tiers.**
  - *1a:* symmetric Dirichlet, free-boundary UV, shared mesh set curated from Thingi10K +
    Shay–Solomon–Stein 2022; feasibility sub-suite on Du-2020 (11,647). Easy/typical/adversarial
    strata (BBOB principle). Hidden tier reserved.
  - *1b:* Stable Neo-Hookean (canonical fixed energy) quasistatic + dynamic; strata over
    resolution × element order (P1/P2) × stiffness, **including a near-incompressible ν sweep**
    (the regime where absolute-filtering claims its edge). Hidden tier reserved.
    - **CONTROL C1 (locking-free element) — mandatory in the ν sweep.** Near-incompressibility
      is a *discretization* problem in engineering (F-bar / mixed u–p / Simo three-field,
      deal.II step-44). Displacement-only P1 tets would confound "solver robustness at high
      Poisson" with volumetric LOCKING — i.e. we'd credit the eigenvalue filter for fixing a
      bad element. The ν-sweep must fix a locking-free formulation (or run both and separate
      the effects). This directly guards the v1 headline experiment.
    - **CONTROL C2 (load-parametrization policy).** Engineering parametrizes the *load*
      (incremental stepping, arc-length/Riks continuation); graphics fixes the load and stresses
      the solver. Declare up front whether load stepping is permitted, apply it uniformly, and
      **do not score divergence at a limit/snap-through point as a solver failure** — that's a
      wrong-parametrization artifact (needs arc-length), not a solver defect.
- **W4 — Metric standardization.** Full deliberation in [`metrics.md`](metrics.md) — the
  over-complete candidate list curated to a per-capability-cell orthogonal core. Backbone: one
  convergence criterion per suite (characteristic gradient norm for 1a; Newton decrement for
  1b); pair a **hardware-dependent** cost (wall-clock) with a **hardware-independent** proxy
  (linear-solves / matrix-vector products) so GPU-vs-CPU can't masquerade as algorithmic; report
  converged cost + fixed-budget error + failure counts; aggregate with performance/data profiles
  (Dolan–Moré; Gould–Scott caveats: multiple metrics, drop-the-winner, no single-profile ranking
  for >2). Equal tuning budget. Iterations never pooled across first/second-order.
- **W5 — Method roster** (from corpus, code-released first as fair-reimpl targets / oracles):
  - *1a direction/accelerator ablation:* AQP, SLIM, Composite Majorization, BCQN, Anderson,
    ABCD, Splitting/ADMM, analytic-Hessian projected Newton (TinyAD-backed). Feasibility:
    TLC, Progressive Embedding, Foldover-free, GOSS.
  - *1b eigenvalue-filter ablation (one skeleton):* full Newton, clamp-to-ε, absolute,
    trust-region (switchboard), project-on-demand, kinetic, progressive/selective, analytic
    eigensystem, spectral shift, eigenvalue blending. **Plus the classical filter/globalization
    controls it must beat:** identity-shift (Levenberg) damping, trust-region Steihaug-CG
    (handles indefiniteness intrinsically — no filter), MINRES on the raw indefinite system,
    and modified-Cholesky (Gill–Murray/Schnabel–Eskow). The scientific question is whether
    per-element spectral filtering genuinely beats these, or is a rebranding.
  - *World-0 honesty baselines (both suites):* GD, nonlinear-CG (PR+/strong-Wolfe), heavy-ball,
    Nesterov, **Adam** (expected to plateau as sign descent — that's the informative control),
    **L-BFGS** and **Newton-CG/Gauss-Newton** (strong generic baselines the bespoke methods must
    clear), and a **PETSc SNES/TAO Newton-LS + load-stepping** configuration as the
    computational-mechanics baseline. Equal per-instance tuning budget for all.
- **W6 — Author-proxy fairness loop.** Per-method LLM proxy reviews the protocol & files
  objections; curator resolves; log the objection→resolution as the "consensus & breadth"
  audit trail. (Curation prototype already exists — this fan-out.)
- **W7 — Reproducibility/governance.** Release code/data/configs/env; MLPerf closed
  ("same skeleton") vs open ("best effort") divisions; hidden/rotating tier vs overfitting.

### The decomposition experiments (the scientific payoff)
These are the "untangling" results the paper exists to produce:
1. **Filter isolation (1b):** everything else pinned, swap only the filter → does *absolute*
   actually beat *clamp*, and is the win confined to the near-incompressible ν regime? Does
   *trust-region* dominate uniformly or only adaptively? Quantify per stratum.
2. **Seed-claim decomposition:** reproduce each seed paper's headline delta, then re-measure
   how much survives once the confounds it also changed (line search, criterion, energy) are
   held fixed. Attribute the residual to the named contribution.
3. **BCQN triple-split (1a):** separate its three simultaneous changes (barrier-aware line
   search vs blended Sobolev+L-BFGS proxy vs characteristic-gradient-norm criterion) → how
   much of "fastest+most robust" is each?
4. **First- vs second-order honesty:** show where "N× fewer iterations" inverts under
   wall-clock (per-iteration cost gap), across mesh size.
5. **Criterion sensitivity:** show rankings flip with the convergence criterion — the silent
   confound behind most published speed claims.

### Phasing
- **P0 (now):** finalize taxonomy (W1) + freeze v1 protocol (W3/W4) + harness architecture
  spec (W2). Paper skeleton drafted around the decomposition experiments.
- **P1:** implement harness + 1b eigenvalue-filter ablation (near-prebuilt; borrow Pitfalls/
  PPN); produce experiments 1–2, 4–5.
- **P2:** add 1a accelerator + feasibility suites; experiment 3; full performance profiles.
- **P3:** paper; release harness as living-benchmark seed (divisions + hidden tier).
- **v2:** Track 2 contact via the scenario layer; **plus an optional "learned accelerators"
  companion track** (learned warm-starts + preconditioners + neural subspaces) — same
  convergence criterion, same residual axis, orthogonal to the core so it doesn't contaminate
  the apples-to-apples classical comparison.

---

## 12. External scope: unifying view, lineage map, scope ledger

Per the "oversample broadly, then include-or-justify" mandate. Full records in `corpus.md`
(World 0). Three products:

### 12.1 The unifying view (survey spine)
Every method — graphics, classical, ML — is **metric descent** `x' = x − α M⁻¹∇E`, differing
only in the metric `M` (and the globalization). `M = I` → gradient descent; `M = ∇²E` → Newton;
`M = Fisher` → natural gradient (Amari); `M = Laplacian/H¹` → Sobolev gradient (AQP, BCQN);
`M = Killing operator` → AKVF; `M = reweighted energy` → SLIM (IRLS Gauss-Newton). This single
template lets the survey present "Sobolev preconditioning in graphics" and "natural gradient in
ML" as the *same idea under different metrics* — with the honest caveat that the **Fisher metric
is undefined here** (no probabilistic output model), and that PINN **Energy-NGD collapses to
Gauss-Newton** once unknowns are positions not network weights. Refs: Amari 1998; Martens 2020;
Neuberger *Sobolev Gradients* (LNM 1670); Müller–Zeinhofer 2023.

### 12.2 Lineage map — graphics "innovations" ← named classical ancestors
The single most valuable survey contribution: many recent SIGGRAPH results are adaptations of
established technique. Cite as adaptations, not inventions.
- Eigenvalue clamping / per-element PSD projection (Teran 2005 → Analytic Eigensystems 2019 →
  Absolute 2024 → Trust-Region 2024) ⇐ **modified Cholesky (Gill–Murray 1974, Schnabel–Eskow
  1990) / N&W §3.4 eigenvalue modification**; engineering sibling = **modified/damped/trust-region
  Newton + Abaqus artificial viscous stabilization**. Graphics' genuine contribution = *per-element
  locality* + *analytic* eigensystems. Test: race projected-Newton vs trust-region Steihaug-CG.
- Accelerated Quadratic Proxy (2016) ⇐ **Nesterov 1983** over a Laplacian proxy.
- Anderson-accelerated PD (2018) ⇐ **Anderson 1965** (= multisecant quasi-Newton).
- Projective Dynamics / local–global (2014) ⇐ **ADMM (Douglas–Rachford; Boyd 2011) + Gauss–Newton**;
  PD-as-quasi-Newton (2017) ⇐ **L-BFGS**; relaxation lineage ⇐ **dynamic relaxation (Day 1965)**.
- IPC barriers (2020) ⇐ **primal interior-point (Fiacco–McCormick 1968)**; contact-set/friction ⇐
  **augmented-Lagrangian / mortar / active-set contact mechanics (Wriggers; Popp; Hüeber–Wohlmuth)**.
- Sobolev/proxy preconditioners (AQP/AKVF/BCQN/SLIM) ⇐ **natural-gradient / metric descent (Amari
  1998; Neuberger)**; AKVF is the explicit Riemannian instance.
- Near-incompressible handling ⇐ **F-bar / mixed u–p / Simo three-field**.

### 12.3 Scope ledger (what's in the comparison vs justified-out)
- **INCLUDE-AS-BASELINE (race them):** GD, nonlinear-CG, heavy-ball, Nesterov, Adam (headline
  honesty control), L-BFGS, Newton-CG/Gauss-Newton, trust-region Steihaug-CG, Newton-Krylov/JFNK,
  identity-shift filter, PCG/MINRES inner solvers, Newton-Raphson + load stepping (PETSc SNES/TAO),
  nonlinear multigrid (scalability). Standardize (don't race): Armijo/Wolfe line search.
- **INCLUDE-AS-RELATED (cite as ancestor/sibling; not necessarily raced):** all §12.2 ancestors;
  ENGD / Gauss-Newton-NGD / Sketchy-NGD (idea-transfer: sketch the elastic GN/Laplacian metric);
  SR1, Levenberg–Marquardt, Moré–Sorensen, dogleg; learned preconditioners & warm-starts (→ v2 track).
- **EXCLUDE-WITH-JUSTIFICATION (stated reason):**
  - *Dominated/redundant:* DFP (⊂ BFGS), AdamW/AMSGrad/Lion (⊂ Adam; weight-decay biases positions
    to origin), Adagrad (monotone-decay stall).
  - *Wrong problem structure:* K-FAC/Shampoo (need network layer/Kronecker/tensor structure), Sophia
    (stochastic + probabilistic-loss design; diagonal Hessian dominated by full Hessian), semismooth
    Newton / exact IPM (nonsmooth/constrained — contact extension only).
  - *Different problem class (neural):* PINNs & Deep Energy Method (unknowns = weights, meshfree,
    approximate), MeshGraphNets/GNS & DeepONet/FNO (amortized surrogates/operators, offline cost),
    DiffPD-as-contribution (differentiability for outer-loop learning; its inner PD solve is a
    separate classical baseline), neural ROMs incl. Data-Free Kinematics (reduced-subspace minimizer
    ≠ full-space minimizer). Sketchy-NGD-for-PINNs = evidence the PINN solve is its own class.
  - *Applicability caveats:* Gauss-Newton/LM only for least-squares-form energies; FAS delicate on
    nonconvex hyperelastic.
- **Wording for the paper (principled, not hand-wavy):** "We benchmark *per-instance* solvers that
  treat vertex positions as unknowns and converge to the discrete energy minimizer. We exclude
  methods whose unknowns are network weights, whose cost is *amortized offline across instances*,
  or whose minimizer lies in a *learned reduced subspace* — these solve related but distinct
  (approximate or amortized) problems and are not commensurable on our exact-convergence axis."

## Uncertainties — status (verification pass #6, 2026-08)
**Resolved:** Pitfalls of Projection & Convergent IPC are **unpublished preprints** (no venue —
do not assign one). PPN corrected to **Eurographics 2026, CGF 45(2), 10.1111/cgf.70386** (was
mislabeled 2025). Trust-Region filtering venue/DOI (SIGGRAPH Asia 2024, 10.1145/3680528.3687650)
and code (honglin-c/trust-region-newton) confirmed. Code confirmed: ABD (Autodesk/affine-body-
dynamics), StiffGIPC (KemengHuang/Stiff-GIPC), VBD (AnkaChan/Gaia); **no public repo** found for
JGS2 or Second-Order Stencil Descent. Contact classification: **VBD = penalty/soft (NOT
IPC-guaranteed)**; **JGS2 & Second-Order Stencil Descent = IPC-guaranteed**. Barrier-free title
corrected ("Robust *and Efficient*…", 10.1145/3811035). Citations filled: Zhang-Mischaikow-Turk
2005 (10.1145/1037957.1037958), Kraevoy-Sheffer-Gotsman MatchMaker 2003 (10.1145/1201775.882271),
Martin-Weinkauf-Seidel 2013 (10.1111/cgf.12019).
**Still open:** exact title/venue of the "Total Unsigned Area" (Xu et al. ~2011) reference —
pull from the TLC (Du et al. 2020) bibliography; Martin 2013 full author list (medium confidence).
