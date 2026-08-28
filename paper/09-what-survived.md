# 9. What Survived — and the Review Loop as Method

## 9.1 The hardened ledger

After the decomposition experiments and two single-axis verification passes (§9.2), the
superiority-claims graph stands at **2 validated, 39 qualified, 97 self-claimed, and 22 unmeasured**
edges (`claims/hardening.md`). The verification work promoted sixteen edges from self-claimed to
qualified — ten from the contact-free triage backlog and six more from a "try-harder" pass that built
an incremental-potential testbed and faithfully re-implemented part of the simulation-accelerator
family (quasi-Newton/Liu-2017, generalized Projective Dynamics, Chebyshev acceleration, Vertex Block
Descent, and AQP's own AGD ablation), testing their *convergence* claims on hardware-independent
iteration counts (`results/dynamics_solvers.md`, `results/agd_vs_aqp.md`). But — tellingly — that work
added *nothing* to the validated column: it stays at the same two edges,
SLIM over AQP on iteration count (§8.2, grounded on official libigl code) and Anderson acceleration
over ARAP local–global on convergence (§8.6, a reproducible multi-seed × multi-mesh benchmark). Both
of those meet a high bar (official code or a multi-condition profile); the new results, strong as some
are, do not, and we resisted the temptation to inflate them.

That restraint is itself a finding, and it came from turning the review loop on our *own* first
verdicts: an internal adversarial pass caught four edges we had initially marked *validated* and
forced them down to *qualified*. Three are genuinely striking but rest on a **single hand-picked
instance** or a **deliberately un-globalized baseline**: Anderson wrapped around the official SLIM
fixed-point map cuts a slowly-contracting instance from 380 iterations to 10 (a 36–38× reduction,
verified faithful to continuous SLIM and confirmed on an absolute stopping tolerance,
`results/anderson_slim.md`) — but on one instance with no multi-seed sweep; trust-region and
Project-on-Demand filtering each recover 100/100 inverted-initialization starts where an *unfiltered*
Newton recovers 5/100 (`results/filter_robustness.md`) — but that baseline hard-terminates on its
first non-descent direction, so the margin measures *presence of any indefiniteness handling*, not a
filter's edge over a competently globalized Newton; and Stable Neo-Hookean recovers from inverted
configurations that make the classical barrier energy literally `+∞` (`results/stable_nu.md`) — but
that half of the claim is partly true *by the barrier's definition*, and the rotation half is
untested. Each is now `qualified` with its exact limitation recorded. (Likewise the analytic
eigensystem's claim over numerical eigendecomposition stays *qualified*: the projection is provably
equivalent, but the closed form is not faster than a LAPACK eigensolve on a 4×4 — the advantage lives
in avoiding an autodiff Hessian assembly, not a faster eigendecomposition, `results/analytic_eig.md`.)
Two further edges we had tried to score — AQP versus local–global, and Anderson versus AQP — reverted
all the way to `self-claimed`: our attempted comparison was cross-energy (the methods minimize
different objectives to different minima), so it cannot adjudicate them at all, and saying so is more
honest than a manufactured verdict.

The distribution is the finding. The benchmark **qualifies far more than it overturns: no published
claims-graph edge is `refuted`** (the ledger records zero), and even the headline `ν`-claim ends up
*re-validated* once its confounds are controlled. What does not survive are a few *baseline-confounded
or self-derived* speed statements — AQP's "×200 versus L-BFGS" (a MATLAB-baseline artifact) and the
downstream "AQP's single factorization wins at scale" (refuted at tight tolerance, §8.2) — neither of
which is a first-party superiority edge in the graph. This is the opposite of a debunking exercise. What the ledger records is that
the field's superiority claims are, as of this snapshot, overwhelmingly *untested against confound
control* — not wrong, but unearned — and that when they are tested, the honest verdict is usually a
*qualification of regime* rather than a reversal.

## 9.2 What we cannot yet adjudicate, and why

An honest benchmark must also be explicit about the *boundary* of what it can say. We triaged every
self-claimed edge against the contact-free 2D prototype (`results/claim_triage.md`). Fourteen edges
were **testable now** by a single-axis experiment; we have since run all fourteen, and the honest
split — after the self-review of §9.1 — is the point. Ten were **qualified**: some strong (Anderson's
36–38× acceleration of official SLIM; the filtering methods' recovery from inverted starts; Stable
Neo-Hookean's inversion recovery; an intermediate eigenvalue blend beating both clamp and absolute on
a locking-relieved element), each held back from *validated* by a single instance, a weak baseline, or
a partly-definitional claim (§9.1). Four could **not be adjudicated and we say why** rather than
forcing a verdict: two — AQP versus local–global, Anderson versus AQP — are cross-energy
non-comparisons (different objectives, different minima); SLIM versus projected-Newton on a
non-uniform mesh never reaches the *far-from-minimum* regime the claim is about, because our
clamp-projected, line-searched Newton stays well-conditioned (so the result is a statement about our
harness, not evidence against the paper); and AQP-faster-than-Newton is confounded by the C++/Python
wall-clock boundary at small scale. The remaining one, absolute versus clamp on robustness, is a genuine
**tie**. "We cannot adjudicate this, and here is precisely why" is itself a reported result. The
majority of the graph, meanwhile, remains out of reach — and we label each edge with the *specific
reason* rather than dropping it:

- **needs unavailable code** (~28 edges) — the claim requires the paper's own implementation, which we
  will not substitute with a look-alike (that would beg the question, as with Composite Majorization,
  §8.5). A "try-harder" pass (§9.1) reclaimed part of this bucket: where a method's algorithm is fully
  specified we *did* build it faithfully, so the **convergence** claims of the simulation-accelerator
  family (quasi-Newton, Projective-Dynamics-style, Chebyshev, Vertex Block Descent) are now tested
  (`results/dynamics_solvers.md`); what remains here is genuinely code-bound — a specific *majorizer*,
  or a competitor port (an interior-point QP/SOCP for the injective-mapping edges). The corresponding
  GPU-throughput/wall-clock *speed* headlines stay hardware-confounded, below.
- **needs contact physics** (22) — World-3 (IPC barriers, continuous collision detection, friction);
  v1 implements none, so an intersection-free or friction claim has no harness to run in.
- **needs scale** (21) — the claim *is* about 100K–1.5M-element meshes, GPU throughput, or frame-rate
  budgets the dense Python prototype cannot reach.
- **entangled, needs source** (9) — the method bundles several co-changed components that cannot be
  separated without the paper — the very confound this benchmark exists to expose, now limiting it.
- **hardware-confounded** (4), **subjective-quality** (3), **baseline-confounded** (2), **needs 3D**
  (1) — respectively a GPU-vs-CPU wall-clock claim that cannot be made portable, a visual-quality
  claim with no agreed metric, a claim resting on a weak or self-ablation baseline, and an inherently
  3D free-boundary injectivity claim.

This map of the boundary is itself a contribution: for many published claims the honest statement is
"*we cannot yet adjudicate this, and here is precisely why*." Three of the categories — unavailable
code, scale, and contact — are exactly where a *living* benchmark with author-contributed component
ports and a contact track (§10) would move the frontier.

## 9.3 The review loop applied to ourselves

The report's second contribution is methodological, and it emerged from turning the benchmark's own
discipline on our *own* draft conclusions. We ran an **adversarial review loop**: for each result, a
reviewer — protective of the paper whose claim was under test, and separately, skeptical of *our*
measurement — hunted for the confound we had missed. It repeatedly found one, in our own work:

- The first "trust-region beats both filters" result was **our** artifact of a costly global
  eigendecomposition operator; the faithful per-element implementation reversed it (§8.2).
- The first "P2 element fixes the `ν`-claim" result was measured by **us** on the *wrong* energy; the
  honest test needed the element *and* the energy controlled together (§8.1).
- The first "AQP is mesh-independent" reading was **our** loose-tolerance artifact; the τ-sweep flips
  it (§8.2). The *correction* then overreached — "AQP scales worse than L-BFGS" — and CI-gating forced
  **us** to retract that too (§8.2).
- The first BCQN factorial pooled its arms and **hid** a line-search × direction interaction; the
  reviewer caught the pooling and we reported the per-cell effect (§8.3).
- The first hard-boundary feasibility result presented a cross-algorithm iteration ratio as a clean
  "~100× discrimination"; the reviewer flagged the non-comparability and the mis-stated injectivity
  guarantee, and we downgraded both (§8.4).

Each of these was an over-reach in *our own* prior conclusion, caught by applying the confound-untangling
the benchmark is built for reflexively. The lesson generalizes beyond any single claim: **superiority
in this field is entangled with element choice, energy, tolerance, baseline quality, and hardware, and
a single confound rarely acts alone** — the `ν`-claim needed two removed at once. A benchmark that does
not audit itself as adversarially as it audits the literature will simply manufacture new confounded
claims of its own.

## 9.4 The call

We therefore offer the report not as a leaderboard but as a *method*: a taxonomy and unifying view that
name the components, a lineage map that points each to its classical analysis, a machine-readable
claims graph that turns the literature's assertions into testable hypotheses, and a conformance-gated,
single-axis benchmark that promotes or qualifies them — all held to a status ladder that distinguishes
*the paper's word* from *independently measured*. The natural next step for the community is to report
solver claims with this honest status, and to contribute faithful component ports and adversarial
re-measurements to the living benchmark (§10) rather than another confounded speed number.
