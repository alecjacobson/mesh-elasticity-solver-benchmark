# 9. What Survived — and the Review Loop as Method

## 9.1 The hardened ledger

After the decomposition experiments and the single-axis verification pass (§9.2), the
superiority-claims graph stands at **6 validated, 32 qualified, 100 self-claimed, and 22 unmeasured**
edges (`claims/hardening.md`). The six validated edges are: SLIM over AQP on iteration count (§8.2);
Anderson acceleration over ARAP local–global on convergence (§8.6); **Anderson acceleration over
SLIM** (wrapping the official libigl SLIM fixed-point map, a 38× iteration reduction on a
slowly-contracting instance, `results/anderson_slim.md`); **trust-region-adaptive filtering** and
**Project-on-Demand Newton**, each over unfiltered full Newton on a 100-start inverted-init
robustness battery (both recover 100/100 starts versus raw Newton's 5/100, `results/filter_robustness.md`);
and **Stable Neo-Hookean over standard Neo-Hookean** on inversion robustness (the classical barrier
energy is `+∞` at an inverted initialization and cannot even start, while the stable energy recovers,
`results/stable_nu.md`). The qualified edges are dominated by claims the *paper itself* limited to a
regime (2D, a specific energy, a mesh class) that later citations dropped, plus claims our benchmark
reproduced only under stated conditions. (Notably, the analytic eigensystem's claim over numerical
eigendecomposition stays *qualified*, not validated: the projection is provably equivalent, but the
closed form is not faster than a LAPACK eigensolve on a 4×4 — the advantage lives in avoiding an
autodiff Hessian assembly, not in a faster eigendecomposition, `results/analytic_eig.md`.)

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
were **testable now** by a single-axis experiment; we have since run all fourteen (§9.1's promotions
come from this pass). The honest split is the point: four decisive **validations** (trust-region and
Project-on-Demand filtering over full Newton; Anderson over SLIM; Stable- over standard-Neo-Hookean),
four regime-dependent **qualifications** (e.g. an intermediate eigenvalue blend beats both clamp and
absolute only on a locking-relieved element; SLIM beats L-BFGS on iterations but the time-axis
headline is out of scale), three **not-reproduced-with-documented-why** (AQP is *not* faster than
local-global on our back-solve axis; SLIM does *not* out-run projected-Newton on our non-uniform mesh
because a well-safeguarded Newton stays well-conditioned — the Fig.11 stall is a far-from-minimum
pathology; AQP is *not* faster than Newton at small scale), and one **tie** (absolute versus clamp
recover identically under a line search). "Not reproduced here, and here is precisely why" is itself a
reported result. The majority of the graph, however, remains out of reach — and we label each edge
with the *specific reason* rather than dropping it:

- **needs unavailable code** (~34 edges) — the claim requires the paper's own implementation, which we
  will not substitute with a look-alike (that would beg the question, as with Composite Majorization,
  §8.5). The projective-dynamics / ADMM / XPBD / Vertex-Block-Descent simulation-accelerator family
  falls here.
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
