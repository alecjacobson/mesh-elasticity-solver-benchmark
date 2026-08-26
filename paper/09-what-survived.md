# 9. What Survived — and the Review Loop as Method

## 9.1 The hardened ledger

After the decomposition experiments, the superiority-claims graph stands at **2 validated, 23
qualified, 113 self-claimed, and 22 unmeasured** edges (`claims/hardening.md`). The two validated
edges are SLIM over AQP on iteration count (§8.2) and the analytic eigensystem's equivalence to
numerical eigendecomposition; the qualified edges are dominated by claims the *paper itself* limited
to a regime (2D, a specific energy, a mesh class) that later citations dropped, plus claims our
benchmark reproduced only under stated conditions.

The distribution is the finding. The benchmark **qualifies far more than it overturns, and overturns
nothing outright** — even the headline `ν`-claim ends up *re-validated* once its confounds are
controlled, not refuted. This is the opposite of a debunking exercise. What the ledger records is that
the field's superiority claims are, as of this snapshot, overwhelmingly *untested against confound
control* — not wrong, but unearned — and that when they are tested, the honest verdict is usually a
*qualification of regime* rather than a reversal.

## 9.2 The review loop applied to ourselves

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

## 9.3 The call

We therefore offer the report not as a leaderboard but as a *method*: a taxonomy and unifying view that
name the components, a lineage map that points each to its classical analysis, a machine-readable
claims graph that turns the literature's assertions into testable hypotheses, and a conformance-gated,
single-axis benchmark that promotes or qualifies them — all held to a status ladder that distinguishes
*the paper's word* from *independently measured*. The natural next step for the community is to report
solver claims with this honest status, and to contribute faithful component ports and adversarial
re-measurements to the living benchmark (§10) rather than another confounded speed number.
