# 10. Open Problems and the Living Benchmark

The v1 measurements are a 2D prototype, and their scope is a feature of the honesty argument, not an
accident: a smaller, fully-controlled and reflexively-audited set of results is worth more than a
broad set of confounded ones. The report is therefore also a *plan*, and the harness a *seed*.

## 10.1 Open problems

- **Larger scale and a gold-standard locking-free element.** The `ν`-claim (§8.1) is settled to the
  precision a locking-*relieved* P2 element allows; a fully locking-free Taylor–Hood or mixed u–p
  element is the pending gold-standard control, and 3D at scale (where the pure-Python prototype does
  not reach) is required to turn "indicative" into "definitive."
- **Faithful ports of a few key methods.** Where a method's construction is fully specified we now
  build it faithfully rather than substitute a look-alike — most notably **Composite Majorization**,
  which we implemented from its convex-concave singular-value construction and gated on the paper's own
  Proposition 3.1 (§8.5); the honest finding is that its "faster than projected-Newton" is a wall-clock
  claim that does not surface on the iteration axis, while it decisively beats first-order AQP. What
  still resists faithful re-measurement are methods whose *specific per-simplex formula* we could not
  verify without the source (the **injectivity-cohort** — TLC's lifted content, foldover-free's
  regularizer — needed to rank *within* the cohort of §8.4). These remain the clearest invitations for
  original-author contributions.
- **The contact world.** World-3 (IPC and relatives) is surveyed but unmeasured; its four
  solver-external parameters (barrier stiffness, CCD tolerance, friction regularizer, time step) make
  it a benchmark-design problem in its own right, deferred to v2.

## 10.2 The living benchmark

We release the harness, the claims graph, the annotated corpus, and the deterministic figures as the
seed of a living benchmark, governed by three principles carried over from the report:

- **Divisions.** A *closed* division fixes the components and races only the axis under test (the
  single-axis discipline of §7); an *open* division admits any method on a shared problem set and
  reference; a rotating *hidden tier* of held-out instances guards against overfitting to the public
  set. Every submission must clear the same conformance gate (§7.2) — no un-conformant component
  enters a comparison.
- **Hardware-independent verdicts.** Leaderboard rankings are on portable counts (iterations,
  factorizations, matrix-vector products), with wall-clock reported only where the comparison is
  implementation-fair (§7.3). This keeps the benchmark from rewarding engineering over algorithm.
- **The status ladder as the unit of contribution.** A contribution is not "my method is fastest" but
  "this edge of the claims graph moves from self-claimed to validated/qualified, by this single-axis
  experiment, reproducibly." The adversarial-review discipline of §9 is part of the acceptance
  criterion: a submitted result must survive an attempt to find its confound.

An optional **learned-accelerators** companion track (learned warm-starts, preconditioners, neural
subspaces) can join on the same convergence criterion and residual axis, kept orthogonal to the
classical core so it does not contaminate the apples-to-apples comparison.

## 11. Conclusion

A decade of mesh-elasticity solvers is, to a first approximation, a decade of *component swaps inside
one metric-descent iteration* — and the field's superiority claims are entangled with the components
each paper changed but did not credit. We have reorganized the literature around that shared structure,
named each innovation's classical ancestor, encoded the claims as a testable graph, and built a
conformance-gated benchmark that changes one component at a time. Applied to the contact-free track,
it re-validates a reversed headline claim only after separating two confounds, exposes a flagship
method's components as interacting rather than additive, reduces an entire filtering debate to one
analytic scalar, and — most importantly — audits its *own* conclusions as adversarially as the
literature's, retracting several of our own over-reaches in the process.

The result is not a ranking but a method for honest attribution, and a living benchmark that asks the
community to earn its superiority claims one controlled, reflexively-audited experiment at a time.
