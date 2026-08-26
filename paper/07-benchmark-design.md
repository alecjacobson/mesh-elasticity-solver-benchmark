# 7. Benchmark Design

The benchmark's job is to promote or qualify a superiority claim by changing exactly one component
axis while holding the rest fixed. Three design commitments make this possible: a component-factored
harness, a conformance admissibility gate, and a metric discipline that separates algorithm from
hardware.

## 7.1 A configuration is a point in component space

The harness (`bench/`) is factored along the six axes of §3.1: energy, filter, direction, line search,
linear solver, and criterion are independent, swappable *slots*. A configuration is a choice on each
slot — a point in component space — and a **decomposition experiment is a single-axis diff** between
two configurations. This is what lets us attribute a measured difference to one component rather than
to an entangled bundle. Where a paper's method is a bundle (BCQN, §8.3), we implement each of its
components as a slot and run the full factorial, so the bundle's win can be decomposed into
main-effects and interactions rather than asserted.

## 7.2 The conformance admissibility gate

A component is admissible into the benchmark only if it passes a **conformance suite** — the fairness
gate of §3.3. In the v1 harness this is eight gates, all passing: element gradient and Hessian versus
finite differences (~1e-9); the assembled global gradient versus finite differences; the
symmetric-Dirichlet energy versus its canonical form; the Stable-Neo-Hookean gradient, rest-stress,
and finite-through-inversion property; the trust-region blend reproducing Newton/clamp/absolute
exactly; the selective-reduced-integration element's gradient and rest-stress; the barrier line
search's step landing exactly on the inversion boundary; and the untangling penalty's gradient. A
component that fails its gate cannot enter a comparison — this is what prevents a mis-implemented
method from producing a spurious "algorithmic" difference. For methods with released reference code,
we additionally require an **official-code-first** regression: the port must reproduce the source
implementation's energy on a shared instance (as done against libigl's SLIM).

## 7.3 Metric discipline: hardware-independent first

The report's most-repeated methodological rule is that **wall-clock is not an algorithmic quantity**.
Every result is reported on a *hardware-independent* count — iterations, global factorizations,
back-solves, or matrix-vector products — and only *paired* with wall-clock where the comparison is
implementation-fair. This matters because:

- A compiled library and a Python prototype differ in wall-clock by orders of magnitude for reasons
  that are not the algorithm; the iteration and factorization counts are portable and carry the
  verdict (§8.6, the SLIM comparison).
- The *cost structure* differs by method: a projected-Newton iteration is a factorization; an AQP
  iteration is one prefactored back-solve; an L-BFGS iteration is none. "Fewest iterations" therefore
  does not mean "cheapest," and a factorization-weighted cost model can invert an iteration-count
  ranking (§8.2, scale-cost).

Robustness is reported as a **performance profile** (Dolan–Moré) and **data profile** (Moré–Wild)
over a problem set, with pairwise win-fractions per the Gould–Scott caveat that an N-solver profile is
not a total order — never as a single speed number.

## 7.4 The frozen protocol

The v1 protocol fixes, per world: an energy and a convergence criterion (characteristic gradient norm
for the distortion track, gradient/Newton-decrement for the filter track); a problem set stratified
into *easy / typical / adversarial / ill-conditioned*; controls that hold the confounding axes fixed
(the element and the energy, whose entanglement is the subject of §8.1); an **independent reference
solution** `E*` computed by Newton to a tight gradient tolerance — *not* the best final energy among
the compared methods, which would bias toward the strongest solver; an **equal tuning budget** with no
per-problem parameter tuning; and a hidden/rotating tier for the living benchmark (§10). The corpus
that populates these strata spans roughly two decades and all three worlds (Figure 7.1), so the
benchmark is not concentrated in one paper or one corner of the field.

![Corpus breadth](../figures/corpus_breadth.png)

*Figure 7.1. Corpus breadth: papers per year by world, and node totals. The survey spans ~2003–2026
and all three worlds (World-1-heavy).*
