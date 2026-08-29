# 6. The Superiority-Claims Graph

To reason about the literature's claims systematically rather than anecdotally, we extract them into a
machine-readable directed graph. Each **node** is a method; each **edge** `A → B` is a claim that `A`
beats `B` on a stated **dimension** (speed, convergence, robustness, quality, scalability), annotated
with the paper's own evidence, the source, and an **evidentiary status**. The v1 graph has **81 nodes
and 160 claimed-win edges**, consolidated from a per-paper extraction over the corpus (`claims/`).

## 6.1 The status ladder

Every edge starts `self-claimed` — the paper's own assertion — and is only promoted with cited
evidence:

- **self-claimed** — the paper says so; we have not tested it.
- **unmeasured** — extracted but out of the v1 measurement scope (e.g. contact-world edges: v1
  measures no contact).
- **qualified** — the paper itself states a regime limit, *or* the claim rests on a released benchmark
  pending independent re-run, *or* it is an independent (not self-serving) study, *or* our benchmark
  reproduces it only under stated conditions.
- **validated** — independently confirmed by our measurements (or a regression against official code).

![Claims ledger](../figures/claims_ledger.png)

*Figure 6.1. The epistemic scoreboard. Of 160 extracted superiority edges, 77 are the papers' own
word (`self-claimed`), 22 are unmeasured (contact), 59 are qualified, and only 2 are independently
validated. The benchmark **qualifies** far more than it overturns — and refutes no published edge
outright.*

## 6.2 The honesty patterns

Reading the graph as a whole surfaces the recurring ways superiority claims mislead — the patterns the
benchmark is built to test:

- **Fixed-budget versus converged.** A method declared "faster" at a fixed iteration budget may simply
  stop earlier under a criterion that flatters it; to convergence, the ranking can invert (§8.6).
- **Hardware confounds.** A compiled-C++ method compared against a research prototype wins on
  wall-clock for reasons that are not algorithmic; only hardware-independent counts (iterations,
  factorizations, matrix-vector products) are portable (§8.6).
- **Baseline quality.** An order-of-magnitude speed-up against a deliberately weak baseline (a MATLAB
  reference implementation, an un-preconditioned first-order method) says little about the method
  versus a *well-implemented* baseline (§8.2).
- **Entanglement.** A bundled method credits its headline component for a win that its other bundled
  changes, or an interaction between them, actually produce (§8.3).
- **Author disclaimers.** Many papers already state a regime limit (2D only, a particular energy, a
  mesh class) that later citations drop; the graph preserves these, and they account for a large
  fraction of the `qualified` edges.

## 6.3 What the graph is for

The claims graph is not a scoreboard to be topped; it is a *to-do list for honest attribution*. Each
`self-claimed` edge is a hypothesis the benchmark can, in principle, promote or qualify by a
single-axis decomposition experiment. The distribution of statuses — overwhelmingly self-claimed, a
handful validated — is itself the report's central empirical finding about the state of the field:
the community's superiority claims are, as of this snapshot, largely untested against confound
control. §7 describes the instrument that tests them; §8 reports what it found.
