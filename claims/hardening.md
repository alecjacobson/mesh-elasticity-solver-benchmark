# Claim Hardening Ledger

Tracks the promotion of claims-graph edges (`claims.yaml`) from `self-claimed` toward
`validated` / `qualified` / `refuted` (issue #5). An edge is hardened only with cited evidence:
an **independent study**, a **released benchmark** (pending our re-run), or **our own benchmark**
(the E1–E5 decomposition experiments, `docs/experiments.md`). Status is written back into
`claims.yaml` (`status`, `assessed_by`, `notes`).

## Decidability buckets

Where each edge can be settled — so we know what v1 delivers vs what waits.

- **A — decidable in v1 closed division** (contact-free, one-slot swap, conformance-passing).
  The eigenvalue-filter edges (E1), the seed-claim deltas and distortion-accelerator edges
  (E2/E3), the first-vs-second-order and criterion-sensitivity edges (E4/E5). This is the bulk
  of Worlds 1–2.
- **B — decidable in v1 but open division only** (a component with no official code / no oracle
  → not conformance-gated; reported for breadth, not a closed ranking).
- **C — needs v2 (contact).** All World-3 edges targeting `ipc`/`gipc` etc. The GPU-vs-CPU and
  contact-model-swap edges can only be *fairly* hardened once the contact scenario layer +
  capability metrics (non-penetration #52, friction #54) are in.
- **D — not decidable by us / capability-only.** Binary non-penetration guarantees and
  "handles scenes X cannot" capability claims are validated by *reproduction of the failure/
  success*, not by a shared convergence metric; and GPU-vs-CPU speed headlines are marked
  permanently `qualified` (hardware-confounded) unless re-run on matched hardware.

## Already hardened (independent evidence, pre-benchmark) — 16 `qualified` edges

These carry `status: qualified` in `claims.yaml` today, on evidence that exists **without** our
benchmark:

- **Independent study.** `pitfalls-projection→clamp-filtering` (convergence): an independent
  study (not the clamp authors) shows unconditional PSD projection degrades asymptotic rate →
  qualifies clamp's implicit convergence claim. Regime: asymptotic; clamping still robust far
  from the solution.
- **Author-stated regime limit.** `absolute-filtering→clamp-filtering` (small-deformation
  compression caveat); `progressively-projected-newton→clamp-filtering` (PN wins at very large
  steps / quasistatics); the three `splitting-flip-free→{slim,akvf,progressive-param}` edges
  (robust to flips but not consistently faster; can fail on far-constraint inits).
- **Released benchmark, pending our re-run.** `tlc→{lbd,simplex-assembly,foldover-free}` (Du-2020
  11,647-mesh set); `progressive-embedding→{fp-tutte,lbd}` (Thingi10K / Myles set);
  `foldover-free→tlc` (Du-2020); `goss→{tlc,foldover-free}` (FFM/ABCD suites);
  `efficient-bijective-param→scaf` (14,861-model set).

## Honest-caveat edges to re-read (not yet re-statused, but flagged in `notes`)

Captured verbatim from the papers; E4 will convert or `qualify` them:

- **Fixed-budget / per-iteration, not converged:** `fast-mass-spring→full-newton`,
  `projective-dynamics→full-newton`, `quasi-newton-liu2017→full-newton`,
  `vertex-block-descent→full-newton`, `pbng→full-newton`, `jgs2→full-newton`.
- **Author disclaimer of *no* advantage:** `xpbd→pbd` (authors explicitly disclaim any speed/
  convergence win — consistency only). This is the strongest candidate for a **`refuted`** on the
  *speed* dimension if anyone ever asserts it.
- **Parity, not superiority:** `barrier-free-elastodynamics→ipc` (convergence) — authors claim
  robustness *parity* with IPC, superiority only on speed. Guard against over-reading.
- **GPU-vs-CPU (permanently qualified unless matched-hardware re-run):** `abd→ipc`,
  `medial-ipc→ipc`, `gipc→ipc` (95× overall), `barrier-aug-lagrangian→ipc` (80×),
  `second-order-stencil-descent→ipc` (58–129×). Their *same-hardware* sub-claims (GIPC 3×
  eigensystem; StiffGIPC/Guo vs GIPC) are the fair, hardenable parts.

## Experiment → edges map (what v1 will harden)

| experiment | edges it hardens |
|---|---|
| E1 filter isolation | absolute/trust-region/blending/PPN/pitfalls → clamp; analytic→numeric |
| E2 seed decomposition | seed outgoing edges (slim→aqp, cm→{aqp,slim}, filter edges) |
| E3 BCQN triple-split | bcqn→{aqp,slim,composite-majorization,l-bfgs}; bcqn→gradient-descent |
| E4 1st-vs-2nd-order | fixed-budget sim edges → converged or qualified |
| E5 criterion sensitivity | meta-annotate every speed/convergence edge |

## Bookkeeping rules

- One edge = one `(from,to,dimension)`; harden in place, never duplicate.
- On promotion: set `status`, `assessed_by` (citation / `benchmark`), and `notes` = the regime
  under which it holds. A `qualified` edge must state its condition.
- A `refuted` edge stays in the graph (with evidence) — refutations are results, not deletions.
