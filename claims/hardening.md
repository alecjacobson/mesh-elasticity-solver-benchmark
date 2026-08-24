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

## Benchmark-hardened (P1 prototype, `bench/` + `results/`)

First edges annotated with *our own* measured evidence (`assessed_by` now includes `benchmark`):

- **`absolute-filtering → clamp-filtering` (convergence)** — **INDICATIVE (2D)** (`results/p2_nu.md`).
  On P1 elements absolute under-performs clamp and *fails* at ν=0.4999; but on a locking-relieved
  **P2 (quadratic) element**, from the same init, absolute **matches and beats** clamp near
  incompressibility (41 vs 53 it at ν=0.4999). So the P1 "refutation" is *consistent with* a
  **volumetric-locking artifact** rather than a filter defect, and the paper's claim looks sound
  once a proper discretization is used. Status stays `qualified` — conditional on a non-locking
  element. **Scope/caveats (do not over-read this):** 2D only; a *single* stretch scenario/seed;
  small dense meshes; and the crossed-mesh probe is **non-monotone** — the absolute−clamp gap
  collapses to 0 at ν=0.499 but absolute is **worse again at ν=0.4999 (98 vs 64 it)** because the
  crossed mesh is not *fully* locking-free (residual locking still bites at the most extreme ν).
  The clean separation is on the genuinely locking-relieved P2 element; the crossed mesh only
  *points* that way. This is the benchmark doing its core job — separating a plausible solver
  effect from a discretization confound — but "indicative in 2D," not "settled."
- **`pitfalls-projection → clamp-filtering` (convergence)** — our 1b dynamic probe
  (`results/1b_dynamic.md`) is consistent: with inertial regularization, unfiltered Newton
  converged in *fewer* iterations than clamp, i.e. projecting-when-unneeded hurts — this paper's
  thesis, reproduced in miniature. Stays `qualified`, now with `benchmark(1b-dynamic)` evidence.

General P1 finding feeding future hardening: unfiltered full Newton (`none`) fails 25–58% of
static instances (`results/profiles.md`) — quantitative support that *some* Hessian modification
is necessary (the shared premise under the whole eigenvalue-filtering cohort). The *ranking among*
filters remains regime- and criterion-dependent (E5, E1ν), so those edges stay `qualified` pending
the locking-free + more-filters runs.

## Bookkeeping rules

- One edge = one `(from,to,dimension)`; harden in place, never duplicate.
- On promotion: set `status`, `assessed_by` (citation / `benchmark`), and `notes` = the regime
  under which it holds. A `qualified` edge must state its condition.
- A `refuted` edge stays in the graph (with evidence) — refutations are results, not deletions.

## Benchmark verdicts (current) — from the `bench/` prototype

| edge | dim | verdict | evidence |
|---|---|---|---|
| anderson-geometry → local-global | convergence | **validated (2D)** | Anderson 12 it vs local-global 23 it on a non-trivial sheared-target ARAP, mesh-independent, same min; wall-clock speedup < iter speedup (anderson) |
| absolute-filtering → clamp-filtering | convergence | **qualified (indicative, 2D)** | P1 "refutation" consistent with locking; on locking-relieved P2 absolute matches/beats clamp (p2_nu). Crossed-mesh probe non-monotone; single 2D scenario |
| trust-region-filtering → clamp-filtering | convergence | **qualified** | P2: TR beats clamp; P1: degrades to absolute — but P1 is locking-confounded, so non-attributable (world2_filters) |
| trust-region-filtering → absolute-filtering | convergence | **qualified** | P2: TR beats absolute; P1: identical (world2_filters) |
| aqp → l-bfgs | speed | **qualified (unreproduced)** | AQP loses to a well-implemented L-BFGS; ×200 was a MATLAB-baseline confound (e2). Not tested from a Tutte-far init (AQP's regime) — see #29 |
| sobolev-lbfgs → l-bfgs | convergence | **qualified** | isolated Sobolev-preconditioning component (D0=L⁻¹) helps only ill-conditioned (34 vs 55) not well-cond (42 vs 40) (e2) |
| bcqn → l-bfgs | convergence | **self-claimed** | REVERTED: only 1 of BCQN's 3 components (Sobolev) was isolated (see sobolev-lbfgs→l-bfgs); full method + E3 unimplemented |
| pitfalls-projection → clamp-filtering | convergence | **qualified** | consistent with 1b-dynamic (projecting-when-unneeded hurts). NB: measures iteration count, not the paper's asymptotic *rate*/affine-invariance claim — see #39 |

Plus the general premise **validated**: unfiltered Newton fails 25–58% of static instances (profiles),
and the ν-claim is a discretization artifact on P1, real on P2 (p2_nu, 3d_nu, world2_filters).
