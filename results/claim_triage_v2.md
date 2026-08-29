# Claim triage v2 — the "try harder" pass on the self-claimed backlog

The v1 triage (`results/claim_triage.md`) labelled ~103 edges untestable and stopped. This v2 pass
re-examines that backlog with a higher bar: **a faithful implementation from a paper's specification
is exactly what a benchmark should build**, so "we haven't ported the code" is not, by itself, a
reason to call a claim untestable. Many of the self-claimed edges are *convergence* claims about
methods that are just different inner minimizers of an energy we already have — those are testable on
hardware-independent iteration counts. Only the genuinely hardware/scale/contact-bound claims remain
out of reach, and those now carry a *tight* reason.

## Newly TESTED (V2.1 — see the ledger for current status)

Built `bench/incremental.py` (a shared implicit-Euler incremental-potential testbed, 10th conformance
gate) + `bench/run_dynamics_solvers.py`, and the AQP self-ablation (`bench/run_agd_vs_aqp.py`). Six
edges moved self-claimed → qualified with measured, iteration-axis evidence:

| edge | result | file |
|---|---|---|
| quasi-newton-liu2017 → l-bfgs | Laplacian init 6 it vs scaled-identity 78 | `dynamics_solvers.md` |
| quasi-newton-liu2017 → projective-dynamics | history m=5 (6) beats m=0 fixed-proxy (10) | `dynamics_solvers.md` |
| chebyshev → projective-dynamics | Chebyshev-PD 7 < PD 10 (accel direction) | `dynamics_solvers.md` |
| quasi-newton-liu2017 → chebyshev | m=2 (7) == Chebyshev (7), no ρ estimate | `dynamics_solvers.md` |
| vertex-block-descent → jacobi | GS converges 0.52×, Jacobi diverges 9.75× | `dynamics_solvers.md` |
| aqp → accelerated-gradient-descent | crossover: proxy helps only ill-conditioned | `agd_vs_aqp.md` |

## NEWLY TESTABLE — queued (the testbed reaches these; not yet run)

- **XPBD / primal-XPBD / pbng → {full-newton, xpbd}** and **jgs2 → {vbd, pd, xpbd} (convergence)** —
  add an XPBD constraint-projection inner solver and a nonlinear-Gauss-Seidel (pbng/jgs2) solver to
  `incremental.py`; compare on the same Φ. Convergence-axis only.
- **second-order-stencil-descent → {gradient-descent, gauss-seidel} (convergence)** — a 2nd-order
  block sweep (≈ our VBD-GS) vs first-order GD (`descent.solve_gd`) and vs a GS baseline, per unit
  work (careful: a VBD *sweep* ≠ a global *iteration*; report per-work, not per-sweep).
- **admm-pd → projective-dynamics / aa-admm → admm (convergence)** — ADMM is well-specified
  (Overby 2017 ADMM-PD; the x-update reuses the PD global system, the z-update is a per-spring prox,
  the dual is a running sum). Anderson-accelerated ADMM wraps the ADMM fixed point in our existing
  `anderson_accelerate` core. Both testable on the V2.2 mass-spring substrate.
- **second-order-stencil-descent → {gradient-descent, gauss-seidel}** — GD exists; a 2nd-order block
  sweep ≈ VBD-GS. Report per-work, not per-sweep.

**BLOCKED on the source paper (will NOT fake — same discipline as #14 Composite Majorization):**
- **tlc → tua / foldover-free → tlc (robustness)** — the Total Lifted Content energy (Du 2020) has a
  specific per-simplex lifted-content formula (a lifting parameter ε combined with the signed area)
  that we could not verify from the homepage/README, only the paper. TUA (Σ|Aₜ|) is trivial, but
  testing "TLC fixes TUA's stuck minima" needs the faithful TLC energy — implementing a look-alike
  would repeat the round-1 unfaithful-substitution mistake. Needs the paper in hand; stays
  self-claimed. (Our `untangle.py` is a *one-sided* penalty, NOT TUA's Σ|A|, so it is not a substitute.)

## Genuinely OUT OF REACH — with a tight reason (not "we didn't try")

- **GPU-throughput / wall-clock speed headlines** (jgs2 "8000×/step", "40–173× faster"; vbd "10×
  XPBD"; descent-gpu, second-order-stencil "two-orders speedup"): the *convergence* half is testable
  (above); the *speed* half is a parallel-hardware throughput claim with no hardware-independent
  proxy — **hardware-confounded**, permanent unless re-run on matched hardware.
- **Competitor-code comparisons** (simplex-assembly / lbd / progressive-* / smith-schaefer /
  advanced-mips vs weber-zorin, aigerman-lipman-2013, kovalsky-2014, lipman-2012, LIM, MatchMaker):
  each needs a faithful port of the *named competitor* (an interior-point QP/SOCP, a specific
  embedding method). Portable in principle but each is its own project — **needs-competitor-code**,
  a concrete v2 backlog item, not a v1 gap.
- **Contact / IPC family** (ipc, gipc, c-ipc, abd, rigid-ipc, barrier-*, ogc, medial-ipc): still need
  a barrier + continuous-collision-detection harness the 2D prototype lacks — **needs-contact-physics**.
- **True-scale claims** (>100K–1.5M elements, FPS budgets): the dense 2D prototype cannot reach them;
  some *trends* are probed (mesh-independence, scale-cost) but the headline magnitudes are not —
  **needs-scale**.
- **Composite Majorization** (#14): needs the paper's specific per-element convex majorizer of the
  twist eigenvalue; substrate ready (`twist_analysis.md`) but the majorizer must not be faked.

## The honest shape

The "try harder" pass converts a chunk of the needs-unavailable-code bucket into *tested* (the
simulation-accelerator convergence edges are implementable and were implemented). What remains
untestable is now dominated by three hard walls — **matched-hardware throughput, ported competitor
code, and contact physics** — each a real capability the v1 prototype lacks, each a v2 target.
