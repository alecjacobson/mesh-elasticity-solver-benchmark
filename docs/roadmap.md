# Roadmap — a durable, pull-from-the-top work queue

A standing backlog so work can continue for a long time without re-deciding each session. **How to use:**
pull the top *un-done* item in the highest-priority thread that is not blocked, do it end-to-end to the
Definition of Done, tick it, move on. Rotate across threads so no single one starves.

## Definition of Done (every item)
1. **Faithful, not a look-alike.** A method is reimplemented from its paper (+ reference code where it
   exists); never substitute a stand-in and call it the method. If a piece is genuinely unrecoverable,
   say so and scope the claim — do not fabricate.
2. **Conformance-gated.** Add a gate in `bench/conformance.py` asserting the method's *defining*
   property (barrier-free, monotone MM, ==reference minimum, intersection-free, rigid-invariance, etc.).
   All gates must stay green.
3. **Adjudicated on the hardware-independent axis** (iterations / factorizations / matvecs / success
   rate), with wall-clock only paired and caveated. Report the honest verdict and its exact regime.
4. **Reconciled.** Update `claims/claims.yaml`, cascade the counts across the paper, regenerate the
   claims figures, add a `results/*.md` and a `results/README.md` row.
5. **CI-green.** `bash paper/build.sh` (via the `build-paper` workflow) must compile; keep the `.tex`
   ASCII-safe (no un-mapped glyphs, no LaTeX-in-backticks).
6. **Periodic adversarial self-review** every ~4–5 promoted edges: turn the confound-untangling on our
   own new verdicts; downgrade anything that over-reaches.

Snapshot at last update: **17 conformance gates; ledger 2 validated / 64 qualified / 73 self-claimed /
21 unmeasured; four faithful reimplementations (Composite Majorization, BCQN, TLC, IPC); sparse 3D to
131K tets; §8.1 flagship complete in 3D.** Read live counts from `claims.yaml`, not from here.

---

## Thread 1 — deepen & stress the empirical core
- [ ] **T1.1 Adversarial 3D scenarios.** Twist/torsion and large-compression on `tet_scale`, sweeping
  stiffness/conditioning — stress tests that *break* solvers, not benign stretches (benchmark rubric:
  small is fine only if adversarial). Adjudicate filter necessity + clamp/absolute/none success in 3D.
- [ ] **T1.2 Filter necessity & first-vs-second-order at 3D scale.** Does unfiltered Newton stall in 3D?
  clamp/absolute/none success + iteration profiles on hard 3D scenes; L-BFGS vs Newton on Φ in 3D.
- [ ] **T1.3 Stable Neo-Hookean in 3D + inversion recovery.** Recover from inverted tets; extend the
  Stable-NH inversion-recovery claim (currently 2D-qualified) to 3D.
- [ ] **T1.4 3D performance/data profiles.** Dolan–Moré + Moré–Wild over a set of 3D scenes.
- [ ] **T1.5 (heavy, optional) Taylor–Hood / mixed u–p 3D element** — the further gold standard for §8.1.

## Thread 2 — faithful reimplementations (adjudicate more claims)
- [ ] **T2.1 Mesh–mesh 2D IPC + friction.** Vertex-vs-edge distance + point-edge CCD (quadratic TOI) and
  a friction term; unlocks contact edges beyond the halfplane guarantee (e.g. friction / C-IPC-style).
- [ ] **T2.2 Faithful foldover-free / SEA (Su et al. 2019)** and/or **LBD (Kovalsky 2015)** — to *rank
  within* the injectivity cohort (TLC vs FF vs LBD vs simplex-assembly), the §8.4 deferred comparison.
- [ ] **T2.3 Progressive Embedding (Shen et al. 2019)** — faithful reimplementation; adjudicate its edges.
- [ ] **T2.4 Accelerated VBD** (Anderson/Chebyshev layer) — revisit `vertex-block-descent→l-bfgs`
  properly (the earlier probe tested base VBD only) on a vectorised harness.
- [ ] **T2.5 AKVF (Killing-operator metric)** in World-1 — the one metric-table method not yet built.

## Thread 3 — paper quality (make it un-rejectable)
- [x] **T3.1 Signature figures.** Dolan–Moré performance + Moré–Wild data profiles already present
  (18 problems, pairwise per Gould–Scott). ADDED a **work-precision** diagram (Figure 8.2c,
  `fig_work_precision`): cost (iterations) to reach k accuracy digits, over the accelerator set —
  Newton/CM flat-cheap, AQP's first-order tail explodes to 250+ iters. Done.
- [ ] **T3.2 Proportionate-framing pass.** Recalibrate now-stale "2D prototype / dense / indicative"
  hedging to the true state (3D to 131K tets, four faithful methods, contact opened, flagship-in-3D) —
  keep every caveat that is still true; remove the ones the evidence has outgrown. Honesty, not spin.
- [ ] **T3.3 Reproducibility-attrition headline (Dacrema-style).** Report the corpus scan → testable-now
  rate as a first-class finding, with the attrition numbers.
- [ ] **T3.4 Unified-notation table** (symbol → meaning) — a signature survey element.
- [ ] **T3.5 §4 read-as-argument pass.** Rewrite the survey-by-axis so each method paragraph answers
  "what problem in the design space / what it trades / how it relates to neighbours," not annotation.
- [ ] **T3.6 Visual convergence gallery** — same scene at N iterations converging toward the Newton
  solution (complements the §8.2b curves).

## Standing / cross-cutting
- Keep the `build-paper` CI green; keep `results/README.md` and the claims figures in sync.
- Every ~4–5 promotions: an adversarial self-review wave (spawn reviewers; downgrade over-reach).
- Never let the ledger drift from the paper counts; never commit a look-alike as a method.

_Next up pointer: T3.1 (signature figures) → T2.2 (cohort ranking) → T1.1 (adversarial 3D). Rotate._
