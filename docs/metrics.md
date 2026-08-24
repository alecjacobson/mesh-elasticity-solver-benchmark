# Performance Metrics

Metrics govern what conclusions the benchmark is *licensed* to make. This document does the
deliberation the design (`design.md` §11 W4) points to: enumerate an **over-complete candidate
list**, then curate to a set that makes each comparison **neither boringly small (one-axis
"X is fastest") nor too big to conclude (every solver wins on *something* → cherry-picking)**.

## The tension, and four levers that resolve it

- **Too few metrics** → collapses onto one axis (usually wall-clock): boring, non-portable,
  often an artifact of one machine/BLAS/GPU.
- **Too many metrics** → a high-dimensional cloud where every method wins somewhere; no
  defensible ranking (amplifies Gould–Scott's "no valid ranking of >2 solvers").

**Lever 1 — Pair every hardware-DEPENDENT cost with a hardware-INDEPENDENT proxy.** Always
report wall-clock *alongside* iteration count / linear-solves / matrix-vector products. If they
agree, the wall-clock result is credible and portable; **if they diverge, the divergence *is*
the finding.** This is the single most important discipline here, because the claims graph
(`claims/`) shows the literature's biggest numbers are partly hardware (GIPC ~95×, JGS2 ~8000×,
Barrier-Aug-Lagrangian 80× — all GPU-vs-CPU). A Hessian-filter or line-search swap changes
iteration count (HW-independent); a linear-solver swap changes per-iteration cost (HW-dependent)
— separating the axes attributes the win to the right component.

**Lever 2 — Index the metric set by *capability cell*, not globally.** Cells = (World ×
component-swapped × capability). Within each head-to-head report only the small orthogonal core
below. Each comparison stays low-dimensional (mitigating Gould–Scott instability, worst with
many comparable solvers) while the *survey* stays broad because there are many cells.

**Lever 3 — Separate PRIMARY (decision) from DIAGNOSTIC (explanation) metrics.** Rankings are
made only on primaries; diagnostics explain *why* and guard confounds but never declare a
winner. This is what prevents "every solver wins on something."

**Lever 4 — Fix the confounds the catalog's pitfall column repeatedly flags:** one convergence
target/tolerance per cell, one budget, **no per-problem hyperparameter tuning** (Beiranvand–
Hare–Lucet / COCO rule), a fixed *and reported* solver set (Gould–Scott), and a fixed
discretization/element type (so volumetric locking doesn't masquerade as a solver effect — this
is control C1 in `design.md`).

---

## Part 1 — Over-complete candidate catalog (80)

Columns: **W** = World(s) 1/2/3 · **HW** = hardware-dependent? · **P/D** = primary/diagnostic ·
pitfall. "std" = textbook/community convention (no single canonical cite); "[?]" = uncertain as
a *standardized* metric.

### A. Convergence / rate
| # | metric | W | HW | P/D | key pitfall |
|--|--|--|--|--|--|
|1|final gradient norm ‖∇E‖|1,2,3|no|P|not affine-invariant; scales with units/stiffness/DOF; state the norm|
|2|relative gradient ‖∇E‖/‖∇E₀‖|1,2,3|no|P|hides absolute floor; large ∇E₀ makes big drop trivial|
|3|**Newton decrement λ²=∇EᵀH⁻¹∇E**|1,2,3|no|P|best affine-invariant test; needs a solve; use filtered H|
|4|energy gap E−E\*|1,2,3|no|P(W1)|E\* usually unknown → best-found proxy biases toward strongest solver|
|5|empirical convergence order p|1,2,3|no|D|asymptotic-only; noisy near float floor; spoiled by inexact solves|
|6|asymptotic rate constant|1,2,3|no|D|fit-window sensitive; not comparable across scaled energies|
|7|**iterations to tolerance**|1,2,3|no|P|HW-indep cost proxy; not comparable across differing per-iter cost|
|8|work-to-accuracy curve|1,2,3|no|P|work-unit choice changes the winner; fix it up front|
|9|affine/scale-invariance check|1,2,3|no|D|guards unit-gaming; a rescale-only win is fragile|
|10|stagnation/plateau detection|1,2,3|no|D|small step = convergence OR line-search collapse; pair w/ ‖∇E‖|

### B. Cost — hardware-INDEPENDENT proxies
| # | metric | W | P/D | key pitfall |
|--|--|--|--|--|
|11|# function (energy) evals|1,2,3|P|penalizes line-search-heavy methods|
|12|# gradient evals|1,2,3|P|some methods reuse gradients|
|13|# Hessian evals/assemblies|1,2,3|P|unfair as sole axis vs quasi-Newton/L-BFGS|
|14|**# linear solves**|1,2,3|P|conflates direct vs iterative|
|15|**# matrix–vector products**|1,2,3|P|best proxy for iterative solve stage; N/A for direct solvers|
|16|# factorizations|2,3|P|ignores factorization size/fill|
|17|# line-search backtracks|1,2,3|D|correlates w/ func-evals; interacts w/ Hessian filter|
|18|# CCD queries|3|P|broad- vs narrow-phase mixing; scene-dependent|
|19|# active-set updates|3|D|scene-dependent; [?] standardization|
|20|FLOP count|1,2,3|D|rarely faithful (memory-bound); hard to count fairly|

### C. Cost — hardware-DEPENDENT
| # | metric | W | P/D | key pitfall |
|--|--|--|--|--|
|21|**wall-clock to tolerance**|1,2,3|P|machine/thread/BLAS/compiler dependent; pair w/ HW-indep proxy|
|22|wall-clock per frame/step|2,3|P|depends on Δt + tolerance; factorization amortization skews|
|23|time-to-first-feasible / visually-converged|1,2,3|D|"good enough" subjective; define target|
|24|assembly-vs-solve breakdown|1,2,3|D|essential for attribution; boundaries differ across impls|
|25|per-iteration wall-clock|1,2,3|D|mean hides early-vs-late variance|
|26|peak RAM|1,2,3|P|fill-in/reordering/library dependent|
|27|peak GPU memory|1,2,3|P|driver/framework overhead confounds|
|28|GPU utilization/occupancy|1,2,3|D|high util ≠ useful throughput|
|29|effective memory bandwidth|1,2,3|D|needs profiler; cross-library hard|
|30|energy/power (Joules)|1,2,3|D|[?] not a graphics convention|

### D. World-1 distortion / parametrization quality
| # | metric | HW | P/D | key pitfall |
|--|--|--|--|--|
|31|symmetric Dirichlet value|no|P|double-counts if it *is* the objective — report raw distortion separately|
|32|MIPS value|no|P|same double-counting caveat|
|33|per-element singular values σᵢ|no|P|diagnostic backbone; aggregation choice hides cherry-picking|
|34|quasi-conformal error σ₁/σ₂|no|P|insensitive to area|
|35|area/scale distortion σ₁σ₂|no|P|insensitive to angles; report with #34|
|36|angle distortion|no|D|many formulations; specify|
|37|**isometric distortion max(σ,1/σ)**|no|P|single scalar conflates over/under-stretch|
|38|**max (L∞) distortion**|no|P|one bad element dominates; pair w/ mean|
|39|mean/median/percentile distortion|no|P|mean hides tails; area- vs uniform-weight differs|
|40|**distortion ECDF over elements**|no|P|recommended reporting form; needs a plot|
|41|**flipped/inverted element count**|no|P|count vs presence; det≈0 ambiguous|
|42|**local-injectivity flag**|no|P|local ≠ global|
|43|global bijectivity flag|no|P|costly to verify; often only asserted|
|44|singular-value bounds satisfied|no|D|only if bounds imposed|
|45|Hausdorff / one-sided distance to target|no|P|when target exists; sampling-density sensitive|
|46|boundary/constraint error|no|D|soft vs hard handling differs|

### E. World-2/3 simulation fidelity
| # | metric | W | P/D | key pitfall |
|--|--|--|--|--|
|47|**trajectory error vs reference**|2,3|P|gold standard; reference must be truly converged; error can be chaotic → short horizons|
|48|total-energy behavior (drift)|2,3|P|implicit Euler dissipates *by design* — interpret per integrator|
|49|linear-momentum conservation|2,3|D|contact/damping/BCs legitimately break it|
|50|angular-momentum conservation|2,3|P|many integrators don't conserve AM → discriminating|
|51|volume/incompressibility error|2,3|P|element locking confound — hold discretization fixed (C1)|
|52|**non-penetration guarantee (binary)**|3|P|IPC headline; one penetration = hard fail|
|53|max/mean penetration depth|3|P|0 by construction for IPC → only splits non-guaranteed methods|
|54|**friction / stick-slip accuracy**|3|P|lagged vs semi-implicit differ; needs analytic benchmark|
|55|contact-force / impulse error|3|D|reference forces hard to define; [?]|
|56|constraint/thickness drift|3|D|only for methods imposing them (C-IPC)|
|57|quasistatic equilibrium residual|2|P|= gradient-norm criterion; tolerance dominates|
|58|qualitative artifacts (locking/jitter)|2,3|D|subjective; supplementary video, not a number|

### F. Robustness / reliability
| # | metric | W | P/D | key pitfall |
|--|--|--|--|--|
|59|**success rate within budget**|1,2,3|P|y-axis of profiles; target τ + budget drive everything|
|60|**failure-mode taxonomy**|1,2,3|P|why runs fail; manual categorization, not scalar|
|61|initialization sensitivity / basin size|1,2,3|P|needs sampled init ensemble; init distribution arbitrary|
|62|**hyperparameter/tuning fragility**|1,2,3|P|fairness-critical; per-problem tuning is a classic sin — fix params across suite|
|63|largest stable time step|2,3|P|entangles integrator+energy+solver; hold others fixed|
|64|robustness across stiffness sweep|2,3|P|couples to linear solver via conditioning|
|65|robustness across Poisson ratio→0.5|2,3|P|locking confound|
|66|robustness across mesh quality|1,2,3|P|fix + report the mesh-quality metric|
|67|robustness across resolution|1,2,3|P|overlaps mesh-independence|
|68|# problems uniquely best/worst|1,2,3|D|Gould–Scott: NOT a valid >2-solver ranking|

### G. Scalability
| # | metric | HW | P/D | key pitfall |
|--|--|--|--|--|
|69|**iterations vs DOFs (mesh-independence)**|no|P|linear solver can restore/destroy it independently|
|70|wall-clock scaling exponent vs DOFs|yes|P|fit range + cache effects|
|71|linear-solve cost vs DOFs|no*|P|direct O(N^~1.5–2) vs multigrid O(N) — different curves|
|72|strong-scaling efficiency|yes|P|Amdahl ceiling; assembly ≠ solve parallelism|
|73|weak-scaling efficiency|yes|P|comm cost grows; hard to keep per-core work constant|
|74|memory vs DOFs (fill-in)|yes|P|ordering heuristic dominates fill — report it|
|75|setup-vs-solve amortization|yes|D|favors constant/lagged-Hessian methods|

### H. Reproducibility
| # | metric | HW | P/D | key pitfall |
|--|--|--|--|--|
|76|**run-to-run variance (fixed seed)**|yes|P|FP non-associativity in parallel reductions → nonzero; report spread|
|77|seed sensitivity|no|P|separate random *inputs* from random *solver*|
|78|cross-hardware/thread determinism|yes|D|bitwise usually impossible → report agreement to tolerance|
|79|bitwise reproducibility flag|yes|D|almost never true on parallel/GPU|
|80|artifact/config completeness (meta)|no|D|omitting solver-set makes profiles non-reproducible|

---

## Part 2 — The curated orthogonal core (per capability cell)

Each core spans **cost / accuracy / robustness / rate / scalability** in ~4–7 primaries.
Diagnostics are reported but never used to rank.

### World 1 — static distortion / parametrization
**Primary:** iterations-to-tol (#7) + #linear-solves/mat-vecs (#14/#15) · wall-clock-to-tol
(#21) · isometric-distortion **max (#38) + median (#39)**, reported as an ECDF (#40) ·
flipped-element count (#41) + local-injectivity flag (#42) · success-rate-in-budget (#59) ·
mesh-independence (#69).
**Diagnostic:** singular-value spectrum (#33), conformal/area split (#34/#35), asymptotic order
(#5), Hausdorff (#45), failure-mode taxonomy (#60), init sensitivity (#61).

### World 2 — quasistatic / dynamic hyperelastic
**Primary:** iters/step (#7) + factorizations/solves-per-step (#16/#14) · wall-clock/frame
(#22) · trajectory error vs reference (#47) [quasistatic: equilibrium residual (#57)] · energy
behavior (#48) + angular-momentum (#50) [+ volume error (#51) near-incompressible] · largest
stable Δt (#63) + success-rate over stiffness/ν sweeps (#59/#64/#65) · Newton decrement (#3) ·
scaling exponent vs DOFs (#70) + mesh-independence (#69).
**Diagnostic:** linear-momentum (#49), assembly/solve breakdown (#24), line-search backtracks
(#17), qualitative artifacts (#58).

### World 3 — contact-coupled dynamics (IPC-style)
**Primary:** non-penetration guarantee **binary** (#52) [+ penetration depth (#53) for
non-guaranteed baselines] · friction/stick-slip accuracy (#54) · iters/step (#7) + CCD queries
(#18) + linear-solves (#14) + wall-clock/step (#22) · trajectory error (#47) + energy behavior
(#48) · largest stable Δt (#63) + success-rate (#59) + failure-mode taxonomy (#60) ·
scaling vs DOFs (#70).
**Diagnostic:** active-set churn (#19), contact-force error (#55), angular-momentum (#50),
peak memory (#26/#27), assembly/solve/CCD breakdown (#24).

### Cross-cutting (report once for the whole suite)
Run-to-run variance as **error bars** (#76) · tuning-fragility statement (#62) · artifact/config
completeness (#80).

---

## Part 3 — Aggregation across a suite

**Primary: DATA PROFILES (Moré–Wild)** — fraction of problems solved to a fixed target τ vs a
**hardware-independent budget** (evals or mat-vecs). Reads directly as "with budget B, solver X
solves p% of the cell," and — because the x-axis is an absolute budget, not a ratio-to-best — it
is far less sensitive than performance profiles to *which* solvers are in the set. Compute a
**paired wall-clock data profile** as the HW-dependent view (Lever 1).

**Performance profiles (Dolan–Moré)** only for isolated **two-solver** comparisons where the
ratio-to-best view adds insight. Gould–Scott caveats we design around:
1. **No valid ranking of >2 solvers** — restrict ranking claims to pairwise, within-cell.
2. **Solver-set sensitivity** — always report the full set (#80); never compare profiles from
   different sets.
3. **Non-monotonic area** — report the per-problem data in a supplement; the profile is a
   summary, not the sole evidence.
4. **τ sensitivity** — fix τ per cell, show a small sweep (e.g. τ∈{1e-3,1e-6}) to prove the
   ordering isn't a cutoff artifact.
5. **CPU-time in profiles is fragile** — prefer the HW-independent budget for the profile;
   report wall-clock separately.

**Deliverable per cell:** one HW-independent data profile + a paired wall-clock data profile +
a table of the orthogonal-core primaries with run-to-run error bars; per-problem data in the
supplement.

## Sources
Dolan–Moré 2002 (performance profiles) · Moré–Wild 2009 (data profiles) · Beiranvand–Hare–Lucet
2017 (best practices) · Gould–Scott 2016 (profile caveats) · COCO/BBOB (fixed-target ECDF) ·
Deuflhard (affine-invariant Newton, mesh-independence) · IPC 2020 / C-IPC (non-penetration,
stick-slip) · symmetric-Dirichlet / SLIM / Advanced MIPS (distortion) · energy–momentum
integrators (Gonzalez lineage) · FP non-associativity / reproducibility. Uncertainty: Gould–Scott
retrieved via secondary sources; metrics #19/#30/#55 uncertain as *standardized* conventions.
