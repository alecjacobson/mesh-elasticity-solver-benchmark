# 8. Results: The Decomposition Experiments

This section is the report's reason to exist. Each result below is a single-axis decomposition on the
contact-free track, produced by the conformance-gated harness and regenerable from `bench/`; every
number cites a `results/*.md` file. Most v1 decompositions run on a 2D prototype (dense solves, small
meshes, few seeds), and those headlines are *indicative*, each stated with its regime of validity. The
near-incompressibility headline (§8.1) additionally runs on a new **sparse, analytic-Hessian 3D
tetrahedral path** whose projected-Newton scales to **131,712 elements / 68,121 DOF with a
mesh-independent iteration count** (4–5 Newton steps flat across a 3k→132k sweep; `results/scale_3d.md`)
— the substrate that takes the 2D lessons toward the scale the claims are actually about. What survives
is less a set of rankings than a set of *lessons about attribution*.

## 8.1 The headline: a near-incompressibility filtering claim, reversed then re-validated

A recent, well-cited result claims that **absolute** eigenvalue filtering [cite:absolute-filtering]
beats **clamping** [cite:clamp-filtering] near
incompressibility. On standard P1 constant-strain elements our harness finds the *opposite*: as
Poisson's ratio `ν → ½`, absolute filtering under-performs clamp and, at `ν = 0.4999`, fails to
converge at all. Taken at face value this refutes the claim. It does not — the reversal is a
**volumetric-locking artifact of the element**, not a property of the filter.

![Volumetric locking](../figures/locking_p1_p2_sri.png)

*Figure 8.1. The confound, made visual. A near-incompressible Neo-Hookean stretch colored by `J = det
F`. The P1 constant-strain element cannot represent the near-isochoric deformation and buckles into
spurious modes (volumetric locking), taking 130 iterations; a locking-relieved P2 element and a
selective-reduced-integration element deform smoothly and converge in 26 and 66 iterations *at this
figure's ν=0.499 stretch instance* (the ν-sweep table in `results/world2_filters.md` reports the
per-ν counts separately). (`results/world2_filters.md`.)*

Untangling the claim requires removing **two entangled confounds at once**: the *element* (which
governs locking) and the *energy* (the paper's method is built on a specific one). Removing only the
element is not enough — an intermediate round of our own review caught that a "P2 fixes it" result was
measured on the *wrong* (classical-barrier) energy. With **both** confounds controlled — a
locking-relieved P2 element **and** the Stable Neo-Hookean energy [cite:stable-neo-hookean] the method actually targets —
absolute filtering *beats* clamp near incompressibility, and its advantage **grows toward the
incompressible limit**: 38 versus 48 iterations at `ν = 0.4999`, widening to 71 versus 113 at `ν =
0.49999` (`results/p2_stable_nu.md`). A locking artifact would *collapse* at the limit; instead it
strengthens, which is the signature of a real effect.

Four independent locking treatments now concur that the P1 "refutation" is a discretization confound
rather than a filter property: a lower-locking crossed mesh (`results/locking.md`), a standard P2
element (`results/p2_nu.md`), the Stable-Neo-Hookean P2 combination above, and a *validated*
selective-reduced-integration element on which absolute crushes clamp **23 versus 250 iterations** at
`ν = 0.4999` (`results/sri_nu.md`). The effect generalizes to **genuine 3D tetrahedra at scale**: on a
10,368-element P1 tet mesh (the scalable analytic-Hessian harness `bench/tet_scale.py`, not a 2D
prototype), projected-Newton iterations climb from 5 at `ν = 0.30` to **93 at `ν = 0.499`** as the
constant-strain element locks, and — reproducing the 2D P1 reversal in 3D — **absolute under-performs
clamp there (172 versus 93 iterations)**, the same locking artifact rather than a filter property
(`results/tet3d_filters.md`). The 3D locking-relieved control (a P2 / mixed u–p tet, on which the 2D
re-validation predicts absolute should again *beat* clamp) is the pending next step.

The re-validation is not a single-initialization accident: across **five genuinely different
deformation problems** — varying the stretch magnitude (1.6×–2.5×) and adding a shear, not merely
jittering one init — absolute beats clamp on **all five** at both ν = 0.499 and ν = 0.4999 (median 22
vs 28 and 38 vs 56 iterations, with wide per-config bands reflecting the real diversity), and the gap
widens toward the incompressible limit, exactly as a real effect should
(`results/p2_stable_multiseed.md`).

**The lesson.** A decade-old superiority claim that *reverses* and then *re-validates* only once two
entangled confounds — element and energy — are separately controlled. Neither confound acts alone;
this is the report's clearest demonstration that single-axis control is not optional. *(Scope: 2D,
single stretch magnitude and τ; the seed confound is removed above, but the P2 element is
locking-*relieved*, not fully locking-free — a Taylor–Hood / mixed u–p element and 3D remain the
pending gold-standard controls, so this is indicative, not a general proof.)*

## 8.2 Innovations that do not survive fair, faithful re-measurement

Several well-cited advantages shrink or invert once the baseline is fair and the bundled changes are
held fixed:

- **Trust-region filtering [cite:trust-region-filtering] "beats both clamp and absolute."** Our own round-1 measurement reproduced
  this — but it was an artifact of an expensive *global* eigendecomposition operator. The faithful
  *per-element* blend (with a principled SPD-probe schedule) reverses it: trust-region wins on the
  locking element, where the plain filters struggle, but is a wash on the locking-relieved element.
  The operative axis is *volumetric locking*, not Hessian conditioning — measured, the P2 Hessian is
  in fact *worse*-conditioned than P1's, yet converges faster (`results/world2_filters.md`).

- **AQP's mesh-independence is a loose-tolerance artifact.** AQP's [cite:aqp] celebrated mesh-independent
  iteration count holds only to *loose* tolerance. A τ-sweep with a CI-gated growth exponent (iters
  ∝ DOF^p) shows p = −0.09 (CI includes 0, mesh-independent) at τ = 1e-3 but **p = +0.68 (clearly
  growing)** at τ = 1e-6 (`results/mesh_independence.md`, Figure 8.2). The Laplacian proxy gives
  mesh-independent *initial* progress but a first-order *tail* that is not. Honesty cuts both ways:
  the same CI-gating forced us to *retract* our own follow-on claim that "AQP scales worse than
  L-BFGS," which the overlapping confidence intervals do not support.

![Mesh independence](../figures/mesh_independence.png)

*Figure 8.2. AQP's mesh-independence is tolerance-dependent — a flat growth exponent at loose τ, a
clearly growing one at tight τ, with min–max bands and CI-gated exponents.*

- **AQP's single-factorization "wins at scale" — refuted at tight tolerance.** Measured factorization
  and back-solve counts plus a sparse-Cholesky cost model show AQP's iteration count blowing up
  (49 → 206 over the size range) at tight τ, making it 1.5–2.2× the cost of mesh-independent Newton
  and rising (`results/scale_cost.md`).

- **AQP "×200 faster than L-BFGS" — a baseline-quality confound.** The celebrated factor was measured
  against a MATLAB reference L-BFGS; against a *well-implemented* L-BFGS, AQP does not win on raw
  iteration count in either a well- or ill-conditioned regime (`results/e2.md`). AQP's genuine,
  separable claim is cheap mesh-independent *initial* progress, not raw iterations versus a strong
  baseline.

- **What *does* validate: SLIM > AQP.** To a fair relative-energy tolerance, official libigl SLIM [cite:slim]
  reaches the symmetric-Dirichlet minimum in **6 iterations versus AQP's 19** (counts aligned to a
  common pre-step convention), and a seed × mesh profile confirms SLIM's worst case stays below AQP's
  best case at every one of four resolutions — one of only two independently validated edges (§9.1).
  The soft-versus-hard-constraint confound was checked and cleared; the wall-clock is
  C++/Python-confounded, so the *counts* carry the verdict, and the real trade-off is SLIM's 6
  factorizations against AQP's single one (`results/slim.md`).

That SLIM-versus-AQP trade-off — few expensive iterations versus many cheap ones — is the whole field
in miniature, and Figure 8.2b makes it concrete across *all* the faithful distortion solvers at once,
on a single controlled instance.

![Running example](../figures/running_example.png)

*Figure 8.2b. The controlled running example — one symmetric-Dirichlet scene, six faithfully
implemented solvers, convergence to a shared minimum shown both ways. **Left (vs iteration):**
projected-Newton and Composite Majorization reach the minimum in the fewest steps, BCQN close behind,
AQP slowest — second-order and superlinear methods win the iteration axis. **Right (vs cost):** the
ranking compresses and re-orders, because Newton and CM refactor a coupled Hessian every iteration
while AQP and BCQN factor once and reuse it — "fewest iterations" is not "cheapest." Wall-clock is
comparable only within this pure-Python group (libigl SLIM's compiled C++ is excluded from the time
axis); iteration counts remain the portable, hardware-independent verdict, and the cost panel is
illustrative of the per-iteration-cost structure the benchmark keeps separate from the algorithm.*

## 8.3 Bundled methods entangle rather than add

BCQN [cite:bcqn] claims "fastest and most robust" from three simultaneous changes — a blended Sobolev/L-BFGS
proxy, a barrier-aware line search, and a characteristic-gradient criterion. The full 2³ factorial
(one unified solver over all three axes) shows the components **interact rather than sum**. The
Sobolev *direction* is the only factor that moves the iteration count, and only in its regime: it is
a wash pooled but beats L-BFGS on all six ill-conditioned problems. The barrier *line search* does not
add iteration speed and can *cancel* the direction — its inversion cap binds on the Sobolev
direction's large early steps (about 12 caps per solve), so the same comparison that reads L-BFGS 34 →
Sobolev 26 under backtracking becomes L-BFGS 33 → **Sobolev 37** under the barrier arm. The *criterion*
only re-times the stop (`results/e3.md`, Figure 8.3). A pooled main-effects table would have hidden
this interaction; an adversarial review of our own factorial caught exactly that, and we report the
per-cell effect instead.

![E3 factorial](../figures/e3_factorial.png)

*Figure 8.3. The BCQN direction factor is regime-gated — the Sobolev proxy's iteration reduction grows
with ill-conditioning and vanishes elsewhere — and interacts with the line-search factor.*

**The lesson.** BCQN's bundle is one strong (regime-gated) factor plus two minor ones that interact,
not three co-equal contributions. On this barrier energy, the barrier-aware line search is moreover
partly *redundant* with the energy's own `+∞`-at-inversion barrier.

**The assembled method, faithfully.** To test the *whole* method rather than its factored parts, we
reimplemented BCQN end-to-end from the paper and the authors' reference code — the cotan-Laplacian
proxy (scaled by two) factored once and applied per coordinate, the secant/Laplacian *blend* of
Eq. 13 (which mixes the L-BFGS secant with the Laplacian-applied step under a curvature-ratio weight
clamped to the unit interval), the "cured" barrier-aware direction filter (a per-element no-inversion
QP solved by damped projected Jacobi), the inversion-free plus Armijo line search, and the
characteristic-gradient stop — and conformance-gated it on the blend weight staying in the unit
interval, monotone descent, and convergence to the projected-Newton minimum (`bench/bcqn.py`). On
symmetric Dirichlet over six mesh/seed scenarios (`results/bcqn.md`), full BCQN reaches the shared
energy tolerance in **8.0 iterations** — versus AQP's 26.3, its own no-blend Sobolev-L-BFGS ablation's
12.7, and a well-implemented L-BFGS's 12.0 — so `bcqn to aqp` and the blend's contribution both
**reproduce on the hardware-independent axis**, and the paper's headline over AQP is earned there (its
7-fold wall-clock figure is a separate, hardware-confounded claim). Against the second-order methods
the ordering **inverts**, exactly as expected: BCQN needs **more** iterations than projected-Newton
(6.8) and Composite Majorization (7.0), because it descends a *fixed* scalar-Laplacian proxy while they
refactor a coupled Hessian each step. BCQN's claim over projected-Newton and CM is therefore the same
shape as Projective Dynamics versus Newton (§8.7): a *cheaper-per-iteration*, factor-once argument that
lives in wall-clock and memory-at-scale, not a fewer-iterations one — so it stays
`qualified` on the mechanism, not the iteration axis.

## 8.4 Injectivity is a capability axis, not a speed contest

The World-1 injectivity methods (TLC, foldover-free, progressive embedding) are barrier-free untangling
energies — the classical maximize-minimum-area lineage of §5. The benchmark's feasibility suite makes
the capability distinction concrete: a **barrier** distortion energy is a definitional non-starter from
a folded map (its energy is `+∞` there; given a *feasible* start it converges normally, but it can
only *polish* an injective map, never *find* one from folds), while **barrier-free** energies untangle
folded initializations 100% of the time — the classical area penalty and Stable Neo-Hookean both
recover, the latter in far fewer iterations owing to a better elastic basin (`results/injectivity.md`,
Figure 8.4).

![Injectivity](../figures/injectivity.png)

*Figure 8.4. A folded initialization (108 inverted elements, red) untangled to all-valid by two
barrier-free energies; the barrier symmetric-Dirichlet energy is `+∞` at folds and cannot start.*

On a *hard* non-convex boundary — a wavy warp whose injective target is guaranteed by exact discrete
area preservation — both barrier-free energies still succeed, but the raw area penalty needs far more
first-order steps to first-crossing than the elastic energy needs Newton steps; we report this as
suggestive of a shallower basin rather than a clean ratio, since the two use different algorithms and
their iteration counts are not work-comparable. This is exactly the capability axis — untangle from
folds — that separates the injectivity cohort from distortion-barrier minimizers.

We now port one cohort member *faithfully*. **Total Lifted Content** (TLC, Du et al. 2020
[cite:tlc]) — reimplemented from the paper and its reference code as the exact lifted-content energy
(the Cayley–Menger form on lifted squared edge lengths, uniform auxiliary, auto-scaled lifting `α`) and
conformance-gated on its defining properties (finite and smooth at a fold, `α → 0` equal to total
unsigned area, untangles to full injectivity; `bench/tlc.py`) — settles TLC's own key ablation
`tlc → tua`. On eight folded initializations (convex and non-convex targets), TLC untangles **6/8**
where its `α = 0` limit (Total Unsigned Area) untangles only **1/8**, and where both succeed TLC's
lifted gradient reaches injectivity in fewer iterations (median 34 vs 60) — a clean single-axis
confirmation of the paper's Proposition 4.3, that the *lifting* is what turns the degenerate
unsigned-area plateau into an injective minimizer (`results/tlc.md`). The barrier symmetric-Dirichlet
energy remains `+∞` at a fold and cannot start at all — the capability distinction, now drawn with the
real TLC energy rather than a classical area-penalty stand-in. Ranking TLC *against the other cohort
members* (foldover-free, LBD, simplex-assembly) still needs their code, and TLC's large-scale
100%-success headline is not adjudicated here.

## 8.5 The clamp-versus-absolute question is one analytic scalar

Built on the *validated* analytic eigensystem [cite:analytic-eigensystems] (which matches a finite-difference Hessian to ~1e-10),
we establish the structural fact under the entire World-2 filter debate: the 2D symmetric-Dirichlet
element Hessian's **only sign-indefinite eigenmode is the twist**, `λ_t = (g(σ₁)+g(σ₂))/(σ₁+σ₂)`. Over
250,000 samples of the singular-value plane, the two stretching modes and the flip mode are *never*
negative; the twist is negative over 37.8% of the plane, all of it under compression, and exactly zero
at the isometry (`results/twist_analysis.md`, Figure 8.5). Therefore every projected-Newton filter is
*identical except on the twist*: clamp sends it to ε, absolute to `|λ_t|`, raw Newton keeps it
(indefinite), and Composite Majorization [cite:composite-majorization] majorizes it. The entire `ν → ½` filter verdict of §8.1 is
**one scalar per element**, active only under compression — precisely the regime a near-incompressible
material enters as it necks.

We now implement Composite Majorization *faithfully* — its singular-value convex-concave construction
(`bench/composite_majorization.py`), conformance-gated on the paper's own **Proposition 3.1** (the CM
Hessian majorizes the true Hessian, $H \succeq \nabla^2 f$), on monotone majorize–minimize descent, and on
convergence to the *same* minimum as projected-Newton, for both symmetric Dirichlet and symmetric
ARAP. Testing it settles the long-deferred `composite-majorization` edges (`results/composite_majorization.md`):
CM decisively beats first-order **AQP** (9 versus ~780 iterations, `→` qualified), but its headline
**"4× faster than projected Newton" does not reproduce on the hardware-independent iteration axis** —
CM takes 9.0 iterations versus projected-Newton's 8.8, essentially tied. This is exactly what a
*majorizer* must do: because $H \succeq \nabla^2 f$, CM takes conservative guaranteed-descent steps, whereas the
clamp filter minimally projects only the indefinite twist. The paper's speed advantage is a
*wall-clock* claim resting on its cheap analytic Hessian — which it also uses for its own
projected-Newton, so it is not the algorithmic differentiator. An honest close to the one edge we had
left deliberately unmeasured (§9.1).

![Twist phase](../figures/twist_phase.png)

*Figure 8.5. The twist eigenvalue over the singular-value plane (left; blue = negative = indefinite,
all under compression) and the clamp↔absolute gap `|λ_t|` (right) — the only place, and the only
amount, by which the filter choice matters.*

## 8.6 Confounds the benchmark quantifies

Finally, the harness measures several confounds directly:

- **Filtering is necessary.** Unfiltered full Newton non-descent-stalls: it solves only 30/40
  symmetric-Dirichlet and 5/12 Neo-Hookean instances, while the eigenvalue filters reach 100%
  (`results/profiles.md`) — the concrete reason the filter axis exists.
- **First- versus second-order inverts under wall-clock.** Newton wins on iterations (~10) but
  **L-BFGS wins on wall-clock** (~50 iterations, each skipping a Hessian factorization); Adam plateaus
  above tight tolerances — the honesty control (`results/e4.md`).
- **Criterion sensitivity.** The same three filter runs, re-scored under four convergence criteria,
  produce **three different "fastest" filters** — the ranking is a criterion artifact
  (`results/e5.md`).
- **Projection breaks affine invariance.** Unfiltered Newton's step is affine-covariant to ~1e-13;
  every eigenvalue projection breaks it (an O(1) covariance residual) — the Pitfalls-of-Projection
  thesis, shown directly and untestable by an iteration-count comparison (`results/pitfalls.md`).
- **The full 1a accelerator profile.** Over 18 problems on a shared reference, Newton dominates the
  performance profile on iteration count; AQP's first-order tail lengthens at tight τ; the Sobolev
  proxy is a pooled wash but wins within the ill-conditioned stratum — a regime structure the pooled
  profile hides and the per-stratum pairwise surfaces (`results/1a_profiles.md`).
- **Anderson acceleration validates.** Wrapping ARAP local–global [cite:local-global] in Anderson mixing [cite:anderson-geometry] reaches
  the same minimum in **13 versus 24 iterations** (a 1.85× iteration speedup, robust across three seeds
  and three meshes — it never collapses to 1×), the second of the two validated edges. Each iteration
  is one back-solve for both, so the iteration ratio is the hardware-independent work ratio; the same
  acceleration core also speeds an unrelated Jacobi fixed-point (a generality check), and — wrapped
  instead around the *official libigl SLIM* fixed-point map — cuts a deliberately slow-contracting
  instance from **380 iterations to 10** (a 36–38× reduction, verified faithful to continuous SLIM,
  `results/anderson_slim.md`). That last result is only **qualified**, not validated: it rests on a
  single hand-picked instance, so the *direction* (Anderson wraps and speeds SLIM) is solid but the
  *magnitude* is instance-selected. The wall-clock speedup is smaller than the iteration speedup owing
  to Anderson's per-iteration
  least-squares (`results/anderson.md`).

Every one of these is a place where a single, unstated component choice — a filter, a criterion, an
implementation language, a Hessian modification — governs a published "advantage."

## 8.7 The simulation-accelerator family, on one shared potential

A large part of the corpus — projective dynamics, XPBD/PBD, vertex block descent, quasi-Newton,
Chebyshev, ADMM, and their Anderson accelerations — had been triaged "needs the paper's code." A
*try-harder* pass shows most of it is not code-bound at all: these methods are simply different inner
minimizers of the **same** implicit-Euler incremental potential (or, for the position-based family, the
same mass-spring system). Building that shared testbed once — conformance-gated so each solver's
per-vertex block equals the assembled Hessian block, its projective-dynamics global system is exact
local/global, and its XPBD update is the exact compliance form — lets every method be compared on the
*hardware-independent* iteration/quality axis, faithfully. The convergence claims largely **reproduce**;
the GPU/throughput *speed* headlines (jgs2's "8000×/step", VBD's "10× XPBD") stay hardware-confounded
and are not adjudicated.

- **Second-order and preconditioning beat first order, as claimed.** Quasi-Newton with a mass+Laplacian
  initial metric converges in **6 iterations versus 78** for scaled-identity L-BFGS; adding L-BFGS
  history to the fixed-proxy step (Projective-Dynamics-style) improves it (6 vs 10); Chebyshev and
  Anderson each shave the proxy's tail (7 and 6 vs 10) (`results/dynamics_solvers.md`). Composite
  Majorization — implemented faithfully (§8.5) — needs **9 iterations versus AQP's ~780**
  (`results/composite_majorization.md`).
- **XPBD's compliance is real; its residual is not the physics.** On the mass-spring testbed XPBD's
  constraint sweep **stagnates** on the incremental-potential residual (it omits the momentum coupling),
  while local/global, Newton, and nonlinear Gauss–Seidel drive it to zero — so a *primal* method
  "reaches tolerance where XPBD stagnates" reproduces. Yet XPBD's *positions* stay within **0.7%** of the
  true Newton solution at a soft operating point — "visually indistinguishable," but only there: at
  stiff-cloth stiffness × large timestep the error climbs to **65%**. And XPBD's constraint violation is
  **iteration-count independent** (compliance) where PBD stiffens ~23× with iterations
  (`results/massspring_solvers.md`).
- **Vertex Block Descent and the proxy ablations.** Gauss–Seidel block updates converge where an
  under-relaxed block-Jacobi crawls; AQP's own AGD ablation (proxy disabled) *beats* AQP when
  well-conditioned but blows up when ill-conditioned — across a 1000× sweep of the momentum parameter,
  so the proxy's value is real but conditional, and the "baseline-confounded" flag is defused
  (`results/agd_vs_aqp.md`). Anderson-accelerated ADMM cuts plain ADMM 11→6 iterations
  (`results/admm_ms.md`).

The recurring shape: on the axis a 2D prototype can measure honestly — iterations, constraint
violation, position error — the simulation family's *convergence and quality* claims mostly hold, each
with its regime spelled out, while the *wall-clock/GPU* headlines remain out of reach. "Mostly" is
literal, not a hedge: several specific sub-claims did **not** reproduce on the iteration axis and we
say so — Anderson-ADMM does not beat plain Projective Dynamics on iterations (11 vs 8), Composite
Majorization ties rather than beats projected-Newton (9.0 vs 8.8, §8.5), and Projective Dynamics needs
~5× *more* iterations than Newton, not fewer (16–36 vs 4, `results/pd_vs_newton.md`) — its
interactive-speed edge is factorization reuse (one prefactored constant system vs Newton's
per-iteration refactorization), a per-step-cost story that resolves to wall-clock, not a fewer-steps win. Those stay `qualified` on the
*direction* they establish, not the headline margin. This pass moved
**forty-one edges** from the field's own word to `qualified`, and added *nothing* to `validated`
(§9.1) — the honest yield of trying hard without inflating.

## 8.8 Opening the contact world: IPC's guarantee, minimally but faithfully

World-3 — contact — had been surveyed but entirely deferred, its 22 edges all `unmeasured`. We now
open it with a **minimal but faithful 2D IPC** (`bench/ipc.py`): the exact C² log-barrier
`b(d) = -(d-dhat)^2 * ln(d/dhat)` (conformance-gated on its shape and on its first and second
derivatives versus finite differences), a
**CCD-filtered line search** that caps each step short of any wall crossing, and an implicit-Euler
incremental potential minimized by projected Newton — the three ingredients that make IPC IPC. The
scope is honest: vertex-versus-half-plane contact (linear CCD) and a mass-spring body settling into a
fixed wedge; mesh–mesh CCD, friction, and 3D/GPU scale are not implemented.

That minimal harness is already enough to adjudicate IPC's *defining* claim, `ipc → prior-rigid-engines`
(robustness): the **guaranteed intersection-free trajectory**. Thrown into the wedge at impact speeds
from 4 to 30 (with a large `dt = 1/30`), IPC keeps every wall distance **strictly positive at every
speed** (minimum distance 0.013–0.016) — the barrier is `+∞` at contact and the CCD cap makes a
crossing impossible *by construction*, independent of velocity — while a classical quadratic penalty
method (finite stiffness, no CCD) **tunnels on all five impacts**, worse as the impact hardens
(penetration depth 0.09 → 0.28; `results/ipc.md`). This is the distinction between a *guarantee* and a
*tuned parameter*: no stiffness or timestep makes IPC penetrate, whereas the penalty's safety is
speed-dependent. `ipc → prior-rigid-engines` moves off `unmeasured` to `qualified` — the first
contact-world edge the benchmark adjudicates. IPC's *speed/throughput* edges over GIPC, ABD, and
medial-IPC remain `unmeasured`: those are GPU-scale, mesh–mesh claims a minimal 2D harness cannot reach.
