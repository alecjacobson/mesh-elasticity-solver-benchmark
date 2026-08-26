# Measured results (P1 prototype)

Real, reproducible numbers from the [`bench/`](../bench) prototype harness — every figure
produced by `python -m bench.<script>`, gated on `bench/conformance.py` (analytic derivatives
vs finite differences, ~1e-9). These are **small controlled slices** (2D, dense solve, few
filters, no locking-free element yet), not the finished benchmark — but they exercise the full
loop and already reproduce several of the design's predicted effects. Caveats are in each file.

**Figures** for these results (deterministic, `python -m bench.run_figures`) are indexed in
[`figures/README.md`](../figures/README.md) — convergence curves, the locking visualization, the
mesh-independence log-log with CI, Dolan–Moré/data profiles, the claims-graph render, and the
polyscope-headless 3D tet.

## Findings so far

| # | experiment | file | headline (measured) |
|---|---|---|---|
| E1 | filter isolation (symmetric Dirichlet) | [`e1.md`](e1.md) | unfiltered full Newton **non-descent-stalls**; clamp/absolute/identity-shift converge to the same minimum |
| E1ν | near-incompressible ν-sweep (Neo-Hookean) | [`e1_nu.md`](e1_nu.md) | absolute **under**performs clamp as ν→½ — the **volumetric-locking confound** (motivates control C1), not a refutation |
| — | data profiles over problem sets | [`profiles.md`](profiles.md) | filtering is **necessary**: full Newton solves only **30/40** (SD) and **5/12** (NH); filters reach 100% |
| E4 | first- vs second-order | [`e4.md`](e4.md) | Newton wins **iterations** (~10) but **L-BFGS wins wall-clock** (~50 it, skips Hessian); Adam **plateaus** at \|g\|~1e-2 |
| E5 | criterion sensitivity | [`e5.md`](e5.md) | **3 different "fastest" filters across 4 criteria** — ranking is a criterion artifact |
| 1b | dynamic incremental potential | [`1b_dynamic.md`](1b_dynamic.md) | inertia regularizes the Hessian → unfiltered Newton **beats** clamp (opposite of statics; Pitfalls-of-Projection theme) |
| E-lock | locking sensitivity (crossed mesh) | [`locking.md`](locking.md) | on a lower-locking crossed mesh the absolute−clamp gap at ν=0.499 **collapses 141→0 it**, and absolute **converges at ν=0.4999** where it had failed — strong evidence the ν-result was a locking artifact (C1) |
| scale | mesh-independence + CG conditioning (sparse) | [`scaling.md`](scaling.md) | Newton iters **mesh-independent [7,11] over DOFs 98→3042**, while unpreconditioned **CG mat-vecs/iter grow 53→348 (~√DOFs)** (Jacobi-PCG cuts it to 251) — the quantitative case for preconditioning; sparse-direct wall ~DOFs¹·⁰¹ |
| LS | linear-solver axis (direct vs CG) | [`ls.md`](ls.md) | same Newton iterations, but **wall-clock ranks the two solvers oppositely across scenarios** while the mat-vec count stays consistent — rank on the HW-independent count, not wall-clock |
| TR | trust-region vs filtering | [`tr.md`](tr.md) | classical **trust-region (Steihaug-CG, no filter) converges across the whole ν-sweep** — incl. ν=0.4999 where absolute fails — supporting the lineage claim that filtering ≈ modified-Newton/trust-region |
| LSrch | line-search axis (null result) | [`linesearch.md`](linesearch.md) | backtracking ≡ full-step for clamp-projected Newton — a strong filter makes the **line-search axis inert** (axes interact); it becomes decisive only for first-order/aggressive steps |
| **P2** ⭐ | **disentangling the ν-claim (indicative) (P1 vs P2 element)** | [`p2_nu.md`](p2_nu.md) | **on P1 absolute fails vs clamp; on a locking-relieved P2 element absolute matches/BEATS clamp** — the P1 result was a discretization artifact; the paper's claim looks sound once locking is removed (indicative, 2D) |
| W2filt | filter head-to-head (clamp/absolute/trust-region, P1 vs P2) | [`world2_filters.md`](world2_filters.md) | **fair per-element TR** (same cost as filters): beats both on **locking P1** (iters+wall) but **loses to both on locking-free P2** — round-1 'beats-both-on-P2' was a costly-`eigh` artifact, now **reversed**; discretization-dependent (#42/#44) |
| E2 | World-1 accelerators across regimes | [`e2.md`](e2.md) | Sobolev-init helps only in the ill-conditioned regime (validated); **AQP does NOT beat a well-implemented L-BFGS** (paper's ×200 was vs MATLAB — a baseline-quality confound) |
| 3Dν | ν-claim in 3D (P1 tets) | [`3d_nu.md`](3d_nu.md) | the P1-locking confound **generalizes to 3D** — absolute under-performs clamp near-incompressible on P1 tets (worse than 2D) |
| **SLIM** | official libigl SLIM vs AQP | [`slim.md`](slim.md) | to a fair energy-tol **SLIM (5 it) beats AQP (19)** — `slim->aqp` **validated on iterations/factorizations** (soft-constraint drift 4e-16, confound cleared); **wall-clock is C++/Python-confounded** so counts carry the verdict; real tradeoff = 5 factorizations vs AQP's 1 |
| W1prof | World-1 accelerators — **rigorous** (multi-seed, indep. E*, τ-sweep) | [`world1_profiles.md`](world1_profiles.md) | 10 instances, spread + τ-sweep: **Sobolev-L-BFGS > L-BFGS** (slower on these well-conditioned instances, reinforcing 'helps only ill-conditioned'); AQP's tail **blows up at tight τ** (9→77 it); pairwise, not total-order (#48/#49/#51) |
| AA | Anderson vs local-global + generality | [`anderson.md`](anderson.md) | Anderson **~12 it vs ~23 it** (1.92× [1.85–2.00] over **3 seeds × 3 meshes** — robust, #47); wall-clock speedup < iteration speedup; **generality:** same AA core accelerates a Jacobi solve **374→62→37 it** (#36) |
| INJ | injectivity/feasibility — untangle a folded init (barrier vs barrier-free) | [`injectivity.md`](injectivity.md) | **capability axis** (not a speed race): barrier symmetric-Dirichlet is a *definitional* non-starter from folds (+∞ at J≤0; 0 it from a feasible rest start — it polishes, never *finds*), while barrier-free energies untangle **100%**. On the shared **iters-to-first-injective** metric Stable NH (2–3 it) beats the area-penalty (18–40 it). On a **hard non-convex boundary** (wavy warp; injective target guaranteed by exact discrete area-preservation, so A tunes non-convexity not feasibility) both still succeed 100%, but first-crossing takes 610 L-BFGS-B iters vs 6 Newton iters — *suggestive* that the raw penalty's first-order basin is shallow there, though these are **different-algorithm counts, not work-comparable**. Separates the injectivity cohort from distortion-barrier minimizers |
| 1aP ⭐ | full 1a accelerator performance profiles (Newton/L-BFGS/Sobolev/AQP) | [`1a_profiles.md`](1a_profiles.md) | Dolan–Moré + Moré–Wild over **18 problems** (easy/typical/ill-cond × 2 meshes × 3 seeds), shared E*, pairwise (Gould–Scott): **Newton dominant** (median 5 it, beats all); AQP slowest (median 61, wins 6% pairwise) with a first-order tail that **lengthens at tight τ** (12→61 it, 5× vs Newton's ~1×); **Sobolev proxy is a wash vs L-BFGS at this scale** (regime-gated, needs stronger ill-conditioning — honest small-mesh limitation) |
| E3 ⭐ | BCQN triple-split — **full 2³ factorial** (line-search × direction × criterion) | [`e3.md`](e3.md) | one unified solver over all 3 factors: the **factors ENTANGLE, not add** — the Sobolev **direction** cuts iterations most (largest ill-conditioned) but the **barrier line-search cancels it** (its inversion cap binds ~12×/solve on Sobolev's large early steps, reversing typical 26→37), and the **criterion** only re-times the stop. So BCQN's win is not the sum of independent components; `bcqn→l-bfgs` qualified (direction axis reproduces, interaction-gated). n=3 seeds, medians indicative |
| AE | analytic vs numeric eigensystem (2D+3D) | [`analytic_eig.md`](analytic_eig.md) | projection **provably identical** (~1e-15); but with a **fair** baseline the closed form is **~3.7×/5× *slower*** than LAPACK `eigh` — the old "3.3× faster" was FD-assembly cost; the real advantage is autodiff-free SPD projection, not a faster eigensolve (#37) |
| TW ⭐ | the twist eigenvalue **is** the clamp-vs-absolute-vs-CM story (analytic, gated) | [`twist_analysis.md`](twist_analysis.md) | on the validated eigensystem the **only** indefinite element-Hessian mode is the **twist** (negative over 38% of σ-space, all of it compression, zero at the isometry); so clamp (→ε) / absolute (→\|λ_t\|) / Newton (→λ_t) / CM (majorizes it) differ **only there** — the entire ν→½ filter verdict is one scalar per element. Substrate for **#14** without faking CM |
| PoP | Pitfalls: affine-invariance + rate | [`pitfalls.md`](pitfalls.md) | **definitive:** unfiltered Newton affine-invariant (3e-13); clamp/absolute/global-PDN all break it (60.8/0.21/60.8) — projection depends on coordinates, a claim iteration-count can't test (#39) |
| MI | AQP mesh-independence — **rigorous, CI-gated** (multi-seed, indep. E*, τ-sweep) | [`mesh_independence.md`](mesh_independence.md) | **τ-DEPENDENT:** AQP mesh-independent at loose τ (p=−0.09, CI incl. 0) but **GROWS at tight τ (p=+0.68, CI [0.57,0.80])** — round-1 'mesh-independent' was a loose-tolerance artifact. Whether it's *worse than L-BFGS* is **not resolved** (CIs overlap, #R1) (#48/#50/#51/#52) |
| NHeig | NH sweep: FD-vs-complex-step Hessian (clamp/abs decision) | [`nh_eig_check.md`](nh_eig_check.md) | FD eigenvalues match a machine-precision complex-step reference to ~1e-9; the clamp/absolute decision **never flips across ~20k states** — the ν-sweep ranking is a real effect, not FD noise (#32) |
| **P2s** ⭐ | **definitive absolute-vs-clamp: P2 element + Stable NH (both confounds removed)** | [`p2_stable_nu.md`](p2_stable_nu.md) | with the locking-free element AND the correct energy, **absolute BEATS clamp** near-incompressible (38 vs 48 it at ν=0.4999) — the paper's claim reproduces once both confounds are controlled (review-r3) |
| P2sMS | multi-config hardening of the headline (removes the 'single init' caveat) | [`p2_stable_multiseed.md`](p2_stable_multiseed.md) | across **5 genuinely diverse deformation problems** (stretch 1.6×–2.5× + shear, not jitter), **absolute beats clamp 5/5** at ν=0.499 (22 vs 28) and ν=0.4999 (38 vs 56), gap **widening** toward the limit — headline robust to real initial-condition diversity (wide clamp bands 25–37, 49–65); still 2D, single-τ, locking-*relieved* element |
| SC | factorization-vs-iteration cost at scale | [`scale_cost.md`](scale_cost.md) | measured count structure + sparse-Cholesky model: at tight τ **Newton (mesh-indep., 4 factorizations) is cheapest** while AQP's iters blow up (49→206) → AQP 1.5–2.2× Newton and rising — **refutes** slim.md's 'AQP wins at scale' at tight τ (#3) |
| SRI ⭐ | absolute-vs-clamp on a **validated locking-relieved** element (SRI-P2, #74) | [`sri_nu.md`](sri_nu.md) | SRI validated (lower energy → locking relieved; no hourglass — same minimum): on it **absolute crushes clamp 23 vs 250 it** at ν=0.4999 — a **4th independent locking treatment** confirming the P1 'absolute worse' result was a locking artifact |
| SNH | Stable Neo-Hookean absolute-vs-clamp | [`stable_nu.md`](stable_nu.md) | on the **correct** (stable, finite-for-all-J) energy the ν-sweep shows the same locking confound; and in the **inverted-init** regime absolute is designed for (barrier NH is +∞ there) absolute-vs-clamp is a **near-null** — within 1–2 iters, no decisive win (#31) |

## What these already demonstrate for the benchmark's thesis

1. **Confounds are real and measurable.** The near-incompressible filter comparison is confounded
   by volumetric locking — on a lower-locking mesh the absolute−clamp gap at ν=0.499 collapses
   from 141 iterations to 0 (E1ν, E-lock) — and "N× fewer iterations" does not survive translation
   to wall-clock (E4). The two central confounds the survey argues about, reproduced and quantified.
2. **Metric choice changes conclusions.** The same runs rank filters differently under different
   convergence criteria (E5) — empirical support for the metric protocol (`docs/metrics.md`).
3. **Regime matters.** The filter axis behaves oppositely in static (filtering necessary; profiles)
   vs dynamic (inertia makes it optional; 1b) cells — justifying the per-cell taxonomy.
4. **Honesty baselines bite.** Full-batch Adam plateaus above tight tolerance exactly as predicted
   (E4), validating its role as the "did the bespoke method beat a tuned generic optimizer?" control.

## Reproduce

```bash
python -m bench.conformance      # gate
python -m bench.run_e1           # E1        -> e1.md
python -m bench.run_e1_nu        # E1 ν      -> e1_nu.md
python -m bench.run_profiles     # profiles  -> profiles.md   (~3 min, dense)
python -m bench.run_e4           # E4        -> e4.md
python -m bench.run_e5           # E5        -> e5.md
python -m bench.run_1b_dynamic   # dynamic   -> 1b_dynamic.md
python -m bench.run_locking      # locking   -> locking.md
python -m bench.run_scaling      # scaling   -> scaling.md
python -m bench.run_ls           # LS axis   -> ls.md
python -m bench.run_tr           # trust-region -> tr.md
python -m bench.run_linesearch   # line-search -> linesearch.md
python -m bench.run_p2_nu        # SETTLE ν-claim (P1 vs P2) -> p2_nu.md
python -m bench.run_world2_filters # filter head-to-head -> world2_filters.md
python -m bench.run_e2           # World-1 accelerators -> e2.md
python -m bench.run_3d_nu        # 3D ν-claim -> 3d_nu.md
python -m bench.run_slim         # SLIM vs AQP (official) -> slim.md
python -m bench.run_world1_profiles # World-1 profiles -> world1_profiles.md
python -m bench.run_anderson     # Anderson vs local-global -> anderson.md
python -m bench.run_stable_nu    # Stable Neo-Hookean absolute-vs-clamp -> stable_nu.md
python -m bench.run_p2_stable_nu # P2 + stable NH (both controls) -> p2_stable_nu.md
python -m bench.run_scale_cost   # factorization-vs-iteration cost at scale -> scale_cost.md
python -m bench.run_sri_nu       # absolute-vs-clamp on locking-relieved SRI-P2 -> sri_nu.md
python -m bench.run_pitfalls     # affine-invariance + rate -> pitfalls.md
python -m bench.run_mesh_independence # AQP mesh-independence -> mesh_independence.md
python -m bench.run_nh_eig_check  # NH FD-vs-complex-step eig check -> nh_eig_check.md
python -m bench.analytic_eig     # analytic eigensystem check
```

## Honest limitations (→ next P1 steps)

Dense solve (small meshes); 2D only; filters = {none, clamp, absolute, project-on-demand,
identity-shift, global-pdn} + a trust-region (Steihaug-CG) solver (analytic-eigensystem, eigenvalue-blending not yet);
a locking-relieving **P2 element now disentangles the ν-claim (indicative)** (a fully locking-free Taylor–Hood P2–P1 is future work); single scenario/seed for most experiments; no official-code
regression yet (grounding is the FD conformance test). Sparse solve, a Taylor–Hood/MINI
locking-free element, more filters, and porting an official reference (TinyAD/libigl) are the
next steps (`docs/experiments.md`, `docs/protocol.md`).

### Methodological caveats surfaced by round-2 review (apply repo-wide)

These are known, not-yet-closed limitations of the *prototype* measurements — stated here once so
individual result files don't over-read:

- **Single-run, no error bars (#48).** The data profiles (`profiles.md`, `world1_profiles.md`) are
  one run per (instance, budget) with no seed-averaging and no per-instance supplement, though
  `docs/metrics.md` mandates variance bands + auditable per-problem tables. Small profile gaps are
  therefore not significant; treat profiles as descriptive until seed-averaged.
- **Single tolerance, no τ-sweep (#50).** Most verdicts are at one convergence tolerance. E5
  (`e5.md`) shows this can flip the "fastest" method, so any single-τ ranking is provisional until
  shown τ-stable at (e.g.) τ=1e-3 and 1e-6.
- **Self-referential E\* (#51).** Energy-tolerance criteria (`e5`, `world1_profiles`,
  `mesh_independence`, `slim`) score against E\* = best final energy *among the compared methods*,
  which biases toward the strongest solver (`metrics.md #4`). The protocol's independent
  high-accuracy reference (TinyAD/PETSc to τ=1e-12) is the fix; until then energy-tol rows are
  indicative, and pairwise gaps are safer than an E\*-anchored absolute.
- **Rankings are pairwise, not total orders (#49).** Per Gould–Scott, a single data profile
  supports pairwise comparisons, not an N-solver total order; any "A < B < C" phrasing is a
  descriptive summary of the plot, not a ranking claim.
- **Wall-clock at prototype scale is diagnostic-only.** Pure-Python callback overhead (and the
  C++/Python boundary for libigl SLIM) makes millisecond timings unreliable; the HW-independent
  counts (iterations, factorizations, mat-vecs) carry cross-method verdicts until the sparse/
  compiled harness lands.
