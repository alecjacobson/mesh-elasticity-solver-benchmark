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

**Claim triage** ([`claim_triage.md`](claim_triage.md)): every self-claimed edge classified **testable-now** (14 edges → the verification backlog) vs **untestable** with a categorized reason (needs-unavailable-code ~34, needs-contact-physics 22, needs-scale 21, entangled-needs-source 9, hardware-confounded 4, subjective-quality 3, baseline-confounded 2, needs-3D 1) — the honest boundary of what a 2D contact-free prototype can adjudicate.

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
| BLEND | eigenvalue-blending filter (clamp<->absolute interpolation) | [`blend_filter.md`](blend_filter.md) | the blend is **exactly clamp@w=0.5, absolute@w=1.0** (gated to 0); on the locking-relieved **P2** element an intermediate blend **beats both endpoints** (w=0.625->20 vs clamp/abs 23; w=0.875->35 vs 53/41) so `eigenvalue-blending->{clamp,absolute}` **reproduces** there; on locking **P1** clamp dominates and blending only interpolates. Regime-dependent (qualified) |
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
| ROB | filter/method **robustness** — success rate from inverted starts (P5.2 #2,#3,#12) | [`filter_robustness.md`](filter_robustness.md) | 100-start inverted-init recovery battery (5 severities × 20 seeds): **trust-region & project-on-demand recover 100/100 vs unfiltered Newton 5/100** (+95 each) — `trust-region→full-newton` and `pitfalls-PDN→full-newton` **validated**; but **absolute==clamp (100=100, tie)** — the absolute→clamp *robustness* edge is **not distinguished** (a line search equalizes them; its edge is per-step descent-quality, not basin-of-attraction) |
| A-SLIM | **Anderson acceleration OF official libigl SLIM** (P5.2 #9) | [`anderson_slim.md`](anderson_slim.md) | wrapping the **official SLIM** fixed-point map in the Anderson core on a slow-contracting instance: plain SLIM **380 it** to cut the residual to 1e-3, Anderson (m=5) **10 it** — a **38× reduction** → `anderson→slim` **validated**. Key: SLIM's *energy* saturates in ~1 step, so the win lives in the residual **tail** — measuring energy hides it |
| AQP-LG | AQP vs local-global vs Anderson-LG — back-solves to own minimum (P5.2 #5,#6) | [`aqp_localglobal.md`](aqp_localglobal.md) | on a shared bend instance (equal per-iter cost): **Anderson-LG 4.0 < local-global 5.6 < AQP 11.4** back-solves. So **#5 `anderson→aqp` reproduces** (Anderson cheaper, cross-energy caveat) but **#6 `aqp→local-global` does NOT** (AQP needs *more* solves here) — both qualified/indicative (SD vs ARAP, different minima) |
| SLIM-NU | SLIM vs projected-Newton on a **non-uniform** mesh (P5.2 #7) | [`slim_nonuniform.md`](slim_nonuniform.md) | on a 306×-aspect jittered mesh + 2× stretch, **projected-Newton reaches the tol in 5 it; SLIM does not in 60** (drift 2e-9, not a soft-BC artifact) — the **reverse** of the claim. Well-safeguarded (clamp-projected) Newton stays near-quadratic; SLIM crawls a slow linear tail. `slim→projected-newton` **not reproduced** — the Fig.11 stall is a *far-from-min* pathology a stretch can't create (qualified, documented why) |
| DYN ⭐ | **inner-solver convergence on one incremental-potential timestep** (V2.1) | [`dynamics_solvers.md`](dynamics_solvers.md) | many simulation accelerators are inner minimizers of the **same** implicit-Euler Φ (conformance-gated: ∇Φ vs FD 1e-9, VBD block == Hessian block 0). On a stiff dt=1 timestep: **quasi-Newton L-BFGS (Laplacian init) 6 it vs scaled-identity 78** (`quasi-newton→l-bfgs` reproduces, ~13×, the strongest); **history m=5 (6) beats fixed-proxy m=0 (10)** (`quasi-newton→projective-dynamics`, a clean self-ablation); **Chebyshev 7 < proxy 10** (`chebyshev→PD` — direction only, ρ-tuned); **L-BFGS m=2 (7) ties Chebyshev (7)** but needs no ρ-estimate (`quasi-newton→chebyshev` — the no-tuning half, not "faster"); **VBD Gauss-Seidel (0.52×) beats under-relaxed Jacobi (0.93×)** per 24 sweeps (`vbd→jacobi`; earlier un-relaxed "Jacobi diverges" was a strawman, fixed). 5 edges → qualified (single-timestep, HW-independent iteration axis; GPU/wall-clock headlines stay hardware-confounded) |
| AGD | AQP vs its own ablation (proxy on/off) across conditioning (V2.1) | [`agd_vs_aqp.md`](agd_vs_aqp.md) | AGD = `solve_aqp(use_proxy=False)`, the paper's literal self-ablation (`d=−g` not `L⁻¹(−g)`). Sweeping anisotropic stretch: **clear crossover** — well-conditioned (s=1) AGD *wins* 61 vs 573 (proxy hurts), ill-conditioned (s=2.5) AGD blows up 937 vs 256, s=3.5 AGD maxiters. **Not a θ artifact:** AGD stays ~940 across a 1000× η/θ sweep at s=2.5. `aqp→accelerated-gradient-descent` **reproduces (regime-dependent)** → qualified; and it **defuses the baseline-confound flag** — the ablation is a fair competitor, not a straw-man |
| MS ⭐ | **constraint-projection solvers on one mass-spring timestep** (V2.2) | [`massspring_solvers.md`](massspring_solvers.md) | faithful mass-spring Φ (conformance-gated; PD/local-global is *exact* here, not a stand-in). Newton (2)/local-global (8)/nonlinear-GS (14) converge; **XPBD STAGNATES** at 0.14·r₀ (omits the momentum residual) and **PBD converges to a too-stiff state** (residual ~4·r₀, over-constrains). Stiffness test: **XPBD `\|C\|`=0.10 flat across K=2→80** (iteration-independent) while **PBD `\|C\|` shrinks 0.091→0.004** (~23×, stiffens with iterations). 7 edges → qualified: `primal-xpbd→xpbd`, `pbng→xpbd`, `fast-mass-spring→pbd`, `projective-dynamics→pbd`, `xpbd→pbd`, plus **quality**: `xpbd→full-newton` (XPBD **0.7%** off Newton positions — visually indistinguishable despite stagnating — but **only at low stiffness**: →**65%** at k=1e5×dt=1/8, i.e. it FAILS in the stiff-cloth regime XPBD targets) and `vbd→xpbd` (VBD matches true implicit-Euler, XPBD plateaus) |
| ADMM | ADMM-PD & Anderson-ADMM on the mass-spring timestep (V2.3) | [`admm_ms.md`](admm_ms.md) | ADMM-PD (Overby 2017): x-update reuses PD's constant system, per-spring prox z-update, running dual. **Anderson-ADMM 6 it vs plain ADMM 11** → `aa-admm→admm` **reproduces**; but plain ADMM (best 11 over ρ-sweep) is **not** faster than PD (8) on iterations → `admm-pd→projective-dynamics` **not reproduced** (its "faster" is a wall-clock/weight claim, confounded). 2 edges → qualified |

| PROP | structural solver properties — simplicity/generality/factorization count (V2.6) | [`solver_properties.md`](solver_properties.md) | architectural facts on the testbeds: **PD is monotone with no line-search/filter/SVD** (`projective-dynamics→full-newton` simplicity); **PD & quasi-Newton prefactor once (1 factorization) vs Newton's 5** (`fast-mass-spring→full-newton`, `quasi-newton→full-newton` speed *mechanism*, HW-independent — wall-clock × not claimed); **PD-style & L-BFGS run on FEM Neo-Hookean** not just mass-spring (`quasi-newton→projective-dynamics` generality). 4 edges → qualified (`projective-dynamics→fast-mass-spring` reverted to self-claimed — our FEM-PD is the fixed-proxy approximation, not exact PD's constraint generality) |

| PPN | progressively-projected-newton — indefinite-element fraction (V2.7) | [`ppn_fraction.md`](ppn_fraction.md) | PPN claims it projects **<10% of elements** vs clamp projecting all. Measured along a Newton solve the indefinite fraction is **iteration-dependent: 54–57% far from the min → 2–5% near convergence (~33% mean)** — the `<10%` holds only near the solution, not through the hard early iterations. `progressively-projected-newton→clamp-filtering` **regime-specific** (mechanism holds; the <10% doesn't on a stretch) → qualified |

| CM ⭐ | **Composite Majorization — faithful impl, closes #14** (V3) | [`composite_majorization.md`](composite_majorization.md) | the real CM (Shtengel 2017): singular values via similarity/anti-similarity == SVD (1e-15), PSD convex-majorizer Hessian (eq. 9), **conformance-gated on the paper's Proposition 3.1 (H ⪰ ∇²f)**, monotone majorize-minimize, **same minimum as projected-Newton**; also symmetric ARAP. Honest finding on the iteration axis: **CM 9 ≈ projected-Newton 8.8** (`cm→projected-newton` NOT reproduced — a majorizer takes conservative steps; the '4× faster' is wall-clock on the shared analytic Hessian) but **CM 9 ≪ AQP 777** (`cm→aqp` reproduces, ~86×); SLIM 5 faster (`cm→slim` needs-scale). 3 edges → qualified |

| BCQN | **Faithful full BCQN** (Zhu-Bridson-Kaufman 2018) vs competitors | [`bcqn.md`](bcqn.md) | reimplemented from paper + authors' reference code (proxy `L=2·cotan` factored once + blend Eq.13 + cured DPJ direction filter + inversion-free/Armijo LS + characteristic-norm stop), conformance-gated (gate 14: β∈[0,1], monotone, ==p-Newton min). Shared energy-tol, 6 scenarios: **BCQN 8.0 iters vs AQP 26.3** (`bcqn→aqp` reproduces), vs its own no-blend Sobolev-L-BFGS 12.7 (blend earns ~1.6×), vs L-BFGS 12.0. But **BCQN 8.0 vs projected-Newton 6.8 / CM 7.0** — MORE iters (`bcqn→{pn,cm}` NOT reproduced on iterations; wall-clock/factor-once story). 3 edges → qualified |

| scale | **3D harness scaling ceiling** | [`scale_3d.md`](scale_3d.md) | sparse analytic-Hessian projected-Newton, `bench/tet_scale.py`. Reaches **131,712 tets / 68,121 free DOF**; Newton iteration count **flat at 4–5 across a 3k→132k sweep** (mesh-independent, as 2nd-order predicts); wall-clock grows only with sparse-factorization fill-in (implementation, not algorithm). Refutes the "2D toy" critique for the filtering headline (§8.1) |

| 3D⬆ | **Off the 2D toy — clamp vs absolute on genuine 3D tets at scale** | [`tet3d_filters.md`](tet3d_filters.md) | scalable analytic-Hessian sparse harness (`bench/tet_scale.py`, gate 13). On a **10,368-tet** P1 mesh, projected-Newton iters climb 5→93 as ν→0.499 (**volumetric locking, in 3D**), and **absolute under-performs clamp near incompressibility (172 vs 93)** — reproducing the 2D §8.1 P1 reversal in genuine 3D. Harness verified to 82,944 tets / 43K DOF. Pending: P2/mixed 3D control |

| PD/N | Projective Dynamics vs full Newton on the shared incremental potential | [`pd_vs_newton.md`](pd_vs_newton.md) | same Φ, same residual, so iterations compare directly. **NOT reproduced on the iteration axis:** PD needs ~5× **more** iters than Newton (PD 16–36 vs Newton 4, to 1e-5) — first-order local/global (linear rate) vs second-order (quadratic). The mechanism behind PD's interactive-speed reputation is on the **factorization** axis: PD prefactors its **constant** system **once** (1 factorization) vs Newton refactorizing every iter (4). `projective-dynamics→full-newton` (speed) → qualified (per-iteration-cost/wall-clock story, same shape as CM→PN: cheaper-per-step ≠ fewer-step) |

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
