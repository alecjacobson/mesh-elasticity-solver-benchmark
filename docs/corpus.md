# Corpus — Mesh-Elasticity Solver Literature (~130 papers)

Built from a 5-way research fan-out. Grouped by the three problem-class "worlds" (see
`design.md`). Per entry: **Title** | authors year venue | primary axis | claim-type |
comparability note | code. `[?]` = detail the researcher could not verify — confirm before
citing. Metric-comparability is the load-bearing field.

Legend — axes: E=energy, S=search-direction, H=Hessian/eigenvalue-filtering, L=line-search/feasibility,
LS=linear-solver/preconditioner, C=convergence-criterion, A=acceleration.

---

## WORLD 1 — Static geometry / distortion optimization (no inertia, no contact)

### Energy models & foundations
- **A simple geometric model for elastic deformations** | Chao, Pinkall, Sanan, Schröder 2010 TOG | E | ARAP-style corotational energy; bridges static energy ↔ time integration. code:[?]
- **As-Rigid-As-Possible Surface Modeling** | Sorkine, Alexa 2007 SGP | E,S | introduces local-global template. code: libigl
- **A Local/Global Approach to Mesh Parameterization** | Liu et al. 2008 SGP/CGF | S,LS | ARAP/ASAP, prefactored global. code: reimplemented
- **Bounded distortion mapping spaces** | Lipman 2012 TOG | E,LS | convex bounded-distortion set. code:[?]

### Injectivity / feasibility cluster (metric = success-rate from bad init)
- **Injective and bounded distortion mappings in 3D** | Aigerman, Lipman 2013 TOG | L,E | closed-form BD projection 3D. code: yes
- **Locally Injective Parametrization w/ Arbitrary Fixed Boundaries** | Schüller et al. 2014 TOG | E,L | flip barrier. code: libigl
- **Computing Locally Injective Mappings by Advanced MIPS** | Fu, Liu, Guo 2015 TOG | E,S | advanced MIPS + block CD. code:[?]
- **Bijective Parameterization with Free Boundaries** | Smith, Schaefer 2015 TOG | E,L | dual barriers. code: reimplemented
- **Large-Scale Bounded Distortion Mappings** | Kovalsky et al. 2015 TOG | LS,L | ~100× BD projection. code: yes
- **Inversion-Free Mappings by Simplex Assembly** | Fu, Liu 2016 TOG | L,S | per-simplex projection + reassembly. code: page
- **Simplicial Complex Augmentation (Scaffold)** | Jiang, Schaefer, Panozzo 2017 TOG | E,L | scaffold → global bijectivity from local. code: yes
- **Progressive Embedding** | Shen et al. 2019 TOG | L | float-robust Tutte-like. code: yes
- **Lifting Simplices to Find Injectivity (TLC)** | Du et al. 2020 TOG | E,L | smooth energy, 100% success on 11,647-mesh benchmark. code+data: yes ★
- **Foldover-free maps in 50 lines** | Garanzha et al. 2021 TOG | E,L | compact untangling. code: yes
- **Practical lowest distortion mapping** | Garanzha et al. 2022 arXiv | E,C | polyconvex hyperelastic. code:[?]
- **Progressive Parameterizations** | Liu et al. 2018 TOG | E,L | progressive reference. code: page
- **OptCuts** | Li et al. 2018 TOG | E,+topology | joint cut+param. code: yes
- **Efficient Bijective Parameterizations** | Su et al. 2020 TOG | S,LS | 2nd-order, ~6×. code:[?]

### Solver / accelerator cluster (metric = wall-clock-to-tolerance + distortion) — THE clean World-1 ablation
- **Accelerated Quadratic Proxy (AQP)** | Kovalsky, Galun, Lipman 2016 TOG | S,A,LS | Laplacian proxy + Nesterov; single-axis. code: yes ★
- **Isometry-Aware Preconditioning (AKVF)** | Claici et al. 2017 CGF/SGP | S,LS | Killing-operator preconditioner; single-axis. code: yes
- **Scalable Locally Injective Mappings (SLIM)** | Rabinovich et al. 2017 TOG | S | reweighted-ARAP proxy; standard baseline. code: yes ★
- **Geometric Optimization via Composite Majorization** | Shtengel et al. 2017 TOG | H,E | convex majorizer; ENTANGLED (surrogate+energy), 2D. code: yes
- **Blended Cured Quasi-Newton (BCQN)** | Zhu, Bridson, Kaufman 2018 TOG | L,A,C | ENTANGLED — changes line-search+proxy+criterion at once; characteristic-gradient-norm criterion worth adopting. code: yes ★
- **Anderson Acceleration for Geometry Opt & Physics** | Peng et al. 2018 TOG | A | Anderson on local-global fixed-point; single-axis; spans World 1&2. code: yes (AASolver)
- **Adaptive Block Coordinate Descent (ABCD)** | Naitsat et al. 2020 CGF/SGP | E,S,C | ENTANGLED (repair energy+block CD+criterion); handles inverted init. code: yes
- **A Splitting Scheme for Flip-Free Distortion Energies** | Stein, Li, Solomon 2021 SIIMS | S,LS | ADMM; single-axis direction. code: yes
- **Geometric Optimisation via Spectral Shifting (GOSS)** | Poya et al. 2023 TOG | E,H,L | ENTANGLED (spectral-shift energy + analytic Hessian); recovers from folded init. code: page

### Tooling
- **TinyAD** | Schmidt et al. 2022 CGF/SGP | H | compile-time AD → sparse grad/Hessian; enables projected Newton generally. code: yes ★

### Datasets/benchmarks
- **Locally-Injective-Mappings Benchmark** | Du et al. 2020 | 11,647 tri+tet, de-facto success-rate set. data: yes ★
- **A Dataset and Benchmark for Mesh Parameterization** | Shay, Solomon, Stein 2022 arXiv | standardized protocol. data: yes ★

---

## WORLD 2 — Quasistatic/dynamic hyperelasticity (inertia / incremental potential, NO contact)

### Eigenvalue-filtering axis — THE v1 core cohort (one projected-Newton skeleton, swap the filter)
- **Robust Quasistatic FE & Flesh Simulation** | Teran, Sifakis, Irving, Fedkiw 2005 SCA | H,LS | ORIGIN of eigenvalue clamping. baseline every later paper cites.
- **Invertible Finite Elements** | Irving, Teran, Fedkiw 2004 SCA | E | diagonalized-F inversion handling (upstream of analytic eigensystems). code: SOFA
- **Analytic Eigensystems for Isotropic Distortion Energies** | Smith, de Goes, Kim 2019 TOG | H | analytic per-element clamp (no numeric eig). supplies the filtering machinery. code: reimplemented ★
- **Stable Neo-Hookean Flesh Simulation** | Smith, de Goes, Kim 2018 TOG | E,H | inversion/rotation-robust energy + analytic clamp. THE canonical fixed energy. code:[?] ★
- **Anisotropic Elasticity for Inversion-Safety** | Kim, de Goes, Iben 2019 TOG | E,H | closed-form aniso eigensystems.
- **A FE Formulation of Baraff-Witkin Cloth** | Kim 2020 CGF/SCA | E,H | analytic aniso clamp (thin-shell).
- **An Eigenanalysis of Angle-Based Deformation Energies** | Wu, Kim 2023 PACMCGIT | E,H | analytic clamp for bending/strands.
- **Spatial Eigenanalysis of 2D Deformation Energies** | Wu, Wu, Kim 2025 CGF/SGP | H | spatial-domain per-element eigensystems.
- **Stabler Neo-Hookean: Absolute Eigenvalue Filtering** | Chen, Liu, Levin, Zheng, Jacobson 2024 SIGGRAPH | H | λ⁺=|λ|; one-line change on fixed skeleton. code: yes (abs-psd) ★ SEED
- **Trust-Region Eigenvalue Filtering** | Chen, Liu, Jacobson, Levin, Zheng 2024 SA | H,S | UNIFIES Newton/clamp/absolute as one adaptive rule — the "switchboard". code: yes (trust-region-newton)[?] ★ SEED
- **Pitfalls of Projection** | Longva, Löschner, Fernández-Fernández et al. 2023/24 arXiv[?] | H,C,L | shows unconditional PSD projection slows convergence; proposes project-on-demand + kinetic Newton; controlled ablation on both hyperelastic & contact. code:[?] ★ SEED
- **Progressively Projected Newton (PPN)** | Fernández-Fernández, Löschner, Bender 2025 CGF | H,S | selective subset projection (<10% of PN); controlled head-to-head vs PN/PDN incl. contact. code:[?] ★
- **Eigenvalue Blending for Projected Newton** | Cheng, Liu, Fu 2025 CGF | H | blend clamped+absolute via descent constraint. code:[?]
- **Isotropic ARAP Energy Using Cauchy-Green Invariants** | Kim et al. 2022 TOG/SA | E,H | simpler analytic eigensystem. code:[?]

### Reformulation / accelerator / alt-descent (metric = residual-to-tolerance OR fixed-budget)
- **Fast Simulation of Mass-Spring Systems** | Liu et al. 2013 TOG | S,LS | local-global, prefactored; ENERGY-RESTRICTED (mass-spring) — speedup partly artifact. code: 3rd-party
- **Projective Dynamics** | Bouaziz et al. 2014 TOG | E,S,LS | constant prefactored solve; ENERGY-RESTRICTED (quadratic constraint form). code: yes (ShapeOp)
- **Quasi-Newton for Real-Time Hyperelastics** | Liu, Bouaziz, Kavan 2017 TOG | S,E | PD=quasi-Newton; L-BFGS; ~10× vs Newton; GENERAL energy — the key bridge. code: yes
- **Descent Methods for Elastic Body Sim on GPU** | Wang, Yang 2016 TOG | S,A,LS | Jacobi+Chebyshev, dot-product-free; GENERAL. code:[?]
- **A Chebyshev Semi-Iterative Approach** | Wang 2015 TOG | A | Chebyshev on PD/PBD; ≥10×. code:[?]
- **ADMM ⊇ Projective Dynamics** | Narain, Overby, Brown 2016 SCA (TVCG 2017) | S,E,LS | ADMM generalizes PD to general constitutive; residual metric. code: yes (admm-elastic)
- **Accelerating ADMM (Anderson)** | Zhang et al. 2019 TOG | A | Anderson on ADMM fixed-point. code: yes (AA-ADMM)
- **Vertex Block Descent (VBD)** | Chen, Liu, Yang, Yuksel 2024 TOG | S,C | vertex Gauss-Seidel; unconditional stability; GENERAL; fixed-budget claims. code: yes[?]
- **Second-Order Stencil Descent** | Lan et al. 2023 TOG | S,H | per-stencil 2nd-order; up to 100× vs CPU; also World 3. code:[?]
- **JGS2: Near-2nd-order Jacobi/GS for GPU Elastodynamics** | Lan et al. 2025 TOG | S,A | near-quadratic parallel iteration; 50–100× vs prior GPU. code:[?]
- **Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity** | Chen, Han, Teran et al. 2024 TOG | S,A,C | genuinely convergent PB-GS; GENERAL. code:[?]
- **Primal Extended Position Based Dynamics** | Chen, Han, ... Fedkiw, Teran 2023 MIG | C,S | fixes XPBD stagnation on backward-Euler residual. code:[?]
- **Position Based Dynamics** | Müller et al. 2007 | S,C | Gauss-Seidel constraint projection; NO convergence (fixed budget) — foundational caveat. code: yes
- **XPBD** | Macklin, Müller, Chentanez 2016 MIG | E,C | compliance fixes PBD stiffness; still constraint-projection. code: reimplemented
- **A Survey on Position Based Dynamics** | Bender, Müller, Macklin 2017 EG | (survey) | connects PBD/XPBD to variational implicit Euler.

### Linear-solver / multigrid / subspace (accelerate the INNER solve — complementary, not head-to-head)
- **Smoothed Aggregation Multigrid for Cloth** | Tamstorf, Jones, McCormick 2015 TOG | LS | algebraic MG. code:[?]
- **Scalable Galerkin Multigrid** | Xian, Tong, Liu 2019 TOG | LS | skinning-space prolongation. code:[?]
- **GPU Multilevel Additive Schwarz Preconditioner** | Wu et al. 2022 TOG | LS | ~4× PCG. code:[?]
- **Hyper-Reduced Projective Dynamics** | Brandt, Eisemann, Hildebrandt 2018 TOG | subspace,E | model reduction + hyper-reduction. code:[?]
- **Anderson Acceleration for Geometry Opt & Physics** — see World 1 (spans both).

---

## WORLD 3 — Contact-coupled dynamics (barriers, CCD, friction). Metric = non-penetration guarantee + timing + robustness

### IPC family (barrier held as a shared constant → Option-B fairness possible)
- **Incremental Potential Contact (IPC)** | Li et al. 2020 TOG | barrier,L,E | intersection/inversion-free any-material; the contact substrate. code: yes ★
- **Codimensional IPC (C-IPC)** | Li, Kaufman, Jiang 2021 TOG | barrier,E | codim-0/1/2/3 + strain limiting. code: yes
- **Intersection-free Rigid Body Dynamics (Rigid-IPC)** | Ferguson et al. 2021 TOG | E,L | curved-trajectory CCD. code: yes
- **Medial IPC** | Lan et al. 2021 TOG | E,LS | medial reduced elastics + IPC. code:[?]
- **Affine Body Dynamics (ABD)** | Lan et al. 2022 TOG | E | affine reduced DOFs; up to ~10,000× GPU. code:[?]
- **Convergent IPC** | Li et al. 2023 arXiv[?] | barrier,C | continuous formulation converges under joint mesh/Δt/d̂ refinement — evidence d̂ is coupled to accuracy.

### GPU-scaling (SAME barrier fixed, vary inner solve → the fair contact sub-track)
- **GIPC: Gauss-Newton IPC Barrier** | Huang, Yang et al. 2024 TOG | H,LS,S | analytic-eigensystem GN + PCG, GPU. code: yes ★
- **StiffGIPC** | Huang et al. 2024/25 TOG | LS | connectivity-aware MAS preconditioner. code:[?]
- **Barrier-Augmented Lagrangian for GPU Elastodynamic Contact** | Guo et al. 2024 TOG | barrier,LS,C | aug-Lagrangian improves conditioning; inexact Newton-PCG. code:[?]
- **Second-Order Stencil Descent** — see World 2 (interior-point contact variant).
- **Efficient GPU Cloth w/ Non-distance Barriers & Subspace Reuse** | Lan et al. 2024 TOG | barrier,LS | non-distance barrier + subspace reuse. code:[?]
- **Penetration-Free Projective Dynamics on the GPU** | Lan et al. 2022 TOG | E,LS | IPC-in-PD; batched contact-set/CCD. code:[?]
- **Subspace-Preconditioned GPU PD with Contact** | Li et al. 2023 SA | LS | subspace preconditioner. code:[?]

### Contact-MODEL swaps (NOT metric-comparable — capability demos only)
- **A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness** | Ando 2024 TOG | barrier,E | cubic non-log barrier; tight gaps. code:[?]
- **Robust Penetration-Free Elastodynamics without Barriers** | Zheng, Luo, Li 2025 TOG | barrier-free,L,LS | aug-Lagrangian; avoids TOI locking. code: yes[?]
- **Offset Geometric Contact (OGC)** | Chen et al. 2025 TOG | barrier,L | offset geometry, displacement bounds replace CCD; >100×. code: yes[?]
- **Fast But Accurate: Real-Time Hyperelastic w/ Robust Frictional Contact** | Zeng et al. 2025 TOG | barrier,LS | NCP complementarity instead of log-barrier. code:[?]
- **JGS2** — see World 2 (contact guarantee unconfirmed[?]).

### Friction models (reused across the family)
- IPC lagged semi-implicit friction (smoothed static↔kinetic via ε_v) — defined in IPC 2020, reused by C-IPC/Rigid-IPC/ABD/GIPC.
- **Primal-Dual Non-Smooth Friction for Rigid Body Animation** | 2024 SIGGRAPH | interior-point primal-dual log-barrier friction. 
- **Augmented IPC for Sticky Interactions** | Fang, Li et al. 2023 TVCG | adhesion.

---

## Reference syntheses / courses / datasets (cross-world)
- **Dynamic Deformables: Implementation and Production Practicalities** | Kim, Eberle 2020/22 SIGGRAPH course | canonical projected-Newton + clamp skeleton. code: yes (HOBAK) ★
- **Thingi10K** | Zhou, Jacobson 2016 | adversarial real meshes (non-manifold/noisy). data: yes ★
- **SimJEB** | Whalen et al. 2021 | FEM sim dataset w/ ground-truth solutions. data: yes

★ = highest-value node (seed paper, standard baseline, reusable dataset, or the switchboard/skeleton).

---

## WORLD 0 — External / general (non-SIGGRAPH): baselines, ancestors, exclusions

The substrate the graphics methods sit on. Verdicts: **BASE** = include as honesty-baseline
(race it), **REL** = include-as-related (named ancestor of a graphics method; cite, don't
necessarily race), **EXCL** = exclude-with-justification. Unifying view: every method is
metric descent `x' = x − α M⁻¹∇E`; the axis it touches is which `M` (or globalization) it sets.

### Classical numerical-optimization canon (refs: Nocedal-Wright; Boyd-Vandenberghe; Gill-Murray-Wright; Saad; Kelley)
- **Gradient descent / steepest descent** | Cauchy 1847 | BASE — the floor; exposes energy ill-conditioning.
- **Nonlinear CG (Fletcher-Reeves, Polak-Ribière+)** | 1964/69 | BASE — matrix-free first-order; strong-Wolfe.
- **Newton + backtracking** | N&W §3.3 | BASE — = graphics "Projected Newton" once Hessian filtered.
- **L-BFGS** | Liu-Nocedal 1989 | BASE ★ — the key large-scale generic baseline; ancestor of PD-as-quasi-Newton (Liu 2017).
- **BFGS** | 1970 | BASE (small/medium). **SR1** | REL (trust-region companion). **DFP** | EXCL — dominated by BFGS.
- **Gauss-Newton** | REL — the structure behind ARAP local-global (constant-Laplacian normal equations); BASE only for least-squares-form energies (ARAP, sym-Dirichlet-like).
- **Levenberg-Marquardt** | 1944/63 | REL — `H+λI` is the provenance of identity-shift Hessian regularization.
- **Newton-Krylov / inexact Newton** | Dembo-Eisenstat-Steihaug 1982; Kelley | BASE — matrix-free 2nd-order; underlies GPU solvers.
- **Modified Cholesky (Gill-Murray 1974; Schnabel-Eskow 1990)** | REL ★ — DIRECT ANCESTOR of eigenvalue clamping/PSD projection.
- **Hessian eigenvalue modification (spectral clamp/flip)** | N&W §3.4 | REL ★ — textbook provenance of graphics per-element clamping (graphics' contribution = per-element locality).
- **Identity-shift / Levenberg damping** | N&W Alg 3.3 | BASE + REL — cheapest filter to race clamping against (isotropic vs spectral-selective).
- **Trust-region: Steihaug-CG** | Steihaug 1983 | BASE — handles indefinite H intrinsically; natural alt to line-search+clamping. **Moré-Sorensen** REL (too costly at scale). **Dogleg** REL.
- **Line search: Armijo / Wolfe / strong-Wolfe** | 1966/69 | BASE (mandatory axis, standardize it); IPC's CCD-truncated line search is the domain-specialized version.
- **Nesterov accel. / heavy-ball momentum** | Polyak 1964; Nesterov 1983 | REL ★ — DIRECT ANCESTOR of AQP; also BASE (acceleration-axis control, with restart).
- **Anderson / Aitken acceleration** | Anderson 1965; Walker-Ni 2011 | REL ★ — ANCESTOR of Anderson-for-geometry (Peng 2018); = multisecant quasi-Newton.
- **Interior-point / barrier** | Fiacco-McCormick 1968; Nesterov-Nemirovskii | REL ★ — ANCESTOR of IPC barriers. EXCL from unconstrained race.
- **Proximal / operator-splitting / ADMM** | Douglas-Rachford; Boyd 2011 | REL — ANCESTOR of Projective Dynamics (ADMM ⊇ PD).
- **Semismooth Newton** | Qi-Sun 1993 | EXCL (smooth unconstrained); REL only for friction/complementarity.
- **PCG / MINRES / GMRES** | Hestenes-Stiefel 1952; Paige-Saunders 1975; Saad-Schultz 1986 | BASE — inner-solver axis (CG needs SPD-filtered H; MINRES handles raw indefinite).
- **Nonlinear/algebraic multigrid & FAS** | Brandt 1977 | REL + optional-BASE — ancestor of graphics multigrid elasticity; FAS on nonconvex energy is delicate.

### ML optimizers (deterministic full-batch regime is decisive)
- **SGD (= full-batch GD)** | Robbins-Monro 1951 | BASE — reference floor.
- **Momentum / heavy-ball** | Polyak 1964 | BASE. **Nesterov** | BASE (see above).
- **Adam** | Kingma-Ba 2015 | BASE ★ — THE headline honesty control. Full-batch Adam ≈ momentum-sign descent (Balles-Hennig 2018; Kunstner 2023; Liu 2024): normalizes gradient magnitude, ignores curvature → expected to plateau/limit-cycle and lose to L-BFGS/Newton on high-accuracy elastic min. That loss is the informative result.
- **RMSProp** | Tieleman-Hinton 2012 | BASE (adaptive sanity control; ≈ sign descent deterministically).
- **AdamW** | EXCL — decoupled weight decay biases positions toward origin (physically wrong); ≡ Adam at λ=0.
- **Adagrad** | Duchi 2011 | EXCL — monotone `√G` decay stalls deterministically (negative control only).
- **AMSGrad** | EXCL — its fix is a stochastic/online phenomenon; ≈ Adam deterministically.
- **Lion** | Chen 2023 | EXCL (or optional sign-descent control) — explicit fixed-ℓ∞ sign momentum + origin-biasing weight decay.
- **Hessian-free / Newton-CG** | Martens 2010 | BASE — = classical Newton-CG (cleaner here; exact Hv products).
- **Gauss-Newton / GGN** | Schraudolph 2002 | BASE (as GN / projective-dynamics-style). Fisher interpretation is ML-specific, does NOT transfer.
- **K-FAC** | Martens-Grosse 2015 | EXCL — intrinsic layer/Kronecker structure; no mesh analog.
- **Shampoo** | Gupta 2018 | EXCL — per-tensor-mode structure degenerates on flat mesh-DOF vector; dominated by exact Hessian.
- **Sophia** | Liu 2023 | EXCL — stochastic + probabilistic-loss design; diagonal Hessian dominated by available full Hessian.

### Natural gradient & PINN optimizer line
- **Natural Gradient Descent** | Amari 1998 | REL ★ — parent abstraction (`M=Fisher`). NOT runnable here: no probabilistic output model ⇒ Fisher undefined. Unifying refs: Martens 2020; Neuberger, *Sobolev Gradients and Differential Equations* (LNM 1670).
- **Energy Natural Gradient Descent (ENGD)** | Müller-Zeinhofer 2023 ICML | REL ★ — metric = energy 2nd-derivative (= generalized Gauss-Newton). With positions-as-unknowns the pullback Jacobian is identity ⇒ **collapses to Newton/Gauss-Newton on the elastic energy** (the existing graphics baseline).
- **Gauss-Newton NGD for PINNs** | Jnini-Vella-Zeinhofer 2024 | REL — confirms function-space NGD ≡ Gauss-Newton; matrix-free trick transferable.
- **Near-optimal Sketchy Natural Gradients for PINNs** | Best Mckay, Kaur, Greif, Wetton, ICML 2025 | REL ★ (user-named) — randomized low-rank sketch of the change-of-coordinate Gram matrix (proven rapidly-decaying eigenvalues). NOT a mesh-elasticity solver (sketches the *parameter-space* Gram `JᵀGJ` that exists only via the network); transfers only as an IDEA — sketch the elastic Gauss-Newton/Laplacian metric. [sketch primitive unstated in public abstract]

### Graphics Sobolev/proxy methods, re-cast as natural-gradient-under-a-metric (cross-ref World 1)
- **AQP** metric `M` = mesh Laplacian `L⊗I` (constant) + Nesterov | **AKVF** `M` = Killing operator `K(x)` (reassembled; explicit Riemannian/H¹ framing) | **BCQN** `M` = blend of Laplacian (Sobolev gradient, `D₀=L⁻¹`) + L-BFGS secant | **SLIM** `M` = reweighted (weighted-Laplacian) IRLS proxy. All four = REL to Amari NGD; AKVF is the crispest literal natural-gradient instance. Broader statement: Yu-Schumacher-Crane "Repulsive Curves/Surfaces" (Sobolev-like energy-matched inner product).

### Computational-mechanics FEM (engineering solves the SAME problem)
- **Newton-Raphson + incremental load stepping** | Crisfield 1991; Abaqus/Standard | BASE ★ — canonical mechanics baseline; DECIDE whether stepping is permitted (changes robustness numbers).
- **Arc-length / continuation (Riks 1979; Crisfield 1981; Wempner)** | BASE (for snap-through/buckling) / else ACKNOWLEDGE — divergence at a limit point is a wrong-parametrization artifact, NOT a solver defect.
- **Crisfield line search** | 1991 | BASE — shared globalization; standardize it.
- **Dynamic relaxation** | Day 1965; Barnes; Underwood | REL — ANCESTOR of position-based/projective dynamics.
- **Newton-Krylov / JFNK** | Knoll-Keyes 2004; PETSc `-snes_mf` | BASE — matrix-free baseline.
- **Nonlinear multigrid (FAS) / Newton-MG** | Brandt; Trottenberg | BASE (scalability) + REL — FAS on nonconvex hyperelastic is delicate.
- **Modified/damped/trust-region Newton; Abaqus artificial viscous stabilization** | Deuflhard; PETSc SNESNEWTONTR | REL ★ — DIRECT engineering antecedent of eigenvalue/definiteness filtering (same disease, different cure — race projected-Newton vs trust-region-Newton to test if filtering is genuinely better).
- **Contact: penalty / augmented-Lagrangian / mortar / active-set** | Wriggers; Popp-Gee-Wall; Hüeber-Wohlmuth | REL ★ — ANCESTOR of IPC; BASE if contact in scope (v2).
- **Near-incompressible: F-bar / mixed u-p / Simo three-field (deal.II step-44); EAS; selective reduced integration** | REL ★ — CRITICAL: near-incompressibility is a *discretization* problem; eigenvalue-filtering claims live exactly here. Displacement-only tets confound "solver robustness at high Poisson" with volumetric LOCKING.
- **Libraries — harness vs oracle:** PETSc SNES/TAO = best shared HARNESS (SNES Newton-LS/TR+Krylov+PC; TAO optimization view matches energy-min) + oracle. FEniCS/FEniCSx, deal.II (step-44), MFEM (ex10/ex19), Trilinos NOX/LOCA (arc-length) = independent ORACLEs. Abaqus/ANSYS = closed (oracle/convention source only, EXCL as harness).

### Learned / neural (mostly EXCL from core; a v2 companion track exists)
Core-inclusion rule: unknowns = mesh vertex positions, minimize the fixed discrete energy to a per-instance minimizer at controllable tolerance.
- **PINNs for elasticity** | Abueidda 2021; E.Zhang 2022 | EXCL — unknowns are network weights, meshfree collocation, approximate (no exact convergence).
- **Deep Energy Method (DEM)** | Nguyen-Thanh 2020 | EXCL (borderline; mention) — same *energy* objective but weights-not-positions, meshfree quadrature.
- **MeshGraphNets / GNS** | Pfaff 2021; Sanchez-Gonzalez 2020 | EXCL — amortized dynamics surrogate; offline training; error drift.
- **DeepONet / FNO operators** | Lu 2021; Li 2021 | EXCL — amortized operator across an instance family.
- **DiffPD** | Du 2021 | EXCL as contribution (differentiability for outer-loop learning ≠ per-instance convergence); its inner PD solve is a classical baseline on its own.
- **Neural ROM: CROM / LiCROM / Neural Modes / Data-Free Kinematics** | Chen 2023; Wang 2024; Sharp-Romero-Jacobson 2023 | EXCL as core — solves a *reduced-subspace* min ≠ full-space minimizer (even the data-free ones). REL: neural subspace plugs into reduced-coord Newton IF full-space residual is reported.
- **Learned preconditioners (NeuralPCG, Deep Conjugate Direction)** | Li 2023 | REL — drop-in for the preconditioner, exact-convergent by construction (mostly demoed on linear SPD; nonlinear-elastic use unproven).
- **Learned warm-starts (NOWS; Spectrally-Safe Newton warm-starts)** | 2026 | REL ★ — most benchmark-compatible neural contribution: changes only x₀, same convergence axis. → **v2 companion track: "learned accelerators of classical solves"** (warm-starts + preconditioners + neural subspaces), optional, same metrics, orthogonal to core.
- **Sketchy-NGD for PINNs** | Best Mckay 2025 | evidence that the PINN "solve" is a distinct problem class (justifies excluding PINNs) — cited, not benchmarked.

★ (World 0) = load-bearing ancestor, headline baseline, harness/oracle, or critical confound-control.
