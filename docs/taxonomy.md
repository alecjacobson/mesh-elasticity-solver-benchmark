# Taxonomy

A taxonomy is only useful if it is *non-arbitrary*: its dimensions should derive from one
governing purpose, be orthogonal, and license which comparisons are meaningful. We follow the
Nickerson et al. (2013) construction (pick a meta-characteristic; iterate empirical↔conceptual
to explicit ending conditions) and evaluate with the Unterkalmsteiner–Adbeen (2023) quality
attributes (orthogonality, mutual-exclusiveness, conciseness, extensibility).

> Status: framework + classification of the load-bearing methods. Full per-entry classification
> of the ~180 corpus entries is ongoing (issue #1). Extend via `docs/corpus.md` tags.

## Meta-characteristic

**"What does this method change to improve elastic-energy minimization, and under what problem
class is the claim measured?"**

Everything below derives from this. It has two faces — a *method* face (which component does
it swap?) and a *problem* face (which world / capability?). Comparability is governed by the
problem face; contribution is located on the method face.

## Face A — method axes (which component is swapped)

Every method is metric descent `x' = x − α · M⁻¹ ∇E(x)` plus a globalization. The axes:

| axis | symbol | values (non-exhaustive) |
|---|---|---|
| Energy `ψ` | **E** | ARAP, symmetric Dirichlet/MIPS, (Stable) Neo-Hookean, corotated, StVK, barrier-augmented |
| Search direction | **S** | full Newton, Gauss-Newton, projected Newton, L-BFGS, AQP proxy, AKVF/Sobolev, local-global/PD, ADMM, coordinate descent, Anderson |
| Hessian filter / SPD projection | **H** | none, identity-shift (Levenberg), clamp-to-ε, absolute, trust-region, project-on-demand, progressive, analytic eigensystem, spectral shift, blending |
| Line search / feasibility | **L** | backtracking, Armijo/Wolfe, exact, injectivity-barrier-aware, CCD-filtered |
| Linear solver / preconditioner | **LS** | direct Cholesky, PCG, MINRES, multigrid, additive Schwarz, subspace, learned |
| Convergence criterion | **C** | Newton decrement, characteristic gradient norm, backward-Euler residual, fixed budget |

These are the metric `M` (E via ψ; S/H/LS choose and modify `M`; L/C govern the step and stop).

## Face B — problem class × capability cells (which comparisons are licensed)

| dim | values |
|---|---|
| **World** (problem class) | 1 static distortion · 2 quasistatic/dynamic hyperelastic · 3 contact-coupled |
| Inertia | none (static) · incremental-potential (implicit) |
| Contact | none · barrier/IPC · model-swap |
| Energy generality | restricted (mass-spring/PD-form) · general hyperelastic · distortion |
| Codimension | volumetric (codim-0) · shells/rods (codim-1/2) |
| Feasibility goal | converge-to-tolerance · injectivity/success-from-bad-init · non-penetration guarantee |

**Fairness gate:** two methods earn a head-to-head *number* only when they occupy the same
cell on the load-bearing dims (World, Contact, Energy-generality, Feasibility-goal). Across
cells we report *coverage / robustness* via performance profiles, never a single speed number.
This is what keeps a broad survey fair (see `design.md` §3–4).

## Classification of load-bearing methods

Primary axis = the axis the paper *claims* as its contribution. **⚠ = entangled** (moves >1
axis; attribution unclear — a decomposition-experiment target, issue #7).

### World 1 — static distortion / parametrization
| method | primary axis | feasibility goal | notes |
|---|---|---|---|
| AQP | S + A (Nesterov) + LS (Laplacian) | converge | single-axis (proxy) |
| AKVF | S/LS (Killing preconditioner) | converge | per-iter linear solve |
| SLIM | S (reweighted proxy) | converge | scalability claim |
| Composite Majorization | H (convex majorizer) + E | converge | ⚠ surrogate + energy; 2D |
| BCQN | L + S + C | converge | ⚠⚠ line-search + proxy + criterion at once |
| Anderson-for-geometry | A (Anderson) | converge | single-axis (on fixed local-global) |
| Splitting (Stein 2021) | S (ADMM) | converge | single-axis (direction) |
| TLC | E (lifted content) | injectivity | released benchmark |
| Progressive Embedding | L (topological robustness) | injectivity | feasibility tool |
| Foldover-free (Garanzha) | E (regularized barrier) | injectivity | |
| GOSS | E (spectral shift) + H | injectivity + converge | ⚠ energy + analytic Hessian |
| Advanced MIPS | E + S | injectivity | |

### World 2 — quasistatic / dynamic hyperelasticity
| method | primary axis | energy generality | notes |
|---|---|---|---|
| clamp-filtering (Teran'05 / analytic'19) | H (clamp) | general | the baseline filter |
| Stable Neo-Hookean | E + H (analytic clamp) | general | canonical fixed energy |
| Absolute filtering | H (absolute) | general | one-line vs clamp |
| Trust-Region filtering | H (adaptive) + S | general | switchboard (Newton/clamp/abs) |
| Pitfalls of Projection | H (project-on-demand/kinetic) + C + L | general | ⚠ study; changes filter + criterion + line-search |
| Progressively Projected Newton | H (selective) + S | general | regime-dependent |
| Eigenvalue Blending | H (clamp⊕absolute) | general | |
| Projective Dynamics | E + S + LS | restricted (PD-form) | speedup partly artifact of restricted energy |
| Quasi-Newton (Liu 2017) | S (L-BFGS on PD) + E | general | bridge: PD = quasi-Newton |
| ADMM ⊇ PD | S (ADMM) + E | general | superset of PD |
| Vertex Block Descent | S (coordinate descent) + C | general | fixed-budget FPS claims |
| JGS2 / SOSD | S + LS | general | near-2nd-order parallel |
| XPBD / Primal-XPBD | E (compliance) + C | restricted | non-convergent (fixed budget) → Primal fixes |

### World 3 — contact-coupled dynamics
| method | primary axis | capability | notes |
|---|---|---|---|
| IPC | barrier/E + L (CCD) | non-penetration guarantee | the substrate |
| C-IPC / Rigid-IPC / Medial-IPC | E (codim/reduced) | + codim / rigid | barrier held ~fixed |
| ABD | E (affine reduced) | stiff/near-rigid | |
| GIPC / StiffGIPC / Barrier-Aug-Lagrangian | LS/H + S | GPU scalability | fixed barrier → fair inner-solve comparison |
| OGC / cubic-barrier / barrier-free / Fast-But-Accurate | barrier-model swap | non-penetration guarantee | capability demo, NOT shared-metric convergence |

### World 0 — external substrate (baselines & ancestors)
See `docs/corpus.md` World-0 and `design.md` §12. Classical/ML optimizers occupy the S/H/L/LS/A
axes directly; they are the honesty-baselines and the named ancestors (lineage map).

## Evaluation of the taxonomy

- **Orthogonality.** Face-A axes are independent components of the descent iteration (you can
  swap the filter without changing the energy, the line search without changing the direction,
  etc.). Face-B dims are independent problem attributes. The two faces are themselves
  orthogonal (a filter innovation can be tested in World 1 or 2).
- **Mutual-exclusiveness within a dimension.** Each axis takes exactly one value per method
  *per component*; where a method takes several (⚠), that is *recorded as entanglement*, not
  smeared across buckets — the taxonomy makes entanglement legible rather than hiding it.
- **Conciseness / completeness.** 6 method axes + 6 problem dims classify every corpus entry
  (Worlds 0–3) without a residual "other" bucket; ending condition (Nickerson): no new axis was
  needed to place the last ~40 entries added in the external-scope fan-out.
- **Extensibility.** New methods slot in by tagging axes + cell; v2 contact adds no new axis
  (Contact dim already present). Learned methods are handled by the same axes plus the scope
  ledger (`design.md` §12.3).
- **Explanatory power.** The taxonomy *predicts* comparability failures: any two methods in
  different Face-B cells whose papers report a single head-to-head speed number are flagged as
  a likely unfair comparison — which is exactly the pattern the benchmark exists to correct.

## The two survey products this backbone feeds

1. **Unifying view** — metric descent `x' = x − α M⁻¹∇E`; the axes are how each method picks/
   modifies `M`. (`design.md` §12.1)
2. **Lineage map** — Face-A innovations traced to named classical ancestors (eigenvalue filter ←
   modified Cholesky; AQP ← Nesterov; PD ← ADMM/GN; IPC ← interior-point; …). (`design.md` §12.2)
