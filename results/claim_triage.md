# Claim triage — what the prototype can test, and why the rest it cannot

Every `self-claimed` / `unmeasured` edge of the superiority-claims graph, triaged against the **2D
Python/NumPy prototype** harness (no contact/IPC, no GPU, no large-scale, cannot run un-ported
official code). The point is honesty: a claim we leave unadjudicated is labelled with the *specific
reason* it is out of reach, not silently dropped. This drives the remaining verification work
(`results/*` for the testable) and the paper's "what we cannot yet adjudicate" section.

## TESTABLE-NOW (single-axis experiment on the current harness) — 14 edges

Top-10 by leverage (harness reaches it cleanly on a hardware-independent count):

| # | edge (dimension) | experiment |
|---|---|---|
| 1 | eigenvalue-blending → absolute-filtering & → clamp-filtering (convergence) | add one blending value to the filter slot → 3-way head-to-head in `run_world2_filters` (twist substrate ready); **two edges, one change** |
| 2 | absolute-filtering → clamp-filtering (**robustness**) | success-rate (not iters) sibling of the validated convergence edge; extend `run_stable_nu`/`run_p2_stable_nu` |
| 3 | trust-region-filtering → full-newton (robustness) | E1 already runs `none`(=full-Newton) vs TR; score success at indefinite points |
| 4 | slim → l-bfgs (convergence) | official SLIM + L-BFGS both present; add the L-BFGS arm to `run_slim` |
| 5 | anderson-geometry → aqp (convergence) | both slots present; add AQP arm to `run_anderson` |
| 6 | aqp → local-global (speed) | both in harness; iteration/back-solve head-to-head |
| 7 | slim → projected-newton (robustness) | add a non-uniform-triangulation stratum to `run_slim` |
| 8 | stable-neo-hookean → standard-neo-hookean (robustness) | both energies present; inverted-init success-rate |
| 9 | anderson-geometry → slim (convergence) | wrap official SLIM's fixed point with the map-agnostic Anderson core |
| 10 | aqp → full-newton (speed) | `run_1a_profiles` already has both; read the iteration/wall-clock crossover |

Plus: anderson-geometry → l-bfgs (speed); pitfalls-projection → full-newton (robustness axis only);
and the reachable half of the `bcqn → {aqp,slim,composite-majorization,projected-newton}` targets via
the E3 factorial (direction/line-search/criterion axes; full attribution is entangled, below).

**Honesty note on wall-clock:** the speed edges above are adjudicated on hardware-independent counts
(iterations, factorizations, back-solves) — pure-Python + libigl-C++ timing is diagnostic-only, so a
raw wall-clock number would be hardware-confounded even where the count carries the verdict.

## UNTESTABLE — categorized reasons (why verification or rejection is impossible *here*)

This is not "did not try": each category is a concrete capability the v1 prototype lacks.

| category | count | why it is genuinely out of reach | representative edges |
|---|---:|---|---|
| **needs-unavailable-code** | ~34 | requires the paper's specific implementation, which we cannot faithfully reproduce without it (a look-alike would beg the very question). The PD/ADMM/Chebyshev/XPBD/VBD simulation-accelerator family and several parametrization competitors are not ported. | composite-majorization→{projected-newton,slim,aqp} (#14); the `xpbd/pbd/pbng/vertex-block-descent→*` family; `quasi-newton-liu2017`, `chebyshev-semi-iterative`, `admm-pd`, `aa-admm`, `projective-dynamics→*`; progressive-embedding→{slim,matchmaker} |
| **needs-contact-physics** | 22 | World-3: IPC barriers, continuous collision detection, friction — v1 implements none. An intersection-free or friction claim has no harness to run in. | ipc→prior-rigid-engines; c-ipc→ipc; gipc/stiffgipc/abd/medial-ipc/rigid-ipc/ogc/cubic-barrier-ando/barrier-*→{ipc,gipc} |
| **needs-scale** | 21 | the claim *is* about large meshes / GPU throughput / wall-clock at a scale the dense Python prototype cannot reach (100K–1.5M elements, FPS budgets). | slim→martin-multiscale (>2100×); bcqn→{composite-majorization,projected-newton} (1.5M-tri); progressive-param, simplex-assembly, scaf, lbd, efficient-bijective-param speed edges; jgs2/vertex-block-descent throughput |
| **entangled-needs-source** | 9 | the method bundles ≥2 co-changed components we cannot separate without the paper (the confound the benchmark exists to expose, turned against us). | bcqn→{aqp,slim,composite-majorization} full attribution; abcd→{slim,akvf,bcqn,composite-majorization,projected-newton} (a framework wrapping other solvers) |
| **hardware-confounded** | 4 | a GPU-vs-CPU wall-clock claim that cannot be made hardware-independent; the paper's own iteration-parity concession shows the win is throughput, not algorithm. | descent-gpu→{nonlinear-cg,projective-dynamics}; second-order-stencil-descent→{gradient-descent,full-newton} |
| **subjective-quality** | 3 | a visual/quality claim with no agreed scalar metric ("fleshy appearance", "no spurious degeneracies", "as-low-as-possible distortion"). | stable-neo-hookean→standard-neo-hookean (quality); analytic-eigensystems→numeric (quality); advanced-mips→projected-newton (quality) |
| **baseline-confounded** | 2 | the claim rests on a weak/unnamed/self-ablation baseline, so even a reproduction would not adjudicate the method. | aqp→accelerated-gradient-descent (AGD = AQP-with-proxy-off, a self-ablation); advanced-mips→projected-newton (unnamed SOTA) |
| **needs-3D** | 1 | inherently a 3D + free-boundary injectivity capability; only a 2D untangling analog exists. | foldover-free→tlc (generality) |

_(Some World-2/3 simulation edges satisfy two categories — e.g. GPU-IPC edges are both `needs-scale`
and `needs-unavailable-code`; each is counted under its tightest primary reason. Totals are therefore
approximate where those families overlap.)_

## What this means

The 14 testable edges are the concrete next verification work (`P5.2`). The untestable majority is not
a gap to apologize for but a **map of the boundary of what a contact-free 2D prototype can honestly
say** — and three of the categories (`needs-unavailable-code`, `needs-scale`, `needs-contact-physics`)
are exactly where a *living* benchmark with author-contributed ports and a contact track (v2) would
extend the frontier. The honest statement a survey can make about a claim is sometimes "we cannot yet
adjudicate this, and here is precisely why" — which is itself a contribution.
