# Measured results (P1 prototype)

Real, reproducible numbers from the [`bench/`](../bench) prototype harness — every figure
produced by `python -m bench.<script>`, gated on `bench/conformance.py` (analytic derivatives
vs finite differences, ~1e-9). These are **small controlled slices** (2D, dense solve, few
filters, no locking-free element yet), not the finished benchmark — but they exercise the full
loop and already reproduce several of the design's predicted effects. Caveats are in each file.

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
```

## Honest limitations (→ next P1 steps)

Dense solve (small meshes); 2D only; filters = {none, clamp, absolute, project-on-demand,
identity-shift, global-pdn} + a trust-region (Steihaug-CG) solver (analytic-eigensystem, eigenvalue-blending not yet);
**no locking-free element** (so the ν-claim is not settled —
only shown to be locking-confounded); single scenario/seed for most experiments; no official-code
regression yet (grounding is the FD conformance test). Sparse solve, a Taylor–Hood/MINI
locking-free element, more filters, and porting an official reference (TinyAD/libigl) are the
next steps (`docs/experiments.md`, `docs/protocol.md`).
