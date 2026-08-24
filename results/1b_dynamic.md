# World-2 dynamic (1b) - implicit-Euler incremental potential (measured)

Hanging Neo-Hookean sheet (ν=0.45, top edge pinned) under gravity, implicit Euler, dt=0.04. Each step minimizes the incremental potential `1/(2dt²)(x−x̃)ᵀM(x−x̃) + E(x)`. Newton iterations per step, `clamp` vs `none`. Run: `python -m bench.run_1b_dynamic`.

- **clamp**: 12 steps completed, Newton iters/step = [3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6] (avg 4.8), statuses=['ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok']
- **none**: 12 steps completed, Newton iters/step = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] (avg 3.0), statuses=['ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok', 'ok']

## Observed

- The inertia term (SPD, +M/dt²) regularizes the elastic Hessian, so each step converges in a handful of Newton iterations (avg 4.8 for clamp) -- dynamics is better conditioned than the static/quasistatic cell, where unfiltered Newton outright failed 25-58% of instances (results/profiles.md).
- Strikingly, `none` (unfiltered full Newton) completed all 12 steps in avg 3.0 iters/step -- **fewer than clamp** (4.8) -- because the inertia term keeps the Hessian near-SPD, so projection is unnecessary and clamp's conservatism only adds iterations. This is the **opposite** of the static cell (where `none` failed 25-58% of instances) and a concrete instance of the Pitfalls-of-Projection thesis: projecting when you don't need to *hurts* convergence. It is why the filter axis must be studied per-regime (static 1a vs dynamic 1b), and why dt matters -- larger dt weakens inertial regularization and brings filtering back.

_Caveat: small sheet, dense solve, single dt; a dt-sweep (large dt -> less inertial regularization -> filtering matters more) is the natural next probe._
