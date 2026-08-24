# Data profiles (measured aggregate over problem sets)

Moré-Wild data profiles: fraction of a problem SET solved to `|g|inf<1e-6` within a budget, reported in a **hardware-independent** unit (grad+Hessian assemblies) AND in **wall-clock** (docs/metrics.md pairing). Run: `python -m bench.run_profiles`.

## Set 1 — World-1 symmetric Dirichlet (perturbation-recovery)

40 instances. Converged: **none** 30/40, **clamp** 40/40, **absolute** 40/40, **identity-shift** 40/40.

Data profile — budget in **assemblies** (HW-independent):

| filter | 8 | 16 | 32 | 64 | 128 | 256 | 400 |
|---|---|---|---|---|---|---|---|
| none | 0.20 | 0.68 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| clamp | 0.28 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| absolute | 0.23 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| identity-shift | 0.28 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Data profile — budget in **wall-clock seconds** (HW-dependent):

| filter | 0.05s | 0.1s | 0.25s | 0.5s | 1.0s | 2.0s | 5.0s |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.10 | 0.23 | 0.45 | 0.72 | 0.75 |
| clamp | 0.00 | 0.00 | 0.12 | 0.33 | 0.78 | 1.00 | 1.00 |
| absolute | 0.00 | 0.00 | 0.12 | 0.28 | 0.70 | 0.97 | 1.00 |
| identity-shift | 0.00 | 0.00 | 0.07 | 0.30 | 0.65 | 0.95 | 0.97 |

## Set 2 — Neo-Hookean stretch (nu in {0.30,0.45,0.49})

12 instances. Converged: **none** 5/12, **clamp** 12/12, **absolute** 12/12, **identity-shift** 12/12.

Data profile — budget in **assemblies** (HW-independent):

| filter | 8 | 16 | 32 | 64 | 128 | 256 | 400 |
|---|---|---|---|---|---|---|---|
| none | 0.33 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 |
| clamp | 0.33 | 0.67 | 0.67 | 0.92 | 1.00 | 1.00 | 1.00 |
| absolute | 0.33 | 0.58 | 0.67 | 0.75 | 0.92 | 1.00 | 1.00 |
| identity-shift | 0.33 | 0.50 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |

Data profile — budget in **wall-clock seconds** (HW-dependent):

| filter | 0.05s | 0.1s | 0.25s | 0.5s | 1.0s | 2.0s | 5.0s |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.25 | 0.33 | 0.42 | 0.42 | 0.42 |
| clamp | 0.00 | 0.00 | 0.08 | 0.50 | 0.75 | 0.83 | 1.00 |
| absolute | 0.00 | 0.00 | 0.17 | 0.42 | 0.67 | 0.75 | 1.00 |
| identity-shift | 0.00 | 0.00 | 0.25 | 0.50 | 0.75 | 0.92 | 1.00 |

_Note: the assemblies-profile and wall-profile agree closely here because the dense per-iteration cost is near-uniform across filters; a linear-solver swap (direct vs CG vs multigrid) is where they would diverge -- and that divergence would be the finding (metrics.md Lever 1)._
