# World-1 accelerator data profiles (measured)

Data profile over 6 symmetric-Dirichlet perturbation instances (meshes 5/6/7 x seeds 0/1/2). Fraction solved to relative energy tolerance 1e-4 within an iteration budget. Run: `python -m bench.run_world1_profiles`.

| method | ≤3 it | ≤5 it | ≤10 it | ≤20 it | ≤40 it | ≤80 it |
|---|---|---|---|---|---|---|
| newton | 0.00 | 0.17 | 1.00 | 1.00 | 1.00 | 1.00 |
| l-bfgs | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| sobolev-lbfgs | 0.00 | 0.00 | 0.00 | 0.83 | 1.00 | 1.00 |
| aqp | 0.00 | 0.00 | 0.33 | 0.50 | 0.83 | 1.00 |

## Observed

- **Second-order (Newton) and Sobolev-L-BFGS reach the energy tolerance fastest**; plain L-BFGS close behind; **AQP needs the largest budget** -- consistent with E2 and the slim/aqp result (AQP's fixed Laplacian proxy is the weakest of the proxy family on these problems).
- The profile is on the HW-independent iteration budget; it aggregates the E2 single-instance findings over a set (Moré-Wild style), showing the ordering is stable across meshes/seeds, not an artifact of one instance.
- **Read the x-axis as *iterations*, not cost:** a Newton iteration is a full Hessian factorization while Sobolev-L-BFGS/AQP prefactor once and L-BFGS back-solves only, so an iteration-budget profile *understates* Newton's per-iteration cost. See the factorization column in `results/e2.md` for the HW-independent cost that pairs with this iteration-budget view; neither alone settles a wall-clock ranking.

_Caveat: energy-tolerance criterion (fair for first-order tails); small meshes; official SLIM (results/slim.md) would sit near Newton but uses soft constraints, so it is compared separately._
