# Line-search axis - Armijo backtracking vs full-step (measured)

Only the **line-search slot** varies (clamp filter fixed). `full-step` takes the (feasibility-truncated) Newton step with no sufficient-decrease test; `backtracking` enforces Armijo. Run: `python -m bench.run_linesearch`.

| scenario | backtracking | full-step | final \|g\|inf (bt / fs) |
|---|---|---|---|
| SD 10x10 | 10 it | 10 it | 3.0e-10 / 3.0e-10 |
| NH ν=0.45 | 9 it | 9 it | 9.8e-07 / 9.8e-07 |
| NH ν=0.49 | 52 it | 52 it | 8.9e-07 / 8.9e-07 |
| NH ν=0.499 | 112 it | 112 it | 9.2e-07 / 9.2e-07 |

## Observed (a null result -- and it's informative)

- Backtracking and full-step are **identical** across every scenario here (confirmed): with a **clamp-projected** Hessian the full Newton step already satisfies the Armijo sufficient-decrease test, so backtracking never activates. A strong Hessian filter makes the line-search axis **inert** -- a concrete case of two taxonomy axes *interacting* rather than being independent.
- The corollary is the important part: the line search becomes **decisive exactly where the step is NOT already reliable** -- first-order / aggressive methods. That is precisely what E4 (results/e4.md) shows (plain gradient descent needs its line search; Adam, which has none, plateaus) and what BCQN's headline >10x-from-the-line-search-filter claim is about. So the axis matters, but its effect is *conditional on the search-direction/filter*, which is why the benchmark must vary one axis while fixing the rest and report the fixed choices.

_Caveat: two globalization variants (Armijo vs none) on Newton only; the effect shows on first-order methods (E4) and would show for unfiltered/aggressive steps. Wolfe / trust-region (tr.md) / injectivity-barrier line searches are extensions._
