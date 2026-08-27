# SLIM vs projected-Newton on a NON-UNIFORM triangulation (measured, P5.2 #7)

Grid rest mesh with interior vertices strongly jittered: element areas span 5.9e-05–1.8e-02 (**aspect 306×**), all valid. Stretch ×2, boundary pinned. Fair criterion: iterations to relative symmetric-Dirichlet energy `(E-E*)/(E0-E*) < 1e-4` (E\*=6.52315 projected-Newton reference, E₀=6.630). OFFICIAL libigl SLIM vs our clamp/projected-Newton. Run: `python -m bench.run_slim_nonuniform`.

| method | iterations to energy-tol |
|---|---:|
| SLIM (libigl, official) | did-not-reach |
| projected-Newton (clamp) | 5 |

SLIM boundary drift ‖UV[b]−bc‖∞ = 2.2e-09 (negligible; soft penalty ≈ hard BC, comparison fair).

## Observed

- **Not reproduced — if anything, reversed here:** projected-Newton reaches the energy-tol in **5** iterations while SLIM did **not** reach it within 60 iterations (still at E=6.6156 vs E\*=6.5231). On this non-uniform instance (aspect 306×) a well-safeguarded (clamp-projected + line-searched) Newton stays in a **good, near-quadratic basin** and converges fast, whereas SLIM's reweighted first-order-like map crawls down a **slow linear tail** — the opposite ordering to the paper's Fig.11. That figure's SLIM≫Newton result is a **far-from-minimum pathology** (Newton's raw Hessian unreliable there); a 2× stretch does not put Newton far enough from the minimum to stall it. Honest verdict: the edge is **not reproduced** in this harness — the regime that produces it (Newton stalling far from the minimum on a bad mesh) is out of reach of a stretch that keeps Newton well-conditioned.

_Caveat: 2D, single non-uniform instance/seed, moderate stretch; official libigl SLIM grounds the base (D3); wall-clock is C++/Python-confounded so iteration counts carry the verdict; our projected-Newton is clamp-filtered + backtracking-line-searched, which is already fairly robust, so a null here is about instance difficulty, not method identity._
