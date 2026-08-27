# SLIM vs projected-Newton on a NON-UNIFORM triangulation (P5.2 #7 — regime not reached)

Grid rest mesh with interior vertices strongly jittered: element areas span 5.9e-05–1.8e-02 (**aspect 306×**), all valid. Stretch ×2, boundary pinned. Criterion: iterations to relative symmetric-Dirichlet energy `(E-E*)/(E0-E*) < 1e-4` (E\*=6.52315 projected-Newton reference, E₀=6.630). OFFICIAL libigl SLIM vs our clamp/projected-Newton. Run: `python -m bench.run_slim_nonuniform`.

> ⚠️ **This does NOT adjudicate the claim, and is not evidence against the paper.** The claim (Fig.11) is about Newton **stalling far from the minimum** on a bad mesh. Our Newton is clamp-**projected** (SPD-safeguarded) + line-searched, and a 2× stretch keeps it well-conditioned — a **strawman** for a raw-Hessian-far-from-min claim. We report this to document *why the regime is out of reach*, not to rank the methods; the edge stays `self-claimed`.

| method | iterations to energy-tol |
|---|---:|
| SLIM (libigl, official) | did-not-reach |
| projected-Newton (clamp) | 5 |

SLIM boundary drift ‖UV[b]−bc‖∞ = 2.2e-09 (negligible; soft penalty ≈ hard BC, comparison fair).

## Observed

- **Regime not reached (the claim's stall never occurs here):** projected-Newton reaches the energy-tol in **5** iterations while SLIM did **not** reach it within 60 iterations (still at E=6.6156 vs E\*=6.5231). On this non-uniform instance (aspect 306×) a well-safeguarded (clamp-projected + line-searched) Newton stays in a **good, near-quadratic basin** and converges fast, whereas SLIM's reweighted first-order-like map crawls down a **slow linear tail**. Crucially, this is **not evidence against the paper**: its SLIM≫Newton result (Fig.11) is a **far-from-minimum pathology** where Newton's *raw* Hessian is unreliable, and a 2× stretch on a projected+line-searched Newton never enters that regime. Honest verdict: the edge is **not adjudicable** in this harness — the configuration that produces the claim (raw-Hessian Newton stalling far from the minimum on a bad mesh) is out of reach here, so we report the limitation rather than a ranking. The edge stays `self-claimed` / needs-harder-instance.

_Caveat: 2D, single non-uniform instance/seed, moderate stretch; official libigl SLIM grounds the base (D3); wall-clock is C++/Python-confounded so iteration counts carry the verdict; our projected-Newton is clamp-filtered + backtracking-line-searched, which is already fairly robust, so a null here is about instance difficulty, not method identity._
