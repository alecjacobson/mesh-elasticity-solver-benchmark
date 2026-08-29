# SLIM (official libigl) vs AQP / L-BFGS / Newton (measured)

All minimize symmetric Dirichlet; SLIM is libigl's official implementation. Fair shared criterion: iterations to reach relative energy tolerance `(E-E*)/(E0-E*) < 1e-4`, **paired with wall-clock and a HW-independent cost (global factorizations)** per docs/metrics.md. Run: `python -m bench.run_slim` (needs libigl).

E\* = 4.000000 (hard-constrained Newton reference), E₀ = 5.3061.

| method | iters to energy-tol | wall (ms) | global factorizations |
|---|---|---|---|
| SLIM (libigl, official) | 6 | 2.0 | 6 |
| AQP | 19 | 329.2 | 1 |
| L-BFGS | 14 | 133.3 | 0 |
| Newton | 5 | 270.6 | 5 |

## Constraint-satisfaction check (soft-vs-hard confound)

SLIM pins the boundary with a **soft** penalty (`soft_p=1e8`); the other methods use **hard** pinned BCs and `E*` is the hard-constrained minimum. Measured SLIM boundary drift `||UV[b] − bc||∞ = 4.44e-16` (**negligible** — the stiff penalty effectively enforces the hard BC, so the shared elastic-energy metric and hard `E*` are fair for SLIM).

## Observed

- **On the HW-independent axis (iterations / factorizations) `slim->aqp` reproduces:** SLIM reaches the tol in **6 iterations** vs AQP's **19**, with the OFFICIAL libigl SLIM. SLIM is a **reweighted (IRLS / Gauss-Newton) second-order-ish proxy** that refactorizes a global system each iteration -- *not* a first-order method like AQP; that is why it needs far fewer iterations.
- **⚠️ Do NOT read the raw wall-clock across the SLIM row:** libigl SLIM is compiled **C++**, our AQP/L-BFGS/Newton are pure **Python/NumPy**. SLIM does per-iteration work comparable to Newton (both refactorize a global system each iteration: SLIM 6 factorizations, Newton 5) yet reports ~137× less wall-clock -- that gap is the **compiled-vs-interpreted implementation confound**, not an algorithmic property. Wall-clock is only comparable *within* the Python group (there L-BFGS 133ms < Newton 271ms < AQP 329ms).
- **The real SLIM-vs-AQP tradeoff is factorizations vs iterations:** SLIM does **6 full factorizations**; AQP does **1** (it prefactors its fixed Laplacian once) plus 19 cheap back-solves; L-BFGS does **0**. On small meshes a factorization is cheap so SLIM's few-factorization route wins. We had speculated AQP's single-factorization route becomes more attractive at scale, but results/scale_cost.md MEASURES the cost structure and REFUTES that at tight tau (AQP's iteration/back-solve count blows up with mesh size, outrunning the few mesh-independent factorizations a Newton-class method needs; the factorize-once win holds only at loose tau).

_Caveat: energy-tolerance criterion; this headline row is a single 8×8 scenario at the default seed (7, distinct from the multi-seed profile's seeds 0–4 below); SLIM's scale- and mesh-independence and no-flip headlines are NOT tested here (see #29). Official-code SLIM grounds this comparison (D3), but the C++/Python wall-clock boundary means the HW-independent counts carry the verdict, not raw milliseconds._

## Seed × mesh profile — multi-seed AND multi-mesh (review-r2 #47)

The `slim→aqp` *validated* edge previously rested on a single 8×8 scenario; its note flagged a **seed-averaged, mesh-swept profile with ranges** as the pending hardening step. Here it is: SLIM vs AQP iterations to the same energy-tol `(E-E*)/(E0-E*)<1e-4`, over **5 seeds × 4 mesh resolutions** (official libigl SLIM, D3).

| mesh | vertices | SLIM iters, mean [min–max] | AQP iters, mean [min–max] | SLIM factor |
|---|---:|---|---|---:|
| 6×6 | 49 | 5.0 [3–6] | 22.8 [7–52] | 4.6× |
| 8×8 | 81 | 5.4 [5–6] | 38.4 [8–134] | 7.1× |
| 10×10 | 121 | 5.4 [4–6] | 28.4 [12–73] | 5.3× |
| 12×12 | 169 | 5.4 [4–6] | 18.2 [8–55] | 3.4× |

The SLIM-beats-AQP iteration gap **holds on every seed at every resolution: the per-mesh ranges never overlap** — SLIM's *worst* case (6 iters) stays below AQP's *best* case (7) at all 4 resolutions. SLIM's Gauss-Newton count is nearly flat (~5 iters, range 3–6, mesh-independent); AQP's first-order count is far larger and **high-variance** (single-seed values span 7–134), and its *mean* is **non-monotonic** in mesh size (it does not grow cleanly with resolution — do not read a scaling law into it). What the profile *does* establish is that `slim→aqp` is neither a single-seed nor a single-resolution artifact: the ordering is uniform. This is the seed×mesh profile the note required; combined with the official-code grounding (D3) it upholds the *validated* status.
