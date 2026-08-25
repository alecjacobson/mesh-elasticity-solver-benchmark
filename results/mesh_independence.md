# AQP mesh-independence — rigorous (measured)

Round-2 hardening of the mesh-independence test (#48/#50/#51/#52): a wider sweep with **3 seeds** (mean [min–max] spread), an **independent high-accuracy E\*** (Newton to |g|<1e-9 per instance — *not* best-of-compared, removing the bias toward the strongest solver), and a **τ-sweep** (τ∈{1e-3,1e-6}). Fixed continuous problem (unit square, right edge stretched to x=1.5), refined. Iterations to `(E−E*)/(E0−E*)<τ`. Run: `python -m bench.run_mesh_independence`.

The quantitative test is the **growth exponent p** in `iters ∝ DOF^p` (p≈0 → mesh-independent; p>0 → grows with resolution).

### τ = 0.001

| mesh | free dof | newton (mean [min–max]) | l-bfgs (mean [min–max]) | aqp (mean [min–max]) |
|---|---|---|---|---|
| 6×6 | 70 | 2.7 [2–3] | 15.7 [15–16] | 11.7 [9–14] |
| 9×9 | 160 | 3.0 [3–3] | 22.7 [22–23] | 15.0 [12–17] |
| 12×12 | 286 | 3.0 [3–3] | 30.0 [28–32] | 13.7 [11–17] |
| 15×15 | 448 | 2.7 [2–3] | 38.3 [37–40] | 10.0 [9–11] |

growth exponent p (iters∝DOF^p): **newton p=+0.01**, **l-bfgs p=+0.48**, **aqp p=-0.06**

### τ = 1e-06

| mesh | free dof | newton (mean [min–max]) | l-bfgs (mean [min–max]) | aqp (mean [min–max]) |
|---|---|---|---|---|
| 6×6 | 70 | 4.0 [4–4] | 29.3 [28–31] | 48.7 [43–54] |
| 9×9 | 160 | 4.0 [4–4] | 37.3 [35–41] | 89.3 [82–100] |
| 12×12 | 286 | 4.3 [4–5] | 49.0 [48–50] | 133.7 [105–154] |
| 15×15 | 448 | 3.7 [3–4] | 69.3 [67–73] | 148.5 [131–166] |

growth exponent p (iters∝DOF^p): **newton p=-0.02**, **l-bfgs p=+0.45**, **aqp p=+0.62**

## Observed

- **AQP's mesh-independence is TOLERANCE-DEPENDENT — the τ-sweep is decisive (review-r2 #50).** At the loose tolerance τ=0.001 AQP's growth exponent is ≈0 (**p=-0.06, mesh-INDEPENDENT**, matching its design claim), but at the tight τ=1e-06 it **GROWS (p=+0.62)** — steeper than L-BFGS (p=+0.45); in absolute terms AQP goes 49→148 iters over the 6.4× DOF increase while L-BFGS goes 29→69. So AQP's Laplacian proxy gives excellent **mesh-independent *initial* progress** but its first-order **asymptotic tail is NOT mesh-independent** (it lengthens with resolution, and to tight tolerance AQP scales *worse* than L-BFGS).
- **This resolves the round-1 over-claim honestly:** 'AQP is mesh-independent' was a **loose-tolerance artifact**. The τ-sweep the round-2 review demanded flips the reading — the ordering is exactly the cutoff artifact Gould–Scott/#50 warn about. Correct status: *mesh-independent to loose tolerance only; not to tight.*
- Newton is mesh-independent at both τ (p≈0, its known property) but pays a factorization per iteration (see e2) — it is the high-accuracy reference here, not a competitor on cost.

_Caveat: 2D, dense, one stretch magnitude; the independent E\* is our own Newton driven to |g|<1e-9 (a high-accuracy reference — its final energy is E\* to ~machine precision — not a third-party oracle like TinyAD/PETSc, which remains the gold standard). Spread is min–max over 3 seeds._
