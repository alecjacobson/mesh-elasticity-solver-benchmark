# AQP mesh-independence — rigorous (measured)

Round-2/3 hardening of the mesh-independence test (#48/#50/#51/#52; #R1/#R2/#R3): a wider sweep with **3 seeds** (median [min–max] + k/N converged), an **independent high-accuracy E\*** (Newton to |g|<1e-9, *not* best-of-compared), a **τ-sweep** (τ∈{1e-3,1e-6}), and a **growth exponent with a 95% CI** — fit on the median over only the sizes where all seeds converged and no solver hit its (raised) iteration cap, so no censored cell enters the fit. Cap-touched any cell: **False**. Run: `python -m bench.run_mesh_independence`.

Test: growth exponent p in `iters ∝ DOF^p` (p≈0 → mesh-independent). A verdict is only asserted when the 95% CI clears the flat band or two CIs separate.

### τ = 0.001

| mesh | free dof | newton median [min–max] (k/3) | l-bfgs median [min–max] (k/3) | aqp median [min–max] (k/3) |
|---|---|---|---|---|
| 6×6 | 70 | 3 [2–3] (3/3) | 16 [15–16] (3/3) | 12 [9–14] (3/3) |
| 9×9 | 160 | 3 [3–3] (3/3) | 23 [22–23] (3/3) | 16 [12–17] (3/3) |
| 12×12 | 286 | 3 [3–3] (3/3) | 30 [28–32] (3/3) | 13 [11–17] (3/3) |
| 15×15 | 448 | 3 [2–3] (3/3) | 38 [37–40] (3/3) | 10 [9–11] (3/3) |

growth exponent p (iters∝DOF^p, ±95% CI): **newton p=+0.00±0.00** (R²=1.00), **l-bfgs p=+0.46±0.03** (R²=1.00), **aqp p=-0.09±0.32** (R²=0.14)

### τ = 1e-06

| mesh | free dof | newton median [min–max] (k/3) | l-bfgs median [min–max] (k/3) | aqp median [min–max] (k/3) |
|---|---|---|---|---|
| 6×6 | 70 | 4 [4–4] (3/3) | 29 [28–31] (3/3) | 49 [43–54] (3/3) |
| 9×9 | 160 | 4 [4–4] (3/3) | 36 [35–41] (3/3) | 86 [82–100] (3/3) |
| 12×12 | 286 | 4 [4–5] (3/3) | 49 [48–50] (3/3) | 142 [105–154] (3/3) |
| 15×15 | 448 | 4 [3–4] (3/3) | 68 [67–73] (3/3) | 166 [131–217] (3/3) |

growth exponent p (iters∝DOF^p, ±95% CI): **newton p=+0.00±0.00** (R²=1.00), **l-bfgs p=+0.45±0.15** (R²=0.95), **aqp p=+0.68±0.11** (R²=0.99)

## Observed (CI-gated)

- **AQP's mesh-independence is TOLERANCE-DEPENDENT (the τ-sweep is decisive).** At loose τ=0.001 its growth exponent is consistent with 0 (p=-0.09, 95% CI [-0.41,+0.23] — mesh-independent), but at tight τ=1e-06 the CI clears the flat band (p=+0.68, CI [+0.57,+0.80]) → it **grows**. So AQP's Laplacian proxy gives mesh-independent *initial* progress but its first-order *asymptotic tail is not* mesh-independent. The round-1 'AQP is mesh-independent' reading was a **loose-tolerance artifact**.
- **Is AQP's tight-τ growth steeper than L-BFGS's?** **Not resolved at this sample size** — AQP p=+0.68 [+0.57,+0.80] and L-BFGS p=+0.45 [+0.30,+0.60] have overlapping 95% CIs, so 'AQP scales worse than L-BFGS' is NOT supported (review-r3 #R1). Both grow; the ordering between them is within noise.
- Newton is mesh-independent at both τ (p≈0, its known property) but pays a factorization per iteration (see e2) — it is the high-accuracy reference here, not a competitor on cost.

_Caveat: 2D, dense, one stretch magnitude; the independent E\* is our own Newton driven to |g|<1e-9 (a high-accuracy reference — its final energy is E\* to ~machine precision — not a third-party oracle like TinyAD/PETSc, which remains the gold standard). Spread is min–max over 3 seeds._
