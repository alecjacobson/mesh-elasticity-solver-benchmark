# Analytic eigensystems vs numeric eigendecomposition (measured)

Hardens `analytic-eigensystems -> numeric`. Closed-form SPD projection (Smith-de Goes-Kim 2019) vs `numpy.linalg.eigh`. **Fair baseline (review-r1 #37):** the numeric path consumes the *analytically assembled exact Hessian* (assembled once, outside the timed region) and we time only `eigh + clamp + reassemble` — NOT a finite-difference Hessian assembly. The analytic path is timed from `F` (`svd + closed-form + clamp + reassemble`). Excluding H assembly from the numeric timing *favours* the numeric path (the analytic route never assembles the raw Hessian), so the multiplier below is a **conservative lower bound**. Run: `python -m bench.analytic_eig`.

| dim | eigenvalues vs FD (rel) | projection vs numeric (rel) | analytic (svd+closed-form) | numeric (eigh only) | analytic/numeric |
|---|---|---|---|---|---|
| 2D | 1.4e-10 | 1.5e-15 | 350 ms | 94 ms | **3.71× slower** |
| 3D | 2.3e-10 | 2.4e-15 | 730 ms | 146 ms | **5.00× slower** |

## Observed (an honest surprise)

- **Equivalence — validated.** The analytic eigenpairs match a finite-difference Hessian of the energy to ~1e-10, and the analytic projection matches the numeric one to ~1e-15 — the two produce the **same projected Hessian**, so a solver using either takes the *same iteration count*. The *direction/correctness* of the paper is not in dispute.
- **Speed — does NOT reproduce at the eigendecomposition-kernel level.** Once the numeric baseline is made fair (fed the already-assembled analytic Hessian, timing only `eigh + clamp`, no finite differences), `numpy.linalg.eigh` on a 4×4/9×9 is **faster** than the scalar closed-form projection (analytic is ~3.7× slower in 2D, ~5× in 3D). LAPACK's symmetric eig on tiny matrices is extremely fast, and the per-element Python closed form carries SVD + allocation overhead.
- **So where does the paper's speedup live?** Entirely in the *Hessian-assembly / differentiation* cost that this fair kernel comparison **excludes**: the paper's baseline is autodiff/numerical differentiation of the Hessian (expensive), which the analytic route skips. Our *old* number (3.3×) was measuring exactly that FD-assembly cost — a straw-man. The eigendecomposition itself is not where the analytic method wins at these sizes.

**Takeaway:** the analytic eigensystem's value is a **guaranteed-SPD projection without autodiff**, not a faster eigensolve. Its wall-clock advantage is real only when the alternative pays to *build* the Hessian (autodiff/FD) or when a batched/compiled/GPU kernel amortizes the closed form — neither is a scalar NumPy `eigh` of a 4×4. This is the benchmark separating *what* is faster (assembly-free projection) from a mis-attributed *eigendecomposition* speedup.

_Caveat: micro-benchmark; scalar Python/NumPy (no batching/SIMD/GPU), which would shift the kernel comparison. End-to-end solver integration (identical iterations by construction; only per-iteration assembly cost moves) tracked in #37/#32._
