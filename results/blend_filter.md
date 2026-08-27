# Eigenvalue-blending filter — does an intermediate blend beat clamp/absolute? (measured)

Tests `eigenvalue-blending -> {clamp, absolute}` (convergence). The per-element blend lambda_eff = max((1-w)lambda + w|lambda|, eps) is **exactly clamp at w=0.5 and absolute at w=1.0** (conformance-verified to 0), so w in [0.5,1] is the blending principle. Neo-Hookean stretch, 8x8 mesh, near-incompressible nu. Iterations to converge (tol 1e-6). Run: `python -m bench.run_blend_filter`.

### P1 (locking) — iterations vs blend weight w

| nu | w=0.5 (clamp) | w=0.625 (blend-0.625) | w=0.75 (blend-0.75) | w=0.875 (blend-0.875) | w=1 (absolute) |
|---|---|---|---|---|---|
| 0.499 | 139 | 176 | 163 | 196 | 314 |
| 0.4999 | 242 | 307 | 359 | maxiter | maxiter |

### P2 (locking-relieved) — iterations vs blend weight w

| nu | w=0.5 (clamp) | w=0.625 (blend-0.625) | w=0.75 (blend-0.75) | w=0.875 (blend-0.875) | w=1 (absolute) |
|---|---|---|---|---|---|
| 0.499 | 23 | 20 | 22 | 26 | 23 |
| 0.4999 | 53 | 36 | 40 | 35 | 41 |

## Observed

- **On P1 (locking), no intermediate blend beats clamp** — iterations generally rise as w goes clamp(0.5)->absolute(1.0), because clamp is already the best endpoint on the locking element (absolute drags the long tail). The blend interpolates; it does not dominate.
- **On P2, an intermediate blend beats both endpoints.**
- **Verdict.** An intermediate blend can beat both endpoints in at least one regime, supporting the eigenvalue-blending claim (regime-dependent).

_Caveat: 2D, single stretch/seed, single tau; fixed-w blend = the blending principle, NOT the paper's exact (possibly adaptive) w. Endpoint equivalence to clamp/absolute is exact._
