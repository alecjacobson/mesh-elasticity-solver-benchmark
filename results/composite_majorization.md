# Composite Majorization vs projected-Newton / absolute / AQP / SLIM (measured, closes #14)

**Faithful CM** (Shtengel et al. 2017, `bench/composite_majorization.py`), conformance-gated on the paper's own theorems: Σ,σ via similarity/anti-similarity == SVD (1e-15); PSD majorizer Hessian (eq. 9); **Proposition 3.1: CM Hessian ⪰ true Hessian**; monotone majorize-minimize descent; converges to the SAME minimum as projected-Newton. Symmetric-Dirichlet, 6 scenarios (meshes 6–10, multiple seeds). Iterations to |g|∞<1e-6. Run: `python -m bench.run_composite_majorization`.

| method | iterations to converge, mean [min–max] |
|---|---:|
| Composite Majorization (CM) | 9.0 [7–11] |
| projected-Newton (clamp) | 8.8 [7–11] |
| absolute filtering | 9.0 [7–11] |
| AQP (first-order) | 776.7 [351–1514] |
| SLIM (official libigl, one scenario) | 5 |

Full-step (α=1) acceptance rate (majorize-minimize property): CM **100%** vs clamp 100% (8×8 seed 0).

## Observed — edges adjudicated

- **`composite-majorization → aqp` (speed/convergence) — REPRODUCES (decisively):** CM, a second-order method, converges in **9.0 [7–11]** iterations vs first-order AQP's **776.7 [351–1514]** — CM needs ~86× fewer iterations, exactly the paper's second-order-beats-first-order point (the HW-independent core of its wall-clock claim).
- **`composite-majorization → projected-newton` — NOT reproduced on iterations:** CM takes **9.0 [7–11]** iterations vs projected-Newton's **8.8 [7–11]** (1.02× — CM is slightly MORE, not fewer). This is the expected behaviour of a MAJORIZER: CM's Hessian ⪰ the true Hessian (Prop 3.1), so its steps are conservative-but-guaranteed-descent, whereas clamp minimally projects only the indefinite modes. (CM accepts the full step α=1 100% of the time by its majorize-minimize guarantee; clamp 100% here too, as these mild scenarios are near-quadratic — not a distinguishing edge on the iteration axis.) CM's genuine edge is a cheap ANALYTIC Hessian (no per-element eigendecomposition) — but the paper uses that same analytic Hessian for its projected-Newton too, so it is not the differentiator, and the wall-clock '4× faster than PN' is hardware/energy/scenario-confounded and does not surface on the 2D iteration axis.
- **`composite-majorization → slim` — NOT reproduced on iterations here:** CM **9.0 [7–11]** vs official SLIM **5** on an 8×8 scenario — SLIM's reweighted Gauss-Newton converges in very few iterations; CM does not beat it on this mild instance (the paper's SLIM disadvantage is a large-mesh/far-from-init scalability regime, needs-scale, not reached here).

_Faithfulness note: this is the real CM (gated on Prop 3.1 majorization + same-minimum), also implemented for symmetric ARAP (`cm_element_hessian_sarap`), where the same ordering holds (CM slightly above clamp). The honest finding: CM's headline 'faster than projected-Newton' is a WALL-CLOCK claim resting on the analytic Hessian; on the hardware-independent iteration axis for 2D distortion, CM is a conservative majorizer comparable to eigenvalue filtering and does not beat projected-Newton, while it decisively beats first-order AQP._
