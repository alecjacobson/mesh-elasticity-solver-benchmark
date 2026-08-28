# AQP vs its ablation baseline AGD (proxy on/off) across conditioning (measured, V2.1)

`accelerated-gradient-descent` (AGD) is the AQP paper's OWN ablation: the identical accelerated scheme with the Laplacian proxy disabled (`solve_aqp(use_proxy=False)`, so `d=-g` instead of `d=L⁻¹(-g)` — nothing else changes). We sweep an area-preserving anisotropic stretch `A=diag(s,1/s)` (8×8 grid) that drives symmetric-Dirichlet from well- to ill-conditioned. Iterations to gradient tol 1e-4 (max 3000). Run: `python -m bench.run_agd_vs_aqp`.

| stretch s (conditioning) | AQP (proxy on) | AGD (proxy off) | proxy helps? |
|---|---:|---:|---|
| 1 | 573 | 61 | no (AGD ≤ AQP) |
| 1.5 | 163 | 70 | no (AGD ≤ AQP) |
| 2.5 | 256 | 937 | yes |
| 3.5 | 991 | maxi>3000 | yes |

## Observed

- **`aqp → accelerated-gradient-descent` — REPRODUCES, but regime-dependent (the ablation is NOT a straw-man):** there is a clear crossover. On the **well-conditioned** end (s=1) AGD is *faster* than AQP (61 vs 573) — the Laplacian proxy is the wrong metric there and actually hurts. As the energy becomes **ill-conditioned** (s=2.5) AGD blows up (937) while AQP stays bounded (256), and at s=3.5 AGD maxi>3000. So the proxy earns its keep exactly in the ill-conditioned regime the paper targets, and the claim 'AQP scales where AGD scales poorly as energies become ill-conditioned' reproduces.
- **Honest qualification of the baseline-confound flag:** AGD is a *fair* ablation, not a weak strawman — it beats AQP when the problem is well-conditioned. The proxy's value is conditional on the Laplacian being a good preconditioner (high-distortion / spatially smooth regime), which is a real but bounded claim.

_Caveat: 2D, single mesh size/seed per s, one anisotropy family; iteration-axis (HW-independent). AGD's Nesterov θ is inherited from AQP's η (as in the ablation), not separately tuned — the paper's ablation makes the same choice._
