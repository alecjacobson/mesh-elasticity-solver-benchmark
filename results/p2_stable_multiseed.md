# Multi-config absolute-vs-clamp on the headline element+energy (P2 + Stable NH) — measured

Hardens `results/p2_stable_nu.md` by removing the **single-initial-condition** confound. The near-incompressible ν values are re-run over **5 genuinely different deformation problems** — varying the stretch magnitude (1.6×–2.5×) and a shear (±0.25), plus a small interior jitter — NOT noise-restarts of one init (8×8 P2 mesh, Stable Neo-Hookean). Reports iterations as median [min–max] over the configs, and how often **absolute beats clamp** per-config. Run: `python -m bench.run_p2_stable_multiseed`.

| ν | clamp median [min–max] (k/N conv.) | absolute median [min–max] (k/N) | absolute<clamp / N |
|---|---|---|---|
| 0.499 | 28 [25–37] (5/5) | 22 [17–26] (5/5) | 5/5 |
| 0.4999 | 56 [49–65] (5/5) | 38 [29–45] (5/5) | 5/5 |

## Observed

- **Absolute beats clamp on all 5 diverse configs** at both ν=0.499 and ν=0.4999 (5/5/5/5). Because the configs span different stretch magnitudes (1.6×–2.5×) and shears — not a jittered single init — this is genuine initial-condition diversity, so the 'single stretch/seed' caveat of `p2_stable_nu.md` is removed (the wide clamp min–max band reflects that real diversity).
- The advantage still **grows toward the incompressible limit** across the config set (median clamp−absolute gap widens ν=0.499→0.4999), consistent with a real effect rather than a locking artifact (which would collapse at the limit).

_Caveat: still 2D, single τ=1e-6, and a locking-*relieved* (not fully locking-free) P2 element — a Taylor–Hood / mixed u–p gold-standard control and 3D remain. This removes the initial-condition confound only._
