# Multi-seed absolute-vs-clamp on the headline element+energy (P2 + Stable NH) — measured

Hardens `results/p2_stable_nu.md` by removing the **seed** confound: the near-incompressible ν values re-run with a small per-seed interior perturbation of the stretch init (5 seeds, 8×8 P2 mesh, Stable Neo-Hookean). Reports iterations as median [min–max] over seeds, and how often **absolute beats clamp**. Run: `python -m bench.run_p2_stable_multiseed`.

| ν | clamp median [min–max] (k/N conv.) | absolute median [min–max] (k/N) | absolute<clamp / N |
|---|---|---|---|
| 0.499 | 30 [28–32] (5/5) | 22 [22–23] (5/5) | 5/5 |
| 0.4999 | 60 [53–64] (5/5) | 39 [38–39] (5/5) | 5/5 |

## Observed

- **The absolute-beats-clamp ordering holds across all 5 seeds** at both ν=0.499 and ν=0.4999 (5/5/5/5 seeds), so the headline result is not a single-seed artifact — the 'single seed' caveat of `p2_stable_nu.md` is removed for these ν.
- The advantage still **grows toward the incompressible limit** within this multi-seed set (median clamp−absolute gap widens ν=0.499→0.4999), consistent with a real effect rather than a locking artifact (which would collapse at the limit).

_Caveat: still 2D, single stretch magnitude, single τ=1e-6, and a locking-*relieved* (not fully locking-free) P2 element — a Taylor–Hood / mixed u–p gold-standard control and 3D remain. This removes the seed confound only._
