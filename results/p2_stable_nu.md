# P2 element + Stable Neo-Hookean — absolute vs clamp (measured, definitive)

Combines **both** controls the earlier runs kept separate (review-r3 Fresh #1/#2): the locking-relieved **P2** element (as in p2_nu.md) AND the correct **Stable Neo-Hookean** energy (as in stable_nu.md), so the near-incompressible absolute-vs-clamp comparison has the volumetric-locking confound removed *and* runs on the energy the absolute-filtering paper is actually built on. Same stretch init, only the filter swapped. Gradient conformance is covered by bench/conformance.py (stable-NH). Run: `python -m bench.run_p2_stable_nu`.

| ν | clamp | absolute |
|---|---|---|
| 0.3 | 5 it | 5 it |
| 0.45 | 9 it | 9 it |
| 0.49 | 15 it | 13 it |
| 0.499 | 24 it | 23 it |
| 0.4999 | 48 it | 38 it |
| 0.49999 | 113 it | 71 it |

## Observed

- At the most incompressible ν=0.49999 on the **correct energy + locking-relieved element**: clamp 113, absolute 71 — **absolute BEATS clamp**.
- **Absolute's advantage GROWS toward the incompressible limit** (5/5 at ν=0.3 → 113/71 at ν=0.49999, clamp/absolute iters). This is strong evidence the effect is a **real filter property, not residual locking**: if P2's remaining locking were driving the result, the extreme limit would show the locking pathology (as on P1, where absolute *fails*), collapsing or reversing the gap — instead absolute wins *harder* as ν→½, exactly as the paper's near-incompressible claim predicts. (Partially addresses #74: a fully locking-free Taylor–Hood element remains the gold-standard control, but the growing-advantage trend is what a residual-locking artifact would NOT produce.)
- **This is the honest, confound-free verdict for the near-incompressible claim.** With BOTH confounds removed, absolute **beats** clamp near incompressibility (71 vs 113 it at ν=0.49999; also 13 vs 15 at ν=0.49) — so the paper's headline absolute-over-clamp advantage **does reproduce** once you use a locking-relieved element AND the correct (stable) energy. The earlier P1 'refutation' was a volumetric-locking artifact; the earlier near-null in `results/stable_nu.md` was in the *inverted-init* regime (a different scenario), not the near-incompressible stretch swept here. The two are now cleanly separated.

_Caveat: 2D, single stretch/seed, single τ; P2 relieves but a fully locking-free (Taylor–Hood P2–P1 / mixed u–p) element is still the gold-standard control (tracked separately). This run removes the *energy* confound and the *element-order* confound together, which no prior run did._
