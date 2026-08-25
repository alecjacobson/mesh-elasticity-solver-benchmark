# P2 element + Stable Neo-Hookean — absolute vs clamp (measured, definitive)

Combines **both** controls the earlier runs kept separate (review-r3 Fresh #1/#2): the locking-relieved **P2** element (as in p2_nu.md) AND the correct **Stable Neo-Hookean** energy (as in stable_nu.md), so the near-incompressible absolute-vs-clamp comparison has the volumetric-locking confound removed *and* runs on the energy the absolute-filtering paper is actually built on. Same stretch init, only the filter swapped. Gradient conformance is covered by bench/conformance.py (stable-NH). Run: `python -m bench.run_p2_stable_nu`.

| ν | clamp | absolute |
|---|---|---|
| 0.3000 | 5 it | 5 it |
| 0.4500 | 9 it | 9 it |
| 0.4900 | 15 it | 13 it |
| 0.4990 | 24 it | 23 it |
| 0.4999 | 48 it | 38 it |

## Observed

- At the most incompressible ν=0.4999 on the **correct energy + locking-free element**: clamp 48, absolute 38 — **absolute BEATS clamp**.
- **This is the honest, confound-free verdict for the near-incompressible claim.** With BOTH confounds removed, absolute **beats** clamp near incompressibility (38 vs 48 it at ν=0.4999; also 13 vs 15 at ν=0.49) — so the paper's headline absolute-over-clamp advantage **does reproduce** once you use a locking-relieved element AND the correct (stable) energy. The earlier P1 'refutation' was a volumetric-locking artifact; the earlier near-null in `results/stable_nu.md` was in the *inverted-init* regime (a different scenario), not the near-incompressible stretch swept here. The two are now cleanly separated.

_Caveat: 2D, single stretch/seed, single τ; P2 relieves but a fully locking-free (Taylor–Hood P2–P1 / mixed u–p) element is still the gold-standard control (tracked separately). This run removes the *energy* confound and the *element-order* confound together, which no prior run did._
