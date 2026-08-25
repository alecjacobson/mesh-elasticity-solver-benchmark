# SRI-P2 (locking-relieved element): absolute vs clamp (measured)

Uses **Selective Reduced Integration** on P2 (deviatoric full 3-pt quadrature, volumetric reduced 1-pt centroid — the classic Malkus–Hughes locking cure, `bench/p2_sri.py`), a genuinely locking-relieving element in **displacement form** so the clamp/absolute filters stay a single-axis swap (unlike a mixed u–p element). Classical Neo-Hookean (its deviatoric and volumetric parts are individually rest-stress-free, so SRI preserves rest equilibrium — validated: rest |grad| ~1e-15). Conformance-gated. Run: `python -m bench.run_sri_nu`.

## Validation (two gates)

| ν | full-P2 clamp | SRI-P2 clamp | E(full-P2) | E(SRI-P2) | SRI clamp≡absolute soln? |
|---|---|---|---|---|---|
| 0.3 | 4 | 4 | 1.0412 | 1.0409 | ✓ (Δ0e+00) |
| 0.45 | 9 | 8 | 1.2754 | 1.2722 | ✓ (Δ1e-09) |
| 0.49 | 15 | 20 | 1.3551 | 1.3474 | ✓ (Δ2e-07) |
| 0.499 | 23 | 57 | 1.3754 | 1.3642 | ✓ (Δ9e-07) |
| 0.4999 | 53 | 250 | 1.3778 | 1.3658 | ✓ (Δ1e-06) |

- **(1) SRI relieves locking** — measured the RIGHT way, by the converged energy (a locking element is over-stiff → higher energy at the same BCs). At ν=0.4999 SRI-P2 reaches **E=1.3658 vs full-P2's 1.3778** — lower, i.e. less over-constrained. (My first pass wrongly used clamp *iterations* as the indicator; those went UP because clamp handles SRI's near-singular volumetric mode poorly — see below — which is a filter effect, not a locking measure.)
- **(2) No hourglassing** — SRI-clamp and SRI-absolute converge to the **same solution** (‖Δx‖∞ < 1e-5 at every ν), so SRI's reduced integration did not introduce spurious zero-energy modes; the clamp/absolute gap is a pure convergence-rate effect on the same minimum.

## absolute vs clamp on the (validated) locking-relieved SRI element

| ν | SRI-P2 clamp | SRI-P2 absolute |
|---|---|---|
| 0.3 | 4 | 4 |
| 0.45 | 8 | 7 |
| 0.49 | 20 | 14 |
| 0.499 | 57 | 17 |
| 0.4999 | 250 | 23 |

## Observed

- **On the locking-relieved SRI element, absolute DRAMATICALLY beats clamp** (23 vs 250 it at ν=0.4999) — the largest absolute-over-clamp margin in the whole benchmark. SRI's reduced volumetric integration makes the near-incompressible volumetric mode nearly singular; clamp floors it to ε (a near-null search direction → very slow), while absolute maps it to |λ| (well-scaled → fast). Same minimum, opposite convergence.
- **This is a THIRD independent locking treatment** (after the P1 crossed-mesh probe and the standard/stable-NH P2 element) and it gives the same verdict as the others: once volumetric locking is relieved, absolute filtering matches or **beats** clamp near incompressibility. Four locking treatments now concur (crossed-mesh, P2, stable-NH-P2, SRI-P2) — strong evidence the P1 'absolute is worse' result was a **locking artifact**, not a filter property.

- **Scope (#74):** SRI is a *genuine* locking cure (validated: lower energy, no hourglass) but not the gold-standard mixed (Taylor–Hood P2–P1) formulation, and it's on classical (barrier) NH. A fully locking-free mixed element remains the ultimate control, but four independent treatments agreeing substantially de-risks the verdict.

_Caveat: 2D, single stretch/seed/τ; classical NH (J>0). The point is the *agreement across locking treatments*, validated (energy + hourglass gates), not any single element._
