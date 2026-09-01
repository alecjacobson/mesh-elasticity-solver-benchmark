# Filter necessity under 3D torsion — adversarial stress test (1296 tets, measured)

A 6×6×6 Neo-Hookean bar with the far face **twisted** about the bar axis by increasing angle (the near face pinned). Large rotations make the element Hessian indefinite far from the minimum — a harder stress than a gentle stretch. Projected-Newton to `|g|∞<1e-6`, comparing **no filter** (raw Newton) vs **clamp** vs **absolute** on the scalable analytic-Hessian harness. Run: `python -m bench.run_tet3d_twist`.

| twist (rad) | no filter | clamp | absolute |
|---:|---|---|---|
| 0.3 | 3 | 4 | 5 |
| 0.6 | 4 | 6 | 7 |
| 0.9 | 4 | 7 | 10 |
| 1.2 | 6 | 9 | 13 |

## Observed — filter necessity is scenario-dependent

- **Under smooth 3D torsion, filtering is NOT necessary — and raw Newton is the fastest.** Twisting the far face up to 1.2 rad (~69°) from a *valid* rest start, unfiltered Newton converges on **every** level in the fewest iterations, because the deformation stays in the descent basin: away from element inversion the line search alone keeps the (true) Hessian step productive, and the true Hessian beats any filtered surrogate near a smooth minimum. The eigenvalue filters add iterations here (a conservative projection buys nothing when the raw step is already descent).
- **So the World-2 "filtering is necessary" result (`results/profiles.md`) is a statement about the *regime*, not the dimension:** filtering earns its keep far from the minimum and near inversion (where the raw step is non-descent), not for smooth large-rotation deformation of an initially-valid mesh. Pushing this test to torsion past ~2.5 rad *with* axial compression drives elements toward inversion — the barrier regime, where the un-line-search-capped Newton step is the wrong tool for a different reason (it needs the inversion-aware line search of §4.4, not just an SPD filter).
- **Clamp vs absolute track each other** across the sweep (torsion is a rotation, not a near-incompressibility regime, so §8.1/§8.5's locking/twist distinction is not exercised).

_Scope: single mesh, a torsion stress test mapping WHERE filtering matters (it does not, for smooth large rotation from a valid start). Reuses the conformance-gated 3D harness (gate 13)._
