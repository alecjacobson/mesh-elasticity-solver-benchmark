# E1 (near-incompressible) — absolute vs clamp as ν → ½ (measured, indicative)

Probes the Stabler-Neo-Hookean **absolute vs clamp** claim in its claimed regime (high Poisson + large deformation). Config-diff: only the Hessian filter varies; Neo-Hookean energy, mesh, BC-driven stretch (right edge → x=2.0), Armijo line search, dense solve, `|g|inf<1e-6` fixed. ν is a swept *material* scenario parameter. Run: `python -m bench.run_e1_nu` (NH gradient conformance-gated per ν).

Mesh 8×8; cells = iterations to converge (or failure status).

| ν | λ | clamp | absolute | project-on-demand | none (full Newton) | identity-shift |
|---|---|---|---|---|---|---|
| 0.3000 | 1.5 | 6 it | 6 it | 6 it | 6 it | 6 it |
| 0.4500 | 9.0 | 9 it | 12 it | 9 it | **nondescent** | 13 it |
| 0.4900 | 49.0 | 52 it | 89 it | 52 it | **nondescent** | 41 it |
| 0.4990 | 499.0 | 112 it | 253 it | 112 it | **nondescent** | 84 it |
| 0.4999 | 4999.0 | 234 it | **maxiter** | 234 it | **nondescent** | 125 it |

**Observed (this run):** filters agree at ν=0.3 (well-conditioned, Hessian SPD); as ν→½ the orderings diverge sharply — clamp needs *fewer* iterations than absolute (234 vs non-convergent at ν=0.4999), full Newton (`none`) fails once the Hessian turns indefinite, and the global identity-shift is fastest at high ν. Absolute *under*performing clamp here runs **opposite** to the Stabler-Neo-Hookean claim — but that is the signature of the **volumetric-locking confound** (control C1), not a refutation: on displacement-only elements the volumetric term is artificially stiff, and flipping large negative eigenvalues to positive overshoots that locked direction. This is exactly why the protocol mandates a locking-free element for the ν-sweep — the confound is empirically real and would silently corrupt the comparison.

**Caveat (important):** displacement-only P1 triangles, no locking-free element (control C1 in `docs/protocol.md` is NOT applied here), single scenario/seed, dense solve. At high ν the comparison is partly confounded by **volumetric locking**, so this is an *indicative probe of the harness*, not a settled reproduction of the claim. The proper test (mixed u–p / F-bar element, ν-sweep, official-code regression) is the next P1 step. Whatever the outcome, it is reported as measured — including a null result.
