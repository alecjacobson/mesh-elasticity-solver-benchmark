# AQP vs ARAP local-global vs Anderson-LG — back-solves to each method's own minimum (measured, P5.2 #6 & #5)

10×10 non-affine bend, 5 seeds. Each method does 1 prefactorization + 1 global back-solve per iteration (AQP additionally runs an Armijo line search, so its per-iteration work is a little higher). Iterations to reach each method's OWN `(E-E*)/(E0-E*) < 1e-4` (AQP on symmetric-Dirichlet; local-global & Anderson-LG on ARAP). Run: `python -m bench.run_aqp_localglobal`.

> ⚠️ **This is NOT a fair head-to-head, and does not adjudicate `aqp→local-global` (#6) or `anderson→aqp` (#5).** The methods minimize **different energies** (symmetric-Dirichlet vs ARAP) to **different minima**, so 'back-solves to each own tol' compares distances to two unrelated basins — on this bend the ARAP minimum simply sits nearer the start. A fair same-objective race is not constructible without committing all three to one energy (which the source papers do not specify). The numbers below are **descriptive only**; both edges stay `self-claimed`.

| method (energy) | back-solves to own tol, mean [min–max] | wall (ms) mean |
|---|---:|---:|
| AQP (symmetric-Dirichlet) | 11.4 [9–13] | 2248 |
| local-global (ARAP) | 5.6 [5–6] | 226 |
| Anderson-LG, m=5 (ARAP) | 4.0 [4–4] | 172 |

## Observed (descriptive — not a verdict)

- Each method reaches ITS OWN minimum in: AQP **11.4**, local-global **5.6**, Anderson-LG **4.0** back-solves. Anderson-LG's lower count vs local-global is consistent with the *validated* `anderson→local-global` edge (acceleration of the same ARAP map). But comparing AQP's symmetric-Dirichlet count against the two ARAP counts does **not** adjudicate `aqp→local-global` or `anderson→aqp`: the ARAP minimum being nearer on this bend is an instance property, not a convergence win, and 'lower final energy' is not even comparable across the two objectives.
- Wall-clock (both pure NumPy, diagnostic only): AQP 2248ms, local-global 226ms, Anderson-LG 172ms — reported for completeness; it cannot rank cross-energy methods either.

_Caveat: CROSS-ENERGY, single mesh size, one bend family (5 seeds differing only by tiny jitter ≈ one instance). This runner exists to DOCUMENT WHY `aqp→local-global` (#6) and `anderson→aqp` (#5) are not adjudicable in this harness, not to score them — both edges remain `self-claimed`. A fair test needs all methods committed to a single shared energy._
