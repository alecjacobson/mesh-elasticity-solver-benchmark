# Progressively-Projected-Newton: fraction of elements actually indefinite (measured, V2.7)

`progressively-projected-newton → clamp-filtering` (scalability) claims PPN projects **<10% of elements** vs clamp eigen-projecting ALL of them each iteration. We measure the fraction of elements whose element-Hessian is indefinite (min eigenvalue < 0) — exactly PPN's per-iteration eigendecomposition work — along a clamp-Newton solve (symmetric Dirichlet, 10×10). Run: `python -m bench.run_ppn_fraction`.

| scenario | indefinite-element fraction per Newton iteration | mean | max |
|---|---|---:|---:|
| mild stretch (seed 0) | 57%, 43%, 31%, 24%, 33%, 41%, 42%, 2% | 34% | 57% |
| harder (seed 3) | 54%, 45%, 33%, 31%, 28%, 29%, 34%, 36%, 5% | 33% | 54% |

## Observed

- **`progressively-projected-newton → clamp-filtering` (scalability) — the <10% is REGIME-SPECIFIC, not general:** the indefinite-element fraction is strongly iteration-dependent — it starts high FAR from the minimum (up to 57%) and only drops below 10% (to ~2%) NEAR convergence, averaging **33%** over the solve. So PPN's headline '<10% of elements' holds only in the near-solution regime; through the hard early iterations most of the deformed elements ARE indefinite and PPN would project a large fraction, not <10%. The MECHANISM (project on demand → fewer than all → savings that grow as the solve converges) is real and reproduces; the specific <10% is scenario/iteration-dependent and does NOT hold far from the minimum. Qualified as regime-specific.

_Caveat: 2D symmetric-Dirichlet, single element type; the fraction depends on energy, deformation severity, and distance to the minimum. We measure the fraction, which is the HW-independent core of the scalability claim; the actual eigendecomposition wall-clock saving is confounded._
