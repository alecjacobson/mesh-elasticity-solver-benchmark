# Faithful TLC vs Total Unsigned Area (its ablation) — untangling folded maps (measured)

TLC (`bench/tlc.py`, conformance-gated: barrier-free, analytic grad, α→0 == total unsigned area, untangles) vs its own **α=0 limit (TUA)** — same lifted-content energy, only the lifting scalar differs — on **8 folded initializations** (strengths [1.0, 2.0] × warps [0.0, 0.3] × 2 seeds; convex and non-convex `warp` targets). Metric: untangled to fully injective (all signed areas > 0), and first-injective iteration. Run: `python -m bench.run_tlc`.

| method | untangled / N | median first-injective iters (both-succeed cases) |
|---|---:|---:|
| **TLC (α>0, faithful)** | 6/8 | 34.0 |
| TUA (α=0, ablation) | 1/8 | 60.0 |
| barrier symmetric-Dirichlet (control) | 0/8 | — (`+∞` at folds — cannot start) |

## Observed

- **`tlc → tua` (robustness) reproduces:** TLC untangles **6/8** folded maps vs TUA's **1/8** — the lifting (α>0) is exactly what the paper's Prop. 4.3 says makes the minimizer injective; at α=0 the unsigned-area energy stalls on its degenerate/non-injective plateau (5 cases untangled by TLC but not TUA).
- **The capability axis (§8.4) with faithful TLC:** TLC is finite and smooth at every folded/degenerate configuration, so it untangles from folds (6/8); the barrier symmetric-Dirichlet energy is `+∞` at a fold and cannot even begin — the qualitative distinction the paper draws against barrier methods (SLIM/MIPS), now shown with the real TLC energy rather than a classical area-penalty stand-in.

_Median first-injective: TLC 34.0 vs TUA 60.0 iterations on cases both solve (TLC's lifted gradient is better-conditioned near degeneracies than TUA's kink). Faithfulness: exact lifted-content energy (Cayley–Menger form) + auto α (1e-6 ratio, Tutte auxiliary) + L-BFGS stopping at first injectivity, per the paper and reference code. The large-scale 100%-vs-baselines headline (vs foldover-free/LBD/simplex-assembly on 10k+ maps) needs those competitors' code and is not adjudicated here._
