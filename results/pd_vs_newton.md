# Projective Dynamics vs full Newton — shared mass-spring potential (measured)

Both minimize the **same** implicit-Euler potential `Φ(x) = ½h⁻²(x−x̃)ᵀM(x−x̃) + E(x)` (`bench/massspring.py`) and stop at the **same** residual criterion `max|∇Φ[free]| / initial < 1e-5`, so iteration counts are directly comparable. PD is the exact local/global (Liu 2013) minimizer of this Φ; Newton is projected-SPD with a line search. Run: `python -m bench.run_pd_vs_newton`.

| mesh | free dof | spring k | PD iters | Newton iters | PD factorizations | Newton factorizations |
|---|---:|---:|---:|---:|---:|---:|
| 6×6 | 84 | 1e+03 | 16 | 4 | 1 | 4 |
| 8×8 | 144 | 1e+03 | 16 | 4 | 1 | 4 |
| 8×8 | 144 | 1e+04 | 36 | 4 | 1 | 4 |
| 12×12 | 312 | 1e+03 | 16 | 4 | 1 | 4 |

## Observed — `projective-dynamics → full-newton` (speed) adjudicated

- **NOT reproduced on the iteration axis:** PD needs **more** iterations than Newton on every scenario (~5× more on average) — expected, because PD is a **first-order** local/global fixed-point iteration while Newton is **second-order**. On iterations-to-residual, Newton dominates; the paper's speed claim does *not* hold on this hardware-independent axis.
- **The mechanism, on the OTHER hardware-independent axis (factorizations):** PD prefactors its **constant** system **once** and reuses it for every iteration (**1 factorization** total); Newton refactorizes its changing Hessian **every iteration** (as many factorizations as iterations). This factorization-reuse is the real basis of PD's interactive-speed reputation — a **per-iteration-cost** advantage, not a fewer-steps advantage.

_Honest verdict: the `projective-dynamics → full-newton` speed edge is **qualified** — it does **not** reproduce as fewer iterations (Newton wins that axis), but PD's factorize-once-vs-refactorize-each-iteration structure is real and measured. Whether that converts to a net wall-clock win depends on mesh size, per-iteration cost and hardware (a factorization is cheap on these small meshes but dominates at scale), so the headline ‘faster’ resolves to a **wall-clock/scale-confounded** claim, not an algorithmic one on the iteration axis. Same shape as the CM→projected-Newton finding: a cheaper-per-step method is not a fewer-step method._
