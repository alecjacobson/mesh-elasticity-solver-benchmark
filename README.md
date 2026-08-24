# Mesh-Elasticity Solver Survey & Benchmark

> Untangling a decade of numerical-method claims for **mesh elasticity** problems —
> distortion/parametrization optimization, projected-Newton eigenvalue filtering, and
> IPC-style contact — mostly from the SIGGRAPH literature (~2010–2025), placed against the
> classical optimization, computational-mechanics, and ML-optimizer canon.

**Status:** 🚧 pre-execution design & corpus. This repo currently holds the *survey design*,
the *annotated corpus*, and a *superiority-claims graph*. The benchmark harness comes next.

---

## The one-paragraph pitch

Nearly every "faster / more robust elastic solver" paper is a **component swap** inside one
shared problem — minimize a nonlinear elastic energy `E(x) = Σ_e V_e ψ(F_e(x))` over mesh
vertex positions — yet papers typically change **2–3 things at once and credit one**. This
project builds (a) a defensible **taxonomy**, (b) a **superiority-claims graph** that records
who claims to beat whom and whether that claim is *validated / qualified / unvalidated /
refuted*, and (c) a **component-factored benchmark** that pins the confounds and measures
honest attribution.

## The unifying view

Every method here — graphics, classical, or ML — is **metric descent**

```
x' = x − α · M⁻¹ ∇E(x)
```

differing only in the metric `M` (and its globalization):

| `M` | method |
|---|---|
| `I` | gradient descent |
| `∇²E` | Newton |
| Fisher | natural gradient (Amari) — *undefined here: no output distribution* |
| Laplacian / H¹ | Sobolev gradient — **AQP, BCQN** |
| Killing operator | **AKVF** |
| reweighted energy | **SLIM** (IRLS Gauss-Newton) |

This lets us say precisely: "Sobolev preconditioning in graphics" and "natural gradient in ML"
are the *same idea under different metrics*.

## The three "worlds" (comparability is governed by problem class, not method)

- **World 1 — static distortion / parametrization** (no inertia, no contact). AQP, SLIM,
  BCQN, Composite Majorization, TLC, GOSS.
- **World 2 — quasistatic/dynamic hyperelasticity** (inertia, no contact). The
  eigenvalue-filtering cohort, L-BFGS QN, ADMM, Vertex Block Descent.
- **World 3 — contact-coupled dynamics** (barriers, CCD, friction). IPC, ABD, GIPC, OGC.

Worlds 1–2 share the whole projected-Newton skeleton; World 3 adds four parameters that
belong to no solver (barrier stiffness `d̂`, CCD tolerance, friction `ε_v`, time-step `Δt`).

## Benchmark design in one picture

```
taxonomy → capability cells → tiered problem classes
  → within-cell FAIR head-to-head (fixed energy + criterion, single-axis ablation)
  → cross-cell performance profiles (Dolan–Moré) for robustness
  → equal tuning budget, standardized harness, hidden/rotating tier
```

- **v1** = contact-free **solver track**: *1a* distortion accelerators + *1b* eigenvalue-filter
  ablation on one projected-Newton skeleton. Delivered as a survey+benchmark paper, architected
  toward a living benchmark.
- **v2** = **contact capability track** + optional **learned-accelerators** companion track.

---

## Repository map

| Path | What |
|---|---|
| [`docs/design.md`](docs/design.md) | Full design: scope arguments, recommended two-track benchmark, v1 plan, decomposition experiments, lineage map, scope ledger |
| [`docs/taxonomy.md`](docs/taxonomy.md) | The taxonomy: method axes × capability cells, classification of the corpus, orthogonality evaluation, fairness gate |
| [`docs/metrics.md`](docs/metrics.md) | Performance-metric deliberation: over-complete 80-metric catalog → per-cell orthogonal core; HW-dependent/independent pairing; data-profile aggregation |
| [`docs/harness.md`](docs/harness.md) | Harness architecture: component-slot contracts, config-as-point-in-component-space, conformance-suite admissibility gate, official-code-first mapping, scenario layer |
| [`docs/corpus.md`](docs/corpus.md) | Annotated corpus (~180 entries) across Worlds 0–3, with axis tags, comparability notes, code availability |
| [`claims/`](claims/) | **Superiority-claims graph** — machine-readable edges (`claims.yaml`) + rendered Mermaid graph & tables ([`claims/README.md`](claims/README.md)) + [`schema`](claims/schema.md) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Conventions for **humans and agents** adding papers, claims, and findings |

## The superiority-claims graph

A directed graph where **nodes = papers/methods** and **edges = "A claims superiority over B"**,
each edge annotated with the *dimension* of the claim (speed / robustness / convergence /
quality / generality), the *evidence* offered, and a *validation status*:

- `self-claimed` — asserted by the authors, not independently checked here
- `validated` — reproduced/confirmed (by an independent study or our benchmark)
- `qualified` — true only under stated conditions (regime, energy, mesh, metric)
- `unvalidated` — no sufficient evidence either way
- `refuted` — contradicted by evidence

The endgame: **harden** self-claims into validated / qualified / refuted as the benchmark
produces data. See [`claims/README.md`](claims/README.md).

## How this is built

Progress is tracked in **GitHub issues**; commits reference the issue they address. Today,
research and extraction are done by a mix of human curation and **LLM agents** (corpus breadth
+ claim extraction). The benchmark harness is a **common framework of hot-swappable
components**; components are reimplemented **from official code where it exists** and
**regression-tested against that official reference (or an independent oracle)** — and will
increasingly be **agent-generated** under that same rule. The invariant: benchmark numbers are
always *measured against a validated implementation*, never asserted by a model. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Caveats

Corpus entries carry `[?]` uncertainty flags where a detail (venue, code release, exact
method) was not independently verified. Some 2025–2026 arXiv items are unconfirmed. Claims are
recorded as *authors' assertions* until marked otherwise — inclusion is not endorsement.
