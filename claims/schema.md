# Superiority-Claims Graph — Schema

The graph lives in [`claims.yaml`](claims.yaml) as two lists: `nodes` (methods/papers) and
`edges` (directed superiority claims). [`README.md`](README.md) renders it (Mermaid + tables).

## Node

```yaml
- id: absolute-filtering          # stable kebab-case key, unique
  title: "Stabler Neo-Hookean: Absolute Eigenvalue Filtering for Projected Newton"
  authors: "Chen, Liu, Levin, Zheng, Jacobson"
  year: 2024
  venue: "SIGGRAPH"
  world: 2                         # 0 external | 1 distortion | 2 hyperelastic | 3 contact
  ref: "arXiv:2406.05928"          # DOI/arXiv/URL if known; "" if not
```

## Edge  (a directed "A claims superiority over B")

```yaml
- from: absolute-filtering         # node id making the claim
  to: clamp-filtering              # node id being claimed better-than
  dimension: convergence           # see below
  status: self-claimed             # see below
  claim: "Absolute-value eigenvalue filtering needs fewer Newton iterations than clamp-to-zero, especially near-incompressible."
  evidence: "iteration counts on a quasistatic dataset; one-line change holding the projected-Newton skeleton fixed"
  source: "arXiv:2406.05928"       # where the claim is made (usually the 'from' paper)
  assessed_by: "self"              # "self" (author claim) | "corpus" | "benchmark" | citation to independent study
  notes: ""                        # regime/caveats; why status is what it is
```

### `dimension`  — what axis the superiority is claimed on
`speed` (wall-clock) · `convergence` (iterations / rate to tolerance) · `robustness`
(success rate, inversion/injectivity, stability) · `quality` (final energy/distortion,
non-penetration) · `generality` (range of energies/meshes/materials) · `scalability`
(mesh size, GPU) · `simplicity` (implementation effort).

> Prefer one dimension per edge. If a paper claims superiority on several axes vs the same
> baseline, emit **one edge per (dimension)** — this is what makes "N× faster but not more
> robust" legible.

### `status`  — how well the claim holds up (the hardening ladder)
- `self-claimed` — asserted by the `from` authors; not independently checked here.
- `validated` — reproduced/confirmed by an independent study or our benchmark (`assessed_by` says which).
- `qualified` — holds only under stated conditions (regime, energy, mesh, metric); `notes` states them.
- `unvalidated` — insufficient evidence either way.
- `refuted` — contradicted by evidence (`assessed_by`/`notes` cite it).

### Conventions
- A "method" node may be a **baseline family** (e.g. `clamp-filtering`, `projected-newton`,
  `local-global`) even if no single paper "owns" it — needed as a claim target.
- Keep `claim` to the authors' actual assertion; put our judgement in `status` + `notes`.
- Inclusion is not endorsement. Every edge starts `self-claimed` and is *hardened* only with
  cited evidence.
- One edge = one (from, to, dimension). Update `status` in place as evidence arrives; don't
  duplicate.
