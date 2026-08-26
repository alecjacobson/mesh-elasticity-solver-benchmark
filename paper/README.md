# Paper — STAR-style survey + benchmark (v1 draft)

A State-of-the-Art Report (Eurographics/CGF norm) that is *also* a benchmark. The punchline is
**honest attribution**, not a leaderboard. Drafted from the design docs (`docs/`), the annotated
corpus, the superiority-claims graph (`claims/`), the measured experiments (`results/`, 30) and the
deterministic figures (`figures/`, 20).

## Format

Drafted as **markdown section files** here, assembled into [`paper.md`](paper.md) (GitHub-renderable,
easy to iterate, every figure/result link resolves in-repo). LaTeX conversion for a journal submission
is a deferred mechanical step — the prose, structure, and evidence are the content that matters now.

Every quantitative claim in the draft traces to a `results/*.md` file (regenerable via `bench/`); the
reference-integrity check in P3.5 verifies this.

## Sections

| file | § | source docs |
|---|---|---|
| [`00-abstract.md`](00-abstract.md) | Abstract | thesis + headline findings |
| [`01-introduction.md`](01-introduction.md) | 1 Introduction | design.md |
| [`02-unifying-view.md`](02-unifying-view.md) | 2 Unifying view | design.md §12.1 |
| [`03-taxonomy.md`](03-taxonomy.md) | 3 Taxonomy + three worlds | taxonomy.md |
| [`04-survey-by-axis.md`](04-survey-by-axis.md) | 4 Survey by axis | corpus.md |
| [`05-lineage.md`](05-lineage.md) | 5 Lineage map | design.md §12.2 |
| [`06-claims-graph.md`](06-claims-graph.md) | 6 Superiority-claims graph | claims/ |
| [`07-benchmark-design.md`](07-benchmark-design.md) | 7 Benchmark design | harness/metrics/protocol.md |
| [`08-results.md`](08-results.md) | 8 Results — decomposition experiments | results/ |
| [`09-what-survived.md`](09-what-survived.md) | 9 What survived + review-loop-as-method | claims/hardening.md |
| [`10-living-benchmark.md`](10-living-benchmark.md) | 10 Open problems + living benchmark | design.md |

## Build scripts

| script | produces | notes |
|---|---|---|
| `python paper/make_bib.py` | `references.bib` + `11-references.md` | derived from `claims/claims.yaml` (citekey = node id); classical ancestors are in the hand-curated `references_classical.bib` |
| `python paper/assemble.py` | `paper.md` | concatenates the section files + TOC + References |
| `python paper/to_latex.py` | `paper.tex` | deterministic md→LaTeX **skeleton** — *not compiled here* (no local toolchain); manual finishing = venue class + inline `\cite` wiring + recompile (documented in the .tex header) |

## Status

Draft in progress (P3). Scope discipline: this is a *v1 snapshot* — 2D prototype measurements,
indicative not definitive; the paper says so at every headline. The contribution is the **method of
honest attribution** and the taxonomy/lineage/claims-graph scaffolding, not a settled leaderboard.
