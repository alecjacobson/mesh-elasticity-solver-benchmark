# Contributing (humans & agents)

This repo is built by a mix of human curation and LLM agents. Keep everything **plain
Markdown / YAML** so it renders on GitHub and stays diff-friendly.

## Ground rules

1. **Agents extract and organize; they do not invent numerics.** LLM agents are used for
   corpus breadth, self-claim extraction, and cross-checking — never to generate solver
   implementations or fabricate benchmark numbers.
2. **Cite or flag.** Every factual claim carries a source, or a `[?]` uncertainty flag. Do not
   silently upgrade a `[?]`; verify it, then remove the flag in the same commit.
3. **Inclusion ≠ endorsement.** The claims graph records *authors' assertions*. A claim is
   `self-claimed` until independently assessed; harden status only with cited evidence.
4. **One source of truth.** Machine-readable data lives in structured files
   (`claims/claims.yaml`); human-readable renders (`claims/README.md`) are derived — keep them
   in sync in the same commit.

## Workflow

- Work is tracked in **GitHub issues**. Reference the issue in the commit subject or body
  (`... (#4)` / `Closes #4`).
- Branch off `main` for non-trivial work; small doc fixes can go straight to `main`.
- Commit messages: imperative subject, short body explaining *why*. Machine commits should be
  attributed to their tool via a trailer.

## File conventions

| Kind | Where | Format |
|---|---|---|
| Survey / design prose | `docs/*.md` | Markdown, GitHub-flavored |
| Corpus (annotated papers) | `docs/corpus.md` | Markdown, one line per entry with axis/verdict tags |
| Claims graph data | `claims/claims.yaml` | YAML per [`claims/schema.md`](claims/schema.md) |
| Claims graph render | `claims/README.md` | Markdown + Mermaid |
| Diagrams | inline | Mermaid fenced blocks (```` ```mermaid ````) — GitHub renders them |

## Adding a paper

1. Add a one-line tagged entry to the right World section of `docs/corpus.md`.
2. If it makes a superiority claim, add a node + one edge **per (target, dimension)** to
   `claims/claims.yaml`, then reflect it in `claims/README.md`.
3. Carry any uncertainty as `[?]`.

## Corpus entry tag legend

Axes: `E` energy · `S` search-direction · `H` Hessian/eigenvalue-filtering · `L`
line-search/feasibility · `LS` linear-solver/preconditioner · `C` convergence-criterion ·
`A` acceleration. Verdicts (World 0): `BASE` include-as-baseline · `REL` include-as-related
(ancestor) · `EXCL` exclude-with-justification.
