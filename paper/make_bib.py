"""Generate paper/references.bib from the claims graph (claims/claims.yaml).

Each node with a real publication (year > 0, named authors) becomes a BibTeX entry whose citekey is
the node id. DOIs and arXiv ids are parsed from the node's `ref` field. Venue heuristics pick the
entry type. Nodes that are generic World-0 baselines (year 0, author '—'/'classical') are emitted as
@misc only if a concrete source is identifiable, else skipped — they are cited by category, not paper.

This keeps the bibliography a *derived* artifact of the one source of truth (claims.yaml), per the
repo's "one source of truth" rule. Run: `python paper/make_bib.py`.
"""
import os
import re
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CONF = ("SIGGRAPH", "SCA", "SGP", "Eurographics", "CGF", "Symposium", "I3D")
JOURNAL = ("TOG", "CGF", "SIIMS", "TVCG", "Graphical Models")


def _doi(ref):
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", ref)
    if not m:
        return None
    doi = m.group(0)
    # the ref field often has "<doi>; arXiv:..." — a trailing ';' (or '.') breaks doi.org resolution
    return doi.rstrip(".;,")


def _arxiv(ref):
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", ref)
    return m.group(1) if m else None


def _authors_bibtex(a):
    # "Kovalsky, Galun, Lipman" -> "Kovalsky and Galun and Lipman"; drop trailing "et al."/"; ..."
    a = re.split(r";", a)[0].strip()
    a = a.replace(" et al.", "")
    parts = [p.strip() for p in a.split(",") if p.strip()]
    return " and ".join(parts) if parts else a


def _venue_type(venue, ref, arxiv):
    v = venue or ""
    if any(j in v for j in JOURNAL):
        return "article"
    if any(c in v for c in CONF):
        return "inproceedings"
    if arxiv and not _doi(ref):
        return "misc"
    return "article"


def entry(node):
    ref = node.get("ref", "") or ""
    year = node.get("year", 0)
    authors = node.get("authors", "") or ""
    if year <= 0 or authors.strip() in ("—", "", "classical"):
        return None                                   # generic baseline: cited by category, skip
    doi, arxiv = _doi(ref), _arxiv(ref)
    if not (doi or arxiv):
        return None                                   # no locatable source → don't fabricate one
    etype = _venue_type(node.get("venue"), ref, arxiv)
    fields = {
        "author": _authors_bibtex(authors),
        "title": "{" + node["title"].strip() + "}",
        "year": str(year),
    }
    venue = node.get("venue", "")
    if etype == "article":
        fields["journal"] = venue or "ACM Transactions on Graphics"
    elif etype == "inproceedings":
        fields["booktitle"] = "Proc. " + (venue or "SIGGRAPH")
    else:
        fields["howpublished"] = "arXiv"
    if doi:
        fields["doi"] = doi
    if arxiv:
        fields["eprint"] = arxiv
        fields["archivePrefix"] = "arXiv"
    body = ",\n  ".join(f"{k} = {{{v}}}" if not v.startswith("{") else f"{k} = {v}"
                        for k, v in fields.items())
    return f"@{etype}{{{node['id']},\n  {body}\n}}"


def _parse_bib(text):
    """Regex-parse @type{key, field={val}, ...} into dicts (no bibtexparser dependency)."""
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = dict(re.findall(r"(\w+)\s*=\s*\{+(.*?)\}+\s*,?\s*\n", body + "\n"))
        out.append((key, etype, fields))
    return out


def render_md(bib_paths):
    """Human-readable References section (§) from the .bib files, sorted by citekey."""
    items = []
    for p in bib_paths:
        items += _parse_bib(open(p).read())
    items.sort(key=lambda x: x[0])
    lines = ["# References", "",
             "_Derived from `paper/references.bib` (auto-generated from `claims/claims.yaml`) and "
             "`paper/references_classical.bib` (hand-curated classical ancestors). Citekey = "
             "claims-graph node id; formal `\\cite` wiring is in the LaTeX render. "
             f"{len(items)} works._", ""]
    for key, _etype, f in items:
        au = f.get("author", "").replace(" and ", ", ")
        venue = f.get("journal") or f.get("booktitle") or f.get("publisher") or ""
        doi = f.get("doi")
        cite = f"**[{key}]** {au}. *{f.get('title','')}*. {venue} ({f.get('year','')})."
        if doi:
            cite += f" doi:{doi}"
        lines.append(cite + "  ")
    return "\n".join(lines) + "\n"


def main():
    d = yaml.safe_load(open(os.path.join(ROOT, "claims", "claims.yaml")))
    nodes = d["nodes"]
    entries, skipped = [], []
    for n in nodes:
        e = entry(n)
        (entries.append(e) if e else skipped.append(n["id"]))
    header = ("% references.bib — DERIVED from claims/claims.yaml (do not hand-edit; run\n"
              "% `python paper/make_bib.py`). Citekey = claims-graph node id.\n"
              f"% {len(entries)} entries; {len(skipped)} nodes skipped (generic baselines / no locatable source).\n\n")
    with open(os.path.join(HERE, "references.bib"), "w") as f:
        f.write(header + "\n\n".join(entries) + "\n")
    md = render_md([os.path.join(HERE, "references.bib"),
                    os.path.join(HERE, "references_classical.bib")])
    with open(os.path.join(HERE, "11-references.md"), "w") as f:
        f.write(md)
    print(f"wrote paper/references.bib: {len(entries)} entries, {len(skipped)} skipped; "
          f"paper/11-references.md rendered")
    print("skipped:", ", ".join(skipped))
    return entries, skipped


if __name__ == "__main__":
    main()
