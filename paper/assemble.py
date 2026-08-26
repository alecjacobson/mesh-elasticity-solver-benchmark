"""Assemble the section files into paper/paper.md (with a table of contents).

Section figure links are `../figures/...`, relative to paper/, so they resolve unchanged in the
assembled paper.md (also in paper/). Run: `python paper/assemble.py`.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["00-abstract", "01-introduction", "02-unifying-view", "03-taxonomy", "04-survey-by-axis",
         "05-lineage", "06-claims-graph", "07-benchmark-design", "08-results", "09-what-survived",
         "10-living-benchmark", "11-references"]


def main():
    parts = []
    toc = ["## Contents", ""]
    for name in ORDER:
        path = os.path.join(HERE, name + ".md")
        text = open(path).read().rstrip()
        # collect the top-level heading for the TOC (first '# ' line)
        m = re.search(r"^#\s+(.+)$", text, re.M)
        if m and name != "00-abstract":
            title = m.group(1)
            anchor = re.sub(r"[^a-z0-9 ]", "", title.lower()).replace(" ", "-")
            toc.append(f"- [{title}](#{anchor})")
        parts.append(text)
    body = "\n\n---\n\n".join(parts)
    # insert the TOC right after the abstract (first section)
    head, _, rest = body.partition("\n\n---\n\n")
    out = head + "\n\n---\n\n" + "\n".join(toc) + "\n\n---\n\n" + rest
    with open(os.path.join(HERE, "paper.md"), "w") as f:
        f.write(out + "\n")
    print(f"assembled paper/paper.md ({len(out.splitlines())} lines, {len(ORDER)} sections)")


if __name__ == "__main__":
    main()
