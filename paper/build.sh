#!/usr/bin/env bash
# Build paper/paper.pdf from the markdown sources. Requires pdflatex + bibtex (TeX Live).
set -e
cd "$(dirname "$0")"
python make_bib.py && python assemble.py && python to_latex.py
rm -f paper.aux paper.bbl paper.blg paper.out paper.toc
pdflatex -interaction=nonstopmode paper.tex >/dev/null
bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdfinfo paper.pdf | grep Pages
echo "built paper/paper.pdf"
