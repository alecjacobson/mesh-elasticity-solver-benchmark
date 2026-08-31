"""Deterministic markdown -> LaTeX converter, targeting the Eurographics/CGF STAR class (egpubl).

Produces paper/paper.tex from the section files; `bash paper/build.sh` compiles it to paper/paper.pdf
(pdflatex + bibtex + eg-alpha-doi.bst). The remaining manual pre-submission steps are wiring inline
\\cite (the markdown carries no citation markers yet — \\nocite{*} lists the full bibliography for
now) and filling \\author/\\teaser.

Handles the constructs this paper uses: ATX headings (leading manual numbers stripped — the class
numbers), bold/italic/inline-code, images -> figure, pipe tables -> tabular, itemize, verbatim code
(unicode ASCII-transliterated), links -> text, and a unicode -> LaTeX map for the maths glyphs
(‖ σ λ ν × ⇐ → ∇ ² ≤ …). Run: `python paper/to_latex.py`.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["01-introduction", "02-unifying-view", "03-taxonomy", "04-survey-by-axis", "05-lineage",
         "06-claims-graph", "07-benchmark-design", "08-results", "09-what-survived",
         "10-living-benchmark"]

UNI = {
    "‖": r"\textbar\textbar ", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "⁻¹": r"\textsuperscript{-1}",
    "×": r"$\times$", "⇐": r"$\Leftarrow$", "→": r"$\rightarrow$", "↔": r"$\leftrightarrow$",
    "∇": r"$\nabla$", "≤": r"$\le$", "≥": r"$\ge$", "≈": r"$\approx$", "≠": r"$\ne$", "√": r"$\sqrt{}$",
    "σ": r"$\sigma$", "λ": r"$\lambda$", "ν": r"$\nu$", "ε": r"$\epsilon$", "α": r"$\alpha$",
    "μ": r"$\mu$", "ψ": r"$\psi$", "ρ": r"$\rho$", "κ": r"$\kappa$", "τ": r"$\tau$", "δ": r"$\delta$",
    "π": r"$\pi$", "φ": r"$\phi$", "Σ": r"$\Sigma$", "½": r"$\tfrac12$", "…": r"\dots",
    "⭐": "", "—": "---", "–": "--", "‑": "-", "’": "'", "“": "``", "”": "''",
    "₁": r"\textsubscript{1}", "₂": r"\textsubscript{2}", "ᵢ": r"\textsubscript{i}",
    "θ": r"$\theta$", "·": r"$\cdot$", "∞": r"$\infty$", "∝": r"$\propto$", "†": r"$\dagger$",
    "§": r"\S", "Δ": r"$\Delta$", "−": r"$-$", "¹": r"\textsuperscript{1}", "é": r"\'{e}",
    "⁻": r"\textsuperscript{-}", "̂": "", "∈": r"$\in$", "∉": r"$\notin$", "γ": r"$\gamma$",
    "β": r"$\beta$", "≫": r"$\gg$", "∂": r"$\partial$", "⇔": r"$\Leftrightarrow$",
}


# ASCII transliteration for verbatim/code blocks (inputenc utf8 cannot render utf8 in verbatim).
VERB = {"α": "alpha", "β": "beta", "γ": "gamma", "ε": "eps", "λ": "lambda", "μ": "mu", "ν": "nu",
        "σ": "sigma", "ψ": "psi", "ρ": "rho", "τ": "tau", "δ": "delta", "π": "pi", "φ": "phi",
        "θ": "theta", "Σ": "Sum", "∇": "grad", "−": "-", "·": "*", "×": "x", "⁻¹": "^-1", "²": "^2",
        "³": "^3", "‖": "||", "≤": "<=", "≥": ">=", "≈": "~=", "→": "->", "⇐": "<=", "√": "sqrt",
        "∈": " in ", "½": "1/2", "∞": "inf", "≠": "!=", "∝": "~", "↔": "<->", "⇔": "<=>", "…": "...",
        "d̂": "dhat", "̂": "", "Δ": "Delta", "₁": "1", "₂": "2", "ᵢ": "i", "§": "S", "’": "'"}


def _denum(s):
    """Strip a leading manual section number ('1. ', '2.1 ', '8.1 ', '12.2 ') — the EG class numbers
    sections itself, so keeping the literal number would double it ('1. 1. Introduction')."""
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", s)


def deverb(s):
    for k, v in VERB.items():
        s = s.replace(k, v)
    return "".join(c if ord(c) < 128 else "?" for c in s)   # any stray non-ascii -> '?'


def esc(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    # escape LaTeX specials once (inline() leaves them raw so this runs exactly once per line)
    for ch in ("&", "%", "#", "_"):
        s = s.replace(ch, "\\" + ch)
    s = s.replace("^", r"\textasciicircum{}").replace("~", r"\textasciitilde{}")
    return s


def inline(s):
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)                       # images handled at block level
    s = re.sub(r"\[cite:([a-z0-9,\-]+)\]", r"\\cite{\1}", s)         # [cite:key] / [cite:k1,k2] -> \cite
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)                   # links -> text
    s = re.sub(r"`([^`]+)`", lambda m: "\x00" + m.group(1) + "\x01", s)   # protect code spans
    s = re.sub(r'"([^"]*)"', r"``\1''", s)                                # straight quotes -> ``''
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    s = esc(s)
    s = s.replace("\x00", r"\texttt{").replace("\x01", "}")               # restore code (post-esc)
    return s


def convert_block(md):
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^#\s", ln):
            out.append(r"\section{" + inline(_denum(ln[2:].strip())) + "}")
        elif re.match(r"^##\s", ln):
            out.append(r"\subsection{" + inline(_denum(ln[3:].strip())) + "}")
        elif re.match(r"^###\s", ln):
            out.append(r"\subsubsection{" + inline(_denum(ln[4:].strip())) + "}")
        elif ln.strip().startswith("$$") and ln.strip().endswith("$$") and len(ln.strip()) > 4:
            eq = ln.strip()[2:-2].strip()                            # display math: pass through RAW
            out += [r"\begin{equation}", "  " + eq, r"\end{equation}"]
        elif ln.strip() == "---":
            pass
        elif ln.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            cap, path = m.group(1), os.path.basename(m.group(2))
            # figure* spans both columns (all our figures are wide multi-panel plots)
            out += [r"\begin{figure*}[t]\centering",
                    r"  \includegraphics[width=\textwidth]{" + path + "}",
                    r"  \caption{" + inline(cap) + "}", r"\end{figure*}"]
        elif ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[-:| ]+\|", lines[i + 1]):
            rows, j = [], i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j]); j += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            ncol = len(cells[0]); header = cells[0]; body = cells[2:]
            if ncol >= 5:
                # wide table: full-width tabularx with wrapping (ragged) columns so long cells wrap
                spec = "l" + "Y" * (ncol - 1)
                out.append(r"\begin{table*}[t]\centering\footnotesize"
                           r"\begin{tabularx}{\textwidth}{" + spec + "}")
                out.append(r"\toprule " + " & ".join(inline(h) for h in header) + r" \\ \midrule")
                for row in body:
                    out.append(" & ".join(inline(c) for c in (row + [""] * ncol)[:ncol]) + r" \\")
                out.append(r"\bottomrule\end{tabularx}\end{table*}")
            else:
                out.append(r"\begin{table*}[t]\centering\footnotesize\begin{tabular}{" + "l" * ncol + "}")
                out.append(r"\toprule " + " & ".join(inline(h) for h in header) + r" \\ \midrule")
                for row in body:
                    out.append(" & ".join(inline(c) for c in (row + [""] * ncol)[:ncol]) + r" \\")
                out.append(r"\bottomrule\end{tabular}\end{table*}")
            i = j
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(r"  \item " + inline(lines[i][2:])); i += 1
            out += [r"\begin{itemize}"] + items + [r"\end{itemize}"]
            continue
        elif ln.strip() == "":
            out.append("")
        elif ln.startswith("```"):
            out.append(r"\begin{verbatim}"); i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(deverb(lines[i])); i += 1
            out.append(r"\end{verbatim}")
        else:
            out.append(inline(ln))
        i += 1
    return "\n".join(out)


PREAMBLE = r"""%% paper.tex -- GENERATED by paper/to_latex.py from the markdown sections.
%% Eurographics / Computer Graphics Forum STAR format (egpubl class, vendored in paper/).
%% Build: bash paper/build.sh  (pdflatex + bibtex + eg-alpha-doi.bst).
%% Manual pre-submission polish: fill \author / \teaser, drop in the editor's journal metadata.
\documentclass{egpubl}
\STAR
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx,amsmath,amssymb,booktabs}
\usepackage{tabularx}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage{hyperref}
\graphicspath{{../figures/}}
\title[Untangling a Decade of Mesh-Elasticity Solvers]%
      {Untangling a Decade of Mesh-Elasticity Solvers:\\ A Component-Factored Survey and Benchmark}
\author[Anonymous]{Anonymous submission}
\begin{document}
\maketitle
"""


def main():
    abstract = open(os.path.join(HERE, "00-abstract.md")).read()
    m = re.search(r"## Abstract\s*(.+)", abstract, re.S)
    abs_body = "\n\n".join(inline(p) for p in m.group(1).strip().split("\n\n")) if m else ""
    body = "\n\n".join(convert_block(open(os.path.join(HERE, n + ".md")).read()) for n in ORDER)
    tex = (PREAMBLE + r"\begin{abstract}" + "\n" + abs_body + "\n" + r"\end{abstract}" + "\n\n"
           + body + "\n\n"
           + r"\nocite{*}" + "\n"
           + r"\bibliographystyle{eg-alpha-doi}" + "\n"
           + r"\bibliography{references,references_classical}" + "\n"
           + r"\end{document}" + "\n")
    with open(os.path.join(HERE, "paper.tex"), "w") as f:
        f.write(tex)
    print(f"wrote paper/paper.tex ({len(tex.splitlines())} lines) (CGF STAR / egpubl); build with paper/build.sh")


if __name__ == "__main__":
    main()
