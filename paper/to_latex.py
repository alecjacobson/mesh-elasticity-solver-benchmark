"""Deterministic markdown -> LaTeX converter for the STAR draft (submission skeleton).

Produces paper/paper.tex from the section files. There is NO local LaTeX toolchain here, so paper.tex
is NOT compiled/verified — it is a deterministic, inspectable *starting point* for a journal
submission, not a finished camera-ready. The manual finishing steps (documented at the top of the
generated .tex) are: pick the venue class (e.g. eg-article / CGF), wire inline \\cite commands (the
markdown carries no citation markers yet — \\nocite{*} lists the full bibliography for now), and
verify tables/math after compilation.

Handles the constructs this paper actually uses: ATX headings, bold/italic/inline-code, images ->
figure, pipe tables -> tabular, horizontal rules, links -> text, and a unicode -> LaTeX map for the
maths glyphs (‖ σ λ ν × ⇐ → ∇ ² ≤ …). Run: `python paper/to_latex.py`.
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
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)                   # links -> text
    s = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1) + "}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    return esc(s)


def convert_block(md):
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^#\s", ln):
            out.append(r"\section{" + inline(ln[2:].strip()) + "}")
        elif re.match(r"^##\s", ln):
            out.append(r"\subsection{" + inline(ln[3:].strip()) + "}")
        elif re.match(r"^###\s", ln):
            out.append(r"\subsubsection{" + inline(ln[4:].strip()) + "}")
        elif ln.strip() == "---":
            pass
        elif ln.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            cap, path = m.group(1), os.path.basename(m.group(2))
            out += [r"\begin{figure}[t]\centering",
                    r"  \includegraphics[width=\linewidth]{" + path + "}",
                    r"  \caption{" + inline(cap) + "}", r"\end{figure}"]
        elif ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[-:| ]+\|", lines[i + 1]):
            rows, j = [], i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j]); j += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            ncol = len(cells[0]); header = cells[0]; body = cells[2:]
            out.append(r"\begin{table}[t]\centering\small\begin{tabular}{" + "l" * ncol + "}")
            out.append(r"\hline " + " & ".join(inline(h) for h in header) + r" \\ \hline")
            for row in body:
                out.append(" & ".join(inline(c) for c in (row + [""] * ncol)[:ncol]) + r" \\")
            out.append(r"\hline\end{tabular}\end{table}")
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
%% NOT compiled/verified here (no local LaTeX toolchain). This is a deterministic submission
%% SKELETON. Manual finishing steps before submission:
%%   1. Replace `article` with the venue class (e.g. eg-article.cls / CGF style).
%%   2. Wire inline \cite{key} commands -- the markdown carries no citation markers yet, so
%%      \nocite{*} below lists the full bibliography as a placeholder.
%%   3. Recompile and fix any table/math artifacts the converter could not resolve.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,amsmath,amssymb,hyperref,booktabs,textcomp}
\usepackage[margin=1in]{geometry}
\graphicspath{{../figures/}}
\title{Untangling a Decade of Mesh-Elasticity Solvers:\\ A Component-Factored Survey and Benchmark}
\author{(authors)}
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
           + r"\bibliographystyle{plain}" + "\n"
           + r"\bibliography{references,references_classical}" + "\n"
           + r"\end{document}" + "\n")
    with open(os.path.join(HERE, "paper.tex"), "w") as f:
        f.write(tex)
    print(f"wrote paper/paper.tex ({len(tex.splitlines())} lines) -- NOT compiled here; skeleton only")


if __name__ == "__main__":
    main()
