"""Analytic eigensystem for the 2D symmetric-Dirichlet Hessian (Smith-de Goes-Kim 2019).

The 4x4 energy Hessian d^2 psi/d vec(F)^2 has 4 closed-form eigenpairs in the SVD F=U S V^T
(twist / flip / two scaling modes). This lets us project (clamp/absolute) analytically without a
numeric eigendecomposition -- the paper's speed claim (analytic-eigensystems -> numeric, ~7x).

`python -m bench.analytic_eig` verifies the analytic eigenvalues match numpy.eigh of the numeric
Hessian and measures the analytic-vs-numeric projection speedup (hardens the edge).
"""
import time
import numpy as np
from .energy import grad_psi, hess_psi


def _eigpairs(F):
    U, s, Vt = np.linalg.svd(F)
    s1, s2 = s
    Lm = np.array([[0.0, -1.0], [1.0, 0.0]])
    Pm = np.array([[0.0, 1.0], [1.0, 0.0]])
    g = lambda a: 2 * a - 2 / a**3          # dpsi/dsigma
    gp = lambda a: 2 + 6 / a**4             # d2psi/dsigma^2
    lt = (g(s1) + g(s2)) / (s1 + s2)                                   # twist
    lf = (g(s1) - g(s2)) / (s1 - s2) if abs(s1 - s2) > 1e-9 else gp(s1)  # flip
    return [
        (lt, (U @ Lm @ Vt) / np.sqrt(2.0)),
        (lf, (U @ Pm @ Vt) / np.sqrt(2.0)),
        (gp(s1), U @ np.diag([1.0, 0.0]) @ Vt),
        (gp(s2), U @ np.diag([0.0, 1.0]) @ Vt),
    ]


def project_analytic(F, kind="clamp", eps=1e-9):
    """Analytically SPD-project the 4x4 symmetric-Dirichlet Hessian (no numeric eigendecomp)."""
    H = np.zeros((4, 4))
    for lam, D in _eigpairs(F):
        lam = max(lam, eps) if kind == "clamp" else max(abs(lam), eps)
        v = D.reshape(4)
        H += lam * np.outer(v, v)
    return H


def project_numeric(F, kind="clamp", eps=1e-9):
    """Numeric reference: eigendecompose the FD Hessian and clamp/abs."""
    w, V = np.linalg.eigh(hess_psi(F))
    w = np.maximum(w, eps) if kind == "clamp" else np.maximum(np.abs(w), eps)
    return (V * w) @ V.T


def run():
    rng = np.random.default_rng(0)
    worst_eig = worst_proj = 0.0
    Fs = []
    for _ in range(300):
        F = np.eye(2) + 0.3 * rng.standard_normal((2, 2))
        if np.linalg.det(F) <= 0.2:
            continue
        Fs.append(F)
        w_num = np.sort(np.linalg.eigvalsh(hess_psi(F)))
        w_ana = np.sort([lam for lam, _ in _eigpairs(F)])
        worst_eig = max(worst_eig, np.max(np.abs(w_num - w_ana)) / (np.max(np.abs(w_num)) + 1e-12))
        for kind in ("clamp", "absolute"):
            Pa, Pn = project_analytic(F, kind), project_numeric(F, kind)
            worst_proj = max(worst_proj, np.max(np.abs(Pa - Pn)) / (np.max(np.abs(Pn)) + 1e-12))
    # timing
    t = time.perf_counter()
    for F in Fs * 20:
        project_analytic(F, "clamp")
    t_ana = time.perf_counter() - t
    t = time.perf_counter()
    for F in Fs * 20:
        project_numeric(F, "clamp")
    t_num = time.perf_counter() - t
    ok = worst_eig < 1e-6 and worst_proj < 1e-5
    print(f"[analytic-eig] eigenvalues vs eigh:   max rel err {worst_eig:.2e} -> {'PASS' if worst_eig<1e-6 else 'FAIL'}")
    print(f"[analytic-eig] projection vs numeric: max rel err {worst_proj:.2e} -> {'PASS' if worst_proj<1e-5 else 'FAIL'}")
    print(f"[analytic-eig] speed: analytic {t_ana*1e3:.0f}ms vs numeric-FD {t_num*1e3:.0f}ms "
          f"-> {t_num/t_ana:.1f}x")
    print(f"[analytic-eig] {'ALL PASS' if ok else 'FAILED'}")
    return ok, t_num / t_ana


if __name__ == "__main__":
    import sys
    ok, _ = run()
    sys.exit(0 if ok else 1)
