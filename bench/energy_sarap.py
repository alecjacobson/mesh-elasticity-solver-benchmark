"""Symmetric ARAP distortion energy (Shtengel et al. 2017, eq. 25): the energy CM is designed for.

    psi(F) = (Σ-1)² + (σ⁻¹-1)²      (Σ, σ = max/min singular values of F)

Unlike symmetric Dirichlet (whose density is fully convex in Σ,σ), the σ⁻¹ term of symmetric ARAP is
convex on (0,1.5] and CONCAVE on (1.5,∞) — a genuine convex-concave structure that Composite
Majorization exploits, which is why the CM paper features this energy. Inversion-barrier: +inf at
det F <= 0. Gradient via the singular-value chain rule (df/dF = U diag(dg/dσ_i) Vᵀ).
"""
import numpy as np


def _svd_signed(F):
    """SVD with a rotation-only U,V (det=+1 each) so singular values are correctly oriented."""
    U, s, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1; s = s.copy(); s[-1] *= 1  # keep s>=0; sign folded via V below
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    # ensure U,V are rotations: if det(U)det(V) flipped, the smaller singular value is "signed"
    return U, s, Vt


def psi(F):
    J = np.linalg.det(F)
    if J <= 0.0:
        return np.inf
    s = np.linalg.svd(F, compute_uv=False)
    S, sg = s[0], s[1]
    return (S - 1.0) ** 2 + (1.0 / sg - 1.0) ** 2


def grad_psi(F):
    """dpsi/dF (2x2) via F = U diag(S,σ) Vᵀ, dpsi/dF = U diag(dpsi/dS, dpsi/dσ) Vᵀ."""
    U, s, Vt = np.linalg.svd(F)
    S, sg = s[0], max(s[1], 1e-12)
    dS = 2.0 * (S - 1.0)
    dsg = -2.0 * sg ** -3 + 2.0 * sg ** -2          # d/dσ (σ⁻¹-1)²
    return U @ np.diag([dS, dsg]) @ Vt


def hess_psi(F, h=1e-6):
    Ff = F.reshape(4).astype(float); H = np.zeros((4, 4))
    for k in range(4):
        fp = Ff.copy(); fp[k] += h; fm = Ff.copy(); fm[k] -= h
        H[:, k] = (grad_psi(fp.reshape(2, 2)).reshape(4) - grad_psi(fm.reshape(2, 2)).reshape(4)) / (2 * h)
    return 0.5 * (H + H.T)


def element_terms(x_elem, B, area):
    F = (B @ x_elem).reshape(2, 2)
    J = float(np.linalg.det(F))
    if J <= 0.0:
        return np.inf, None, None, J
    E = area * psi(F)
    g = area * (B.T @ grad_psi(F).reshape(4))
    H = area * (B.T @ hess_psi(F) @ B)
    return E, g, H, J


def element_eg(x_elem, B, area):
    F = (B @ x_elem).reshape(2, 2)
    J = float(np.linalg.det(F))
    if J <= 0.0:
        return np.inf, None, J
    return area * psi(F), area * (B.T @ grad_psi(F).reshape(4)), J
