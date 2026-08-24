"""Energy slot: 2D symmetric Dirichlet distortion energy.

psi(F) = ||F||_F^2 * (1 + 1/det(F)^2)  -> +inf as det F -> 0 (inversion barrier).

Element gradient uses the analytic dpsi/dF; the element Hessian of psi is formed by
finite-differencing that analytic gradient (cheap, 4x4 per element) and is verified against
FD-of-energy in bench/conformance.py -- the grounding test that stands in for official-code
regression until an official reference is ported (D3 / harness.md conformance suite).
"""
import numpy as np


def psi(F):
    J = np.linalg.det(F)
    if J <= 0.0:
        return np.inf
    I1 = float(np.sum(F * F))
    return I1 * (1.0 + 1.0 / (J * J))


def grad_psi(F):
    """Analytic dpsi/dF (2x2). d/dF [ I1 + I1 J^-2 ] with dJ/dF = J F^-T."""
    J = np.linalg.det(F)
    I1 = float(np.sum(F * F))
    Finv = np.linalg.inv(F)
    return 2.0 * F * (1.0 + 1.0 / (J * J)) - 2.0 * I1 / (J * J) * Finv.T


def hess_psi(F, h=1e-6):
    """4x4 Hessian of psi wrt vec(F), by central FD of the analytic gradient."""
    Ff = F.reshape(4).astype(float)
    H = np.zeros((4, 4))
    for k in range(4):
        fp = Ff.copy(); fp[k] += h
        fm = Ff.copy(); fm[k] -= h
        gp = grad_psi(fp.reshape(2, 2)).reshape(4)
        gm = grad_psi(fm.reshape(2, 2)).reshape(4)
        H[:, k] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)


def element_B(Minv):
    """Linear map B (4x6): vec(F) = B @ x_elem, x_elem = [x0x,x0y,x1x,x1y,x2x,x2y].

    F = Ds Minv with Ds = [x1-x0, x2-x0]; F is linear in x_elem and F(0)=0, so the
    columns of B are just vec(F(e_d)) for unit dofs e_d.
    """
    B = np.zeros((4, 6))
    for d in range(6):
        e = np.zeros(6); e[d] = 1.0
        xt = e.reshape(3, 2)
        Ds = np.column_stack((xt[1] - xt[0], xt[2] - xt[0]))
        B[:, d] = (Ds @ Minv).reshape(4)
    return B


def element_terms(x_elem, B, area):
    """Return (E, g(6), H(6x6), detF) for one triangle. E=inf, others None if inverted."""
    F = (B @ x_elem).reshape(2, 2)
    J = float(np.linalg.det(F))
    if J <= 0.0:
        return np.inf, None, None, J
    E = area * psi(F)
    g = area * (B.T @ grad_psi(F).reshape(4))
    H = area * (B.T @ hess_psi(F) @ B)
    return E, g, H, J
