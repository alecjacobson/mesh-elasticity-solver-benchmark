"""Energy slot: STABLE Neo-Hookean (Smith, Kim, de Goes 2018), 2D/3D, parameterized by (mu, lam).

    psi(F) = mu/2 (I_C - d) + lam/2 (J - alpha)^2 - mu/2 log(I_C + 1),
    I_C = ||F||^2 >= 0,  J = det F,  alpha = 1 + mu*d / ((d+1) lam)  (rest-stability correction).

Unlike the classical log-barrier Neo-Hookean in energy_neohookean.py (psi = ... - mu log J + ...,
which is +inf for J <= 0), this energy is **finite and smooth for ALL F, including inverted
elements (J <= 0)** -- the log is on I_C+1 (always > 0), not on J. This is the energy the
"Stabler Neo-Hookean / absolute eigenvalue filtering" work is actually built on, and it is the
regime where absolute filtering is designed to help (it lets Newton pass THROUGH inverted states
that the barrier energy forbids). review-r1 #31.

alpha is chosen so F = I is stress-free: at rest grad = [mu*d/(d+1) + lam(1-alpha)] I = 0.
Near-incompressible = large lam (Poisson ratio -> 1/2): lam = 2 mu nu / (1 - 2 nu).

Analytic gradient (inversion-safe cofactor, no matrix inverse); element Hessian by FD of the
gradient (as in energy.py). `make(mu, lam)` returns (element_terms, psi, grad_psi, element_eg).
"""
import numpy as np


def lam_from_nu(nu, mu=1.0):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def _dJdF(F):
    """Cofactor matrix dJ/dF (= J F^-T when F invertible), computed WITHOUT inversion so it is
    valid at and beyond J = 0 -- essential for the inverted-element regime."""
    d = F.shape[0]
    if d == 2:
        return np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            minor = np.delete(np.delete(F, i, 0), j, 1)
            C[i, j] = ((-1.0) ** (i + j)) * np.linalg.det(minor)
    return C


def make(mu=1.0, lam=1.0):
    def _alpha(d):
        return 1.0 + mu * d / ((d + 1.0) * lam)

    def psi(F):
        d = F.shape[0]
        Ic = float(np.sum(F * F))
        J = float(np.linalg.det(F))
        a = _alpha(d)
        return 0.5 * mu * (Ic - d) + 0.5 * lam * (J - a) ** 2 - 0.5 * mu * np.log(Ic + 1.0)

    def grad_psi(F):
        d = F.shape[0]
        Ic = float(np.sum(F * F))
        J = float(np.linalg.det(F))
        a = _alpha(d)
        return mu * F + lam * (J - a) * _dJdF(F) - mu * F / (Ic + 1.0)

    def hess_psi(F, h=1e-6):
        d = F.shape[0]
        n = d * d
        Ff = F.reshape(n).astype(float)
        H = np.zeros((n, n))
        for k in range(n):
            fp = Ff.copy(); fp[k] += h
            fm = Ff.copy(); fm[k] -= h
            H[:, k] = (grad_psi(fp.reshape(d, d)).reshape(n)
                       - grad_psi(fm.reshape(d, d)).reshape(n)) / (2 * h)
        return 0.5 * (H + H.T)

    def element_terms(x_elem, B, area):
        d = 2
        F = (B @ x_elem).reshape(d, d)
        J = float(np.linalg.det(F))
        E = area * psi(F)                                   # FINITE even for J <= 0 (the point)
        g = area * (B.T @ grad_psi(F).reshape(d * d))
        H = area * (B.T @ hess_psi(F) @ B)
        return E, g, H, J

    def element_eg(x_elem, B, area):
        d = 2
        F = (B @ x_elem).reshape(d, d)
        J = float(np.linalg.det(F))
        return area * psi(F), area * (B.T @ grad_psi(F).reshape(d * d)), J

    return element_terms, psi, grad_psi, element_eg
