"""Energy slot: 2D Neo-Hookean with a volumetric term (plane-strain), parameterized by (mu, lam).

psi(F) = mu/2 (I1 - 2) - mu log J + lam/2 (log J)^2,   I1 = ||F||^2,  J = det F > 0.
Near-incompressible = large lam (Poisson ratio -> 1/2): lam = 2 mu nu / (1 - 2 nu).

This is the 1b cell used to probe the *headline* absolute-vs-clamp claim in the regime the
Stabler-Neo-Hookean paper targets (high nu + large deformation). Analytic gradient; element
Hessian by FD of the gradient (as in energy.py). `make(mu, lam)` returns (element_terms, psi).
"""
import numpy as np


def lam_from_nu(nu, mu=1.0):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def make(mu=1.0, lam=1.0):
    def psi(F):
        J = np.linalg.det(F)
        if J <= 0.0:
            return np.inf
        I1 = float(np.sum(F * F))
        lnJ = np.log(J)
        return 0.5 * mu * (I1 - 2.0) - mu * lnJ + 0.5 * lam * lnJ * lnJ

    def grad_psi(F):
        J = np.linalg.det(F)
        lnJ = np.log(J)
        FinvT = np.linalg.inv(F).T
        return mu * F + (lam * lnJ - mu) * FinvT

    def hess_psi(F, h=1e-6):
        Ff = F.reshape(4).astype(float)
        H = np.zeros((4, 4))
        for k in range(4):
            fp = Ff.copy(); fp[k] += h
            fm = Ff.copy(); fm[k] -= h
            H[:, k] = (grad_psi(fp.reshape(2, 2)).reshape(4)
                       - grad_psi(fm.reshape(2, 2)).reshape(4)) / (2 * h)
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

    return element_terms, psi, grad_psi, element_eg
