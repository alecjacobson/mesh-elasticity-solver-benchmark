"""P2 element with SELECTIVE REDUCED INTEGRATION (SRI) — a genuinely locking-relieving element,
in DISPLACEMENT form so eigenvalue filtering (clamp/absolute) stays well-posed (review-r3 #74).

Volumetric locking near ν→½ comes from the volumetric energy being over-constrained by full
quadrature. SRI integrates the DEVIATORIC part at full (3-point) quadrature and the VOLUMETRIC part
at reduced (1-point centroid) quadrature — the classic locking cure (Malkus-Hughes 1978). Unlike a
mixed u-p (Taylor-Hood) element, SRI stays purely in displacement DOFs, so the projected-Newton
eigenvalue filters apply exactly as on the standard element and the absolute-vs-clamp comparison is
still a single-axis swap.

We build it on the CLASSICAL Neo-Hookean split ψ = ψ_dev + ψ_vol with
    ψ_dev(F) = μ/2 (‖F‖² − 2) − μ log J     (grad_dev(I) = μI − μI = 0),
    ψ_vol(F) = λ/2 (log J)²                  (grad_vol(I) = 0),
because BOTH parts are rest-stress-free individually — so integrating them at different quadrature
does NOT break rest equilibrium. (Stable-NH's parts are NOT individually rest-stress-free — its
α-correction entangles them — so naive SRI there would introduce spurious rest stress; classical NH
avoids that.) Valid for J>0 (the near-incompressible stretch sweep stays inversion-free).

Conformance-gated: `python -m bench.p2_sri` runs an FD gradient check.
"""
import numpy as np
from .p2 import _dNref, _edofs, grid_mesh_p2, _QP, _QW


def _Bmat(dNdX):
    """vec(F) = B @ x_elem (12,), F = x_elem^T-reshaped @ dNdX."""
    B = np.zeros((4, 12))
    for d in range(12):
        ue = np.zeros(12); ue[d] = 1.0
        B[:, d] = (ue.reshape(6, 2).T @ dNdX).reshape(4)
    return B


def rest_quantities_sri(rest, elems):
    """Per element: (dev_quad = [(B,w)]×3 full points, vol_B = centroid B, vol_w = element volume)."""
    out = []
    for e in elems:
        Xr = rest[e]
        dev = []
        for (r, s), w in zip(_QP, _QW):
            dNref = _dNref(r, s); Jref = Xr.T @ dNref
            dNdX = dNref @ np.linalg.inv(Jref)
            dev.append((_Bmat(dNdX), w * abs(np.linalg.det(Jref))))
        # reduced (centroid) point for the volumetric term
        dNref_c = _dNref(1 / 3, 1 / 3); Jref_c = Xr.T @ dNref_c
        vol_B = _Bmat(dNref_c @ np.linalg.inv(Jref_c))
        vol_w = sum(w for _, w in dev)                     # element volume = ∫ dV
        out.append((dev, vol_B, vol_w))
    return out


def _FinvT(F):
    a, b, c, d = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
    det = a * d - b * c
    return np.array([[d, -c], [-b, a]]) / det              # (F^-1)^T for 2x2


def make_sri_terms(mu=1.0, lam=1.0):
    def gdev(F):                                            # d/dF [μ/2‖F‖² − μ logJ]
        return mu * F - mu * _FinvT(F)

    def gvol(F):                                            # d/dF [λ/2 (logJ)²]
        J = np.linalg.det(F)
        return lam * np.log(J) * _FinvT(F)

    def _fd_h(g, F, h=1e-6):
        Ff = F.reshape(4); H = np.zeros((4, 4))
        for k in range(4):
            fp = Ff.copy(); fp[k] += h; fm = Ff.copy(); fm[k] -= h
            H[:, k] = (g(fp.reshape(2, 2)).reshape(4) - g(fm.reshape(2, 2)).reshape(4)) / (2 * h)
        return 0.5 * (H + H.T)

    def psi_dev(F):
        return 0.5 * mu * (float(np.sum(F * F)) - 2.0) - mu * np.log(np.linalg.det(F))

    def psi_vol(F):
        return 0.5 * lam * np.log(np.linalg.det(F)) ** 2

    def element_terms(x_elem, quaddata):
        dev, vol_B, vol_w = quaddata
        E = 0.0; g = np.zeros(12); H = np.zeros((12, 12)); minJ = np.inf
        for B, w in dev:                                   # deviatoric: full 3-point quadrature
            F = (B @ x_elem).reshape(2, 2); J = float(np.linalg.det(F)); minJ = min(minJ, J)
            if J <= 0.0:
                return np.inf, None, None, J
            E += w * psi_dev(F)
            g += w * (B.T @ gdev(F).reshape(4))
            H += w * (B.T @ _fd_h(gdev, F) @ B)
        Fc = (vol_B @ x_elem).reshape(2, 2); Jc = float(np.linalg.det(Fc))  # volumetric: 1-point centroid
        if Jc <= 0.0:
            return np.inf, None, None, Jc
        E += vol_w * psi_vol(Fc)
        g += vol_w * (vol_B.T @ gvol(Fc).reshape(4))
        H += vol_w * (vol_B.T @ _fd_h(gvol, Fc) @ vol_B)
        return E, g, H, min(minJ, Jc)

    return element_terms


def _conformance(seed=0, h=1e-6):
    """FD gradient check of the SRI element on a small mesh (grounding gate)."""
    rng = np.random.default_rng(seed)
    nodes, elems = grid_mesh_p2(3, 3)
    quad = rest_quantities_sri(nodes, elems)
    et = make_sri_terms(mu=1.0, lam=5.0)
    x = (nodes + 0.02 * rng.standard_normal(nodes.shape)).reshape(-1)

    def energy(xv):
        E = 0.0
        for t, e in enumerate(elems):
            Ee, *_ = et(xv[_edofs(e)], quad[t])
            if not np.isfinite(Ee):
                return np.inf
            E += Ee
        return E

    # assembled gradient vs FD
    nv = nodes.shape[0]; g = np.zeros(2 * nv)
    for t, e in enumerate(elems):
        _, ge, _, _ = et(x[_edofs(e)], quad[t]); g[_edofs(e)] += ge
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h; xm = x.copy(); xm[k] -= h
        gfd[k] = (energy(xp) - energy(xm)) / (2 * h)
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    # rest state must be stress-free (SRI preserves it because both parts are individually so)
    xr = nodes.reshape(-1); gr = np.zeros(2 * nv)
    for t, e in enumerate(elems):
        _, ge, _, _ = et(xr[_edofs(e)], quad[t]); gr[_edofs(e)] += ge
    rest = float(np.max(np.abs(gr)))
    return rel, rest


if __name__ == "__main__":
    import sys
    rel, rest = _conformance()
    ok = rel < 1e-5 and rest < 1e-8
    print(f"[p2-sri] assembled grad vs FD: {rel:.2e}  |  rest |grad|: {rest:.2e}  -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
