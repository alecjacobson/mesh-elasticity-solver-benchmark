"""Composite Majorization (Shtengel, Poranne, Sorkine-Hornung, Kovalsky, Lipman, SIGGRAPH 2017).

A FAITHFUL implementation of the CM convex-approximate-Hessian for the 2D symmetric-Dirichlet
distortion energy, following the paper's construction (this closes the deliberately-unfaked #14).

For a distortion energy  f(x) = Σ_i h(Σ_i, σ_i)·area_i  with Σ,σ the singular values of the
per-element Jacobian, the paper writes it as Σ_i h_i∘g_i and gives the PSD "majorizer Hessian"
(their eq. 9):

    H = Jg^T ∇²h⁺ Jg  +  Σ_j [ (∂h/∂u_j)_+ ∇²g⁺_j  +  (∂h/∂u_j)_- ∇²g⁻_j ]

For symmetric Dirichlet  h(u,v)=u²+u⁻²+v²+v⁻²  (convex on u,v>0, so h⁺=h), and (eqs. 21–24)
    Σ = α+β,  σ = α−β,     α = ½‖(a+d, c−b)‖,  β = ½‖(a−d, c+b)‖    (J=[[a,b],[c,d]] entries)
with the convex–concave decomposition g⁺=(α+β, α), g⁻=(0, −β) (α, β are norms of linear
functions of x, hence convex). The Hessian of a norm ½‖w(x)‖ (w linear) is ½ Bᵀ(I−ŵŵᵀ)B/‖w‖ (PSD).

The paper's Proposition 3.1 guarantees H ⪰ 0 and H ⪰ ∇²f(x₀); the osculating quadric built from H
is a genuine convex majorizer of f over a neighbourhood — verified in `_conformance` below (this is
what distinguishes CM from a mere local eigenvalue clamp, whose quadric majorizes only infinitesimally).

Used as the `filt="composite-majorization"` Hessian in bench/solver.py (a modified-Newton with the
CM Hessian; the true gradient drives the step, so it converges to the same minimum as clamp-Newton).
"""
import numpy as np


def singular_values_ab(F):
    """Σ, σ (max/min singular value) via the similarity/anti-similarity magnitudes α, β
    (Chien et al. 2016): Σ=α+β, σ=α−β. Returns (Sigma, sigma, alpha, beta, p, q, r, s)."""
    a, b, c, d = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
    p, q = a + d, c - b            # similarity part (a+d, c-b)
    r, s = a - d, c + b            # anti-similarity part (a-d, c+b)
    alpha = 0.5 * np.hypot(p, q)
    beta = 0.5 * np.hypot(r, s)
    return alpha + beta, alpha - beta, alpha, beta, p, q, r, s


def _g_components(F, B, eps=1e-9):
    """The energy-INDEPENDENT pieces of the CM Hessian: (Σ, σ, Jg, ∇²α, ∇²β). ∇²α, ∇²β are PSD
    (Hessians of the similarity/anti-similarity norms); Jg = ∂(Σ,σ)/∂x."""
    Sig, sig, alpha, beta, p, q, r, s = singular_values_ab(F)
    dp = B[0] + B[3]; dq = B[2] - B[1]; dr = B[0] - B[3]; ds = B[2] + B[1]
    Ba = np.vstack([dp, dq]); Bb = np.vstack([dr, ds])
    na = max(2.0 * alpha, eps); nb = max(2.0 * beta, eps)
    wa = np.array([p, q]) / na; wb = np.array([r, s]) / nb
    I2 = np.eye(2)
    Ha = 0.5 * Ba.T @ (I2 - np.outer(wa, wa)) @ Ba / na
    Hb = 0.5 * Bb.T @ (I2 - np.outer(wb, wb)) @ Bb / nb
    gA = 0.5 * (wa @ Ba); gB = 0.5 * (wb @ Bb)
    Jg = np.vstack([gA + gB, gA - gB])
    return Sig, sig, Jg, Ha, Hb


def _components(F, B, eps=1e-9):
    """Symmetric-Dirichlet h-derivatives + the shared g-pieces (h(u,v)=u²+u⁻²+v²+v⁻², fully convex)."""
    Sig, sig, Jg, Ha, Hb = _g_components(F, B, eps)
    sg = sig if abs(sig) > eps else (eps if sig >= 0 else -eps)
    dhS = 2.0 * Sig - 2.0 / Sig ** 3
    dhs = 2.0 * sg - 2.0 / sg ** 3
    Hh = np.diag([2.0 + 6.0 / Sig ** 4, 2.0 + 6.0 / sg ** 4])          # ∇²h⁺ = ∇²h (h convex)
    return Sig, sig, dhS, dhs, Hh, Jg, Ha, Hb


def _cm_from_parts(dhS, dhs, HhPlus, Jg, Ha, Hb, area):
    """Assemble the CM Hessian (eq. 9) from the energy's ∂h/∂Σ, ∂h/∂σ, ∇²h⁺ and the g-pieces."""
    H = (Jg.T @ HhPlus @ Jg
         + max(dhS, 0.0) * (Ha + Hb)          # (∂h/∂Σ)_+ ∇²(α+β)   [g⁺_Σ = α+β]
         + max(dhs, 0.0) * Ha                 # (∂h/∂σ)_+ ∇²α       [g⁺_σ = α]
         + min(dhs, 0.0) * (-Hb))             # (∂h/∂σ)_- ∇²(−β)    [g⁻_σ = −β]
    return area * 0.5 * (H + H.T)


def cm_element_hessian_sarap(F, B, area, eps=1e-9):
    """CM Hessian for SYMMETRIC ARAP  h(u,v)=(u-1)²+(v⁻¹-1)²  (eq. 25). The v⁻¹ term is convex on
    (0,1.5] and concave on (1.5,∞); its Hessian split is ∇²h⁺_vv = φ''(v) if v≤1.5 else 0, with
    φ''(v)=6v⁻⁴-4v⁻³ (the paper permits forming H from the Hessian split alone, §3.3)."""
    Sig, sig, Jg, Ha, Hb = _g_components(F, B, eps)
    sg = sig if sig > eps else eps
    dhS = 2.0 * (Sig - 1.0)
    dhs = -2.0 * sg ** -3 + 2.0 * sg ** -2                 # ∂h/∂σ = φ'(σ)
    phi2 = 6.0 * sg ** -4 - 4.0 * sg ** -3                 # φ''(σ)
    HhPlus = np.diag([2.0, phi2 if sg <= 1.5 else 0.0])    # convex part of ∇²h
    return _cm_from_parts(dhS, dhs, HhPlus, Jg, Ha, Hb, area)


def analytic_element_hessian(F, B, area):
    """EXACT (analytic) element Hessian of area·psi(F) via the α,β / singular-value chain rule --
    a byproduct of the CM construction, more accurate than the FD hess_psi at extreme singular
    values. ∇²f = Jgᵀ∇²h Jg + (∂h/∂Σ)∇²Σ + (∂h/∂σ)∇²σ, ∇²Σ=∇²α+∇²β, ∇²σ=∇²α−∇²β."""
    _, _, dhS, dhs, Hh, Jg, Ha, Hb = _components(F, B)
    H = Jg.T @ Hh @ Jg + dhS * (Ha + Hb) + dhs * (Ha - Hb)
    return area * 0.5 * (H + H.T)


def cm_element_hessian(F, B, area, eps=1e-9):
    """The 6×6 Composite-Majorization Hessian for one triangle (paper eq. 9), symmetric Dirichlet.
    B (4×6): vec(F)=B·x_elem, rows = ∂[a,b,c,d]/∂x_elem. PSD by construction, and ⪰ the true
    element Hessian (Prop 3.1): CM − true = ((∂h/∂Σ)₊−∂h/∂Σ + (∂h/∂σ)₊−∂h/∂σ)∇²α +
    ((∂h/∂Σ)₊−∂h/∂Σ + (∂h/∂σ)₊)∇²β, a non-negative combination of the PSD matrices ∇²α, ∇²β."""
    _, _, dhS, dhs, Hh, Jg, Ha, Hb = _components(F, B, eps)
    return _cm_from_parts(dhS, dhs, Hh, Jg, Ha, Hb, area)


# ----- conformance / faithfulness gates -------------------------------------------------------
def _conformance():
    from .energy import element_terms
    from .mesh import grid_mesh, rest_quantities
    rng = np.random.default_rng(0)
    rest, tris = grid_mesh(4, 4)
    Bs, areas = rest_quantities(rest, tris)
    worst_sv = 0.0; worst_psd = 1e9; worst_majorize = 1e9; worst_analytic = 0.0
    for _ in range(60):
        x = (rest + 0.25 * rng.standard_normal(rest.shape)).reshape(-1)
        for t, tri in enumerate(tris):
            dofs = np.array([2 * tri[0], 2 * tri[0] + 1, 2 * tri[1], 2 * tri[1] + 1,
                             2 * tri[2], 2 * tri[2] + 1])
            B = Bs[t]; xe = x[dofs]; F = (B @ xe).reshape(2, 2)
            if np.linalg.det(F) <= 0.05:
                continue
            Sig, sig, *_ = singular_values_ab(F)
            sv = np.linalg.svd(F, compute_uv=False)
            worst_sv = max(worst_sv, abs(Sig - sv[0]) + abs(sig - sv[1]))          # (1) Σ,σ vs SVD
            Hcm = cm_element_hessian(F, B, areas[t]); Han = analytic_element_hessian(F, B, areas[t])
            worst_psd = min(worst_psd, np.linalg.eigvalsh(Hcm).min())              # (2) H PSD
            # (3) Prop 3.1: CM ⪰ true (use the EXACT analytic Hessian; FD is unreliable near σ→0)
            worst_majorize = min(worst_majorize, np.linalg.eigvalsh(Hcm - Han).min())
            # validate the analytic Hessian against FD only where FD is reliable (moderate σ)
            if sig > 0.4 and Sig < 3.0:
                _, _, Hfd, _ = element_terms(xe, B, areas[t])
                worst_analytic = max(worst_analytic,
                                     np.abs(Han - Hfd).max() / (np.abs(Hfd).max() + 1e-9))
    # (4) operational: CM modified-Newton monotonically decreases the energy AND reaches the same
    #     minimum as clamp-Newton (majorize-minimize is faithful end-to-end).
    from .solver import solve
    from .energy import element_terms as sd
    from .run_e1 import build_scenario
    sc = build_scenario(nx=6, ny=6, seed=0)
    rc = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], "composite-majorization",
               eterms=sd, tol=1e-8, max_iter=200)
    rn = solve(sc["x0"], sc["tris"], sc["Bs"], sc["areas"], sc["free"], "clamp",
               eterms=sd, tol=1e-8, max_iter=200)
    En = [e["energy"] for e in rc["log"]]
    monotone = all(En[i + 1] <= En[i] + 1e-9 for i in range(len(En) - 1))
    same_min = abs(rc["final_energy"] - rn["final_energy"]) / abs(rn["final_energy"])
    return worst_sv, worst_psd, worst_majorize, worst_analytic, monotone, same_min, rc["status"]


if __name__ == "__main__":
    import sys
    sv, psd, maj, ana, mono, same, status = _conformance()
    ok = sv < 1e-9 and psd > -1e-7 and maj > -1e-7 and ana < 1e-4 and mono and same < 1e-6
    print(f"[CM conformance] Σ,σ vs SVD {sv:.1e} | H PSD {psd:.1e} | CM⪰true (Prop 3.1) {maj:.1e} | "
          f"analytic vs FD {ana:.1e} | MM monotone {mono} | same-min-as-clamp {same:.1e} ({status}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
