"""Hessian-filter slot: the axis E1 ablates.

Per-element filters project a 6x6 element Hessian to (near-)SPD via its eigensystem:
  - none            : identity (raw Hessian; full Newton -- may be indefinite)
  - clamp           : lambda -> max(lambda, eps)          [Teran 2005 / analytic-eigensystems]
  - absolute        : lambda -> max(|lambda|, eps)        [Stabler Neo-Hookean 2024]
Global filters act on the ASSEMBLED matrix in the solver:
  - identity-shift : Levenberg tau*I shift (a modified-Newton globalization, NOT a projection)
  - global-pdn     : the FAITHFUL projected Newton -- try Cholesky (true Newton when SPD), else
                     eigendecompose the assembled Hessian and clamp its eigenvalues (a true PSD
                     PROJECTION of the global operator, not an identity shift).

NAMING NOTE (review-r1 #39): the per-element `project-on-demand` is a *per-element* on-demand
variant (project only the elements whose 6x6 block is indefinite). It is NOT the faithful
"Projected Newton" of the Pitfalls-of-Projection literature, which projects the ASSEMBLED Hessian
-- that is `global-pdn`. Read `project-on-demand` as "per-element-on-demand"; when comparing
against the Pitfalls thesis use `global-pdn`. (The string is kept stable so existing results
remain reproducible.) See claims/claims.yaml for the edges E1 hardens.
"""
import numpy as np

# 'project-on-demand' == per-element-on-demand (see NAMING NOTE above); the faithful assembled
# Projected Newton is the GLOBAL filter 'global-pdn'.
PER_ELEMENT = ("none", "clamp", "absolute", "project-on-demand")
GLOBAL = ("identity-shift", "global-pdn", "trust-region")  # trust-region: adaptive clamp/absolute per step
ALL = PER_ELEMENT + GLOBAL


def project_element_blend(H, w, eps=1e-9):
    """Per-element trust-region blend lambda_eff = (1-w)lambda + w|lambda|, w in {0,0.5,1} ->
    {raw Newton, clamp, absolute}, floored at eps (the standalone filters' floor) when w>0. This is
    ONE 6x6 (or per-element) eigendecomposition -- the SAME per-iteration cost as clamp/absolute --
    so the trust-region switchboard is a fair per-step comparison, not a global assembled eigh
    (review-r2 #42/#44). The adaptive w is a GLOBAL per-step choice (from the model-fit ratio) but
    the projection is applied per element, matching how eigenvalue filtering actually works."""
    if w == 0.0:
        return H
    lam, V = np.linalg.eigh(H)
    lam = np.maximum((1.0 - w) * lam + w * np.abs(lam), eps)
    return (V * lam) @ V.T


def project_element(H, kind, eps=1e-9):
    if kind == "none":
        return H
    w, V = np.linalg.eigh(H)
    if kind == "project-on-demand":
        if w.min() >= eps:            # already (near-)SPD -> leave raw (keep true Newton curvature)
            return H
        w = np.maximum(w, eps)        # only project the indefinite ones (per-element PDN variant)
    elif kind == "clamp":
        w = np.maximum(w, eps)
    elif kind == "absolute":
        w = np.maximum(np.abs(w), eps)
    else:
        raise ValueError(f"not a per-element filter: {kind}")
    return (V * w) @ V.T
