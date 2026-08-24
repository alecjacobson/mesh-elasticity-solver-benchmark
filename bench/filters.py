"""Hessian-filter slot: the axis E1 ablates.

Per-element filters project a 6x6 element Hessian to (near-)SPD via its eigensystem:
  - none            : identity (raw Hessian; full Newton -- may be indefinite)
  - clamp           : lambda -> max(lambda, eps)          [Teran 2005 / analytic-eigensystems]
  - absolute        : lambda -> max(|lambda|, eps)        [Stabler Neo-Hookean 2024]
The global filter (identity-shift / Levenberg) is applied to the assembled matrix in the
solver, since it is not a per-element operation. See claims/claims.yaml for the edges E1 hardens.
"""
import numpy as np

PER_ELEMENT = ("none", "clamp", "absolute")
GLOBAL = ("identity-shift",)
ALL = PER_ELEMENT + GLOBAL


def project_element(H, kind, eps=1e-9):
    if kind == "none":
        return H
    w, V = np.linalg.eigh(H)
    if kind == "clamp":
        w = np.maximum(w, eps)
    elif kind == "absolute":
        w = np.maximum(np.abs(w), eps)
    else:
        raise ValueError(f"not a per-element filter: {kind}")
    return (V * w) @ V.T
