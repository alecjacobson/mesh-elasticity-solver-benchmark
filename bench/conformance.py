"""Conformance / grounding tests (harness.md admissibility gate).

For the classical analytic energy, correctness of the derivatives IS the reference: we verify
(1) the analytic dpsi/dF against a finite-difference of psi, and (2) the assembled global
gradient against a finite-difference of the global energy. This stands in for official-code
regression until an official reference (e.g. TinyAD / libigl) is ported; it is the gate that a
component -- human- or agent-written -- must pass before entering a comparison.
"""
import numpy as np
from .energy import psi, grad_psi
from .mesh import grid_mesh, rest_quantities
from .solver import energy_only, assemble


def check_grad_psi(n=200, h=1e-6, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        F = np.eye(2) + 0.3 * rng.standard_normal((2, 2))
        if np.linalg.det(F) <= 0.2:
            continue
        G = grad_psi(F).reshape(4)
        Ff = F.reshape(4)
        Gfd = np.zeros(4)
        for k in range(4):
            fp = Ff.copy(); fp[k] += h
            fm = Ff.copy(); fm[k] -= h
            Gfd[k] = (psi(fp.reshape(2, 2)) - psi(fm.reshape(2, 2))) / (2 * h)
        worst = max(worst, np.max(np.abs(G - Gfd)) / (np.max(np.abs(Gfd)) + 1e-12))
    return worst, worst < tol


def check_global_gradient(nx=4, ny=4, h=1e-6, tol=1e-5, seed=1):
    rng = np.random.default_rng(seed)
    rest, tris = grid_mesh(nx, ny)
    Bs, areas = rest_quantities(rest, tris)
    x = (rest + 0.05 * rng.standard_normal(rest.shape)).reshape(-1)
    _, g, _ = assemble(x, tris, Bs, areas, "none")
    gfd = np.zeros_like(g)
    for k in range(g.size):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        gfd[k] = (energy_only(xp, tris, Bs, areas) - energy_only(xm, tris, Bs, areas)) / (2 * h)
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
    return rel, rel < tol


def check_energy_reference(n=2000, tol=1e-12, seed=2):
    """Regression against the CANONICAL symmetric-Dirichlet definition (singular-value form
    sigma1^2+sigma2^2+1/sigma1^2+1/sigma2^2 -- the energy libigl's SLIM implements as
    igl.SYMMETRIC_DIRICHLET), via an independent SVD code path. This is the D3 'official-
    reference' grounding for the energy: our Frobenius-based psi must equal the published
    definition to machine precision, not merely be self-consistent under finite differences."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        F = np.eye(2) + 0.4 * rng.standard_normal((2, 2))
        if np.linalg.det(F) <= 0.05:
            continue
        s = np.linalg.svd(F, compute_uv=False)
        ref = s[0] ** 2 + s[1] ** 2 + 1 / s[0] ** 2 + 1 / s[1] ** 2
        worst = max(worst, abs(psi(F) - ref) / abs(ref))
    return worst, worst < tol


def check_stable_neohookean(n=200, h=1e-6, tol=1e-5, seed=4):
    """Stable Neo-Hookean (Smith-Kim-de Goes 2018): analytic gradient vs FD, rest state F=I
    stress-free, and FINITENESS under inversion (J<=0) -- the property that distinguishes it from
    the classical barrier NH and admits the inverted-element regime (review-r1 #31)."""
    from .energy_stable_neohookean import make, lam_from_nu
    et, psi_s, grad_s, _ = make(mu=1.0, lam=lam_from_nu(0.45))
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        F = np.eye(2) + 0.3 * rng.standard_normal((2, 2))
        G = grad_s(F).reshape(4); Ff = F.reshape(4); Gfd = np.zeros(4)
        for k in range(4):
            fp = Ff.copy(); fp[k] += h; fm = Ff.copy(); fm[k] -= h
            Gfd[k] = (psi_s(fp.reshape(2, 2)) - psi_s(fm.reshape(2, 2))) / (2 * h)
        worst = max(worst, np.max(np.abs(G - Gfd)) / (np.max(np.abs(Gfd)) + 1e-12))
    rest = float(np.max(np.abs(grad_s(np.eye(2)))))
    Finv = np.diag([-0.5, 1.0])
    finite = bool(np.isfinite(psi_s(Finv)) and np.all(np.isfinite(grad_s(Finv))))
    ok = worst < tol and rest < 1e-8 and finite
    return worst, rest, finite, ok


def check_trust_region_blend(n=300, tol=1e-9, seed=5):
    """Regression grounding for the PER-ELEMENT trust-region blend (review-r1 #38; per-element root
    fix review-r2 #42/#44): the operator project_element_blend(H, w) must EQUAL the actual standalone
    filters -- w=0 raw Newton, w=0.5 = project_element('clamp'), w=1 = project_element('absolute') --
    at the same eps=1e-9 floor. Comparing the projected MATRICES (well-conditioned) proves TR's states
    ARE those filter operators at the same per-element cost, so tol is 1e-9."""
    from .filters import project_element_blend, project_element
    rng = np.random.default_rng(seed)
    wn = wc = wa = 0.0
    for _ in range(n):
        m = 6; A = rng.standard_normal((m, m)); H = (A + A.T) / 2
        wn = max(wn, np.linalg.norm(project_element_blend(H, 0.0) - H) / (np.linalg.norm(H) + 1e-12))
        C = project_element(H, "clamp")
        wc = max(wc, np.linalg.norm(project_element_blend(H, 0.5) - C) / (np.linalg.norm(C) + 1e-12))
        Ab = project_element(H, "absolute")
        wa = max(wa, np.linalg.norm(project_element_blend(H, 1.0) - Ab) / (np.linalg.norm(Ab) + 1e-12))
    ok = wn < tol and wc < tol and wa < tol
    return wn, wc, wa, ok


def run():
    r1, ok1 = check_grad_psi()
    r2, ok2 = check_global_gradient()
    r3, ok3 = check_energy_reference()
    s_grad, s_rest, s_fin, ok4 = check_stable_neohookean()
    tn, tc, ta, ok5 = check_trust_region_blend()
    from .p2_sri import _conformance as _sri_conf
    sri_grad, sri_rest = _sri_conf()
    ok6 = sri_grad < 1e-5 and sri_rest < 1e-8      # SRI-P2 element: analytic grad + rest-stress-free
    from .barrier_ls import _conformance as _bar_conf
    ok7, bar_at, bar_past = _bar_conf()            # barrier line-search: step is the tight inversion bound
    from .untangle import _conformance as _unt_conf
    unt_err = _unt_conf(); ok8 = unt_err < 1e-5    # untangling area-penalty gradient vs FD
    from .filters import project_element as _pe, project_element_blend as _peb
    _rng = np.random.default_rng(9); _bc = _ba = 0.0; _bmin = 1e9
    for _ in range(100):                           # blend filter: ==clamp at w=.5, ==absolute at w=1, SPD between
        _A = _rng.standard_normal((6, 6)); _H = _A + _A.T
        _bc = max(_bc, np.abs(_peb(_H, 0.5) - _pe(_H, "clamp")).max())
        _ba = max(_ba, np.abs(_peb(_H, 1.0) - _pe(_H, "absolute")).max())
        _bmin = min(_bmin, np.linalg.eigvalsh(_peb(_H, 0.75)).min())
    ok9 = _bc < 1e-12 and _ba < 1e-12 and _bmin > 0
    from .incremental import _conformance as _inc_conf
    inc_g, inc_a0, inc_blk = _inc_conf()           # incremental potential: gradPhi vs FD; A0 SPD; VBD block==Hessian block
    ok10 = inc_g < 1e-5 and inc_a0 > 0 and inc_blk < 1e-9
    from .massspring import _conformance as _ms_conf
    ms_g, ms_a0 = _ms_conf()                        # mass-spring: gradPhi vs FD; PD global system SPD
    ok11 = ms_g < 1e-5 and ms_a0 > 0
    from .composite_majorization import _conformance as _cm_conf
    cm_sv, cm_psd, cm_maj, cm_ana, cm_mono, cm_same, _ = _cm_conf()   # CM: Σσ=SVD; PSD; Prop 3.1; MM monotone; same min
    ok12 = cm_sv < 1e-9 and cm_psd > -1e-7 and cm_maj > -1e-7 and cm_ana < 1e-4 and cm_mono and cm_same < 1e-6
    from .tet_scale import _conformance as _t3_conf
    t3_g, t3_h, t3_re, t3_rg, t3_de, t3_dg = _t3_conf()   # 3D scalable tet: analytic grad/Hess vs FD; rigid inv; ==dense tet.py
    ok13 = t3_g < 1e-5 and t3_h < 1e-4 and t3_re < 1e-9 and t3_rg < 1e-8 and t3_de < 1e-10 and t3_dg < 1e-10
    from .bcqn import _conformance as _bcqn_conf
    bq_beta, bq_mono, bq_same, bq_st, bq_it = _bcqn_conf()   # faithful BCQN: β∈[0,1]; monotone; ==p-Newton min
    ok14 = bool(bq_beta) and bool(bq_mono) and bq_same < 1e-5 and bq_st == "converged"
    from .tlc import _conformance as _tlc_conf
    tlc_fin, tlc_g, tlc_tua, tlc_inj, _tlc_fi, _tlc_left = _tlc_conf()   # faithful TLC: barrier-free; grad; α→0==TUA; untangles
    ok15 = bool(tlc_fin) and tlc_g < 1e-5 and tlc_tua < 1e-6 and bool(tlc_inj)
    print(f"[conformance] dpsi/dF vs FD:        max rel err {r1:.2e}  -> {'PASS' if ok1 else 'FAIL'}")
    print(f"[conformance] global grad vs FD:    max rel err {r2:.2e}  -> {'PASS' if ok2 else 'FAIL'}")
    print(f"[conformance] psi vs canonical SD:  max rel err {r3:.2e}  -> {'PASS' if ok3 else 'FAIL'}")
    print(f"[conformance] stable-NH grad/rest/inv: {s_grad:.2e} / {s_rest:.1e} / finite={s_fin} "
          f"-> {'PASS' if ok4 else 'FAIL'}")
    print(f"[conformance] TR blend=Newton/clamp/abs: {tn:.1e}/{tc:.1e}/{ta:.1e} "
          f"-> {'PASS' if ok5 else 'FAIL'}")
    print(f"[conformance] SRI-P2 grad/rest: {sri_grad:.1e} / {sri_rest:.1e} -> {'PASS' if ok6 else 'FAIL'}")
    print(f"[conformance] barrier-LS step bound: area@α={bar_at:.1e} (≈0), past<0={bar_past:.1e} "
          f"-> {'PASS' if ok7 else 'FAIL'}")
    print(f"[conformance] untangle-penalty grad vs FD: max rel err {unt_err:.1e} -> {'PASS' if ok8 else 'FAIL'}")
    print(f"[conformance] blend filter =clamp@.5/=abs@1/SPD: {_bc:.1e}/{_ba:.1e}/min={_bmin:.1e} "
          f"-> {'PASS' if ok9 else 'FAIL'}")
    print(f"[conformance] incremental gradPhi/A0-SPD/VBD-block: {inc_g:.1e}/{inc_a0:.1e}/{inc_blk:.1e} "
          f"-> {'PASS' if ok10 else 'FAIL'}")
    print(f"[conformance] mass-spring gradPhi/PD-SPD: {ms_g:.1e}/{ms_a0:.1e} "
          f"-> {'PASS' if ok11 else 'FAIL'}")
    print(f"[conformance] composite-majorization Σσ/PSD/Prop3.1/analytic/monotone/same-min: "
          f"{cm_sv:.0e}/{cm_psd:.0e}/{cm_maj:.0e}/{cm_ana:.0e}/{cm_mono}/{cm_same:.0e} "
          f"-> {'PASS' if ok12 else 'FAIL'}")
    print(f"[conformance] 3D-scale tet analytic grad/Hess vs FD, rigid, =dense: "
          f"{t3_g:.0e}/{t3_h:.0e}/{t3_re:.0e}/{t3_rg:.0e}/{t3_de:.0e}/{t3_dg:.0e} "
          f"-> {'PASS' if ok13 else 'FAIL'}")
    print(f"[conformance] faithful BCQN β∈[0,1]/monotone/==p-Newton min: "
          f"{bq_beta}/{bq_mono}/{bq_same:.0e} ({bq_it} it) -> {'PASS' if ok14 else 'FAIL'}")
    print(f"[conformance] faithful TLC barrier-free/grad-FD/α→0=TUA/untangles: "
          f"{tlc_fin}/{tlc_g:.0e}/{tlc_tua:.0e}/{tlc_inj} -> {'PASS' if ok15 else 'FAIL'}")
    ok = (ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8 and ok9 and ok10 and ok11
          and ok12 and ok13 and ok14 and ok15)
    print(f"[conformance] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
