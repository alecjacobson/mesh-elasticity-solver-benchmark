"""Figure pipeline (review viz phase): consistent matplotlib style + a polyscope-headless helper.

All figures are deterministic and written to `figures/`. 2D meshes/plots use matplotlib (the right
tool for our 2D benchmark); the genuinely-3D tet example uses polyscope headless (EGL). Regenerate
everything with `python -m bench.run_figures`.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

FIGDIR = "figures"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

# per-method / per-filter colours, stable across all figures
COL = {
    "clamp": "#1f77b4", "absolute": "#d62728", "trust-region": "#2ca02c", "none": "#999999",
    "newton": "#111111", "l-bfgs": "#9467bd", "sobolev-lbfgs": "#8c564b", "aqp": "#ff7f0e",
    "slim": "#17becf", "local-global": "#7f7f7f", "anderson": "#e377c2",
}
# redundant (non-colour) encoding so red/green filters stay distinguishable for CVD readers
LS = {"clamp": "-", "absolute": (0, (5, 2)), "trust-region": (0, (1, 1.4)), "none": "-",
      "newton": "-", "l-bfgs": (0, (5, 2)), "sobolev-lbfgs": (0, (1, 1.4)), "aqp": (0, (4, 1, 1, 1))}
WORLD_COL = {0: "#bdbdbd", 1: "#4c78a8", 2: "#e45756", 3: "#f2a900"}
STATUS_COL = {"self-claimed": "#bbbbbb", "qualified": "#e6a817", "validated": "#2ca02c",
              "unmeasured": "#8888cc", "refuted": "#d62728"}


def save(fig, name, caption=None):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name + ".png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def trimesh(ax, V, F, values=None, cmap="viridis", edge="k", lw=0.35, vmin=None, vmax=None,
            norm=None, title=None, label=None):
    """Render a 2D triangle mesh; if `values` (per-face) given, colour by them. Pass `norm` (e.g. a
    TwoSlopeNorm centred at J=1) to encode deviation honestly without clipping the tail."""
    tri = mtri.Triangulation(V[:, 0], V[:, 1], np.asarray(F))
    tpc = None
    if values is not None:
        kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
        tpc = ax.tripcolor(tri, facecolors=np.asarray(values), cmap=cmap, shading="flat",
                           edgecolors=edge, linewidth=lw, **kw)
    else:
        ax.triplot(tri, color=edge, lw=lw)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
    return tpc


def face_detF_2d(V, F, B_or_rest=None):
    """Per-face det(F) where F maps the REST triangle to the current V. Uses the rest = an
    equilateral-ish reference from the undeformed connectivity if not given; for our grid meshes we
    pass the rest coordinates via `B_or_rest` (the rest V)."""
    rest = B_or_rest
    Js = []
    for f in F:
        Xr = rest[f]; Xc = V[f]
        e_r = np.array([Xr[1] - Xr[0], Xr[2] - Xr[0]]).T   # 2x2 rest edges
        e_c = np.array([Xc[1] - Xc[0], Xc[2] - Xc[0]]).T
        Fmat = e_c @ np.linalg.inv(e_r)
        Js.append(np.linalg.det(Fmat))
    return np.array(Js)


# ---- polyscope headless (EGL) for the 3D example ----
_PS = {"init": False}


def ps_headless():
    import polyscope as ps
    if not _PS["init"]:
        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(3)
        ps.set_view_projection_mode("orthographic")
        _PS["init"] = True
    return ps


def ps_shot(name, pad=24):
    import polyscope as ps
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name + ".png")
    ps.screenshot(p, transparent_bg=False)
    _trim_white(p, pad)
    print(f"  wrote {p}")
    return p


def _trim_white(path, pad=24):
    """Crop the near-white border off a polyscope screenshot so the mesh fills the frame."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        l, t, r, b = bbox
        l = max(0, l - pad); t = max(0, t - pad)
        r = min(im.width, r + pad); b = min(im.height, b + pad)
        im.crop((l, t, r, b)).save(path)
