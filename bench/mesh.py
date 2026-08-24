"""Scenario geometry: a triangulated rectangle + per-triangle rest quantities."""
import numpy as np
from .energy import element_B


def grid_mesh(nx, ny, W=1.0, H=1.0):
    xs = np.linspace(0.0, W, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    verts = np.array([[x, y] for y in ys for x in xs], dtype=float)  # (nv,2)

    def idx(i, j):
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            v00, v10 = idx(i, j), idx(i + 1, j)
            v01, v11 = idx(i, j + 1), idx(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    return verts, np.array(tris, dtype=int)


def boundary_mask(verts, W=1.0, H=1.0, tol=1e-9):
    """Vertex mask: True on the rectangle boundary."""
    x, y = verts[:, 0], verts[:, 1]
    return (np.abs(x) < tol) | (np.abs(x - W) < tol) | (np.abs(y) < tol) | (np.abs(y - H) < tol)


def rest_quantities(rest_verts, tris):
    """Precompute (B, area) per triangle from the REST configuration."""
    Bs, areas = [], []
    for tri in tris:
        X = rest_verts[tri]  # (3,2)
        Dm = np.column_stack((X[1] - X[0], X[2] - X[0]))  # 2x2
        area = 0.5 * abs(np.linalg.det(Dm))
        Bs.append(element_B(np.linalg.inv(Dm)))
        areas.append(area)
    return Bs, np.array(areas)
