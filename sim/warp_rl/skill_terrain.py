"""Terrain for the autonomous skills: a grid of 8 m tiles (rows = difficulty 0-5, columns = type).

The ground is a coarse heightfield (rough, slopes); sharp features are boxes so edges stay sharp:
  flat       flat
  rough      smooth noise up to +-3 cm
  slope      ramp pyramid up to 15 deg
  stairs     stair pyramid: 5 steps up to a platform (spawn outside -> climb, on top -> descend); rise 4-19 cm
  curbs      raised platforms (curbs, single steps) 4-20 cm
  gaps       floor islands separated by trenches 10-45 cm wide (heightfield carved 0.5 m deep below)
  overhang   bars to pass under; bar bottom from 0.80 m (no need to crouch) down to 0.42 m

Besides the collision geometry it returns, for perception, a fine `ground` height grid and a `ceiling` grid
(lowest overhead surface, +inf where open), both [nrow(y), ncol(x)] at `fine_res`.
"""
import numpy as np

TYPES = ("flat", "rough", "slope", "stairs", "curbs", "gaps", "overhang")
STAIR_RISE = (0.04, 0.07, 0.10, 0.13, 0.16, 0.19)
CURB_RISE = (0.04, 0.07, 0.10, 0.13, 0.17, 0.20)
GAP_WIDTH = (0.10, 0.17, 0.24, 0.31, 0.38, 0.45)
BAR_BOTTOM = (0.80, 0.68, 0.60, 0.52, 0.46, 0.42)
TREAD = 0.30
STAIR_HALF = 2.9                # half size of the bottom step: leaves a 1.1 m run-up inside the tile


def _hfield_tile(kind, lv, n, res, rng):
    d = lv / 5.0
    u = (np.arange(n) - (n - 1) / 2) * res
    X, Y = np.meshgrid(u, u, indexing="xy")
    half = (n - 1) / 2 * res
    if kind == "rough":
        amp = 0.005 + 0.025 * d
        z = rng.uniform(-amp, amp, (n // 4 + 2, n // 4 + 2))
        xi = np.linspace(0, z.shape[1] - 1, n)
        zz = np.array([np.interp(xi, np.arange(z.shape[1]), row) for row in z])
        return np.array([np.interp(xi, np.arange(z.shape[0]), col) for col in zz.T]).T
    if kind == "slope":
        grade = np.tan(np.radians(2.0 + 13.0 * d))
        r = np.maximum(np.abs(X), np.abs(Y))
        return grade * np.clip(half - np.maximum(r, 0.25 * half), 0, None)
    if kind == "gaps":
        return np.full((n, n), -0.5)            # everything is a pit; the walkable floor is boxes
    return np.zeros((n, n))


def _box_tile(kind, lv, cx, cy, rng):
    """Boxes (center xyz, half sizes xyz) of a tile centred at (cx, cy); `bars` are overhead."""
    boxes, bars = [], []
    if kind == "stairs":
        r = STAIR_RISE[lv]
        for k in range(1, 6):
            s = STAIR_HALF - (k - 1) * TREAD
            boxes.append(((cx, cy, k * r / 2), (s, s, k * r / 2)))
    elif kind == "curbs":
        h = CURB_RISE[lv]
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(0, 0.5):
            px, py = cx + 2.6 * np.cos(ang), cy + 2.6 * np.sin(ang)
            sx, sy = rng.uniform(0.5, 0.9, 2)
            boxes.append(((px, py, h / 2), (sx, sy, h / 2)))
        boxes.append(((cx, cy, h / 2), (0.8, 0.8, h / 2)))          # a platform in the middle too
    elif kind == "gaps":
        g, t = GAP_WIDTH[lv], 0.25                              # gap width, slab half thickness
        boxes.append(((cx, cy, -t), (0.9, 0.9, t)))              # centre island
        inner = 0.9
        while inner < 3.9:                                      # rings of 4 boxes separated by gaps
            a = inner + g
            b = a + 0.7 if a + 0.7 + g + 0.5 < 4.0 else 4.0       # the outermost ring reaches the tile edge
            m, w = (a + b) / 2, (b - a) / 2
            boxes += [((cx, cy + m, -t), (b, w, t)), ((cx, cy - m, -t), (b, w, t)),
                      ((cx + m, cy, -t), (w, a, t)), ((cx - m, cy, -t), (w, a, t))]
            inner = b
    elif kind == "overhang":
        b = BAR_BOTTOM[lv]
        for rad in (1.6, 2.6, 3.4):
            for (ox, oy, sx, sy) in ((0, rad, rad, 0.15), (0, -rad, rad, 0.15), (rad, 0, 0.15, rad), (-rad, 0, 0.15, rad)):
                bars.append(((cx + ox, cy + oy, b + 0.03), (sx, sy, 0.03)))
    return boxes, bars


def make_terrain(n_levels=6, types=TYPES, tile=8.0, res=0.08, border=4.0, fine_res=0.04, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(tile / res)) + 1
    nb = int(round(border / res))
    ny, nx = n_levels * (n - 1) + 1 + 2 * nb, len(types) * (n - 1) + 1 + 2 * nb
    H = np.zeros((ny, nx))
    size_x, size_y = (nx - 1) * res, (ny - 1) * res
    centers = np.zeros((n_levels, len(types), 2))
    boxes, bars = [], []
    for lv in range(n_levels):
        for t, kind in enumerate(types):
            i0, j0 = nb + lv * (n - 1), nb + t * (n - 1)
            H[i0:i0 + n, j0:j0 + n] = _hfield_tile(kind, lv, n, res, rng)
            cx = -size_x / 2 + (j0 + (n - 1) / 2) * res
            cy = -size_y / 2 + (i0 + (n - 1) / 2) * res
            centers[lv, t] = (cx, cy)
            b, o = _box_tile(kind, lv, cx, cy, rng)
            boxes += b
            bars += o
    ter = dict(heights=H.astype(np.float32), size_x=size_x, size_y=size_y, res=res, tile=tile, n_levels=n_levels,
               types=tuple(types), centers=centers.astype(np.float32), border=border, boxes=boxes, bars=bars,
               fine_res=fine_res)
    ter["ground"], ter["ceiling"] = rasterize(ter)
    return ter


def rasterize(ter):
    """Fine ground-height and ceiling grids for perception (boxes + heightfield)."""
    fr, sx, sy = ter["fine_res"], ter["size_x"], ter["size_y"]
    nx, ny = int(round(sx / fr)) + 1, int(round(sy / fr)) + 1
    xs, ys = np.linspace(-sx / 2, sx / 2, nx), np.linspace(-sy / 2, sy / 2, ny)
    H = ter["heights"]
    hx, hy = np.linspace(-sx / 2, sx / 2, H.shape[1]), np.linspace(-sy / 2, sy / 2, H.shape[0])
    G = np.array([np.interp(xs, hx, row) for row in H])
    G = np.array([np.interp(ys, hy, col) for col in G.T]).T
    C = np.full((ny, nx), np.inf)
    to_idx = lambda v, lo, n: int(np.clip(np.round((v + lo) / fr), 0, n - 1))
    for (c, s) in ter["boxes"]:
        j0, j1 = to_idx(c[0] - s[0], sx / 2, nx), to_idx(c[0] + s[0], sx / 2, nx)
        i0, i1 = to_idx(c[1] - s[1], sy / 2, ny), to_idx(c[1] + s[1], sy / 2, ny)
        G[i0:i1 + 1, j0:j1 + 1] = np.maximum(G[i0:i1 + 1, j0:j1 + 1], c[2] + s[2])
    for (c, s) in ter["bars"]:
        j0, j1 = to_idx(c[0] - s[0], sx / 2, nx), to_idx(c[0] + s[0], sx / 2, nx)
        i0, i1 = to_idx(c[1] - s[1], sy / 2, ny), to_idx(c[1] + s[1], sy / 2, ny)
        C[i0:i1 + 1, j0:j1 + 1] = np.minimum(C[i0:i1 + 1, j0:j1 + 1], c[2] - s[2])
    return G.astype(np.float32), C.astype(np.float32)
