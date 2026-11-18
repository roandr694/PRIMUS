"""Terrain for the autonomous skills: a grid of 8 m tiles (rows = difficulty 0-5, columns = type).

The ground is a coarse heightfield (rough, slopes); sharp features are boxes so edges stay sharp:
  flat       flat
  rough      smooth noise up to +-3 cm
  slope      ramp pyramid up to 15 deg
  stairs     stair pyramid: 5 steps up to a platform (spawn outside -> climb, on top -> descend); rise 4-19 cm
  curbs      raised platforms (curbs, single steps) 4-20 cm
  gaps       floor islands separated by trenches 10-45 cm wide (heightfield carved 0.5 m deep below)
  overhang   bars to pass under; bar bottom from 0.80 m (no need to crouch) down to 0.42 m
  stairs_wide  the stairs with deeper treads that narrow with the level (0.50 m at level 0 -> 0.32 m at level 5):
             room to learn to land and chain jumps, then the real depth (a training aid; real stairs are ~0.30)
  hurdles    thin walls (bars/boxes to jump over, 6-24 cm high, 10 cm thick) in three square rings
  gaps_wide  like gaps with trenches 30-80 cm wide: they need a real jump (run at speed, push, tuck)
  stairs_long  8 steps (instead of 5) up to a small platform: the policy must not stop half way
  stairs_narrow  real stairwells: three straight flights of 12 steps between side walls (0.95-1.3 m wide) from a
             run-up to a top platform, closed at both ends. No room to climb sideways. The middle flight has a
             0.9 m landing after its 6th step.
  bumps      everyday ground for the assist regime: a cross slope (roof along x, 2-10 deg), speed bumps 2-10 cm
             both ways, cable-like ridges 1-3 cm and a raised plateau with 3-8 cm edges

Variants (`variants=True`, default; drawn from the terrain seed, so another seed is unseen geometry): stair treads
per side 0.26-0.34 m (stairs_long 0.25-0.31, stairs_wide its level's tread +-3 cm, walled flights 0.26-0.34 per
flight), rise jitter +-1 cm per step, a step nosing of 1-3 cm on half of the open stairs and on the first walled
flight, hurdle thickness 3-15 cm per ring. `variants=False` is the v11 geometry.

Besides the collision geometry it returns, for perception, a fine `ground` height grid and a `ceiling` grid
(lowest overhead surface, +inf where open), both [nrow(y), ncol(x)] at `fine_res`.
"""
import numpy as np

TYPES = ("flat", "rough", "slope", "stairs", "curbs", "gaps", "overhang", "stairs_wide", "hurdles", "gaps_wide",
         "stairs_long", "stairs_narrow", "bumps")
STAIR_RISE = (0.04, 0.07, 0.10, 0.13, 0.16, 0.19)
CURB_RISE = (0.04, 0.07, 0.10, 0.13, 0.17, 0.20)
GAP_WIDTH = (0.10, 0.17, 0.24, 0.31, 0.38, 0.45)
HURDLE_H = (0.06, 0.09, 0.12, 0.15, 0.19, 0.24)
GAP_WIDE = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
BAR_BOTTOM = (0.80, 0.68, 0.60, 0.52, 0.46, 0.42)
TREAD = 0.30
TREAD_WIDE = (0.50, 0.46, 0.42, 0.38, 0.35, 0.32)    # stairs_wide tread per level
# stairs_narrow (tile-local, flights along +y): wall centre lines, per flight (x centre, inner width, tread)
NARROW_STEPS, NARROW_Y0, NARROW_WALL = 12, -2.4, 0.04          # steps, y of the first riser, wall half thickness
NARROW_WALLS_X = (-1.90, -0.52, 0.66, 1.69)
NARROW_FLIGHTS = ((-1.21, 1.30, 0.30), (0.07, 1.10, 0.32), (1.175, 0.95, 0.28))
NARROW_TOP_Y, NARROW_FOOT_Y = 2.0, -2.9                         # past these the flight is done (up / down)
NARROW_LANDING_AFTER = 6                                       # middle flight: 0.9 m landing after this step
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
    if kind in ("gaps", "gaps_wide"):
        return np.full((n, n), -0.5)            # everything is a pit; the walkable floor is boxes
    if kind == "bumps":
        z = np.tan(np.radians(2.0 + 8.0 * d)) * np.clip(half - np.abs(X), 0, None)     # cross slope driving along y
        hb = 0.02 + 0.08 * d
        bump = lambda u: np.clip(1.0 - (u / 0.25) ** 2, 0, None)                        # 0.5 m wide speed bump
        for yb in rng.uniform(-3.2, 3.2, 3):
            z += hb * bump(Y - yb)
        for xb in rng.uniform(-3.2, 3.2, 2):
            z += hb * bump(X - xb)
        for _ in range(3):                                                              # cables: one-cell ridges
            a, c = rng.uniform(0, np.pi), rng.uniform(-3.0, 3.0)
            z += (0.01 + 0.02 * d) * (np.abs(np.cos(a) * X + np.sin(a) * Y - c) < 0.05)
        px, py = rng.uniform(-2.0, 2.0, 2)                                              # plateau: small drops
        z += (0.03 + 0.05 * d) * ((np.abs(X - px) < 1.0) & (np.abs(Y - py) < 1.0))
        return z
    return np.zeros((n, n))


RISE_JITTER = 0.01                       # +- per step [m]; make_terrain(rise_jitter=...) overrides it


def _rises(r, n, rng, variants):
    """Cumulative step heights with +-RISE_JITTER per step."""
    return np.cumsum(r + (rng.uniform(-RISE_JITTER, RISE_JITTER, n) if variants else np.zeros(n)))


def _nosing(rng, variants, p=0.5):
    return rng.uniform(0.01, 0.03) if variants and rng.uniform() < p else 0.0


def _box_tile(kind, lv, cx, cy, rng, variants=True):
    """Boxes (center xyz, half sizes xyz) of a tile centred at (cx, cy); `bars` are overhead."""
    boxes, bars, walls = [], [], []
    if kind in ("stairs", "stairs_wide", "stairs_long"):
        r = STAIR_RISE[lv]
        n = 8 if kind == "stairs_long" else 5
        base = TREAD_WIDE[lv] if kind == "stairs_wide" else TREAD
        if not variants:
            t = (base,) * 4
        elif kind == "stairs":
            t = rng.uniform(0.26, 0.34, 4)
        elif kind == "stairs_long":
            t = rng.uniform(0.25, 0.31, 4)
        else:
            t = base + rng.uniform(-0.03, 0.03, 4)
        hs, nose = _rises(r, n, rng, variants), _nosing(rng, variants)
        th = min(0.025, 0.5 * r)
        e = [STAIR_HALF] * 4                                    # extent from the centre: +x, -x, +y, -y
        for k in range(n):
            if k:
                e = [e[i] - t[i] for i in range(4)]
            x0, x1, y0, y1 = cx - e[1], cx + e[0], cy - e[3], cy + e[2]
            c, hx, hy = ((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0) / 2, (y1 - y0) / 2
            boxes.append(((c[0], c[1], hs[k] / 2), (hx, hy, hs[k] / 2)))
            if nose:                                            # a slab under the tread, sticking out over the riser
                boxes.append(((c[0], c[1], hs[k] - th / 2), (hx + nose, hy + nose, th / 2)))
    elif kind == "stairs_narrow":
        r, w = STAIR_RISE[lv], NARROW_WALL
        top = 0.0
        for fi, (xc, width, tread) in enumerate(NARROW_FLIGHTS):
            if variants:
                tread = rng.uniform(0.26, 0.30) if fi == 1 else rng.uniform(0.26, 0.34)
            hs = _rises(r, NARROW_STEPS, rng, variants)
            nose = _nosing(rng, variants, p=1.0) if fi == 0 else 0.0
            th = min(0.025, 0.5 * r)
            y0 = NARROW_Y0
            for k in range(1, NARROW_STEPS + 1):                 # step k: from its riser to the end of the tile
                boxes.append(((cx + xc, cy + (y0 + 3.9) / 2, hs[k - 1] / 2), (width / 2, (3.9 - y0) / 2, hs[k - 1] / 2)))
                if nose:
                    boxes.append(((cx + xc, cy + (y0 - nose + 3.9) / 2, hs[k - 1] - th / 2),
                                  (width / 2, (3.9 - y0 + nose) / 2, th / 2)))
                y0 += 0.9 if (variants and fi == 1 and k == NARROW_LANDING_AFTER) else tread
            top = max(top, hs[-1])
        zw = (top + 0.5) / 2
        for xw in NARROW_WALLS_X:                                # side walls, then the two ends
            walls.append(((cx + xw, cy, zw), (w, 3.9 + w, zw)))
        mid, half = (NARROW_WALLS_X[0] + NARROW_WALLS_X[-1]) / 2, (NARROW_WALLS_X[-1] - NARROW_WALLS_X[0]) / 2
        walls.append(((cx + mid, cy + 3.9, zw), (half, w, zw)))
        walls.append(((cx + mid, cy - 3.9, 0.3), (half, w, 0.3)))
    elif kind == "curbs":
        h = CURB_RISE[lv]
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(0, 0.5):
            px, py = cx + 2.6 * np.cos(ang), cy + 2.6 * np.sin(ang)
            sx, sy = rng.uniform(0.5, 0.9, 2)
            boxes.append(((px, py, h / 2), (sx, sy, h / 2)))
        boxes.append(((cx, cy, h / 2), (0.8, 0.8, h / 2)))          # a platform in the middle too
    elif kind == "hurdles":
        h = HURDLE_H[lv]
        for rad in (1.5, 2.4, 3.3):
            q = rng.uniform(0.015, 0.075) if variants else 0.05            # half thickness
            for (ox, oy, sx, sy) in ((0, rad, rad + q, q), (0, -rad, rad + q, q),
                                     (rad, 0, q, rad + q), (-rad, 0, q, rad + q)):
                boxes.append(((cx + ox, cy + oy, h / 2), (sx, sy, h / 2)))
    elif kind in ("gaps", "gaps_wide"):
        g, t = (GAP_WIDTH if kind == "gaps" else GAP_WIDE)[lv], 0.25                              # gap width, slab half thickness
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
    return boxes, bars, walls


def make_terrain(n_levels=6, types=TYPES, tile=8.0, res=0.08, border=4.0, fine_res=0.04, seed=0, variants=True,
                 rise_jitter=0.01):
    global RISE_JITTER
    RISE_JITTER = rise_jitter
    rng = np.random.default_rng(seed)
    n = int(round(tile / res)) + 1
    nb = int(round(border / res))
    ny, nx = n_levels * (n - 1) + 1 + 2 * nb, len(types) * (n - 1) + 1 + 2 * nb
    H = np.zeros((ny, nx))
    size_x, size_y = (nx - 1) * res, (ny - 1) * res
    centers = np.zeros((n_levels, len(types), 2))
    boxes, bars, walls = [], [], []
    for lv in range(n_levels):
        for t, kind in enumerate(types):
            i0, j0 = nb + lv * (n - 1), nb + t * (n - 1)
            H[i0:i0 + n, j0:j0 + n] = _hfield_tile(kind, lv, n, res, rng)
            cx = -size_x / 2 + (j0 + (n - 1) / 2) * res
            cy = -size_y / 2 + (i0 + (n - 1) / 2) * res
            centers[lv, t] = (cx, cy)
            b, o, w = _box_tile(kind, lv, cx, cy, rng, variants)
            boxes += b
            bars += o
            walls += w
    ter = dict(heights=H.astype(np.float32), size_x=size_x, size_y=size_y, res=res, tile=tile, n_levels=n_levels,
               types=tuple(types), centers=centers.astype(np.float32), border=border, boxes=boxes, bars=bars,
               walls=walls, fine_res=fine_res)
    ter["ground"], ter["ceiling"], ter["floor"] = rasterize(ter)
    return ter


def rasterize(ter):
    """Fine grids (boxes + heightfield): `ground` and `ceiling` for perception (walls are in `ground`, a camera sees
    them), and `floor` = the walkable surface without the walls (rewards, spawn, climbed height)."""
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
    F = G.copy()
    for (c, s) in ter.get("walls", []):
        j0, j1 = to_idx(c[0] - s[0], sx / 2, nx), to_idx(c[0] + s[0], sx / 2, nx)
        i0, i1 = to_idx(c[1] - s[1], sy / 2, ny), to_idx(c[1] + s[1], sy / 2, ny)
        G[i0:i1 + 1, j0:j1 + 1] = np.maximum(G[i0:i1 + 1, j0:j1 + 1], c[2] + s[2])
    for (c, s) in ter["bars"]:
        j0, j1 = to_idx(c[0] - s[0], sx / 2, nx), to_idx(c[0] + s[0], sx / 2, nx)
        i0, i1 = to_idx(c[1] - s[1], sy / 2, ny), to_idx(c[1] + s[1], sy / 2, ny)
        C[i0:i1 + 1, j0:j1 + 1] = np.minimum(C[i0:i1 + 1, j0:j1 + 1], c[2] - s[2])
    return G.astype(np.float32), C.astype(np.float32), F.astype(np.float32)
