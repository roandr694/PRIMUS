"""Site and landscape: hillside terrain (heightfield for the physics + textured mesh), sea and seabed, shore rocks,
neighbouring houses with clay-tile roofs (photos 2, 3, 5), Atlantic-forest vegetation on the slopes and the
mountains/islands of the Angra dos Reis bay on the horizon (photos 3, 7).
"""
import math

import numpy as np

import structure as ST
import vegetation as V
from geom import Frame, rot_z

# terrain grid (collision heightfield and visual mesh)
TX0, TX1, TY0, TY1, RES = -64.0, 84.0, -46.0, 62.0, 0.5


def _profile(x):
    xs = [-64, -36, -30, -27.5, -25, -22, -17, -13, -6, 20, 24, 28.5, 33, 38, 48, 58, 62, 75, 84]
    zs = [-16, -10.5, -7.6, -6.8, -5.4, -4.3, -3.9, -1.2, -0.3, -0.3, 3.2, 4.2, 8.4, 12.8, 13.0, 15.5, 15.9, 24, 27]
    return np.interp(x, xs, zs)


def _fbm(X, Y, seed=0, octaves=5, scale=18.0):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    amp = 1.0
    for o in range(octaves):
        k = rng.uniform(0, 2 * math.pi, 4)
        f = (2 ** o) / scale
        out += amp * (np.sin(X * f * 1.3 + k[0] + np.cos(Y * f * 0.7 + k[1])) *
                      np.cos(Y * f * 1.1 + k[2] + np.sin(X * f * 0.6 + k[3])))
        amp *= 0.5
    return out


def heights():
    xs = np.arange(TX0, TX1 + 1e-9, RES)
    ys = np.arange(TY0, TY1 + 1e-9, RES)
    X, Y = np.meshgrid(xs, ys)                   # rows = y
    H = _profile(X)
    yn = ST.y_north(X)
    inside_y = (Y > -2.0) & (Y < yn + 0.3)
    dist = np.maximum(0, np.maximum(-2.0 - Y, Y - yn - 0.3))
    noise = _fbm(X, Y, 1) * (0.25 + 0.9 * np.clip(dist / 15.0, 0, 1))
    noise *= np.clip((H + 8.0) / 3.0, 0.2, 1.0)
    H = H + noise
    # neighbours: terraced pads
    for (x0, x1, y0, y1, z) in ((-8, 11, -15, -3.2, 0.2), (-4, 13, 21.5, 32, 0.9), (29, 41, -14, -3.5, 9.6),
                                (30, 42, 23, 33, 10.4), (-16, -6, -12, -2.5, -2.6), (-18, -6, 21, 30, -2.2)):
        m = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
        H = np.where(m, z + 0.05 * noise, H)
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, 2.2 / RES * 0.5 * 2, mode="nearest")
    # carve under every built surface of the property
    def clamp(x0, x1, y0, y1, zmax):
        nonlocal H
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        H = np.where(m, np.minimum(H, zmax), H)
    clamp(-27.2, -13.3, -2.0, 17.5, ST.Z_DECK - 0.45)
    clamp(-13.4, 21.3, -2.0, 20.5, -0.4)
    clamp(-0.5, 18.0, -2.0, 0.1, -0.4)
    clamp(21.2, 28.7, -2.0, 21.0, ST.Z_UP - 0.4)
    clamp(-32.2, -26.8, 1.8, 11.6, ST.Z_PIER - 0.5)
    # lift corridor cut: below the cabin chassis
    th = ST.LIFT_THETA
    zf = ST.Z_UP + (X - ST.LIFT_X0) * math.tan(th)
    m = (X > 26.0) & (X < 40.5) & (Y > 15.6) & (Y < 18.4)
    H = np.where(m, np.minimum(H, zf - 2.1), H)
    # parking deck, driveway, street
    clamp(37.8, 48.2, 0.8, 19.8, ST.Z_PARK - 0.45)
    corridor = (X > 26.0) & (X < 40.5) & (Y > 15.6) & (Y < 18.4)
    m = (X > 37.0) & (X < 49.0) & (Y > -1.0) & (Y < 21.5) & ~((X > 37.8) & (X < 48.2) & (Y > 0.8) & (Y < 19.8)) \
        & ~corridor
    H = np.where(m, np.clip(H, ST.Z_PARK - 0.6, ST.Z_PARK + 0.3), H)
    H = np.where(corridor, np.minimum(H, zf - 2.1), H)
    ramp = ST.Z_PARK + (X - 48.0) / 14.0 * (ST.Z_STREET - ST.Z_PARK)
    m = (X > 48.0) & (X < 62.0) & (Y > 11.5) & (Y < 17.5)
    H = np.where(m, np.minimum(H, ramp - 0.45), H)
    m = (X > 60.3) & (X < 70.2)
    H = np.where(m, ST.Z_STREET - 0.55, H)
    return xs, ys, H


def terrain(s, lite=False):
    xs, ys, H = heights()
    ny, nx = H.shape
    s.hfields.append(dict(name="terrain", heights=H, half=((TX1 - TX0) / 2, (TY1 - TY0) / 2),
                          centre=((TX0 + TX1) / 2, (TY0 + TY1) / 2), friction=(1.0, 0.03, 0.004)))
    # visual mesh (every 2nd sample in lite mode), triangles split between rock and soil by height and slope
    st = 2 if lite else 1
    Hs = H[::st, ::st]
    X, Y = np.meshgrid(xs[::st], ys[::st])
    V3 = np.stack([X, Y, Hs], -1).reshape(-1, 3)
    h, w = Hs.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    p = V3[tris]
    zc = p[:, :, 2].mean(1)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    slope = np.hypot(nrm[:, 0], nrm[:, 1]) / (np.abs(nrm[:, 2]) + 1e-9)
    # hide triangles fully under the house platforms (cheaper rendering)
    cx, cy = p[:, :, 0].mean(1), p[:, :, 1].mean(1)
    hidden = ((cx > -13.0) & (cx < 21.0) & (cy > -1.5) & (cy < 16.2) & (zc < -0.3))
    rock = (zc < ST.Z_SEA + 1.8) | (slope > 1.9)
    grassy = ~rock & (slope < 0.3)
    for mat, m in (("rock", rock & ~hidden), ("ground_forest", ~rock & ~grassy & ~hidden),
                   ("soil_grass", grassy & ~hidden)):
        T = tris[m]
        if len(T):
            used, inv = np.unique(T.ravel(), return_inverse=True)
            s.trimesh(mat, V3[used], inv.reshape(-1, 3), smooth=True)
    return xs, ys, H


def far_terrain(s, step=6.0):
    """Coarse visual continuation of the hillside and coastline beyond the detailed terrain (no collision)."""
    xs = np.arange(-70.0, 420.0 + 1e-9, step)
    ys = np.arange(-360.0, 380.0 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    w = np.clip((np.abs(Y - 8.0) - 48.0) / 60.0, 0, 1)
    shift = (22 * np.sin(Y / 95.0) + 12 * np.sin(Y / 41.0 + 1.0)) * w
    Xp = X - shift
    prof = np.interp(Xp, [-70, -36, -30, -27.5, -25, -22, -17, -6, 20, 28, 48, 62, 84, 150, 260, 420],
                     [-18, -10.5, -7.6, -6.8, -5.4, -4.3, -3.9, -0.3, -0.3, 4.0, 13.0, 15.9, 27, 75, 170, 260])
    big = _fbm(X * 0.25, Y * 0.25, 7, 5, 18.0)
    Z = prof + (2.0 + 0.08 * np.clip(Xp, 0, None)) * big * np.clip((prof + 6) / 10, 0, 1)
    inner = (X > TX0 + step) & (X < TX1 - step) & (Y > TY0 + step) & (Y < TY1 - step)
    V3 = np.stack([X, Y, Z], -1).reshape(-1, 3)
    h, wd = Z.shape
    idx = np.arange(h * wd).reshape(h, wd)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    cell_inner = (inner[:-1, :-1] & inner[:-1, 1:] & inner[1:, 1:] & inner[1:, :-1]).ravel()
    tris = np.concatenate([np.stack([a, b, c], 1)[~cell_inner], np.stack([a, c, d], 1)[~cell_inner]])
    zc = V3[tris][:, :, 2].mean(1)
    for mat, m in (("rock", zc < ST.Z_SEA + 1.5), ("forest", zc >= ST.Z_SEA + 1.5)):
        T = tris[m]
        used, inv = np.unique(T.ravel(), return_inverse=True)
        s.trimesh(mat, V3[used], inv.reshape(-1, 3), smooth=True)


def shore_rocks(s, H, xs, ys):
    """Granite boulders along the waterline and under the deck (Angra's coast)."""
    rng = np.random.default_rng(12)
    for k in range(70):
        y = rng.uniform(-30, 45)
        x = np.interp(ST.Z_SEA + rng.uniform(-0.6, 1.6), [-16, -10.5, -7.6, -6.8, -5.4, -4.3],
                      [-64, -36, -30, -27.5, -25, -22]) + rng.normal(0, 1.2)
        if -32.5 < x < -26.5 and 1.5 < y < 12.0:
            continue
        r = rng.uniform(0.6, 2.2)
        j, i = int((y - ys[0]) / (ys[1] - ys[0])), int((x - xs[0]) / (xs[1] - xs[0]))
        z = H[np.clip(j, 0, H.shape[0] - 1), np.clip(i, 0, H.shape[1] - 1)]
        V.boulder(s, (x, y, z - 0.15 * r), r, seed=k, flat=rng.uniform(0.45, 0.7))


def sea(s):
    s.planes.append(dict(name="sea", mat="water_sea", pos=(-1000.0, 8.0, ST.Z_SEA), half=(5000.0, 5000.0)))
    s.collider("box", (-1000.0, 8.0, ST.Z_SEA - 4.0), (5000.0, 5000.0, 0.5), name="seabed")


def mountain(s, centre, rx, ry, height, seed, res=56, mat="forest"):
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.6, 1.6, res)
    U, Vv = np.meshgrid(u, u)
    R2 = U ** 2 + Vv ** 2
    base = np.exp(-R2 * 1.4) * (1 + 0.35 * _fbm(U * 30, Vv * 30, seed, 5, 9.0))
    ridge = np.exp(-(Vv * 1.8 + 0.3 * np.sin(U * 2.5 + rng.uniform(0, 3))) ** 2) * np.exp(-U ** 2 * 0.5)
    Z = height * np.clip(0.65 * base + 0.55 * ridge * (1 + 0.2 * _fbm(U * 40, Vv * 40, seed + 1, 4, 8.0)), 0, None)
    Z = Z - height * 0.04
    ang = rng.uniform(-0.4, 0.4)
    c, sn = math.cos(ang), math.sin(ang)
    Xw = centre[0] + rx * (U * c - Vv * sn)
    Yw = centre[1] + ry * (U * sn + Vv * c)
    Vt = np.stack([Xw, Yw, ST.Z_SEA + Z], -1).reshape(-1, 3)
    idx = np.arange(res * res).reshape(res, res)
    a, b, cc, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.concatenate([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)])
    keep = Vt[F][:, :, 2].max(1) > ST.Z_SEA - 2
    s.trimesh(mat, Vt, F[keep], smooth=True)


def horizon(s):
    """Islands and mountain ranges around the bay (photos 3 and 7: green hills across the water)."""
    mountain(s, (-2700, 600), 1500, 800, 520, 1)
    mountain(s, (-2200, -1900), 900, 650, 380, 2)
    mountain(s, (-1600, 2300), 900, 600, 300, 3)
    mountain(s, (500, 3600), 2400, 900, 760, 4)
    mountain(s, (900, -3500), 2000, 900, 600, 5)
    mountain(s, (-900, -700), 160, 110, 45, 6)
    mountain(s, (-1200, 900), 220, 120, 60, 7)
    mountain(s, (2600, 400), 1800, 1600, 900, 8)
    s.note("site", "bay horizon: 8 procedural islands / mountain ranges 1-4 km away")


def neighbour_house(s, x0, x1, y0, y1, z0, storeys=1, pool=None, seed=0):
    zt = z0 + 3.0 * storeys
    s.box("neighbour_wall", ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + zt) / 2), ((x1 - x0) / 2, (y1 - y0) / 2, (zt - z0) / 2))
    rng = np.random.default_rng(seed)
    for k in range(storeys):
        zw = z0 + 1.0 + 3.0 * k
        for x in np.arange(x0 + 1.2, x1 - 1.0, 2.4):
            for y in (y0 - 0.01, y1 + 0.01):
                s.box("black_glass", (x, y, zw + 0.6), (0.6, 0.02, 0.7), collide=False)
                s.box("wood_dark", (x - 0.75, y, zw + 0.6), (0.12, 0.03, 0.75), collide=False)
                s.box("wood_dark", (x + 0.75, y, zw + 0.6), (0.12, 0.03, 0.75), collide=False)
    # clay-tile hipped roof with eaves
    ex = 0.8
    a, b, c, d = x0 - ex, x1 + ex, y0 - ex, y1 + ex
    yc = (c + d) / 2
    hw = (d - c) / 2
    zr = zt + hw * 0.55
    P = {"a": (a, c, zt), "b": (b, c, zt), "c": (b, d, zt), "d": (a, d, zt), "e": (a + hw, yc, zr), "f": (b - hw, yc, zr)}
    if b - a < 2 * hw:
        P["e"] = P["f"] = ((a + b) / 2, yc, zr)
    tris = [("a", "b", "f"), ("a", "f", "e"), ("b", "c", "f"), ("c", "d", "e"), ("c", "e", "f"), ("d", "a", "e")]
    Vt = np.array([P[k] for t in tris for k in t], float)
    s.trimesh("terracotta_roof", Vt, np.arange(len(Vt)).reshape(-1, 3), smooth=False)
    for k in range(int((b - a) / 0.25)):                     # tile ridges on the long slopes
        x = a + 0.12 + k * 0.25
        t = min(1.0, (x - a) / hw, (b - x) / hw)
        for sg in (-1, 1):
            ye = yc + sg * hw
            s.tube("terracotta_roof", (x, ye, zt + 0.03), (x, ye + (yc - ye) * t, zt + (zr - zt) * t + 0.02), 0.035,
                   seg=4)
    if pool:
        px0, px1, py0, py1 = pool
        s.box("concrete_light", ((px0 + px1) / 2, (py0 + py1) / 2, z0 - 0.02), ((px1 - px0) / 2 + 0.4,
                                                                                   (py1 - py0) / 2 + 0.4, 0.02),
              collide=False)
        s.planes.append(dict(name=f"npool{seed}", mat="water_pool", pos=((px0 + px1) / 2, (py0 + py1) / 2, z0 + 0.0),
                             half=((px1 - px0) / 2, (py1 - py0) / 2)))


def neighbours(s):
    neighbour_house(s, -4.0, 8.5, -13.5, -5.5, 0.2, storeys=2, pool=(-6.5, -2.0, -4.6, -3.4), seed=1)
    neighbour_house(s, -2.0, 11.0, 22.8, 30.5, 0.9, storeys=1, seed=2)
    neighbour_house(s, 30.5, 39.5, -12.5, -5.0, 9.6, storeys=2, seed=3)
    neighbour_house(s, 31.5, 41.0, 24.0, 32.0, 10.4, storeys=1, seed=4)
    for (x0, x1, y0, y1, z) in ((-16, -6, -12, -2.6, -2.6), (-18, -6, 21, 30, -2.2)):
        s.box("deck_brown", ((x0 + x1) / 2, (y0 + y1) / 2, z - 0.1), ((x1 - x0) / 2, (y1 - y0) / 2, 0.1),
              collide=True)
    s.note("site", "4 neighbouring houses with clay-tile roofs, neighbours' decks and a pool")


def vegetation(s, H, xs, ys, lite=False):
    rng = np.random.default_rng(99)

    def z_at(x, y):
        j = int(np.clip((y - ys[0]) / (ys[1] - ys[0]), 0, H.shape[0] - 1))
        i = int(np.clip((x - xs[0]) / (xs[1] - xs[0]), 0, H.shape[1] - 1))
        return float(H[j, i])

    def blocked(x, y):
        if -28 < x < 49 and -2.3 < y < ST.y_north(x) + 0.5:          # the property itself
            return not (28.8 < x < 37.6 and (y < 15.4 or y > 18.6))    # except the hill beside the lift
        if 60 < x < 71:
            return True
        for (x0, x1, y0, y1) in ((-8, 11, -15, -3.2), (-4, 13, 21.5, 32), (29, 41, -14, -3.5), (30, 42, 23, 33)):
            if x0 - 0.5 < x < x1 + 0.5 and y0 - 0.5 < y < y1 + 0.5:
                return True
        return False

    n_trees = 45 if lite else (150 if V.LOD["level"] == "high" else 70)
    placed = 0
    while placed < n_trees:
        x, y = rng.uniform(-24, 82), rng.uniform(-44, 60)
        if blocked(x, y):
            continue
        z = z_at(x, y)
        if z < ST.Z_SEA + 1.2:
            continue
        k = rng.random()
        if k < 0.25 and z < 6:
            V.coconut_palm(s, (x, y, z), rng.uniform(7, 12), lean=rng.normal(0, 0.5, 2), seed=int(rng.integers(1e6)),
                           fronds=12 if lite else 14)
        elif k < 0.4:
            V.bamboo_clump(s, (x, y, z), rng.uniform(5, 8), culms=8 if lite else 12, seed=int(rng.integers(1e6)))
        elif k < 0.55:
            V.ginger_clump(s, (x, y, z), 1.0, n=7, height=1.8, seed=int(rng.integers(1e6)), mat="leaf_dark")
        else:
            V.broadleaf_tree(s, (x, y, z), rng.uniform(7, 13), rng.uniform(2.5, 4.5), seed=int(rng.integers(1e6)))
        placed += 1
    n_blobs, placed = (70 if lite else (260 if V.LOD["level"] == "high" else 110)), 0
    while placed < n_blobs:
        x, y = rng.uniform(-26, 82), rng.uniform(-44, 60)
        if blocked(x, y):
            continue
        z = z_at(x, y)
        if z < ST.Z_SEA + 1.5:
            continue
        V.forest_blob(s, (x, y, z), rng.uniform(2.0, 3.6), seed=int(rng.integers(1e6)))
        placed += 1
    # garden plants along the boundary walls inside the property
    for k, x in enumerate(np.arange(-12.5, -6.5, 1.5)):
        V.ginger_clump(s, (x, 18.0, 0.0), 0.6, n=6, height=1.5, seed=500 + k)
    for k, y in enumerate(np.arange(-1.0, 19.0, 2.5)):
        V.ginger_clump(s, (28.2 + 0.0 * k, y, ST.Z_UP), 0.4, n=4, height=1.0, seed=520 + k)
    s.note("site", f"{n_trees} trees / palms / bamboo clumps on the slopes around the property")


def build(s, lite=False):
    xs, ys, H = terrain(s, lite)
    far_terrain(s, step=8.0 if lite else 5.0)
    shore_rocks(s, H, xs, ys)
    sea(s)
    horizon(s)
    neighbours(s)
    vegetation(s, H, xs, ys, lite)
