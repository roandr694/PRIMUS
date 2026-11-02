"""One large terrain shared by all robots (like Isaac Lab terrains): a grid of tiles.

Rows = difficulty level (0 = easy), columns = terrain type. All robots are on the same heightfield; each one
spawns at a tile centre and may drive into the neighbours. In MuJoCo Warp each robot is a separate "world"
(they do not see or collide with each other), but the terrain is the same.

Types for a wheeled robot without a camera (obstacles it should cross by feel):
  flat       flat
  rough      smooth noise (gravel, bumps) up to +-3 cm
  slope      ramp pyramid (up and down) up to 15 deg
  waves      smooth waves / speed bumps up to 6 cm
  steps      small ring steps up to 5 cm (the edge spans one heightfield cell)
  blocks     random discrete blocks up to 5 cm
"""
import numpy as np

TYPES = ("flat", "rough", "slope", "waves", "steps", "blocks")


def _tile(kind, level, n_levels, n, res, rng):
    """Heights [m] of an n x n tile (cell `res` m)."""
    d = level / max(n_levels - 1, 1)                 # 0..1
    u = (np.arange(n) - (n - 1) / 2) * res          # coordenadas locais [m]
    X, Y = np.meshgrid(u, u, indexing="xy")
    half = (n - 1) / 2 * res
    if kind == "flat":
        return np.zeros((n, n))
    if kind == "rough":
        amp = 0.005 + 0.025 * d
        z = rng.uniform(-amp, amp, (n // 4 + 2, n // 4 + 2))
        # interpolate coarse noise so it is smooth (stones of ~10-20 cm)
        xi = np.linspace(0, z.shape[1] - 1, n)
        zz = np.array([np.interp(xi, np.arange(z.shape[1]), row) for row in z])
        yi = np.linspace(0, z.shape[0] - 1, n)
        return np.array([np.interp(yi, np.arange(z.shape[0]), col) for col in zz.T]).T
    if kind == "slope":
        grade = np.tan(np.radians(2.0 + 13.0 * d))
        r = np.maximum(np.abs(X), np.abs(Y))
        top = 0.25 * half
        return grade * np.clip(half - np.maximum(r, top), 0, None)
    if kind == "waves":
        amp = 0.01 + 0.05 * d
        wl = 1.5 - 0.6 * d
        return amp * 0.5 * (np.sin(2 * np.pi * X / wl) + np.sin(2 * np.pi * Y / (wl * 1.3)))
    if kind == "steps":
        h = 0.01 + 0.04 * d
        ring = 0.5
        r = np.maximum(np.abs(X), np.abs(Y))
        k = np.floor((half - r) / ring)
        k = np.clip(k, 0, None)
        return h * np.minimum(k, 5 - np.abs(k - 5))   # sobe 5 degraus e desce (anel)
    if kind == "blocks":
        z = np.zeros((n, n))
        hmax = 0.01 + 0.04 * d
        nb = int(20 + 30 * d)
        for _ in range(nb):
            w = rng.integers(int(0.2 / res), int(0.6 / res))
            i, j = rng.integers(0, n - w, 2)
            z[i:i + w, j:j + w] = max(z[i:i + w, j:j + w].max(), rng.uniform(0.3, 1.0) * hmax)
        c = slice(n // 2 - int(0.6 / res), n // 2 + int(0.6 / res))
        z[c, c] = 0.0                                  # clear spawn area
        return z
    raise ValueError(kind)


def make_terrain(n_levels=6, types=TYPES, tile=8.0, res=0.08, border=4.0, seed=0):
    """Builds the heightfield. Returns a dict with heights[nrow(y), ncol(x)], the size and the tile centres.

    res = 8 cm: in MuJoCo Warp the tire x heightfield collision tests every cell below the tire and the cost
    grows fast with resolution (4096 robots: 2.3 ms per step at 8 cm, 4.4 ms at 4 cm, 81 ms at 2 cm).
    Consequence: a step edge becomes an 8 cm wide ramp."""
    rng = np.random.default_rng(seed)
    n = int(round(tile / res)) + 1
    nb = int(round(border / res))
    ny, nx = n_levels * (n - 1) + 1 + 2 * nb, len(types) * (n - 1) + 1 + 2 * nb
    H = np.zeros((ny, nx))
    centers = np.zeros((n_levels, len(types), 2))
    for lv in range(n_levels):
        for t, kind in enumerate(types):
            z = _tile(kind, lv, n_levels, n, res, rng)
            i0, j0 = nb + lv * (n - 1), nb + t * (n - 1)
            H[i0:i0 + n, j0:j0 + n] = z
    size_x, size_y = (nx - 1) * res, (ny - 1) * res
    for lv in range(n_levels):
        for t in range(len(types)):
            centers[lv, t] = (-size_x / 2 + (nb + t * (n - 1) + (n - 1) / 2) * res,
                              -size_y / 2 + (nb + lv * (n - 1) + (n - 1) / 2) * res)
    return dict(heights=H.astype(np.float32), size_x=size_x, size_y=size_y, res=res, tile=tile,
                n_levels=n_levels, types=tuple(types), centers=centers.astype(np.float32), border=border)


def sample_heights(H, size_x, size_y, xy):
    """Terrain height (bilinear) at points xy [..., 2] (torch). H: tensor [nrow(y), ncol(x)]."""
    import torch
    ny, nx = H.shape
    u = ((xy[..., 0] + size_x / 2) / size_x * (nx - 1)).clamp(0, nx - 1.001)
    v = ((xy[..., 1] + size_y / 2) / size_y * (ny - 1)).clamp(0, ny - 1.001)
    j, i = u.long(), v.long()
    fu, fv = u - j, v - i
    h00, h01 = H[i, j], H[i, j + 1]
    h10, h11 = H[i + 1, j], H[i + 1, j + 1]
    return (h00 * (1 - fu) + h01 * fu) * (1 - fv) + (h10 * (1 - fu) + h11 * fu) * fv
