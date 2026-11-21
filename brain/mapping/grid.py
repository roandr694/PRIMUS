"""2-D log-odds occupancy grid in a floor's map frame (5 cm cells, fixed extent), with ray clearing.

    g = Grid(extent=(-30, 50, -20, 40))
    g.integrate(origin_xy, hits_xy, max_xy=None)    # hits: lidar returns; max_xy: beams with no return (clear only)
    g.occupied(), g.free(), g.unknown(), g.likelihood_field(sigma) (for the scan matcher)
"""
import math

import numpy as np
from scipy import ndimage


class Grid:
    def __init__(self, extent=(-30.0, 50.0, -20.0, 40.0), res=0.05, l_hit=0.9, l_miss=-0.35, l_min=-2.5, l_max=3.5,
                 clear_margin=2.5):
        x0, x1, y0, y1 = extent
        self.res, self.clear_margin = res, clear_margin
        self.origin = np.array([x0, y0], float)
        self.nx, self.ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        self.L = np.zeros((self.ny, self.nx), np.float32)
        self.l_hit, self.l_miss, self.l_min, self.l_max = l_hit, l_miss, l_min, l_max
        self.version = 0                                     # bumps on every update (cache key for derived maps)
        self._lf = None

    # ------------------------------------------------------------------------------------------------ indexing
    def ij(self, xy):
        xy = np.asarray(xy, float)
        j = np.floor((xy[..., 0] - self.origin[0]) / self.res).astype(np.int64)
        i = np.floor((xy[..., 1] - self.origin[1]) / self.res).astype(np.int64)
        return i, j

    def xy(self, i, j):
        return np.stack([self.origin[0] + (np.asarray(j) + 0.5) * self.res,
                         self.origin[1] + (np.asarray(i) + 0.5) * self.res], -1)

    def inside(self, i, j):
        return (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)

    # ------------------------------------------------------------------------------------------------ update
    def integrate(self, origin_xy, hits_xy, max_xy=None, max_clear=8.0):
        """Rays from origin to every hit (cleared), hit cells raised; max_xy: end points of beams with no return
        (cleared up to max_clear)."""
        o = np.asarray(origin_xy, float)
        ends = [np.asarray(hits_xy, float).reshape(-1, 2)]
        if max_xy is not None and len(max_xy):
            m = np.asarray(max_xy, float).reshape(-1, 2)
            d = m - o
            r = np.linalg.norm(d, axis=1, keepdims=True)
            ends.append(o + d * np.minimum(1.0, max_clear / np.maximum(r, 1e-6)))
        E = np.concatenate(ends) if len(ends) > 1 else ends[0]
        if len(E) == 0:
            return
        d = E - o
        r = np.linalg.norm(d, axis=1)
        n = int(math.ceil(float(r.max()) / (0.7 * self.res))) + 1
        s = np.linspace(0.0, 1.0, n, endpoint=False)
        pts = o + d[:, None, :] * s[None, :, None]
        keep = (s[None, :] * r[:, None]) < (r[:, None] - self.clear_margin * self.res)   # keep the wall surface
        fi, fj = self.ij(pts[keep])
        ok = self.inside(fi, fj)
        free = np.unique(fi[ok] * self.nx + fj[ok])
        hits = np.asarray(hits_xy, float).reshape(-1, 2)
        hi, hj = self.ij(hits)
        ok = self.inside(hi, hj)
        occ = np.unique(hi[ok] * self.nx + hj[ok])
        free = np.setdiff1d(free, occ, assume_unique=True)
        Lf = self.L.reshape(-1)
        Lf[free] += self.l_miss
        Lf[occ] += self.l_hit
        np.clip(self.L, self.l_min, self.l_max, out=self.L)
        self.version += 1

    # ------------------------------------------------------------------------------------------------ queries
    def occupied(self, th=0.8):
        return self.L > th

    def free(self, th=-0.5):
        return self.L < th

    def unknown(self):
        return ~self.occupied() & ~self.free()

    def likelihood_field(self, sigma=0.08):
        """exp(-d^2 / 2 sigma^2), d = distance to the nearest occupied cell (cached per grid version)."""
        if self._lf is not None and self._lf[0] == self.version:
            return self._lf[1]
        occ = self.occupied()
        if not occ.any():
            lf = np.zeros_like(self.L)
        else:
            dist = ndimage.distance_transform_edt(~occ) * self.res
            lf = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
        self._lf = (self.version, lf)
        return lf

    def known_bbox(self, pad=0.5):
        known = ~self.unknown()
        if not known.any():
            return None
        ii, jj = np.nonzero(known)
        lo = self.xy(ii.min(), jj.min()) - pad
        hi = self.xy(ii.max(), jj.max()) + pad
        return lo, hi

    # ------------------------------------------------------------------------------------------------ io
    def to_dict(self):
        return dict(L=self.L, origin=self.origin, res=self.res)

    @classmethod
    def from_arrays(cls, L, origin, res):
        g = cls.__new__(cls)
        g.res, g.origin = float(res), np.asarray(origin, float)
        g.L = np.asarray(L, np.float32).copy()
        g.ny, g.nx = g.L.shape
        g.l_hit, g.l_miss, g.l_min, g.l_max = 0.9, -0.35, -2.5, 3.5
        g.version, g._lf, g.clear_margin = 0, None, 2.5
        return g
