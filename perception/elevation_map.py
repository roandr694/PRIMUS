"""Robot-centric 2.5D elevation map with memory: ground height + ceiling (lowest overhead surface) per cell.

The map is a fixed n x n window of `res` cells around the robot, stored as a ring buffer indexed by the global
cell coordinates (no copying when the robot moves; a slot that changes cell is cleared). Each depth frame:
  * points are binned into cells; per cell the lowest point starts a "lower layer" (points less than `gap`
    above it): its highest point is the ground measurement; points `gap` or more above the ground estimate
    are ceiling measurements (bars, table tops, the underside of a stair overhang);
  * ground is fused with a 1-D Kalman filter per cell (measurement variance from the depth noise, process
    noise every frame, so old cells slowly lose confidence); a measurement far from the estimate inflates the
    variance first, so a real change (a moved obstacle) wins after a couple of frames;
  * a ground measurement `gap` or more above a confident ground estimate is treated as a ceiling (a bar seen
    without the floor below it in the same frame);
  * a ceiling must be `gap` above the fused ground and seen in `ceil_frames` frames before it counts (single
    frames at stair edges and depth outliers are not overhangs); a frame that sees the cell without anything
    overhead lowers the count.
All heights are in the odometry (world) frame. Numpy only (runs on the Jetson CPU at 15 Hz with ~6k points).
"""
import numpy as np


class ElevationMap:
    def __init__(self, size=6.0, res=0.04, gap=0.25, process_var=1e-6, min_var=4e-6, ceil_frames=3):
        self.n = int(round(size / res))
        self.res, self.gap = res, gap
        self.process_var, self.min_var, self.ceil_frames = process_var, min_var, ceil_frames
        n = self.n
        self.h = np.full((n, n), np.nan, np.float32)          # ground height
        self.var = np.full((n, n), np.inf, np.float32)
        self.ceil = np.full((n, n), np.inf, np.float32)       # lowest overhead surface (+inf = open)
        self.ceil_n = np.zeros((n, n), np.int16)               # frames that saw it
        self.tag = np.full((n, n, 2), np.iinfo(np.int32).min, np.int32)   # global cell held by each slot
        self.center = np.zeros(2, np.int64)                   # global cell of the robot

    # ---------------- indexing ----------------
    def cell(self, xy):
        return np.floor(np.asarray(xy) / self.res).astype(np.int64)

    def move_to(self, xy):
        self.center = self.cell(xy)

    def _in_window(self, c):
        d = c - self.center
        return np.all((d >= -self.n // 2) & (d < self.n - self.n // 2), axis=-1)

    def _slots(self, c):
        return (c[..., 0] % self.n).astype(np.int64), (c[..., 1] % self.n).astype(np.int64)

    # ---------------- update ----------------
    def integrate(self, pts, sigma):
        """pts [N, 3] in the world frame, sigma [N] depth noise of each point."""
        self.var += self.process_var
        c = self.cell(pts[:, :2])
        ok = self._in_window(c)
        pts, sigma, c = pts[ok], sigma[ok], c[ok]
        if len(pts) == 0:
            return
        i, j = self._slots(c)
        # clear slots that now hold another cell
        stale = np.any(self.tag[i, j] != c, axis=-1)
        if stale.any():
            si, sj = i[stale], j[stale]
            self.h[si, sj], self.var[si, sj], self.ceil[si, sj] = np.nan, np.inf, np.inf
            self.ceil_n[si, sj] = 0
            self.tag[si, sj] = c[stale]
        flat = i * self.n + j
        z = pts[:, 2]
        nn = self.n * self.n
        zmin = np.full(nn, np.inf, np.float32)
        np.minimum.at(zmin, flat, z)
        lower = z < zmin[flat] + self.gap
        g = np.full(nn, -np.inf, np.float32)
        np.maximum.at(g, flat[lower], z[lower])
        s = np.full(nn, np.inf, np.float32)
        np.minimum.at(s, flat[lower], sigma[lower])
        upper_z = np.full(nn, np.inf, np.float32)
        np.minimum.at(upper_z, flat[~lower], z[~lower])
        cells = np.unique(flat)
        ci, cj = cells // self.n, cells % self.n
        m, mv = g[cells], np.maximum(np.square(s[cells]), self.min_var)
        h, v = self.h[ci, cj], self.var[ci, cj]
        known = np.isfinite(h) & (v < 0.02 ** 2)
        # a "ground" far above a confident ground = something overhead seen without the floor below
        overhead = known & (m > h + self.gap)
        up = np.where(overhead, m, upper_z[cells])
        up = np.where(known & (up < h + self.gap), np.inf, up)          # must clear the fused ground too
        seen = np.isfinite(up)
        self.ceil[ci, cj] = np.where(seen, np.minimum(self.ceil[ci, cj], up), self.ceil[ci, cj])
        self.ceil_n[ci, cj] = np.clip(self.ceil_n[ci, cj] + np.where(seen, 1, -1), 0, 2 * self.ceil_frames)
        # cleared when nothing overhead is confirmed any more, or the ground reaches it
        gone = (self.ceil_n[ci, cj] == 0) | (~overhead & (m >= self.ceil[ci, cj] - 0.05))
        self.ceil[ci, cj] = np.where(gone, np.inf, self.ceil[ci, cj])
        fuse = ~overhead
        h, v, m, mv, ci, cj = h[fuse], v[fuse], m[fuse], mv[fuse], ci[fuse], cj[fuse]
        new = ~np.isfinite(h)
        h = np.where(new, m, h)
        v = np.where(new, mv, v)
        v = np.maximum(v, np.square(m - h) - mv)              # disagreement -> less confident first
        k = v / (v + mv)
        self.h[ci, cj] = h + k * (m - h)
        self.var[ci, cj] = (1 - k) * v

    # ---------------- queries ----------------
    def lookup(self, xy):
        """Ground height, its variance and ceiling at world points xy [..., 2] (nan / inf where unknown)."""
        c = self.cell(xy)
        i, j = self._slots(c)
        valid = self._in_window(c) & np.all(self.tag[i, j] == c, axis=-1)
        h = np.where(valid, self.h[i, j], np.nan)
        v = np.where(valid, self.var[i, j], np.inf)
        ce = np.where(valid & (self.ceil_n[i, j] >= self.ceil_frames), self.ceil[i, j], np.inf)
        return h, v, ce
