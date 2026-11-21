"""Layered 2-D costmap of the current floor, in the map frame (docs/home_autonomy_architecture.md §4.2).

    static    SLAM grid of the tour (occupied = lethal, unknown = lethal beyond `unknown_ok` metres of the robot)
    live      lidar returns of the last seconds (with decay); cells seen free again and again clear a static
              obstacle (a chair that moved), cells hit block a free one (a bag on the floor)
    terrain   height jumps in the elevation map (> 5 cm between neighbours: step, curb, drop edge) = lethal for
              driving; skills cross them through nav-graph edges, never through the costmap
    keepout   rectangles the owner forbids ("não entra no banheiro")
    inflation robot radius lethal, then a cost that decays to 1 at `soft` metres

    cm = Costmap(slam_grid)
    cm.update_live(scan_points_map_xy, origin_xy, stamp); cm.update_terrain(emap, T_map_odom)
    cost = cm.cost(robot_xy)          # float64 [ny, nx]: inf = forbidden, >= 1 elsewhere (for brain.nav.astar)
"""
import math

import numpy as np
from scipy import ndimage


class Costmap:
    def __init__(self, grid, robot_r=0.31, soft=0.85, unknown_ok=1.0, live_decay=8.0):
        self.g = grid                                     # brain.mapping.grid.Grid (static, from the tour)
        self.res, self.origin = grid.res, grid.origin
        self.ny, self.nx = grid.L.shape
        self.robot_r, self.soft, self.unknown_ok, self.live_decay = robot_r, soft, unknown_ok, live_decay
        self.live_hit = np.zeros((self.ny, self.nx), np.float32)       # time of the last hit
        self.live_hit[:] = -1e9
        self.live_free = np.zeros((self.ny, self.nx), np.float32)      # count of recent free observations
        self.terrain = np.zeros((self.ny, self.nx), bool)          # confirmed height edges
        self.terrain_seen = np.zeros((self.ny, self.nx), bool)
        self.t_edge = np.zeros((self.ny, self.nx), np.float32)       # edge evidence (+1 seen as edge, -1 as flat)
        self.keepout = np.zeros((self.ny, self.nx), bool)
        self.blocked = np.zeros((self.ny, self.nx), bool)              # navigator: corridors found impassable
        self.now = 0.0
        self.terrain_lethal_r = 2.0
        self._cache = None

    def ij(self, xy):
        return self.g.ij(xy)

    def xy(self, i, j):
        return self.g.xy(i, j)

    # ------------------------------------------------------------------------------------------------ layers
    def update_live(self, hits_xy, origin_xy, stamp, max_range=6.0):
        self.now = stamp
        hits = np.asarray(hits_xy, float).reshape(-1, 2)
        o = np.asarray(origin_xy, float)
        if len(hits) == 0:
            return
        d = hits - o
        r = np.linalg.norm(d, axis=1)
        near = r < max_range
        hits, d, r = hits[near], d[near], r[near]
        if len(hits) == 0:
            return
        hi, hj = self.ij(hits)
        ok = self.g.inside(hi, hj)
        self.live_hit[hi[ok], hj[ok]] = stamp
        n = int(math.ceil(float(r.max()) / (0.7 * self.res))) + 1
        s = np.linspace(0.0, 1.0, n, endpoint=False)
        pts = o + d[:, None, :] * s[None, :, None]
        keep = (s[None, :] * r[:, None]) < (r[:, None] - 2.5 * self.res)
        fi, fj = self.ij(pts[keep])
        ok = self.g.inside(fi, fj)
        idx = np.unique(fi[ok] * self.nx + fj[ok])
        lf = self.live_free.reshape(-1)
        lf[idx] = np.minimum(lf[idx] + 1.0, 20.0)
        lh = self.live_hit.reshape(-1)
        lh[idx] = np.minimum(lh[idx], stamp - self.live_decay - 1.0)      # seen free now: forget the old hit
        self._cache = None

    def update_terrain(self, emap, T, jump=0.05, robot_top=0.65):
        """Elevation map (robot-centric, odom frame) -> map-frame terrain layer: height edges between neighbours
        (steps, curbs, drop edges) and low overhead surfaces (chair seats, table tops, shelves) below the robot's
        top `robot_top` (m above the ground). A 2-D lidar at ~0.48 m misses most of those (thin legs, seats under
        its plane); the depth camera sees them."""
        n, res = emap.n, emap.res
        xs = (emap.center[0] + np.arange(-n // 2, n - n // 2)) * res + res / 2
        ys = (emap.center[1] + np.arange(-n // 2, n - n // 2)) * res + res / 2
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        h, _, ceil = emap.lookup(np.stack([X, Y], -1))
        known = np.isfinite(h)
        g_ref = np.nanmedian(h) if known.any() else 0.0
        low = np.isfinite(ceil) & ((ceil - np.where(known, h, g_ref)) < robot_top)
        known = known | low
        hz = np.where(known, h, np.nan)
        edge = np.zeros_like(known)
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = hz
            b = np.roll(np.roll(hz, -di, 0), -dj, 1)
            dd = np.abs(a - b)
            e = np.nan_to_num(dd, nan=0.0) > jump
            edge |= e | np.roll(np.roll(e, di, 0), dj, 1)
        edge |= low
        c, s = math.cos(T.th), math.sin(T.th)
        mx = T.x + c * X - s * Y
        my = T.y + s * X + c * Y
        # only the well-observed part of the window (within 2.5 m of its centre: the depth noise grows with range)
        r = np.hypot(X - emap.center[0] * res, Y - emap.center[1] * res)
        i, j = self.ij(np.stack([mx, my], -1))
        ok = self.g.inside(i, j) & known & (r < 2.5)
        ii, jj, e = i[ok], j[ok], edge[ok]
        self.terrain_seen[ii, jj] = True
        ev = self.t_edge
        ev[ii, jj] = np.clip(ev[ii, jj] + np.where(e, 1.0, -1.0), -3.0, 6.0)
        self.terrain[ii, jj] = ev[ii, jj] >= 3.0                      # an edge seen at least 3 more times than flat
        self._cache = None

    def add_keepout(self, x0, x1, y0, y1):
        i0, j0 = self.ij(np.array([x0, y0]))
        i1, j1 = self.ij(np.array([x1, y1]))
        self.keepout[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1] = True
        self._cache = None

    def block_disk(self, xy, r):
        i, j = self.ij(np.asarray(xy))
        k = int(r / self.res) + 1
        ii, jj = np.mgrid[-k:k + 1, -k:k + 1]
        m = ii ** 2 + jj ** 2 <= k * k
        ii, jj = (ii[m] + i), (jj[m] + j)
        ok = self.g.inside(ii, jj)
        self.blocked[ii[ok], jj[ok]] = True
        self._cache = None

    def clear_blocked(self):
        self.blocked[:] = False
        self._cache = None

    # ------------------------------------------------------------------------------------------------ cost
    def lethal(self, robot_xy=None, layers=("live", "terrain"), near=None):
        """layers: which dynamic layers count; near: only within `near` metres of robot_xy (fallback plans)."""
        static_occ = self.g.occupied()
        unknown = self.g.unknown()
        recent_hit = (self.now - self.live_hit) < self.live_decay
        moved = static_occ & (self.live_free >= 4) & ~recent_hit           # a static obstacle that is gone
        dyn = np.zeros_like(static_occ)
        if "live" in layers:
            dyn |= recent_hit
        if "terrain" in layers and robot_xy is not None:
            # camera edges are lethal only near the robot (fresh, short range); farther they are a soft cost
            dyn |= self.terrain & self._disk(robot_xy, self.terrain_lethal_r)
        elif "terrain" in layers:
            dyn |= self.terrain
        if near is not None and robot_xy is not None:
            i, j = self.ij(np.asarray(robot_xy))
            k = int(near / self.res)
            mask = np.zeros_like(dyn)
            mask[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1] = True
            dyn &= mask
        occ = (static_occ & ~moved) | dyn | self.keepout | self.blocked
        unk = unknown & ~recent_hit & (self.live_free < 2)
        if robot_xy is not None and self.unknown_ok > 0:
            i, j = self.ij(np.asarray(robot_xy))
            k = int(self.unknown_ok / self.res)
            sl = (slice(max(i - k, 0), i + k + 1), slice(max(j - k, 0), j + k + 1))
            unk[sl] = False
        return occ, unk

    def _disk(self, xy, r):
        m = np.zeros((self.ny, self.nx), bool)
        i, j = self.ij(np.asarray(xy))
        k = int(r / self.res)
        m[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1] = True
        return m

    def cost(self, robot_xy=None, layers=("live", "terrain"), near=None):
        occ, unk = self.lethal(robot_xy, layers, near)
        dist = ndimage.distance_transform_edt(~occ) * self.res
        c = 1.0 + 25.0 * np.clip((self.soft - dist) / (self.soft - self.robot_r), 0.0, 1.0) ** 2
        if "terrain" in layers and robot_xy is not None:
            far = ndimage.binary_dilation(self.terrain, iterations=4) & ~self._disk(robot_xy, self.terrain_lethal_r)
            c = c + 15.0 * far
        c[dist <= self.robot_r] = np.inf
        c[unk] = np.inf
        return c

    def nearest_free(self, cost, xy, max_r=1.5):
        """Nearest finite-cost cell to xy (goal inside furniture -> the closest reachable spot)."""
        i, j = self.ij(np.asarray(xy))
        k = int(max_r / self.res)
        i0, i1, j0, j1 = max(i - k, 0), min(i + k + 1, self.ny), max(j - k, 0), min(j + k + 1, self.nx)
        sub = cost[i0:i1, j0:j1]
        ok = np.isfinite(sub)
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        d = (ii + i0 - i) ** 2 + (jj + j0 - j) ** 2 + 0.3 * (sub[ii, jj] - 1.0)
        m = np.argmin(d)
        return (int(ii[m] + i0), int(jj[m] + j0))
