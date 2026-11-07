"""SLAMTEC RPLIDAR A1 (2-D, 360 deg, 5.5 Hz): scan -> world points, obstacles in the travel corridor and a 2-D
occupancy grid (log-odds, ray cleared).

The scan plane is fixed to the chassis, so it tilts with the body pitch: returns that hit the ground (close to
the elevation-map ground, or below the wheel contact) are dropped before they become obstacles.
Mount (CAD export of 2026-10-30): rotation axis at (-0.0116, 0.0942) in the base frame, scan plane 0.066 m above
the base origin (+-4 mm), parallel to the chassis, long axis along Y with the motor to the rear; nothing of the
robot blocks the beam. Here angle 0 = forward, counter-clockwise; the real A1 SDK turns clockwise and its 0 deg
may point forward or to the motor side: check on the robot with an object in front.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RPLidarA1:
    pos_base: tuple = (-0.0116, 0.0942, 0.066)
    n_beams: int = 360
    min_range: float = 0.15
    max_range: float = 12.0

    def directions_base(self):
        """Unit beam directions in the base frame (angle 0 = forward, counter-clockwise)."""
        a = np.arange(self.n_beams) * (2 * math.pi / self.n_beams)
        return np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], 1)

    def sigma(self, r):
        return np.where(r < 3.0, 0.01, np.where(r < 5.0, 0.02, 0.025)) * r

    def add_noise(self, ranges, rng):
        r = ranges.astype(np.float32).copy()
        ok = (r > self.min_range) & (r < self.max_range)
        r[ok] += rng.standard_normal(ok.sum()).astype(np.float32) * self.sigma(r[ok])
        r[~ok | (rng.random(r.shape) < 0.03)] = 0.0
        return r

    def points(self, ranges, base_pos, base_R):
        """ranges [n_beams] (0 = no return) -> (world points [N, 3], sensor origin [3])."""
        origin = base_pos + base_R @ np.asarray(self.pos_base)
        ok = (ranges > self.min_range) & (ranges < self.max_range)
        d = self.directions_base()[ok] @ base_R.T
        return origin + d * ranges[ok, None], origin


def drop_ground(points, ground_fn, contact_z, clearance=0.08, around=0.08):
    """Removes returns on the terrain: lower than `clearance` above the highest map ground within `around`
    (a step riser is terrain, not an obstacle), or below the wheels where the map is unknown."""
    if ground_fn is None or len(points) == 0:
        return points[points[:, 2] > contact_z + clearance] if len(points) else points
    offs = np.array([(0, 0), (around, 0), (-around, 0), (0, around), (0, -around)])
    g = np.stack([ground_fn(points[:, :2] + o) for o in offs], 1)
    ref = np.where(np.isfinite(g).any(1), np.nanmax(np.where(np.isfinite(g), g, -np.inf), 1), contact_z)
    return points[points[:, 2] > ref + clearance]


def corridor_obstacle(points, xy, heading, half_width=0.35, start=0.10):
    """Distance from the axle to the nearest obstacle point in the robot corridor (ahead, behind) [m]."""
    if len(points) == 0:
        return math.inf, math.inf
    fwd = np.array([-math.sin(heading), math.cos(heading)])
    right = np.array([math.cos(heading), math.sin(heading)])
    rel = points[:, :2] - xy
    s, l = rel @ fwd, rel @ right
    inside = np.abs(l) <= half_width
    a, b = s[inside & (s > start)], -s[inside & (s < -start)]
    return (float(a.min()) if len(a) else math.inf), (float(b.min()) if len(b) else math.inf)


class OccupancyGrid:
    """2-D log-odds occupancy grid in the odometry frame (fixed extent; for the pilot map and avoidance)."""

    def __init__(self, size=30.0, res=0.05, l_hit=0.85, l_miss=-0.4, l_min=-2.0, l_max=3.5):
        self.n = int(round(size / res))
        self.res, self.half = res, size / 2
        self.L = np.zeros((self.n, self.n), np.float32)
        self.l_hit, self.l_miss, self.l_min, self.l_max = l_hit, l_miss, l_min, l_max

    def _idx(self, xy):
        return np.floor((np.asarray(xy) + self.half) / self.res).astype(np.int64)

    def update(self, origin_xy, hits_xy):
        if len(hits_xy) == 0:
            return
        o = np.asarray(origin_xy, np.float64)
        d = hits_xy - o
        r = np.linalg.norm(d, axis=1)
        steps = np.arange(0.0, 1.0, self.res / max(float(r.max()), self.res))
        free = (o + d[:, None, :] * steps[None, :, None]).reshape(-1, 2)          # samples along every beam
        fi = self._idx(free)
        hi = self._idx(hits_xy)
        ok = lambda ix: np.all((ix >= 0) & (ix < self.n), axis=1)
        fi, hi = np.unique(fi[ok(fi)], axis=0), np.unique(hi[ok(hi)], axis=0)    # each cell once per scan
        self.L[fi[:, 0], fi[:, 1]] += self.l_miss
        self.L[hi[:, 0], hi[:, 1]] += self.l_hit - self.l_miss * np.isin(hi[:, 0] * self.n + hi[:, 1],
                                                                          fi[:, 0] * self.n + fi[:, 1])
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def occupied(self, threshold=1.0):
        return self.L > threshold
