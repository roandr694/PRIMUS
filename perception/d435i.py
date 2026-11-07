"""Intel RealSense D435i depth camera: mount, intrinsics, noise model and depth image -> point cloud.

Shared by the simulator (rendered depth) and the robot (librealsense frames). Frames:
  * optical frame (as librealsense): x right, y down, z forward (depth);
  * base frame of the robot: x lateral (towards the xp leg), y forward, z up.

Mount (CAD export of 2026-10-30): middle of the front cover, looking straight forward (+Y), no tilt; depth
origin = left imager, 17.5 mm to the camera's left (-X) of the body centre and 4.2 mm behind the glass (from the
datasheet, +-2 mm). Level, it sees the ground from ~0.85 m ahead of the wheel axle (nominal legs); the map
keeps what was seen, so the area under and just ahead of the robot is known from the approach, but right after
start-up it is not: start on open ground. The front arms / knees appear in the lower (or, crouched, upper)
side corners at 0.16-0.22 m, closer than the minimum range, so they drop out.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class D435i:
    pos_base: tuple = (-0.0292, 0.1643, -0.0003)   # [m] depth origin in the base frame (CAD)
    tilt_deg: float = 0.0                    # pitch down from the base forward axis (CAD: level)
    width: int = 212                         # depth stream after decimation (848x480 / 4)
    height: int = 120
    hfov_deg: float = 87.0
    vfov_deg: float = 58.0
    min_range: float = 0.28
    max_range: float = 3.0
    subpixel: float = 0.08                   # stereo matching RMS [px] (Intel: 0.05-0.1)
    baseline: float = 0.050                  # [m]

    @property
    def fx(self):
        return 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self):
        return 0.5 * self.height / math.tan(math.radians(self.vfov_deg) / 2)

    def depth_sigma(self, z):
        """RMS depth error [m]: stereo error grows with the square of the distance (~1.4 cm at 2 m)."""
        return self.subpixel * np.square(z) / (self.fx * 4 * self.baseline)    # fx of the full 848 px stream

    def R_base_optical(self):
        """Rotation optical -> base: optical z = forward tilted down, x = right (base +x), y = down."""
        t = math.radians(self.tilt_deg)
        fwd = np.array([0.0, math.cos(t), -math.sin(t)])
        right = np.array([1.0, 0.0, 0.0])
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd], 1)

    def add_noise(self, depth, rng):
        """Simulated sensor artefacts on a perfect depth image [m] (0 = no return)."""
        d = depth.astype(np.float32).copy()
        valid = (d > self.min_range) & (d < self.max_range)
        d[valid] += rng.standard_normal(valid.sum()).astype(np.float32) * self.depth_sigma(d[valid])
        holes = rng.random(d.shape) < 0.02                         # random dropouts
        edges = np.zeros_like(valid)                               # no matches at depth discontinuities
        jump = np.abs(np.diff(depth, axis=1)) > 0.08 * np.maximum(depth[:, 1:], 0.3)
        edges[:, 1:] |= jump
        edges[:, :-1] |= jump
        d[~valid | holes | edges] = 0.0
        return d

    def points(self, depth, stride=1):
        """Depth image [m] -> (points in the optical frame [N, 3], sigma [N])."""
        v, u = np.mgrid[0:self.height:stride, 0:self.width:stride]
        z = depth[::stride, ::stride]
        ok = (z > self.min_range) & (z < self.max_range)
        ok &= (v > 0) & (v < self.height - 1) & (u > 0) & (u < self.width - 1)     # border pixels are unreliable
        z, u, v = z[ok], u[ok] + 0.5, v[ok] + 0.5
        x = (u - self.width / 2) / self.fx * z
        y = (v - self.height / 2) / self.fy * z
        return np.stack([x, y, z], 1), self.depth_sigma(z)

    def to_world(self, pts_optical, base_pos, base_R):
        """Optical-frame points -> world, given the base pose (position [3], rotation base->world [3, 3])."""
        Rbo = self.R_base_optical()
        origin = base_pos + base_R @ np.asarray(self.pos_base)
        return pts_optical @ (base_R @ Rbo).T + origin, origin
