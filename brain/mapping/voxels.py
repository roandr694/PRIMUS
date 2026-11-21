"""3-D occupancy of a floor from the depth camera (log-odds voxels, map frame), the camera-first obstacle map
(docs/home_autonomy_architecture.md §11, phase C).

The 2-D lidar sits at the top of the robot and sees one plane: table tops, chair seats and legs, low shelves, a
shoe on the floor are invisible to it. The D435i sees them in 3-D. Every depth frame is integrated here (hits
raise a voxel, the ray before the hit lowers the voxels it crosses, so a chair that left is carved away), and the
costmap takes as obstacles every column with an occupied voxel between the wheels' step height and the robot's
current top (it changes with the leg length).

    vm = VoxelMap(grid2d, floor_z)                  # same x/y cells as the floor's lidar grid
    vm.integrate(origin_map3, pts_map3)             # camera origin and depth points in the map frame
    occ = vm.columns(z0=0.06, z1=0.75)              # [ny, nx] bool, heights above the floor
    vm.observed()                                   # columns the camera has seen (free or not)

Memory: int8 log-odds, 0.1 m vertical cells from 0.15 m below to 1.85 m above the floor -> 20 bytes per 5 cm
column (34 MB for the default 95 x 45 m extent; the Jetson keeps one floor loaded).
"""
import math

import numpy as np

L_HIT, L_MISS, L_MIN, L_MAX, L_OCC = 9, -3, -30, 40, 8          # log-odds x10


class VoxelMap:
    def __init__(self, grid, floor_z, dz=0.10, z_lo=-0.15, z_hi=1.85):
        # layer boundaries at floor +-0.05, +0.15, ...: the floor's own returns (noise, a slightly wrong floor
        # height) stay in the layer [-0.05, 0.05) and never count as obstacles
        self.res, self.origin = grid.res, np.asarray(grid.origin, float).copy()
        self.ny, self.nx = grid.L.shape
        self.floor_z, self.dz, self.z_lo = float(floor_z), dz, z_lo
        self.nz = int(round((z_hi - z_lo) / dz))
        self.L = np.zeros((self.ny, self.nx, self.nz), np.int8)
        self.version = 0
        self._col = None

    def _ijk(self, p):
        j = np.floor((p[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        i = np.floor((p[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        k = np.floor((p[:, 2] - self.floor_z - self.z_lo) / self.dz).astype(np.int64)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx) & (k >= 0) & (k < self.nz)
        return i, j, k, ok

    def _flat(self, i, j, k):
        return (i * self.nx + j) * self.nz + k

    def integrate(self, origin, pts, max_range=3.0, carve_stride=2):
        """origin [3], pts [N, 3] (map frame). Points beyond max_range only carve free space up to max_range."""
        o = np.asarray(origin, float)
        pts = np.asarray(pts, float).reshape(-1, 3)
        if len(pts) == 0:
            return
        d = pts - o
        r = np.linalg.norm(d, axis=1)
        hit = r <= max_range
        # carve: samples every 0.6 cell along every carve_stride-th ray, stopping 1.5 cells before the hit
        c = d[::carve_stride]
        rc = r[::carve_stride]
        end = np.minimum(rc, max_range) - 1.5 * self.res
        n = int(math.ceil(float(end.max()) / (0.6 * self.res))) + 1 if (end > 0).any() else 0
        free_idx = np.zeros(0, np.int64)
        if n > 0:
            s = np.arange(n) * 0.6 * self.res
            u = c / np.maximum(rc, 1e-6)[:, None]
            P = o + u[:, None, :] * s[None, :, None]
            keep = s[None, :] < end[:, None]
            i, j, k, ok = self._ijk(P[keep])
            free_idx = np.unique(self._flat(i[ok], j[ok], k[ok]))
        i, j, k, ok = self._ijk(pts[hit])
        hit_idx = np.unique(self._flat(i[ok], j[ok], k[ok]))
        free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)
        Lf = self.L.reshape(-1)
        v = Lf[free_idx].astype(np.int16) + L_MISS
        Lf[free_idx] = np.clip(v, L_MIN, L_MAX).astype(np.int8)
        v = Lf[hit_idx].astype(np.int16) + L_HIT
        Lf[hit_idx] = np.clip(v, L_MIN, L_MAX).astype(np.int8)
        self.version += 1

    def columns(self, z0=0.06, z1=0.75, th=L_OCC):
        """Columns with an occupied voxel between z0 and z1 above the floor."""
        key = (self.version, round(z0, 2), round(z1, 2), th)
        if self._col is not None and self._col[0] == key:
            return self._col[1]
        k0 = max(int(math.ceil((z0 - self.z_lo) / self.dz - 1e-6)), 0)    # layers entirely above z0
        k1 = min(int(math.ceil((z1 - self.z_lo) / self.dz)), self.nz)
        occ = (self.L[:, :, k0:k1] >= th).any(2)
        self._col = (key, occ)
        return occ

    def observed(self):
        return (self.L != 0).any(2)

    def points(self, th=L_OCC):
        """Centres of the occupied voxels [N, 3] (map frame), for views and the semantic memory."""
        i, j, k = np.nonzero(self.L >= th)
        return np.stack([self.origin[0] + (j + 0.5) * self.res, self.origin[1] + (i + 0.5) * self.res,
                         self.floor_z + self.z_lo + (k + 0.5) * self.dz], 1)

    # ------------------------------------------------------------------------------------------------ io
    def save(self, path):
        np.savez_compressed(path, L=self.L, origin=self.origin, res=self.res, floor_z=self.floor_z, dz=self.dz,
                            z_lo=self.z_lo)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        v = cls.__new__(cls)
        v.L = z["L"].astype(np.int8)
        v.origin, v.res = z["origin"], float(z["res"])
        v.floor_z, v.dz, v.z_lo = float(z["floor_z"]), float(z["dz"]), float(z["z_lo"])
        v.ny, v.nx, v.nz = v.L.shape
        v.version, v._col = 0, None
        v.from_tour = True                      # built during the tour: a reference the camera can match against
        return v


def depth_to_map(frame, T, stride=2):
    """Depth frame (odom-frame pose) -> (camera origin, points) in the map frame. T: map <- odom (x, y, th).
    Pixels with no return are skipped: on the D435i they are dropouts and edges as often as open space, so they
    must not carve."""
    cam = frame.cam
    d = frame.depth[::stride, ::stride].astype(np.float64)
    v, u = np.mgrid[0:cam.height:stride, 0:cam.width:stride]
    v, u = v[:d.shape[0], :d.shape[1]] + 0.5, u[:d.shape[0], :d.shape[1]] + 0.5
    ok = (d > cam.min_range) & (v > 1) & (v < cam.height - 1) & (u > 1) & (u < cam.width - 1)
    z, u, v = d[ok], u[ok], v[ok]
    x = (u - cam.width / 2) / cam.fx * z
    y = (v - cam.height / 2) / cam.fy * z
    P = np.stack([x, y, z], 1)
    Rbo = cam.R_base_optical()
    pb, Rb = frame.pose.p, frame.pose.R
    origin = pb + Rb @ np.asarray(cam.pos_base)
    W = (P @ Rbo.T) @ Rb.T + origin                  # odom frame
    c, s = math.cos(T.th), math.sin(T.th)
    Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    tm = np.array([T.x, T.y, 0.0])
    return Rm @ origin + tm, W @ Rm.T + tm


class ColumnGrid:
    """The voxel map's obstacle columns (one height band) seen as a 2-D grid for the correlative matcher: the
    camera's depth points are matched against it like a lidar scan (phase D, camera pose measurement). The
    likelihood field is refreshed at most every `refresh` seconds (a distance transform of the whole floor)."""

    def __init__(self, vm, z0=0.10, z1=1.6, refresh=3.0):
        self.vm, self.z0, self.z1, self.refresh = vm, z0, z1, refresh
        self.origin, self.res = vm.origin, vm.res
        self._lf, self._t = None, -1e9

    def occupied(self):
        return self.vm.columns(self.z0, self.z1)

    def likelihood_field(self, sigma=0.08, now=None):
        from scipy import ndimage
        if self._lf is not None and now is None:                    # the matcher's call: use the cache
            return self._lf[1]
        t = now if now is not None else 0.0
        if self._lf is None or self._lf[0] != sigma or t - self._t > self.refresh:
            occ = self.occupied()
            dist = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, 9.0)
            self._lf = (sigma, np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32))
            self._t = t
        return self._lf[1]


def depth_local_points(frame, floor_z, z0=0.10, z1=1.6, stride=3, max_pts=180):
    """Depth points in a height band, in the robot's local 2-D frame (x forward, y left, math convention at the
    odom heading), for matching against a ColumnGrid."""
    class _T:
        x = y = th = 0.0
    _, W = depth_to_map(frame, _T, stride=stride)            # odom frame
    W = W[(W[:, 2] > floor_z + z0) & (W[:, 2] < floor_z + z1)]
    if len(W) > max_pts:
        W = W[np.linspace(0, len(W) - 1, max_pts).astype(int)]
    p = frame.pose
    th = math.atan2(-p.R[0, 1], p.R[1, 1]) + math.pi / 2          # base forward (+Y) -> math heading
    c, s = math.cos(th), math.sin(th)
    rel = W[:, :2] - p.p[:2]
    return np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], 1), (p.p[0], p.p[1], th)
