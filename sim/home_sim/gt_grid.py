"""Ground-truth traversability grid of the house floors, by ray casting the MuJoCo scene (sim only).

For every 5 cm cell of a floor, one ray goes down from `floor + clear_h`: it must land on a walkable surface near
the floor height (within -0.05 .. +step); otherwise the cell is an obstacle (something between the floor and the
robot's height) or a drop (nothing / far below). Doors and drawers are articulated bodies: they count as obstacles
where they are in the current pose.

    g = GTGrid.build(model, data, floor_z=0.0, bounds=(-10.5, 22.0, -2.0, 17.0))
    g.free  [ny, nx] bool, g.world_to_ij(xy), g.ij_to_world(i, j)

Used to score the robot's maps (P1), and by the tour pilot (a person who knows the house) to plan its route.
"""
import math

import numpy as np
import mujoco
from scipy import ndimage

FLOOR_BOUNDS = {                       # plan-frame extents of the walkable area of each floor (x0, x1, y0, y1)
    "social": (0.0, (-10.5, 22.0, -2.0, 17.0)),
    "upper": (3.6, (-2.6, 27.8, -0.2, 18.8)),
    "sea_deck": (-3.4, (-27.5, -13.0, -2.0, 17.5)),
}


class GTGrid:
    def __init__(self, free, origin, res, floor_z, height):
        self.free, self.origin, self.res, self.floor_z, self.height = free, np.asarray(origin, float), res, floor_z, height

    @classmethod
    def build(cls, m, d, floor_z, bounds, res=0.05, clear_h=0.62, step=0.06, groups=(1, 0, 0, 0, 1, 0)):
        x0, x1, y0, y1 = bounds
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        n = X.size
        geomid = np.zeros(1, np.int32)
        dist = np.zeros(1)
        grp = np.array(groups, np.uint8)
        h = np.full(n, -np.inf)
        top = floor_z + clear_h
        # mj_multiRay casts from one origin; vertical rays need one origin each: use mj_ray in a loop over cells
        # (fast enough: ~1 us per ray with the BVH)
        Xf, Yf = X.ravel(), Y.ravel()
        down = np.array([0.0, 0.0, -1.0])
        for k in range(n):
            dd = mujoco.mj_ray(m, d, np.array([Xf[k], Yf[k], top]), down, grp, 1, -1, geomid)
            if dd >= 0:
                h[k] = top - dd
        h = h.reshape(ny, nx)
        walkable = (h > floor_z - 0.05) & (h < floor_z + step)
        # a ray starting inside a wall does not hit it: rasterise the colliders' oriented boxes on a slice
        # floor + step .. floor + robot height instead
        occ = np.zeros((ny, nx), bool)
        zs = floor_z + np.array([step + 0.04, 0.25, 0.45, 0.65, 0.85])
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        robot_root = m.body_rootid[base]
        for gid in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[gid]] == robot_root or not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                continue
            if m.geom_type[gid] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            c_loc, half = m.geom_aabb[gid, :3], m.geom_aabb[gid, 3:]
            R = d.geom_xmat[gid].reshape(3, 3)
            ctr = d.geom_xpos[gid] + R @ c_loc
            ext = np.abs(R) @ half                                   # world AABB half extents
            if ctr[2] - ext[2] > zs[-1] or ctr[2] + ext[2] < zs[0]:
                continue
            j0, j1 = np.searchsorted(xs, [ctr[0] - ext[0], ctr[0] + ext[0]])
            i0, i1 = np.searchsorted(ys, [ctr[1] - ext[1], ctr[1] + ext[1]])
            if j1 <= j0 or i1 <= i0:
                continue
            P = np.stack(np.meshgrid(xs[j0:j1], ys[i0:i1], indexing="xy"), -1)       # [h, w, 2]
            hit = np.zeros(P.shape[:2], bool)
            for z in zs:
                q = np.concatenate([P, np.full(P.shape[:2] + (1,), z)], -1) - ctr
                loc = q @ R                                                          # world -> geom frame
                hit |= np.all(np.abs(loc) <= half + 1e-6, -1)
            occ[i0:i1, j0:j1] |= hit
        free = walkable & ~occ
        g = cls(free, (x0, y0), res, floor_z, h)
        return g

    def world_to_ij(self, xy):
        xy = np.asarray(xy, float)
        j = np.floor((xy[..., 0] - self.origin[0]) / self.res).astype(int)
        i = np.floor((xy[..., 1] - self.origin[1]) / self.res).astype(int)
        return i, j

    def ij_to_world(self, i, j):
        return np.stack([self.origin[0] + (np.asarray(j) + 0.5) * self.res,
                         self.origin[1] + (np.asarray(i) + 0.5) * self.res], -1)

    def inside(self, i, j):
        return (i >= 0) & (i < self.free.shape[0]) & (j >= 0) & (j < self.free.shape[1])

    def clearance(self):
        """Distance [m] from every cell to the nearest non-free cell."""
        return ndimage.distance_transform_edt(self.free) * self.res

    def save(self, path):
        np.savez_compressed(path, free=self.free, origin=self.origin, res=self.res, floor_z=self.floor_z, height=self.height)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["free"], z["origin"], float(z["res"]), float(z["floor_z"]), z["height"])


def cached(m, d, floor, cache_dir, tag="doors_open", bounds=None):
    """GT grid of a floor with the scene in its current pose (doors!), cached next to the ground truth under `tag`.
    bounds: (floor_z, (x0, x1, y0, y1)); default: the CMA house floors."""
    from pathlib import Path
    path = Path(cache_dir) / f"gt_grid_{floor}_{tag}.npz"
    if path.exists():
        return GTGrid.load(path)
    z, b = bounds if bounds is not None else FLOOR_BOUNDS[floor]
    g = GTGrid.build(m, d, z, b)
    g.save(path)
    return g
