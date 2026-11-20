"""Multi-floor 2-D lidar SLAM / localisation (scan-to-map, no loop closure: a house tour is short).

    slam = FloorSLAM()                         # mapping mode; or FloorSLAM.load(dir, mapping=False) to localise
    slam.on_state(state)                       # link.State (odom pose, pitch, z)
    slam.on_scan(scan)                         # sim/home_sim LidarScan or the driver equivalent (ranges, odom pose)
    slam.pose()                                # (x, y, th) in the map frame, th = math heading
    slam.floor                                 # current floor id ("f0", "f1", ...), slam.floors[id].z

One map frame for all floors (they share x, y; each floor has its own grid). The map frame starts equal to the
odom frame. While the robot is between levels (stairs, lift) scans are not integrated, and a new floor map is
started (or an existing one re-entered) once it has been at a new height for 1 s.
Scans are projected with the full 3-D body pose and only returns 0.12-1.9 m above the floor are kept, so the
planner's look-down pitch (up to -18 deg) still gives usable scans; beyond 20 deg of tilt they are dropped.
Floor heights are ground heights (base z minus the leg and the wheel radius), not base heights.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import Grid
from .matcher import global_search, match, score_at

LIDAR_POS = (-0.0116, 0.0942, 0.066)        # perception.lidar.RPLidarA1.pos_base


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Floor:
    id: str
    z: float
    grid: Grid
    n_scans: int = 0


@dataclass
class Transform2D:
    """map <- odom"""
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0

    def apply(self, x, y, th):
        c, s = math.cos(self.th), math.sin(self.th)
        return self.x + c * x - s * y, self.y + s * x + c * y, wrap(th + self.th)

    def inverse_apply(self, x, y, th):
        c, s = math.cos(self.th), math.sin(self.th)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy, wrap(th - self.th)

    @classmethod
    def between(cls, odom, mp):
        """T such that T.apply(odom) = mp."""
        th = wrap(mp[2] - odom[2])
        c, s = math.cos(th), math.sin(th)
        return cls(mp[0] - (c * odom[0] - s * odom[1]), mp[1] - (s * odom[0] + c * odom[1]), th)


class FloorSLAM:
    def __init__(self, mapping=True, extent=(-40.0, 55.0, -15.0, 30.0), res=0.05, max_tilt_deg=20.0,
                 min_score=0.30, floor_gap=1.2, gain=0.35, gain_th=0.05, deadband=0.03,
                 deadband_th=0.01, slip_margin=0.12):
        self.mapping, self.extent, self.res, self.gain, self.gain_th = mapping, extent, res, gain, gain_th
        self.deadband, self.deadband_th, self.slip_margin = deadband, deadband_th, slip_margin
        self.max_tilt, self.min_score, self.floor_gap = math.radians(max_tilt_deg), min_score, floor_gap
        self.floors = {}
        self.floor = None
        self.T = Transform2D()
        self.state = None
        self.last_integrated = None
        self.settle = 0.0
        self.pending_z = None
        self.bad = 0
        self.score = 0.0
        self.lost = False
        self.stats = dict(scans=0, matched=0, rejected=0, tilted=0, transit=0, relocalised=0)

    # ------------------------------------------------------------------------------------------------ inputs
    @staticmethod
    def ground_z(state):
        return state.z - state.leg_length * math.cos(state.pitch) - 0.1075

    def on_state(self, state):
        prev = self.state
        self.state = state
        z = self.ground_z(state)
        if self.floor is None:
            self._enter_floor(z)
            return
        dz = z - self.floors[self.floor].z
        if abs(dz) < 0.3:
            self.settle, self.pending_z = 0.0, None
            return
        dt = state.stamp - prev.stamp if prev is not None else 0.0
        if self.pending_z is not None and abs(z - self.pending_z) < 0.1:
            self.settle += dt
        else:
            self.settle, self.pending_z = 0.0, z
        if self.settle > 1.0 and abs(dz) > self.floor_gap:
            self._enter_floor(z)

    def _enter_floor(self, z):
        near = [f for f in self.floors.values() if abs(f.z - z) < self.floor_gap * 0.5]
        if near:
            self.floor = near[0].id
        elif self.mapping:
            fid = f"f{len(self.floors)}"
            self.floors[fid] = Floor(fid, round(float(z), 2), Grid(self.extent, self.res))
            self.floor = fid
        self.settle, self.pending_z, self.last_integrated = 0.0, None, None

    def odom_pose(self):
        s = self.state
        return (s.x, s.y, wrap(s.yaw + math.pi / 2)) if s is not None else (0.0, 0.0, 0.0)

    def pose(self):
        return self.T.apply(*self.odom_pose())

    def in_transit(self):
        return self.floor is None or abs(self.ground_z(self.state) - self.floors[self.floor].z) > 0.3

    def on_scan(self, scan, lidar):
        """scan: ranges + odom base pose (p, R) at capture; lidar: perception.lidar.RPLidarA1."""
        if self.state is None or self.floor is None:
            return
        self.stats["scans"] += 1
        R = scan.pose.R
        if math.acos(max(-1.0, min(1.0, R[2, 2]))) > self.max_tilt:
            self.stats["tilted"] += 1
            return
        if self.in_transit():
            self.stats["transit"] += 1
            return
        pts3, origin = lidar.points(scan.ranges, scan.pose.p, R)
        fz = self.floors[self.floor].z
        keep = (pts3[:, 2] > fz + 0.12) & (pts3[:, 2] < fz + 1.9)
        pts3 = pts3[keep]
        if len(pts3) < 20:
            return
        ox, oy, oth = scan.pose.p[0], scan.pose.p[1], wrap(scan.pose.yaw + math.pi / 2)
        c, s = math.cos(oth), math.sin(oth)
        rel = pts3[:, :2] - np.array([ox, oy])
        local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], 1)
        floor = self.floors[self.floor]
        guess = self.T.apply(ox, oy, oth)
        mp = guess
        if floor.grid.occupied().sum() > 200:
            win = 0.6 if self.lost else 0.3
            mp, score = match(floor.grid, local, guess, win_xy=win, win_th=0.15 if not self.lost else 0.3,
                              prior=0.0 if self.lost else 0.08)
            self.score = score
            if score >= self.min_score:
                # odometry over one scan period is good to ~1 cm: move only part of the way to the match
                # heading: the gyro drifts slowly, scan headings are noisy -> a much lower gain
                # dead band: matches within the matcher's own bias (~1-2 cm, ~0.3 deg in sim) are noise; only the
                # part of the disagreement beyond it is corrected, so odometry is trusted over short distances
                # ... unless the match is clearly better than where odometry puts us (wheel slip, a bump): then
                # the odometry is wrong and the whole correction is taken
                slip = score - score_at(floor.grid, local, guess) > self.slip_margin
                if slip:
                    self.stats["slips"] = self.stats.get("slips", 0) + 1
                g, gth = (1.0, 1.0) if (self.lost or slip) else (self.gain, self.gain_th)
                db, dbth = (0.0, 0.0) if (self.lost or slip) else (self.deadband, self.deadband_th)
                dx, dy, dth = mp[0] - guess[0], mp[1] - guess[1], wrap(mp[2] - guess[2])
                r = math.hypot(dx, dy)
                k = max(0.0, 1.0 - db / r) if r > 1e-9 else 0.0
                dth = math.copysign(max(0.0, abs(dth) - dbth), dth)
                mp = (guess[0] + g * k * dx, guess[1] + g * k * dy, wrap(guess[2] + gth * dth))
                self.T = Transform2D.between((ox, oy, oth), mp)
                self.stats["matched"] += 1
                self.bad, self.lost = 0, False
            else:
                self.stats["rejected"] += 1
                self.bad += 1
                mp = guess
                if self.bad > 15 and not self.mapping:
                    self.lost = True
                    self.relocalise(local, (ox, oy, oth))
                return                                        # never integrate a scan we could not place
        if self.mapping:
            moved = self.last_integrated is None or \
                math.hypot(mp[0] - self.last_integrated[0], mp[1] - self.last_integrated[1]) > 0.08 or \
                abs(wrap(mp[2] - self.last_integrated[2])) > math.radians(3) or \
                scan.stamp - self.last_integrated[3] > 1.0
            if moved:
                cm, sm = math.cos(mp[2]), math.sin(mp[2])
                hits = local @ np.array([[cm, sm], [-sm, cm]]) + np.array(mp[:2])
                lo = np.asarray(lidar.pos_base)
                o = (mp[0] + cm * lo[1] + sm * lo[0], mp[1] + sm * lo[1] - cm * lo[0])   # lidar origin (base x right)
                floor.grid.integrate(o, hits)
                floor.n_scans += 1
                self.last_integrated = (*mp, scan.stamp)

    def relocalise(self, local, odom):
        """Global search on the current floor (kidnapped robot)."""
        g = self.floors[self.floor].grid
        free = g.free()
        ii, jj = np.nonzero(free[::8, ::8])
        cand = g.xy(ii * 8, jj * 8)
        if len(cand) == 0:
            return False
        best = global_search(g, local, cand)
        if best and best[0][0] >= self.min_score + 0.1:
            s, x, y, th = best[0]
            self.T = Transform2D.between(odom, (x, y, th))
            self.lost, self.bad = False, 0
            self.stats["relocalised"] += 1
            return True
        return False

    def set_pose(self, x, y, th):
        """Tell the localiser where the robot is (map frame), e.g. from a known start or a user hint."""
        self.T = Transform2D.between(self.odom_pose(), (x, y, th))
        self.lost, self.bad = False, 0

    # ------------------------------------------------------------------------------------------------ io
    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = dict(res=self.res, extent=list(self.extent), floors={})
        for f in self.floors.values():
            np.savez_compressed(d / f"floor_{f.id}.npz", L=f.grid.L, origin=f.grid.origin, res=f.grid.res)
            meta["floors"][f.id] = dict(z=f.z, scans=f.n_scans)
        (d / "slam.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory, mapping=False):
        d = Path(directory)
        meta = json.loads((d / "slam.json").read_text())
        s = cls(mapping=mapping, extent=tuple(meta["extent"]), res=meta["res"])
        for fid, fm in meta["floors"].items():
            z = np.load(d / f"floor_{fid}.npz")
            s.floors[fid] = Floor(fid, fm["z"], Grid.from_arrays(z["L"], z["origin"], z["res"]), fm["scans"])
        return s
