"""The owner's tour of the house, simulated: a person who knows the house drives the robot through every room of a
floor (A* on the ground-truth grid with the true pose, path following), while the robot's own stack maps it.

    driver = TourDriver(sim, floors=["social"])          # visits the rooms of house_gt.json on those floors
    result, sim = runner.run(Scenario("tour"), driver)

It is the sim stand-in for the pilot with a gamepad (P1 mapping, P3 semantic tour). The robot never sees the
ground truth: only the pilot does.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from brain.nav.astar import astar, simplify  # noqa: E402
from brain.nav.follower import PathFollower  # noqa: E402
from brain.runtime import PilotInput  # noqa: E402

import gt_grid  # noqa: E402
from world import GT_PATH  # noqa: E402


def cost_from_free(free, res, robot_r=0.33, soft=0.85):
    """Inflated cost grid: inf within robot_r of an obstacle, rising cost up to soft, 1 elsewhere."""
    d = ndimage.distance_transform_edt(free) * res
    c = np.where(d <= robot_r, np.inf, 1.0 + 25.0 * np.clip((soft - d) / (soft - robot_r), 0.0, 1.0) ** 2)
    return c.astype(np.float64)


def room_waypoints(gt, grid, floor, min_clear=0.45, indoor_only=True):
    """One waypoint per room of the floor: the free cell with the largest clearance inside the room. Outdoor rooms
    are skipped by default (lawn and garden foliage is visual-only in the scene: the ground truth calls it free,
    the robot's camera and lidar see it as obstacles)."""
    clear = grid.clearance()
    pts = []
    for r in gt["rooms"]:
        if r["floor"] != floor or (indoor_only and not r["indoor"]):
            continue
        best = None
        for x0, x1, y0, y1 in r["rects"]:
            i0, j0 = grid.world_to_ij(np.array([x0, y0]))
            i1, j1 = grid.world_to_ij(np.array([x1, y1]))
            i0, i1 = max(i0, 0), min(i1, grid.free.shape[0])
            j0, j1 = max(j0, 0), min(j1, grid.free.shape[1])
            if i1 <= i0 or j1 <= j0:
                continue
            sub = clear[i0:i1, j0:j1]
            k = np.unravel_index(np.argmax(sub), sub.shape)
            if best is None or sub[k] > best[0]:
                best = (sub[k], i0 + k[0], j0 + k[1])
        if best is not None and best[0] >= min_clear:
            pts.append((r["name"], grid.ij_to_world(best[1], best[2])))
    return pts


class TourDriver:
    def __init__(self, sim, floor="social", order=None, v_max=0.5, leg=0.30, dwell=1.5, spin=True):
        self.gt = json.loads(GT_PATH.read_text())
        self.grid = gt_grid.cached(sim.m, sim.d, floor, GT_PATH.parent)
        self.cost = cost_from_free(self.grid.free, self.grid.res)
        wps = room_waypoints(self.gt, self.grid, floor)
        if order:
            wps = sorted([w for w in wps if w[0] in order], key=lambda w: order.index(w[0]))
        self.wps, self.v_max, self.leg, self.dwell, self.spin = wps, v_max, leg, dwell, spin
        self.f = PathFollower(v_max=v_max, lookahead=0.4)
        self.message, self.visited, self.failed = "", [], []
        self.sim = None

    def start(self, sim, brain):
        self.sim, self.k, self.phase, self.t_phase = sim, -1, "wait", 0.0
        self.remaining = self._order(sim.truth().xy)

    def _order(self, start):
        """Greedy nearest-room order from the start (straight-line distance)."""
        left, out, p = list(self.wps), [], np.asarray(start)
        while left:
            k = min(range(len(left)), key=lambda q: np.linalg.norm(left[q][1] - p))
            out.append(left.pop(k))
            p = out[-1][1]
        return out

    def _plan(self, goal):
        tp = self.sim.truth()
        s = self.grid.world_to_ij(tp.xy)
        g = self.grid.world_to_ij(goal)
        cost = self.cost.copy()
        i, j = s
        cost[max(i - 6, 0):i + 7, max(j - 6, 0):j + 7] = np.minimum(cost[max(i - 6, 0):i + 7, max(j - 6, 0):j + 7], 3.0)
        path = astar(cost, s, g)
        if path is None:
            return None
        path = simplify(path, cost)
        return self.grid.ij_to_world(np.array([p[0] for p in path]), np.array([p[1] for p in path]))

    def true_pose(self):
        tp = self.sim.truth()
        return (tp.p[0], tp.p[1], tp.yaw + math.pi / 2)

    def step(self, t, brain, ev):
        if ev.state.mode != 2:
            return PilotInput(leg_length=self.leg, source="pilot")
        if self.phase == "wait":
            self.t_phase = t
            self.phase = "next"
        if self.phase == "next":
            if not self.remaining:
                self.phase = "done"
                return PilotInput(leg_length=self.leg, source="pilot")
            name, goal = self.remaining.pop(0)
            pts = self._plan(goal)
            if pts is None:
                self.failed.append(name)
                return PilotInput(leg_length=self.leg, source="pilot")
            self.cur = name
            self.f.set_path(pts)
            self.message = f"tour: going to {name}"
            self.phase, self.t_phase = "drive", t
        if self.phase == "drive":
            v, w = self.f.step(self.true_pose())
            if self.f.done or t - self.t_phase > 90.0:
                (self.visited if self.f.done else self.failed).append(self.cur)
                self.phase, self.t_phase = ("spin" if self.spin else "dwell"), t
                self.spin_start = self.true_pose()[2]
                self.spun = 0.0
                return PilotInput(leg_length=self.leg, source="pilot")
            return PilotInput(speed=v, yaw_rate=w, leg_length=self.leg, source="pilot")
        if self.phase == "spin":                                   # look around the room (camera + lidar)
            th = self.true_pose()[2]
            self.spun += abs(((th - self.spin_start) + math.pi) % (2 * math.pi) - math.pi)
            self.spin_start = th
            if self.spun > 2 * math.pi - 0.2 or t - self.t_phase > 14.0:
                self.phase, self.t_phase = "dwell", t
            return PilotInput(yaw_rate=0.6, leg_length=self.leg, source="pilot")
        if self.phase == "dwell":
            if t - self.t_phase > self.dwell:
                self.phase = "next"
            return PilotInput(leg_length=self.leg, source="pilot")
        return PilotInput(leg_length=self.leg, source="pilot")

    def done(self):
        if self.phase == "done":
            return (len(self.failed) == 0, f"visited {len(self.visited)}/{len(self.wps)}")
        return None

    def extra(self):
        return dict(visited=self.visited, failed=self.failed)
