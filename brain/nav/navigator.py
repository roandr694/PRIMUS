"""Goal-directed navigation on the current floor: plan (A* on the layered costmap), follow, replan, recover.

    nav = Navigator(brain, costmap)
    nav.go_to((x, y), heading=None, tol=0.2)      # map frame; heading = math angle to face at the end
    pilot = nav.step(now)                          # PilotInput for Brain.step (the edge guard stays underneath)
    nav.status: "idle" | "planning" | "driving" | "arrived" | "failed: <why>"

Replanning: every `replan_s`, when the path crosses a newly lethal cell, or when progress stalls. A stall marks the
spot ahead as blocked for a while and plans around it; three failed plans in a row end the goal as "blocked"
(the agent then asks for help, e.g. to open a door).
"""
import math
import time

import numpy as np

from .astar import astar, simplify
from .follower import PathFollower


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Navigator:
    def __init__(self, brain, costmap, v_max=0.6, leg=0.30, replan_s=2.0, stall_s=4.0):
        self.brain, self.cm = brain, costmap
        self.v_max, self.leg, self.replan_s, self.stall_s = v_max, leg, replan_s, stall_s
        self.follower = PathFollower(v_max=v_max)
        self.status, self.goal, self.path = "idle", None, None
        self.message = ""
        self.stats = dict(plans=0, replans=0, stalls=0, plan_ms=0.0)

    # ------------------------------------------------------------------------------------------------ goals
    def go_to(self, xy, heading=None, tol=0.20, v_max=None, leg=None, label=""):
        self.goal = np.asarray(xy, float)
        self.goal_heading, self.tol, self.label = heading, tol, label
        if v_max is not None:
            self.follower.v_max = v_max
        if leg is not None:
            self.leg = leg
        self.status, self.fails, self.path = "planning", 0, None
        self.t_plan = -1e9
        self.t_progress, self.best_rem = None, math.inf
        self.recover_until = 0.0
        self.cm.clear_blocked()

    def cancel(self):
        self.status, self.goal, self.path = "idle", None, None
        self.follower.done = True

    # ------------------------------------------------------------------------------------------------ planning
    def _plan(self, now):
        """Full costmap first; if that has no path, the dynamic layers only near the robot (a far false edge or a
        person who has since moved must not cut the house in two); last, the static map alone (the edge guard and
        the live replanning protect the robot on the way)."""
        for layers, near in ((("live", "terrain"), None), (("live", "terrain"), 2.5), ((), None)):
            pts, why = self._plan_with(now, layers, near)
            if pts is not None:
                self.plan_level = 0 if near is None and layers else (1 if layers else 2)
                return pts, why
        return pts, why

    def _plan_with(self, now, layers, near):
        x, y, th = self.brain.map_pose()
        t0 = time.time()
        cost = self.cm.cost((x, y), layers, near)
        s = self.cm.ij(np.array([x, y]))
        if not np.isfinite(cost[s]):                     # robot inside the inflation (e.g. after a push): allow
            i, j = s
            k = int(self.cm.robot_r / self.cm.res) + 2
            sl = (slice(max(i - k, 0), i + k + 1), slice(max(j - k, 0), j + k + 1))
            cost[sl] = np.where(np.isfinite(cost[sl]), cost[sl], 12.0)
            occ, _ = self.cm.lethal((x, y), layers, near)
            cost[sl] = np.where(occ[sl], np.inf, cost[sl])
        g = self.cm.nearest_free(cost, self.goal, max_r=1.2)
        if g is None:
            return None, "goal unreachable (no free space near it)"
        # crop the search to the known part of the map plus margin, for speed
        path = astar(cost, s, g)
        self.stats["plan_ms"] = (time.time() - t0) * 1000
        self.stats["plans"] += 1
        if path is None:
            return None, "no path"
        path = simplify(path, cost)
        pts = self.cm.xy(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
        self.cost_snapshot = cost
        return pts, ""

    def _path_blocked(self):
        if self.path is None:
            return False
        occ, _ = self.cm.lethal()
        k0 = max(self.follower.progress, 0)
        for a, b in zip(self.path[k0:k0 + 4], self.path[k0 + 1:k0 + 5]):
            n = int(np.linalg.norm(b - a) / self.cm.res) + 2
            P = a + (b - a) * np.linspace(0, 1, n)[:, None]
            i, j = self.cm.ij(P)
            ok = self.cm.g.inside(i, j)
            if np.any(occ[i[ok], j[ok]]):
                return True
        return False

    # ------------------------------------------------------------------------------------------------ 50 Hz
    def step(self, now):
        from brain.runtime import PilotInput
        idle = PilotInput(leg_length=self.leg, source="nav")
        if self.goal is None or self.status in ("idle", "arrived") or self.status.startswith("failed"):
            return idle
        x, y, th = self.brain.map_pose()
        d_goal = float(np.linalg.norm(self.goal - np.array([x, y])))
        if now < self.recover_until:                      # back off after a stall
            return PilotInput(speed=-0.25, leg_length=self.leg, source="nav")
        need = self.path is None or now - self.t_plan > self.replan_s or self._path_blocked()
        if self.path is None and now - self.t_plan < 1.0:
            need = False                                  # a plan just failed: wait before trying again
        if need and not (self.follower.turning_final if self.path is not None else False):
            pts, why = self._plan(now)
            self.t_plan = now
            if pts is None:
                self.fails += 1
                self.message = f"nav: {why} ({self.fails})"
                if self.fails >= 3:
                    self.status = f"failed: {why}"
                    self.message = f"nav: failed ({why})"
                    return idle
                self.cm.clear_blocked() if self.fails == 2 else None
                return idle
            self.fails = 0
            if self.path is not None:
                self.stats["replans"] += 1
            self.path = pts
            self.follower.set_path(pts, self.goal_heading)
            self.status = "driving"
            if self.t_progress is None:
                self.t_progress = now
        # narrow places (doorways, between furniture): short lookahead and slow, so the path is tracked closely
        f = self.follower
        cs = getattr(self, "cost_snapshot", None)
        if cs is not None:
            i, j = self.cm.ij(np.array([x, y]))
            k = 6
            local = cs[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
            tight = (not np.isfinite(local).all()) or float(np.nanmax(np.where(np.isfinite(local), local, np.nan))) > 6.0
        else:
            tight = False
        f.L = 0.30 if tight else 0.55
        v_cap = 0.30 if tight else None
        v, w = f.step((x, y, th))
        if v_cap is not None:
            v = min(v, v_cap)
        if self.follower.done:
            self.status = "arrived"
            self.message = f"nav: arrived{(' at ' + self.label) if self.label else ''}"
            return idle
        st = self.brain.status()
        if st.get("guard_active") and abs(v) > 0.05 and "guard" in (st.get("message") or ""):
            # the edge guard holds us in front of something the map did not show: mark it and go around
            self.guard_t = getattr(self, "guard_t", now) if getattr(self, "guard_on", False) else now
            self.guard_on = True
            if now - self.guard_t > 1.5:
                ahead = np.array([x + 0.55 * math.cos(th), y + 0.55 * math.sin(th)])
                self.cm.block_disk(ahead, 0.30)
                self.path, self.guard_on = None, False
                self.message = "nav: obstacle in the way, planning around it"
                return PilotInput(leg_length=self.leg, source="nav")
        else:
            self.guard_on = False
        rem = self.follower.remaining
        if rem < self.best_rem - 0.15:
            self.best_rem, self.t_progress = rem, now
        elif self.follower.turning_final:
            self.t_progress = now
        elif now - self.t_progress > self.stall_s:        # no progress: something the map does not explain
            self.stats["stalls"] += 1
            ahead = np.array([x + 0.6 * math.cos(th), y + 0.6 * math.sin(th)])
            self.cm.block_disk(ahead, 0.35)
            self.message = "nav: stuck, backing off and replanning"
            self.recover_until = now + 1.2
            self.t_progress, self.best_rem, self.path = now + 1.2, math.inf, None
            return PilotInput(speed=-0.25, leg_length=self.leg, source="nav")
        self.message = f"nav: {self.label or 'goal'} {d_goal:.1f} m"
        return PilotInput(speed=v, yaw_rate=w, leg_length=self.leg, source="nav")
