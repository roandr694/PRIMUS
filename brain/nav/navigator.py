"""Goal-directed navigation on the current floor: plan (A* on the layered costmap), follow, replan, recover.

    nav = Navigator(brain, costmap)
    nav.go_to((x, y), heading=None, tol=0.2)      # map frame; heading = math angle to face at the end
    pilot = nav.step(now)                          # PilotInput for Brain.step (the edge guard stays underneath)
    nav.status: "idle" | "planning" | "driving" | "arrived" | "failed: <why>"

Replanning: every `replan_s`, when the path crosses a newly lethal cell, or when progress stalls. A stall marks the
spot ahead as blocked for a while and plans around it; three failed plans in a row end the goal as "blocked"
(the agent then asks for help, e.g. to open a door).

Narrow passages (doorways, gaps between furniture narrower than ~1 m; phase E): the robot is 0.58 m wide and the
edge guard watches a 0.70 m corridor, so in a 0.8 m door it must be centred and straight. Before the passage it
stops and turns to the passage direction, drives through slowly with a short lookahead and a guard corridor
narrowed to the robot's width plus 2 cm a side, and if the guard still holds it, it backs off, lines up again and
retries (three times) before it marks the passage blocked and plans another way.
"""
import math
import time

import numpy as np

from .astar import astar, simplify
from .follower import PathFollower


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Navigator:
    def __init__(self, brain, costmap, v_max=0.6, leg=0.30, replan_s=4.0, stall_s=5.0):
        self.brain, self.cm = brain, costmap
        self.v_max, self.leg, self.replan_s, self.stall_s = v_max, leg, replan_s, stall_s
        self.follower = PathFollower(v_max=v_max)
        self.status, self.goal, self.path = "idle", None, None
        self.message = ""
        self.stats = dict(plans=0, replans=0, stalls=0, plan_ms=0.0, aligns=0, retries=0)
        self.narrow_clear = 0.50                # passage half-width below which the narrow-passage mode starts
        self.clear_snapshot = None
        self.passage = None                     # (point, direction) of the next narrow passage
        self.aligned_for = None
        self.retries = 0
        self.backoff_until = 0.0

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
        self.aligned_for, self.retries, self.backoff_until = None, 0, 0.0
        self.cm.clear_blocked()

    def cancel(self):
        self.status, self.goal, self.path = "idle", None, None
        self.follower.done = True

    # ------------------------------------------------------------------------------------------------ planning
    def _plan(self, now):
        """Full costmap first; if that has no path, the dynamic layers only near the robot (a far false edge or a
        person who has since moved must not cut the house in two); last, the static map alone (the edge guard and
        the live replanning protect the robot on the way)."""
        r0 = self.cm.robot_r
        # the camera layer near the robot (chairs, table tops the lidar misses) stays in every level
        levels = ((("live", "terrain"), None, r0), (("live", "terrain"), None, 0.27),
                  (("live", "terrain"), 2.5, 0.27))  # tight radius: doorways blurred in the tour map
        try:
            for k, (layers, near, r) in enumerate(levels):
                self.cm.robot_r = r
                pts, why = self._plan_with(now, layers, near)
                if pts is not None:
                    self.plan_level = k
                    return pts, why
        finally:
            self.cm.robot_r = r0
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
        from scipy import ndimage
        occ, _ = self.cm.lethal((x, y), layers, near)
        self.clear_snapshot = ndimage.distance_transform_edt(~occ) * self.cm.res
        return pts, ""

    def _next_passage(self, x, y, horizon=1.4):
        """First point within `horizon` metres along the path whose clearance is below narrow_clear, with the
        path direction there: (point, direction angle, distance along the path) or None."""
        cl = self.clear_snapshot
        if cl is None or self.path is None:
            return None
        k0 = max(self.follower.progress, 0)
        P = [np.array([x, y])] + [np.asarray(p, float) for p in self.path[k0 + 1:]]
        travelled = 0.0
        for a, b in zip(P[:-1], P[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            n = int(L / (0.5 * self.cm.res)) + 2
            s = np.linspace(0, 1, n)
            Q = a + (b - a) * s[:, None]
            i, j = self.cm.ij(Q)
            ok = self.cm.g.inside(i, j)
            c = np.where(ok, cl[np.clip(i, 0, self.cm.ny - 1), np.clip(j, 0, self.cm.nx - 1)], 9.0)
            d_along = travelled + s * L
            hit = (c < self.narrow_clear) & (d_along > 0.15)
            if hit.any():
                m = int(np.argmax(hit))
                return Q[m], math.atan2(b[1] - a[1], b[0] - a[0]), float(d_along[m])
            travelled += L
            if travelled > horizon:
                break
        return None

    def _route_cost(self, pts, start):
        """Cost of driving from start along pts on the latest cost grid (inf if it crosses a forbidden cell)."""
        c = getattr(self, "cost_snapshot", None)
        if c is None or pts is None or len(pts) == 0:
            return np.inf
        P = [np.asarray(start, float)] + [np.asarray(p, float) for p in pts]
        total, travelled = 0.0, 0.0
        for a, b in zip(P[:-1], P[1:]):
            L = float(np.linalg.norm(b - a))
            n = int(L / self.cm.res) + 2
            s = np.linspace(0, 1, n)
            Q = a + (b - a) * s[:, None]
            i, j = self.cm.ij(Q)
            ok = self.cm.g.inside(i, j)
            v = np.where(ok, c[np.clip(i, 0, self.cm.ny - 1), np.clip(j, 0, self.cm.nx - 1)], np.inf)
            near = travelled + s * L < 0.6                    # the first 0.6 m may cross the inflation around us
            v = np.where(near & ~np.isfinite(v), 12.0, v)
            if not np.all(np.isfinite(v)):
                return np.inf
            total += L * float(np.mean(v))
            travelled += L
        return total

    def _path_blocked(self):
        """A new obstacle on the next 1.2 m of the path (live lidar hits only: the static and camera layers were
        known when the path was planned and are refreshed by the periodic replan)."""
        if self.path is None:
            return False
        cm = self.cm
        recent = (cm.now - cm.live_hit) < cm.live_decay
        x, y, _ = self.brain.map_pose()
        k0 = max(self.follower.progress, 0)
        pts = [np.array([x, y])] + list(self.path[k0 + 1:k0 + 4])
        left, n_hit = 1.2, 0
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            u = min(1.0, left / L)
            n = int(L * u / cm.res) + 2
            P = a + (b - a) * u * np.linspace(0, 1, n)[:, None]
            i, j = cm.ij(P)
            ok = cm.g.inside(i, j)
            n_hit += int(np.sum(recent[i[ok], j[ok]]))
            left -= L * u
            if left <= 0:
                break
        return n_hit >= 2

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
        blocked = self.path is not None and now - self.t_plan > 0.5 and self._path_blocked()
        need = self.path is None or now - self.t_plan > self.replan_s or blocked
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
            if self.path is not None and not blocked:
                # hysteresis: keep the current route unless the new one is clearly cheaper on today's costmap
                old = self._route_cost(self.path[max(self.follower.progress, 0):], (x, y))
                new = self._route_cost(pts, (x, y))
                if np.isfinite(old) and new > 0.75 * old:
                    pts = None
            if pts is None:
                self.stats["kept"] = self.stats.get("kept", 0) + 1
            else:
                if self.path is not None:
                    self.stats["replans"] += 1
                self.path = pts
                self.follower.set_path(pts, self.goal_heading)
            self.status = "driving"
            if self.t_progress is None:
                self.t_progress = now
        if self.path is None:                             # no plan yet (or the last one failed): wait, never "arrive"
            return idle
        # narrow passage ahead: stop and line up with it first, then go through slowly with a narrow guard corridor
        f = self.follower
        auto = getattr(self.brain, "auto", None)
        if now < self.backoff_until:
            return PilotInput(speed=-0.15, leg_length=self.leg, source="nav")
        nxt = self._next_passage(x, y)
        in_narrow = False
        if nxt is not None:
            p_n, dir_n, d_n = nxt
            key = (round(float(p_n[0]), 1), round(float(p_n[1]), 1))
            err = wrap(dir_n - th)
            if d_n < 1.0:
                in_narrow = True
                if self.aligned_for != key and abs(err) > math.radians(6):
                    if abs(err) < math.radians(100):
                        self.message = "nav: lining up with the passage"
                        self.t_progress = now
                        if auto is not None:
                            auto.corridor_half_width = 0.35
                        return PilotInput(speed=0.0, yaw_rate=float(np.clip(2.0 * err, -0.8, 0.8)),
                                          leg_length=self.leg, source="nav")
                if self.aligned_for != key:
                    self.aligned_for = key
                    self.stats["aligns"] += 1
        else:
            cl = self.clear_snapshot
            if cl is not None:
                i, j = self.cm.ij(np.array([x, y]))
                if self.cm.g.inside(i, j) and cl[i, j] < self.narrow_clear:
                    in_narrow = True                      # inside the passage now
        if auto is not None:
            auto.corridor_half_width = 0.31 if in_narrow else 0.35
        # narrow places (doorways, between furniture): short lookahead and slow, so the path is tracked closely
        cs = getattr(self, "cost_snapshot", None)
        if cs is not None:
            i, j = self.cm.ij(np.array([x, y]))
            k = 6
            local = cs[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
            tight = (not np.isfinite(local).all()) or float(np.nanmax(np.where(np.isfinite(local), local, np.nan))) > 6.0
        else:
            tight = False
        f.L = 0.25 if in_narrow else (0.30 if tight else 0.55)
        v_cap = 0.18 if in_narrow else (0.30 if tight else None)
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
            if now - self.guard_t < 4.0:
                # first try to line up with the path (the guard only caps the speed; turning is free)
                f.L = 0.25
                v2, w2 = f.step((x, y, th))
                return PilotInput(speed=0.0, yaw_rate=w2, leg_length=self.leg, source="nav")
            elif in_narrow and self.retries < 3:
                # a doorway: back off, line up again and retry before calling it blocked
                self.retries += 1
                self.stats["retries"] += 1
                self.aligned_for, self.guard_on = None, False
                self.backoff_until = now + 1.5
                self.t_progress = now + 1.5
                self.message = f"nav: tight passage, backing off to line up again ({self.retries})"
                return PilotInput(speed=-0.15, leg_length=self.leg, source="nav")
            else:
                ahead = np.array([x + 0.6 * math.cos(th), y + 0.6 * math.sin(th)])
                self.cm.block_disk(ahead, 0.25)
                self.path, self.guard_on = None, False
                self.message = "nav: obstacle in the way, planning around it"
                return PilotInput(leg_length=self.leg, source="nav")
        else:
            self.guard_on = False
        rem = self.follower.remaining
        if rem < self.best_rem - 0.15:
            self.best_rem, self.t_progress = rem, now
            if not in_narrow:
                self.retries = 0
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
