"""Path following for a balancing two-wheeled robot (pure pursuit with turn-in-place).

    f = PathFollower(); f.set_path([(x, y), ...], final_heading=None)
    speed, yaw_rate = f.step(pose)              # pose (x, y, th) in the path's frame, th = math heading
    f.done, f.progress (index of the next waypoint), f.remaining (m)

The LQR follows a speed target with a ~0.5 s lag and a 1.5 m/s^2 ramp, so the follower slows down before sharp
turns and the goal instead of relying on a fast speed loop.
"""
import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    def __init__(self, v_max=0.7, w_max=1.2, lookahead=0.55, goal_tol=0.15, heading_tol=0.12, k_w=2.2,
                 turn_in_place=math.radians(55), a_lat=0.6):
        self.v_max, self.w_max, self.L, self.goal_tol, self.heading_tol = v_max, w_max, lookahead, goal_tol, heading_tol
        self.k_w, self.turn_in_place, self.a_lat = k_w, turn_in_place, a_lat
        self.path, self.final_heading, self.done = None, None, True
        self.progress, self.remaining = 0, 0.0

    def set_path(self, pts, final_heading=None, v_max=None):
        self.path = np.asarray(pts, float).reshape(-1, 2)
        self.final_heading = final_heading
        self.done = len(self.path) == 0
        self.progress = 0
        self.turning_final = False
        if v_max is not None:
            self.v_max = v_max
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1) if len(self.path) > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])

    def _closest(self, p):
        P = self.path
        best, bi, bt = math.inf, self.progress, 0.0
        for k in range(self.progress, min(len(P) - 1, self.progress + 25)):
            a, b = P[k], P[k + 1]
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - p))
            if d < best:
                best, bi, bt = d, k, t
        return bi, bt, best

    def _lookahead_point(self, k, t):
        P, cum = self.path, self.cum
        s = cum[k] + t * (cum[k + 1] - cum[k]) + self.L
        if s >= cum[-1]:
            return P[-1], cum[-1] - (cum[k] + t * (cum[k + 1] - cum[k]))
        m = int(np.searchsorted(cum, s) - 1)
        u = (s - cum[m]) / max(cum[m + 1] - cum[m], 1e-9)
        return P[m] + u * (P[m + 1] - P[m]), cum[-1] - (cum[k] + t * (cum[k + 1] - cum[k]))

    def step(self, pose):
        if self.done or self.path is None:
            return 0.0, 0.0
        x, y, th = pose
        p = np.array([x, y])
        if len(self.path) == 1:
            k, t, rem = 0, 1.0, float(np.linalg.norm(self.path[0] - p))
            target = self.path[0]
        else:
            k, t, _ = self._closest(p)
            self.progress = k
            target, rem = self._lookahead_point(k, t)
            rem = max(rem, float(np.linalg.norm(self.path[-1] - p)) if k >= len(self.path) - 2 else rem)
        self.remaining = rem
        d_goal = float(np.linalg.norm(self.path[-1] - p))
        if d_goal < self.goal_tol or self.turning_final:
            self.turning_final = True
            if self.final_heading is None:
                self.done = True
                return 0.0, 0.0
            e = wrap(self.final_heading - th)
            if abs(e) < self.heading_tol:
                self.done = True
                return 0.0, 0.0
            return 0.0, float(np.clip(self.k_w * e, -self.w_max * 0.7, self.w_max * 0.7))
        dx, dy = target - p
        alpha = wrap(math.atan2(dy, dx) - th)
        w = float(np.clip(self.k_w * alpha, -self.w_max, self.w_max))
        if abs(alpha) > self.turn_in_place:
            return 0.0, w
        v = self.v_max * max(0.0, math.cos(alpha)) ** 2
        v = min(v, math.sqrt(2 * 0.5 * max(rem - 0.05, 0.0)) + 0.08)     # brake for the goal (0.5 m/s^2)
        curv = 2 * math.sin(alpha) / max(self.L, 1e-3)
        if abs(curv) > 1e-3:
            v = min(v, math.sqrt(self.a_lat / abs(curv)))
        return float(v), w
