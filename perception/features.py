"""Terrain features in front of (and behind) the robot, from the elevation map: stairs, curbs, drops, cliffs,
gaps, overhangs and walls, with what skill crosses each one.

The map is sampled on a corridor as wide as the robot along its heading. The ground profile takes the highest
known sample across the corridor (conservative); the ceiling profile the lowest. Edges are height changes of
more than `edge_min` over two cells; nearby edge cells merge into one edge. Then:
  stairs    two or more edges of the same sign, similar rise, treads 0.15-0.50 m
  gap       the ground leaves the current level (lower or unseen) and is back within `max_gap_detect`
  curb_up / curb_down / cliff / wall     a single edge (by sign and size)
  overhang  free height (ceiling - ground) lower than the robot at its current height
Heading convention of the controller: forward = (-sin(heading), cos(heading)), right = (cos, sin).
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Limits:
    """What the skills can cross (the teacher curriculum tops: stairs 19 cm, curbs 20 cm, gaps 45 cm)."""
    stair_rise: float = 0.19
    curb_up: float = 0.20
    drop_skill: float = 0.30          # deeper drops are cliffs
    gap: float = 0.45
    top_above_leg: float = 0.185      # robot top = leg length + wheel radius + lidar top (0.0767, CAD)
    min_leg: float = 0.18
    crouch_margin: float = 0.03
    # what the edge guard lets the pilot drive over without a skill
    guard_up: float = 0.05
    guard_down: float = 0.06


@dataclass
class Feature:
    kind: str                         # stairs_up, stairs_down, curb_up, curb_down, cliff, gap, overhang, wall
    s: float                          # distance from the wheel axle to the start [m] (along the travel direction)
    end: float                        # distance to the end
    rise: float = 0.0                 # per step [m] (+ up)
    n: int = 1
    tread: float = 0.0
    width: float = 0.0                # gap width
    free: float = math.inf            # free height under an overhang
    feasible: bool = False            # a skill can cross it
    guard: bool = False               # the edge guard stops before it when no skill runs
    edge_heading: float = math.nan    # heading perpendicular to the first edge (to align before a skill)
    extra: dict = field(default_factory=dict)

    @property
    def skill(self):
        return {"stairs_up": "climb", "stairs_down": "descend", "curb_up": "step up", "curb_down": "step down",
                "gap": "jump", "overhang": "crouch"}.get(self.kind) if self.feasible else None

    def text(self):
        cm = lambda v: f"{abs(v) * 100:.0f} cm"
        down = (f"stairs down ~{cm(self.rise)} total" if self.extra.get("estimated")
                else f"stairs down {self.n}x{cm(self.rise)}")
        what = {"stairs_up": f"stairs up {self.n}x{cm(self.rise)}", "stairs_down": down,
                "curb_up": f"curb up {cm(self.rise)}", "curb_down": f"drop {cm(self.rise)}",
                "cliff": "unseen ground" if self.extra.get("unknown") else f"cliff {cm(self.rise)}",
                "gap": f"gap {cm(self.width)}", "overhang": f"overhang {cm(self.free)} free", "wall": "wall"}[self.kind]
        return f"{what} at {self.s:.1f} m"


class FeatureDetector:
    def __init__(self, limits=Limits(), res=0.04, ahead=3.0, behind=1.0, half_width=0.30,
                 edge_min=0.03, max_gap_detect=0.6):
        self.lim, self.res = limits, res
        self.s = np.arange(-behind, ahead + 1e-6, res)
        self.lat = np.linspace(-half_width, half_width, 11)
        self.edge_min, self.max_gap_detect = edge_min, max_gap_detect

    # ---------------- profiles ----------------
    def profiles(self, emap, xy, heading, lat=None):
        lat = self.lat if lat is None else lat
        fwd = np.array([-math.sin(heading), math.cos(heading)])
        right = np.array([math.cos(heading), math.sin(heading)])
        pts = xy + self.s[:, None, None] * fwd + lat[None, :, None] * right       # [S, L, 2]
        h, v, c = emap.lookup(pts)
        known = np.isfinite(h) & (v < 0.03 ** 2)
        frac = known.mean(1)
        g = np.where(frac >= 0.5, np.where(known, h, -np.inf).max(1), np.nan)      # highest: steps, obstacles
        low = np.where(frac >= 0.5, np.where(known, h, np.inf).min(1), np.nan)     # lowest: holes, gaps
        ceil = np.where(np.isfinite(h), c, np.inf).min(1)
        self.low = low
        return g, ceil, h, known

    @staticmethod
    def _ffill(x, max_run):
        """forward-fill nan runs of at most max_run samples"""
        y = x.copy()
        last, run = np.nan, 0
        for k in range(len(y)):
            if np.isfinite(y[k]):
                last, run = y[k], 0
            else:
                run += 1
                if run <= max_run:
                    y[k] = last
        return y

    def _gaps(self, H):
        """Gaps: the ground leaves the current level (lower, or not seen) and comes back to it within
        max_gap_detect. -> [(last cell before, first cell after, width)]; the width brackets the rims (+1 cell)."""
        out, n, i = [], len(H), 0
        while i < n - 1:
            if np.isfinite(H[i]) and (not np.isfinite(H[i + 1]) or H[i + 1] < H[i] - 0.08):
                level, j = H[i], i + 1
                while j < n and (j - i) * self.res <= self.max_gap_detect:
                    if np.isfinite(H[j]) and H[j] >= level - 0.025:
                        break
                    j += 1
                if j < n and (j - i) * self.res <= self.max_gap_detect and abs(H[j] - level) <= 0.025:
                    width = (j - i - 1) * self.res + 0.06     # the camera hides the near side of the far rim
                    if width >= 0.07:
                        out.append((i, j, width))
                        i = j
                        continue
            i += 1
        return out

    def _edges(self, H):
        """Edges (height changes) along a profile that starts at the robot: [(index, rise)]. An unknown run between
        known ground (the ground beyond an edge is hidden) is an edge at its start, with the rise across it."""
        out = []
        k, n = 2, len(H)
        while k < n:
            if not np.isfinite(H[k]) and np.isfinite(H[k - 1]):
                a, j = k, k
                while j < n and not np.isfinite(H[j]):
                    j += 1
                if j >= n:
                    break                                   # unknown to the end of the profile
                rise = float(H[j] - H[a - 1])
                if abs(rise) > self.edge_min:
                    out.append((a, rise, (j - a) * self.res))         # hidden run length
                k = j + 2
                continue
            d = H[k] - H[k - 2]
            if np.isfinite(d) and abs(d) > self.edge_min:
                sign, first = np.sign(d), k
                while k + 1 < n and np.isfinite(H[k + 1] - H[k - 1]) and np.sign(H[k + 1] - H[k - 1]) == sign \
                        and abs(H[k + 1] - H[k - 1]) > 0.5 * self.edge_min:
                    k += 1
                out.append((first - 1, float(H[min(k, n - 1)] - H[first - 2]), 0.0))
                k += 2
            else:
                k += 1
        return out

    # ---------------- detection ----------------
    def detect(self, emap, xy, heading, leg_length, direction=+1):
        """Features along the travel direction (+1 forward, -1 backward), nearest first."""
        g, ceil, _, _ = self.profiles(emap, xy, heading)
        i0 = int(np.argmin(np.abs(self.s)))
        cut = (lambda a: a[i0:]) if direction > 0 else (lambda a: a[:i0 + 1][::-1])
        prof, low, cp, dist = cut(g), cut(self.low), cut(ceil), direction * cut(self.s)
        lim, out = self.lim, []
        H = self._ffill(prof, 2)                            # sensor dropouts
        Hs = H.copy()
        for i, j, w in self._gaps(low):                     # unfilled, lowest across the corridor: conservative
            out.append(Feature("gap", dist[i] + self.res / 2, dist[j] - self.res / 2, width=w, feasible=w <= lim.gap,
                               guard=True))
            Hs[i + 1:j] = H[i]                              # no edges inside a gap
        edges = self._edges(Hs)
        k = 0
        while k < len(edges):
            i, rise, hidden = edges[k]
            # a drop whose lower side is hidden (a level camera cannot see the treads below a stair edge) but with
            # lower ground seen beyond it at a stair-like slope: stairs down, estimated (total drop, not the steps)
            if rise < -0.06 and hidden >= 0.2 and -rise / (hidden + self.res) <= 0.8:
                out.append(Feature("stairs_down", dist[i], dist[i] + hidden, rise=rise, n=0,
                                   feasible=-rise <= 1.2, guard=True, extra=dict(estimated=True, total=-rise)))
                k += 1
                continue
            # stairs: a run of same-sign edges, similar rise, regular treads
            j = k
            while j + 1 < len(edges) and np.sign(edges[j + 1][1]) == np.sign(rise) \
                    and 0.15 <= dist[edges[j + 1][0]] - dist[edges[j][0]] <= 0.50 \
                    and abs(edges[j + 1][1] - rise) < 0.4 * abs(rise) + 0.02:
                j += 1
            if j > k:
                rises = np.array([e[1] for e in edges[k:j + 1]])
                treads = np.diff([dist[e[0]] for e in edges[k:j + 1]])
                up = rises[0] > 0
                worst = float(np.abs(rises).max())          # feasibility on the highest step seen
                feas = worst <= lim.stair_rise if up else worst <= max(lim.stair_rise, 0.20)
                out.append(Feature("stairs_up" if up else "stairs_down", dist[i], dist[edges[j][0]],
                                   rise=float(np.median(rises)), n=j - k + 1, tread=float(treads.mean()),
                                   feasible=feas, guard=True,
                                   extra=dict(continues=edges[j][0] > len(dist) - 15, max_rise=worst)))
                k = j + 1
                continue
            if rise > 0:
                kind = "curb_up" if rise <= lim.curb_up else "wall"
                out.append(Feature(kind, dist[i], dist[i], rise=rise, feasible=kind == "curb_up",
                                   guard=rise >= lim.guard_up))
            else:
                kind = "curb_down" if -rise <= lim.drop_skill else "cliff"
                out.append(Feature(kind, dist[i], dist[i], rise=rise, feasible=kind == "curb_down",
                                   guard=-rise >= lim.guard_down))
            k += 1
        # unknown ground right after known ground, close by, not explained by an edge or gap: no returns (a hole,
        # an edge seen from too far, a black or shiny floor)
        known = np.isfinite(H)
        for i in range(3, min(len(H), int(1.2 / self.res))):
            if known[i - 1] and i + 6 <= len(H) and not known[i:i + 6].any():
                if not any(abs(f.s - dist[i]) < 0.1 for f in out):
                    out.append(Feature("cliff", dist[i], dist[i], rise=-math.inf, guard=True, extra=dict(unknown=True)))
                break
        # overhangs: lower than the robot at its current height
        free = cp - np.where(np.isfinite(prof), prof, np.nanmin(prof) if np.isfinite(prof).any() else 0.0)
        top_now = leg_length + lim.top_above_leg
        low = np.nonzero(free < top_now + lim.crouch_margin)[0]
        if len(low):
            a = low[0]
            b = low[-1]
            fmin = float(free[low].min())
            need = lim.min_leg + lim.top_above_leg + lim.crouch_margin
            out.append(Feature("overhang", dist[a], dist[b], free=fmin, feasible=fmin >= need, guard=True)
                       if fmin > 0.05 else Feature("wall", dist[a], dist[a], guard=True))
        out.sort(key=lambda f: f.s)
        for f in out:
            if direction > 0 and f.kind not in ("overhang",):
                f.edge_heading = self.edge_heading(emap, xy, heading, f.s)
        return out

    def edge_heading(self, emap, xy, heading, s_edge, half=0.45, window=0.20):
        """Heading perpendicular to the edge at distance s_edge (line fit of the edge across the corridor)."""
        lat = np.linspace(-half, half, 10)
        fwd = np.array([-math.sin(heading), math.cos(heading)])
        right = np.array([math.cos(heading), math.sin(heading)])
        s = np.arange(s_edge - window, s_edge + window + 1e-6, self.res)
        pts = xy + s[:, None, None] * fwd + lat[None, :, None] * right
        h, v, _ = emap.lookup(pts)
        h = np.where(np.isfinite(h) & (v < 0.03 ** 2), h, np.nan)
        ls, ss = [], []
        for c in range(len(lat)):
            col = self._ffill(h[:, c], 2)
            d = np.abs(col[2:] - col[:-2])
            if np.isfinite(d).any() and np.nanmax(d) > self.edge_min:
                ls.append(lat[c])
                ss.append(s[int(np.nanargmax(d)) + 1])
        if len(ls) < 4:
            return math.nan
        b = np.polyfit(ls, ss, 1)[0]
        return heading + math.atan(b)
