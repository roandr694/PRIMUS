"""People tracking for "me segue" / "vem aqui": person detections (the memory's detector, class "person") ->
3-D positions in the map frame -> tracks with a constant-velocity model and an appearance embedding (CLIP) to
re-identify the person being followed after an occlusion.

    tracker.update(now, [(xy, emb, score), ...]); tracker.lock(now) picks the person to follow (nearest in front)
    tracker.target(now) -> Track or None (lost for more than 1.5 s); tracker.last_target()
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    id: int
    xy: np.ndarray
    vel: np.ndarray
    emb: np.ndarray
    last: float
    hits: int = 1


class PeopleTracker:
    def __init__(self, gate=1.2, lost_after=1.5):
        self.tracks, self.next_id, self.gate, self.lost_after = {}, 0, gate, lost_after
        self.locked = None
        self._last_locked = None

    def update(self, now, dets):
        used = set()
        for xy, emb, score in sorted(dets, key=lambda d: -d[2]):
            xy = np.asarray(xy[:2], float)
            best, bd = None, math.inf
            for t in self.tracks.values():
                if t.id in used:
                    continue
                pred = t.xy + t.vel * (now - t.last)
                d = float(np.linalg.norm(pred - xy)) - 0.4 * float(t.emb @ emb)
                if d < bd and float(np.linalg.norm(pred - xy)) < self.gate + 0.8 * (now - t.last):
                    best, bd = t, d
            if best is None:
                t = Track(self.next_id, xy, np.zeros(2), emb, now)
                self.tracks[t.id] = t
                self.next_id += 1
                used.add(t.id)
                if self.locked is None and self._last_locked is not None and float(emb @ self._last_locked.emb) > 0.85:
                    self.locked = t.id                       # re-identified after a loss
                continue
            dt = max(now - best.last, 1e-3)
            v = (xy - best.xy) / dt
            best.vel = 0.6 * best.vel + 0.4 * np.clip(v, -2.0, 2.0)
            best.xy, best.last, best.hits = xy, now, best.hits + 1
            best.emb = best.emb * 0.8 + emb * 0.2
            best.emb /= np.linalg.norm(best.emb) + 1e-9
            used.add(best.id)
        for k in [k for k, t in self.tracks.items() if now - t.last > 20.0]:
            del self.tracks[k]

    def lock(self, now, robot_pose):
        """Follow the nearest recently seen person, preferring those in front of the robot."""
        x, y, th = robot_pose
        best, bs = None, math.inf
        for t in self.tracks.values():
            if now - t.last > self.lost_after:
                continue
            d = t.xy - np.array([x, y])
            ang = abs(math.atan2(d[1], d[0]) - th)
            ang = min(ang, 2 * math.pi - ang)
            s = float(np.linalg.norm(d)) + 1.5 * ang
            if s < bs:
                best, bs = t, s
        self.locked = best.id if best is not None else None
        return best

    def target(self, now):
        t = self.tracks.get(self.locked) if self.locked is not None else None
        if t is None or now - t.last > self.lost_after:
            if t is not None:
                self._last_locked = t
                self.locked = None
            return None
        self._last_locked = t
        return t

    def last_target(self):
        return self._last_locked
