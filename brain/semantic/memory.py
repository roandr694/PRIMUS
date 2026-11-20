"""Object memory of the house: open-vocabulary detections fused into 3-D object instances in the map frame, kept
up to date as the robot sees the scene again (OK-Robot's memory + DynaMem's add/remove), and queried by text.

    mem = SemanticMemory()                      # detector + embedder from models.py (lazy)
    mem.on_rgbd(frame, stack)                   # RGB-D frame (odom pose) -> detections -> fused objects
    mem.query("sofa", room="living room")       # -> [(score, obj), ...]
    mem.objects                                 # id -> Obj (class, centroid, extent, embedding, hits, misses, ...)

Association: same floor, centroid distance below 0.5 m + half the object's size, and the same class or a CLIP
similarity above 0.85. An object becomes confirmed after 3 sightings. It is removed after it was expected in view
(inside the frustum, 0.6-4 m, not hidden behind something closer) 4 times without being detected, more often
than half its sightings. Removed objects stay in `history` with their last position ("where did I leave ...").
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .models import HOME_CLASSES, Detector, Embedder


@dataclass
class Obj:
    id: int
    cls: str
    floor: str
    centroid: np.ndarray                     # map frame [3]
    lo: np.ndarray
    hi: np.ndarray
    emb: np.ndarray
    score: float
    hits: int = 1
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    room: str = None
    crop: np.ndarray = None                   # best view (small RGB), for a VLM check
    crop_area: float = 0.0
    votes: dict = field(default_factory=dict)  # class -> accumulated score

    @property
    def confirmed(self):
        return self.hits >= 3

    @property
    def size(self):
        return float(np.max(self.hi[:2] - self.lo[:2]))

    def to_dict(self):
        return dict(id=self.id, cls=self.cls, floor=self.floor, room=self.room,
                    centroid=[round(float(v), 3) for v in self.centroid],
                    lo=[round(float(v), 3) for v in self.lo], hi=[round(float(v), 3) for v in self.hi],
                    hits=self.hits, misses=self.misses, score=round(self.score, 3),
                    first_seen=round(self.first_seen, 1), last_seen=round(self.last_seen, 1))


class SemanticMemory:
    def __init__(self, classes=HOME_CLASSES, detector=None, embedder=None, max_range=4.5, every=1):
        self.classes = list(classes)
        self.det = detector or Detector()
        self.emb = embedder or Embedder()
        self.max_range, self.every = max_range, every
        self.objects, self.history = {}, []
        self.next_id, self.n_frames = 0, 0
        self.keyframes = []                    # (stamp, floor, map pose, small rgb) for room naming / review
        self._last_kf = None
        self.timing = dict(det=0.0, emb=0.0, n=0)
        self.enabled = True
        self.last_people = []

    # ------------------------------------------------------------------------------------------------ geometry
    @staticmethod
    def _map_T(stack):
        T = stack.slam.T
        c, s = math.cos(T.th), math.sin(T.th)
        return lambda P: np.stack([T.x + c * P[:, 0] - s * P[:, 1], T.y + s * P[:, 0] + c * P[:, 1], P[:, 2]], 1)

    def _box_points(self, frame, box, to_map):
        cam, depth = frame.cam, frame.depth
        x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
        mx, my = (x1 - x0) // 5, (y1 - y0) // 5                         # inner 60 % of the box
        sub = depth[y0 + my:y1 - my, x0 + mx:x1 - mx]
        ok = (sub > cam.min_range) & (sub < self.max_range)
        if ok.sum() < 12:
            return None
        d_med = float(np.median(sub[ok]))
        v, u = np.mgrid[y0 + my:y1 - my, x0 + mx:x1 - mx]
        sel = ok & (np.abs(sub - d_med) < 0.45)                          # the foreground surface
        z = sub[sel]
        if len(z) < 10:
            return None
        uu, vv = u[sel] + 0.5, v[sel] + 0.5
        X = (uu - cam.width / 2) / cam.fx * z
        Y = (vv - cam.height / 2) / cam.fy * z
        pts, _ = cam.to_world(np.stack([X, Y, z], 1), frame.pose.p, frame.pose.R)
        return to_map(pts)

    # ------------------------------------------------------------------------------------------------ update
    def on_rgbd(self, frame, stack):
        if not self.enabled or stack.slam.floor is None or stack.slam.in_transit():
            return
        self.n_frames += 1
        if self.n_frames % self.every:
            return
        floor = stack.slam.floor
        to_map = self._map_T(stack)
        t0 = time.time()
        dets = self.det(frame.rgb, self.classes)
        t1 = time.time()
        people = [d for d in dets if self.classes[d[5]] == "person"]
        dets = [d for d in dets if self.classes[d[5]] != "person"]       # people are tracked, not mapped
        self.last_people = []
        for d in people:
            P = self._box_points(frame, d, to_map)
            if P is not None and d[4] > 0.15:
                crop = frame.rgb[int(d[1]):int(d[3]), int(d[0]):int(d[2])]
                e = self.emb.images([crop])[0] if crop.size else np.zeros(512, np.float32)
                self.last_people.append((np.median(P, 0), e, d[4]))
        cands = []
        for d in dets:
            P = self._box_points(frame, d, to_map)
            if P is None:
                continue
            lo, hi = np.percentile(P, 10, 0), np.percentile(P, 90, 0)
            cands.append((d, P.mean(0), lo, hi))
        crops = [frame.rgb[int(d[1]):int(d[3]), int(d[0]):int(d[2])] for d, *_ in cands]
        embs = self.emb.images(crops) if crops else np.zeros((0, 512), np.float32)
        t2 = time.time()
        self.timing["det"] += t1 - t0
        self.timing["emb"] += t2 - t1
        self.timing["n"] += 1
        seen = set()
        for (d, c, lo, hi), e, crop in zip(cands, embs, crops):
            cls = self.classes[d[5]]
            o = self._associate(floor, cls, c, lo, hi, e)
            if o is None:
                o = Obj(self.next_id, cls, floor, c, lo, hi, e, d[4], first_seen=frame.stamp, last_seen=frame.stamp)
                o.votes[cls] = d[4]
                self.objects[o.id] = o
                self.next_id += 1
            else:
                w = 1.0 / (o.hits + 1)
                o.centroid = (1 - w) * o.centroid + w * c
                o.lo, o.hi = (1 - w) * o.lo + w * lo, (1 - w) * o.hi + w * hi
                o.emb = o.emb * (1 - w) + e * w
                o.emb /= np.linalg.norm(o.emb) + 1e-9
                o.hits += 1
                o.misses = max(0, o.misses - 1)
                o.last_seen = frame.stamp
                o.votes[cls] = o.votes.get(cls, 0.0) + d[4]
                o.cls = max(o.votes, key=o.votes.get)
                o.score = max(o.score, d[4])
            area = (d[2] - d[0]) * (d[3] - d[1])
            if area > o.crop_area and crop.size:
                o.crop, o.crop_area = crop[::2, ::2].copy(), area
            seen.add(o.id)
        self._expect(frame, stack, floor, seen)
        self._keyframe(frame, stack, floor)

    def _associate(self, floor, cls, c, lo, hi, e):
        best, bd = None, math.inf
        for o in self.objects.values():
            if o.floor != floor:
                continue
            dist = float(np.linalg.norm(o.centroid[:2] - c[:2]))
            gate = 0.5 + 0.5 * max(o.size, float(np.max(hi[:2] - lo[:2])))
            if dist > gate or abs(o.centroid[2] - c[2]) > 0.8:
                continue
            sim = float(o.emb @ e)
            if o.cls != cls and cls not in o.votes and sim < 0.85:
                continue
            score = dist - 0.5 * sim
            if score < bd:
                best, bd = o, score
        return best

    def _expect(self, frame, stack, floor, seen):
        """DynaMem: objects that should be visible in this frame but were not detected lose credibility."""
        cam, depth = frame.cam, frame.depth
        T = stack.slam.T
        c, s = math.cos(-T.th), math.sin(-T.th)
        Rbo = cam.R_base_optical()
        R = frame.pose.R @ Rbo
        origin = frame.pose.p + frame.pose.R @ np.asarray(cam.pos_base)
        for o in list(self.objects.values()):
            if o.floor != floor or o.id in seen:
                continue
            # map -> odom
            dx, dy = o.centroid[0] - T.x, o.centroid[1] - T.y
            po = np.array([c * dx - s * dy, s * dx + c * dy, o.centroid[2]])
            q = R.T @ (po - origin)                                       # optical frame
            if q[2] < 0.6 or q[2] > 4.0:
                continue
            u = q[0] / q[2] * cam.fx + cam.width / 2
            v = q[1] / q[2] * cam.fy + cam.height / 2
            m = 0.15 * cam.width
            if not (m < u < cam.width - m and 0.1 * cam.height < v < 0.9 * cam.height):
                continue
            dz = depth[int(v), int(u)]
            if dz <= 0 or dz < q[2] - 0.35:                               # unknown or something in front
                continue
            o.misses += 1
            if o.misses >= 4 and o.misses > 0.5 * o.hits:
                self.history.append(dict(o.to_dict(), removed=round(frame.stamp, 1)))
                del self.objects[o.id]

    def _keyframe(self, frame, stack, floor):
        x, y, th = stack.pose()
        if self._last_kf is not None:
            px, py, pth = self._last_kf
            if math.hypot(x - px, y - py) < 0.6 and abs((th - pth + math.pi) % (2 * math.pi) - math.pi) < 0.5:
                return
        self._last_kf = (x, y, th)
        self.keyframes.append((frame.stamp, floor, (x, y, th), frame.rgb[::4, ::4].copy()))

    # ------------------------------------------------------------------------------------------------ queries
    def confirmed(self, floor=None):
        return [o for o in self.objects.values() if o.confirmed and (floor is None or o.floor == floor)]

    def query(self, text, room=None, floor=None, top=5):
        """Text -> best matching confirmed objects [(score, Obj)]: CLIP text-image similarity, plus a bonus when
        the detector's class names the query."""
        objs = [o for o in self.confirmed(floor) if room is None or o.room == room]
        if not objs:
            return []
        t = self.emb.texts([text])[0]
        out = []
        for o in objs:
            s = float(o.emb @ t)
            if text.lower() == o.cls or text.lower() in o.cls:
                s += 0.1
            out.append((s, o))
        out.sort(key=lambda p: -p[0])
        return out[:top]

    def summary(self):
        from collections import Counter
        return dict(Counter(o.cls for o in self.confirmed()))

    def to_dict(self):
        return dict(objects=[o.to_dict() for o in self.confirmed()], history=self.history)
