"""Rooms from the floor map (HOV-SG's floor -> room level, done with plain geometry), named from the objects in them.

    regions = segment_rooms(grid)              # label image [ny, nx] (0 = not free), one label per room
    rooms = describe(regions, grid, memory, floor)   # [Room(id, name, cells, centroid, door connections, ...)]

Segmentation: free space -> distance to obstacles -> seeds where it is wider than `seed_clear` (doorways are
narrower, so they separate the seeds) -> every free cell joins the seed it reaches first through free space
(geodesic growth) -> small regions merge into their largest neighbour.
Naming: each room type has object weights (bed -> bedroom, ...); the best-scoring type names the room, ties and
empty rooms become "hallway" when elongated, else "room". The owner's labels override everything.
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

ROOM_TYPES = {
    "kitchen": {"refrigerator": 3, "oven": 3, "stove": 3, "kitchen sink": 2, "dishwasher": 2, "kitchen island": 2,
                "kitchen cabinet": 1.0, "kitchen counter": 1.5},
    "living room": {"sofa": 3, "television": 2.5, "armchair": 1.2, "coffee table": 1.5},
    "dining room": {"dining table": 3, "chair": 0.6},
    "bedroom": {"bed": 4, "nightstand": 2, "wardrobe": 0.7},
    "bathroom": {"toilet": 4, "bathtub": 3, "shower": 3, "bathroom sink": 2},
    "laundry": {"washing machine": 4},
    "office": {"desk": 3, "office chair": 2, "bookshelf": 0.8},
}
# owner's words -> room type (the longest match inside the label wins: "sala de jantar" before "sala")
LABEL_TYPES = {"cozinha": "kitchen", "sala": "living room", "sala de estar": "living room", "sala de tv": "living room",
               "sala de jantar": "dining room", "quarto": "bedroom", "suite": "bedroom", "suíte": "bedroom",
               "banheiro": "bathroom", "lavabo": "bathroom", "lavanderia": "laundry", "area de servico": "laundry",
               "área de serviço": "laundry", "escritorio": "office", "escritório": "office", "corredor": "hallway",
               "varanda": "balcony", "sacada": "balcony", "deposito": "storage", "depósito": "storage",
               "despensa": "storage"}


def label_type(name):
    t = (name or "").lower()
    hits = [w for w in LABEL_TYPES if w in t]
    return LABEL_TYPES[max(hits, key=len)] if hits else None


TYPE_PT = {"kitchen": "cozinha", "living room": "sala", "dining room": "sala de jantar", "bedroom": "quarto",
           "bathroom": "banheiro", "laundry": "lavanderia", "office": "escritório", "hallway": "corredor",
           "room": "cômodo"}


@dataclass
class Room:
    id: int
    floor: str
    name: str                     # unique name ("bedroom 2"), the owner's label when given
    type: str                     # room type from the objects (or "hallway" / "room")
    centroid: tuple
    area: float
    objects: list = field(default_factory=list)
    neighbours: dict = field(default_factory=dict)   # room id -> doorway (x, y)
    label: str = None             # owner's name ("sala de TV")
    pt: str = ""
    aliases: list = field(default_factory=list)   # other names the owner used for the same room

    def to_dict(self):
        return dict(id=self.id, floor=self.floor, name=self.name, type=self.type, label=self.label,
                    centroid=[round(float(v), 2) for v in self.centroid], area=round(self.area, 1),
                    objects=self.objects, neighbours={str(k): [round(float(a), 2) for a in v]
                                                      for k, v in self.neighbours.items()})


def free_space(grid, close=2, voxels=None, robot_top=0.75):
    """Free floor of the map. With the camera's voxel map: free where either sensor saw free floor and neither
    sees an obstacle (the lidar alone sees over beds and under tables, and leaves small rooms half unknown)."""
    free = grid.free()
    occ = grid.occupied()
    if voxels is not None:
        cam_occ = voxels.columns(0.06, robot_top)
        free = (free | voxels.observed()) & ~cam_occ
        occ = occ | cam_occ
    free = ndimage.binary_opening(free, iterations=1)
    occ = ndimage.binary_dilation(occ, iterations=close)
    return free & ~occ


def segment_rooms(grid, seed_clear=0.5, min_seed_area=0.25, min_room_area=1.5, voxels=None, label_xy=()):
    """Defaults tuned on apartments (0.8 m doors, 3 m2 bathrooms) and the CMA house.
    label_xy: where the owner named rooms during the tour; each is a seed of its own (a small room behind a door
    gets its region even when it has no wide spot), unless it lies in a seed that already holds another label.
    Growth is a watershed on the distance to obstacles: regions meet at the narrowest places (doorways), not
    half-way between the seeds."""
    free = free_space(grid, voxels=voxels)
    res = grid.res
    dist = ndimage.distance_transform_edt(free) * res
    seeds, n = ndimage.label(dist > seed_clear)
    areas = ndimage.sum(np.ones_like(seeds), seeds, index=np.arange(1, n + 1)) * res * res if n else np.zeros(0)
    keep = np.zeros(n + 1, bool)
    keep[1:] = areas >= min_seed_area
    seeds = np.where(keep[seeds], seeds, 0)
    nxt = int(seeds.max()) + 1
    used = set()
    for xy in label_xy:
        i, j = grid.ij(np.asarray(xy))
        if not grid.inside(i, j):
            continue
        s = int(seeds[i, j])
        if s and s not in used:
            used.add(s)
            continue
        k = int(0.25 / res)                               # a disk of free cells around the spot
        ii, jj = np.mgrid[-k:k + 1, -k:k + 1]
        m = ii ** 2 + jj ** 2 <= k * k
        ii, jj = ii[m] + i, jj[m] + j
        ok = grid.inside(ii, jj)
        ii, jj = ii[ok], jj[ok]
        ok = free[ii, jj] & (seeds[ii, jj] == 0)
        seeds[ii[ok], jj[ok]] = nxt
        used.add(nxt)
        nxt += 1
    if seeds.max() == 0:
        return np.zeros_like(seeds)
    labels = watershed(seeds, free, dist)
    return merge_small(labels, free, min_room_area / (res * res), protect=used)


def watershed(seeds, mask, dist):
    """Priority flood from the seeds through `mask`, widest cells first (numba)."""
    return _flood(seeds.astype(np.int64).copy(), mask, dist.astype(np.float64))


def _flood_py(lab, mask, dist):
        ny, nx = lab.shape
        h = [(0.0, 0, 0)]
        h.pop()
        for i in range(ny):
            for j in range(nx):
                if lab[i, j] > 0:
                    heapq.heappush(h, (-dist[i, j], i, j))
        while len(h) > 0:
            _, i, j = heapq.heappop(h)
            L = lab[i, j]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < ny and 0 <= b < nx and mask[a, b] and lab[a, b] == 0:
                        lab[a, b] = L
                        heapq.heappush(h, (-dist[a, b], a, b))
        return lab


try:
    from numba import njit
    _flood = njit(cache=True)(_flood_py)
except ImportError:                                       # pragma: no cover
    _flood = _flood_py


def geodesic_grow(seeds, mask, max_iter=600):
    """Multi-source growth through `mask` (each free cell takes the label of the seed that reaches it first)."""
    lab = seeds.copy()
    struct = np.ones((3, 3), bool)
    for _ in range(max_iter):
        grown = ndimage.grey_dilation(lab, footprint=struct)
        new = (lab == 0) & mask & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
    return lab


def merge_small(lab, mask, min_cells, protect=()):
    while True:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        small = ids[(counts < min_cells) & ~np.isin(ids, list(protect))]
        if len(small) == 0:
            return relabel(lab)
        s = small[0]
        m = lab == s
        ring = ndimage.binary_dilation(m, iterations=2) & ~m & (lab > 0)
        if ring.any():
            nb, c = np.unique(lab[ring], return_counts=True)
            lab[m] = nb[np.argmax(c)]
        else:
            lab[m] = 0


def relabel(lab):
    out = np.zeros_like(lab)
    for k, v in enumerate(np.unique(lab[lab > 0]), 1):
        out[lab == v] = k
    return out


def doorways(lab):
    """Pairs of touching rooms and the centre of their shared boundary (cells, i, j)."""
    pairs = {}
    for di, dj in ((0, 1), (1, 0)):
        a = lab[: lab.shape[0] - di, : lab.shape[1] - dj]
        b = lab[di:, dj:]
        m = (a > 0) & (b > 0) & (a != b)
        ii, jj = np.nonzero(m)
        for i, j in zip(ii, jj):
            k = (min(a[i, j], b[i, j]), max(a[i, j], b[i, j]))
            pairs.setdefault(k, []).append((i, j))
    return {k: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v])), len(v)) for k, v in pairs.items()}


SCENE_PROMPTS = {"kitchen": "a kitchen", "living room": "a living room", "dining room": "a dining room",
                 "bedroom": "a bedroom", "bathroom": "a bathroom", "laundry": "a laundry room",
                 "office": "a home office", "hallway": "a hallway"}


def scene_scores(rid, lab, grid, memory, floor):
    """CLIP scene classification of the keyframes taken inside the room (HOV-SG's room features, simplified):
    mean softmax over the room types, or None without keyframes."""
    if memory is None or not getattr(memory, "keyframes", None):
        return None
    imgs = []
    for stamp, f, (x, y, th), rgb in memory.keyframes:
        if f != floor:
            continue
        if room_at(lab, grid, (x, y)) == rid:
            imgs.append(rgb)
    if not imgs:
        return None
    types = list(SCENE_PROMPTS)
    T = memory.emb.texts([SCENE_PROMPTS[k] for k in types])
    V = memory.emb.images(imgs[:24])
    p = np.exp(100.0 * (V @ T.T))
    p /= p.sum(1, keepdims=True)
    return dict(zip(types, p.mean(0)))


def describe(lab, grid, memory, floor, labels=None):
    """Rooms with their objects (from the memory), a type and a unique name; owner labels {room_id: name}."""
    res = grid.res
    rooms = []
    ids = np.unique(lab[lab > 0])
    objs = memory.confirmed(floor) if memory is not None else []
    obj_room = {}
    for o in objs:
        i, j = grid.ij(o.centroid[:2])
        r = int(lab[i, j]) if grid.inside(i, j) else 0
        if r == 0:                                      # furniture is not free space: nearest room within 0.6 m
            k = int(0.6 / res)
            sub = lab[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
            if (sub > 0).any():
                vals, cnt = np.unique(sub[sub > 0], return_counts=True)
                r = int(vals[np.argmax(cnt)])
        obj_room[o.id] = r
    for rid in ids:
        m = lab == rid
        ii, jj = np.nonzero(m)
        c = grid.xy(np.array([ii.mean()]), np.array([jj.mean()]))[0]
        inside = [o for o in objs if obj_room.get(o.id) == rid]
        scores = {t: sum(w.get(o.cls, 0.0) * min(1.0, o.hits / 3) for o in inside) for t, w in ROOM_TYPES.items()}
        if not any(o.cls in ("bed", "nightstand") for o in inside):
            scores["bedroom"] = 0.0                     # no bed seen: not a bedroom (wardrobes alone mislead)
        try:
            sc = scene_scores(rid, lab, grid, memory, floor)
        except Exception:
            sc = None
        if sc:                                          # views: only a confident scene counts (renders are hard)
            top = sorted(sc.values(), reverse=True)
            if top[0] > 0.45 and top[0] > 1.8 * top[1]:
                for k, v in sc.items():
                    scores[k] = scores.get(k, 0.0) + 2.5 * v
        best = max(scores, key=scores.get) if scores else None
        if best is not None and best == "hallway" and scores[best] >= 2.0:
            pass
        elif best is None or scores[best] < 2.0:
            h, w = ii.max() - ii.min() + 1, jj.max() - jj.min() + 1
            elong = max(h, w) / max(1, min(h, w))
            best = "hallway" if elong > 3.0 or m.sum() * res * res < 6 else "room"
        rooms.append(Room(int(rid), floor, best, best, (float(c[0]), float(c[1])), float(m.sum() * res * res),
                          objects=[o.id for o in inside]))
    # unique names: "bedroom", "bedroom 2", ...
    count = {}
    for r in sorted(rooms, key=lambda r: -r.area):
        count[r.type] = count.get(r.type, 0) + 1
        r.name = r.type if count[r.type] == 1 else f"{r.type} {count[r.type]}"
        r.pt = TYPE_PT.get(r.type, r.type) + ("" if count[r.type] == 1 else f" {count[r.type]}")
    byid = {r.id: r for r in rooms}
    for (a, b), (i, j, n) in doorways(lab).items():
        if n < 3 or a not in byid or b not in byid:
            continue
        xy = grid.xy(np.array([i]), np.array([j]))[0]
        byid[a].neighbours[b] = (float(xy[0]), float(xy[1]))
        byid[b].neighbours[a] = (float(xy[0]), float(xy[1]))
    for rid, names in (labels or {}).items():
        if rid in byid:
            names = names if isinstance(names, list) else [names]
            byid[rid].label, byid[rid].aliases = names[0], names[1:]
            lt = label_type(names[0])
            if lt is not None:                          # the owner said what it is: that is its type
                byid[rid].type = lt
    for o in objs:
        o.room = byid[obj_room[o.id]].name if obj_room.get(o.id) in byid else None
    return rooms


def room_at(lab, grid, xy):
    i, j = grid.ij(np.asarray(xy))
    if not grid.inside(i, j):
        return 0
    if lab[i, j]:
        return int(lab[i, j])
    k = int(0.8 / grid.res)
    sub = lab[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
    if (sub > 0).any():
        vals, cnt = np.unique(sub[sub > 0], return_counts=True)
        return int(vals[np.argmax(cnt)])
    return 0
