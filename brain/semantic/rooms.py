"""Rooms from the floor map (HOV-SG's floor -> room level, done with plain geometry), named from the objects in them.

    regions = segment_rooms(grid)              # label image [ny, nx] (0 = not free), one label per room
    rooms = describe(regions, grid, memory, floor)   # [Room(id, name, cells, centroid, door connections, ...)]

Segmentation: free space -> distance to obstacles -> seeds where it is wider than `seed_clear` (doorways are
narrower, so they separate the seeds) -> every free cell joins the seed it reaches first through free space
(geodesic growth) -> small regions merge into their largest neighbour.
Naming: each room type has object weights (bed -> bedroom, ...); the best-scoring type names the room, ties and
empty rooms become "hallway" when elongated, else "room". The owner's labels override everything.
"""
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

    def to_dict(self):
        return dict(id=self.id, floor=self.floor, name=self.name, type=self.type, label=self.label,
                    centroid=[round(float(v), 2) for v in self.centroid], area=round(self.area, 1),
                    objects=self.objects, neighbours={str(k): [round(float(a), 2) for a in v]
                                                      for k, v in self.neighbours.items()})


def free_space(grid, close=2):
    free = grid.free()
    free = ndimage.binary_opening(free, iterations=1)
    occ = ndimage.binary_dilation(grid.occupied(), iterations=close)
    return free & ~occ


def segment_rooms(grid, seed_clear=0.62, min_seed_area=1.0, min_room_area=2.5):
    free = free_space(grid)
    res = grid.res
    dist = ndimage.distance_transform_edt(free) * res
    seeds, n = ndimage.label(dist > seed_clear)
    if n == 0:
        return np.zeros_like(seeds)
    areas = ndimage.sum(np.ones_like(seeds), seeds, index=np.arange(1, n + 1)) * res * res
    keep = np.zeros(n + 1, bool)
    keep[1:] = areas >= min_seed_area
    seeds = np.where(keep[seeds], seeds, 0)
    labels = geodesic_grow(seeds, free)
    return merge_small(labels, free, min_room_area / (res * res))


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


def merge_small(lab, mask, min_cells):
    while True:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        small = ids[counts < min_cells]
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
        best = max(scores, key=scores.get) if scores else None
        if best is None or scores[best] < 2.0:
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
    for rid, name in (labels or {}).items():
        if rid in byid:
            byid[rid].label = name
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
