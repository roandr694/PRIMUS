"""Scene graph of the house: floors -> rooms -> objects, plus named places and the room connections the navigator
uses (docs/home_autonomy_architecture.md §4.4). Built from the SLAM floor maps and the object memory; it resolves
what the owner asks for ("cozinha", "do lado do sofá na sala", "carregador") into a goal pose.

    sg = SceneGraph(); sg.rebuild(stack)
    goal = sg.resolve(room="sala", obj="sofa", relation="near", robot_xy=(x, y))   # -> Goal or None (+ why)
    sg.to_text()            # compact description for the LLM prompt
    sg.label_room(xy, "sala de TV"); sg.add_place("carregador", (x, y, th), floor)
"""
import json
import math
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import rooms as R

# Portuguese / English words -> room types
ROOM_WORDS = {
    "cozinha": "kitchen", "kitchen": "kitchen", "sala": "living room", "sala de estar": "living room",
    "sala de tv": "living room", "living": "living room", "living room": "living room", "sala de jantar": "dining room",
    "quarto": "bedroom", "suite": "bedroom", "suíte": "bedroom", "bedroom": "bedroom", "banheiro": "bathroom",
    "lavabo": "bathroom", "bathroom": "bathroom", "lavanderia": "laundry", "area de servico": "laundry",
    "laundry": "laundry", "escritorio": "office", "office": "office", "corredor": "hallway", "hallway": "hallway",
}
# Portuguese object words -> detector classes (the LLM does this for free-form requests)
OBJECT_WORDS = {
    "sofa": "sofa", "sofá": "sofa", "poltrona": "armchair", "cadeira": "chair", "banqueta": "stool", "cama": "bed",
    "mesa de jantar": "dining table", "mesa de centro": "coffee table", "mesa lateral": "side table",
    "mesa": "dining table", "escrivaninha": "desk", "televisao": "television", "televisão": "television",
    "tv": "television", "estante": "bookshelf", "guarda-roupa": "wardrobe", "armario": "wardrobe",
    "geladeira": "refrigerator", "forno": "oven", "fogao": "stove", "fogão": "stove", "pia": "kitchen sink",
    "pia da cozinha": "kitchen sink", "pia do banheiro": "bathroom sink", "vaso": "toilet", "privada": "toilet",
    "banheira": "bathtub", "chuveiro": "shower", "maquina de lavar": "washing machine",
    "máquina de lavar": "washing machine", "lava-loucas": "dishwasher", "lava-louças": "dishwasher",
    "planta": "potted plant", "vaso de planta": "potted plant", "luminaria": "floor lamp", "luminária": "floor lamp",
    "ilha": "kitchen island", "banco": "bench", "criado-mudo": "nightstand", "aparador": "sideboard",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


@dataclass
class Goal:
    xy: tuple
    heading: float                 # math angle to face at the end (None = any)
    floor: str
    label: str
    kind: str                      # room | object | place
    target: dict = field(default_factory=dict)


@dataclass
class Place:
    name: str
    xy: tuple
    heading: float
    floor: str


class SceneGraph:
    def __init__(self):
        self.rooms = {}            # floor -> [Room]
        self.labels = {}           # floor -> list of (xy, name): owner's room labels, re-applied on rebuild
        self.regions = {}          # floor -> label image
        self.places = {}           # name -> Place
        self.grids = {}

    # ------------------------------------------------------------------------------------------------ build
    def rebuild(self, stack):
        mem = stack.memory
        for fid, fl in stack.slam.floors.items():
            g = fl.grid
            lab = R.segment_rooms(g, voxels=getattr(stack, "voxels", {}).get(fid),
                                  label_xy=[xy for xy, _ in self.labels.get(fid, [])])
            self.regions[fid], self.grids[fid] = lab, g
            labels = {}
            for xy, name in self.labels.get(fid, []):
                rid = R.room_at(lab, g, xy)
                if rid and name not in labels.get(rid, []):
                    labels.setdefault(rid, []).append(name)
            self.rooms[fid] = R.describe(lab, g, mem, fid, labels)
        return self

    def all_rooms(self):
        return [r for rs in self.rooms.values() for r in rs]

    def room_of(self, xy, floor):
        if floor not in self.regions:
            return None
        rid = R.room_at(self.regions[floor], self.grids[floor], xy)
        return next((r for r in self.rooms[floor] if r.id == rid), None)

    def label_room(self, xy, floor, name):
        self.labels.setdefault(floor, []).append((tuple(xy), name))
        r = self.room_of(xy, floor)
        if r is not None:
            if r.label and r.label != name:
                r.aliases.append(r.label)
            r.label = name
        return r

    def add_place(self, name, xy, heading, floor):
        self.places[norm(name)] = Place(name, tuple(xy), heading, floor)

    # ------------------------------------------------------------------------------------------------ resolve
    def find_room(self, text, floor=None):
        """Owner label > exact name > room type from a PT/EN word; prefers the current floor."""
        if not text:
            return None
        t = norm(text)
        rooms = self.all_rooms()
        rooms.sort(key=lambda r: (r.floor != floor, -r.area))
        for r in rooms:
            if any(norm(n) == t for n in [r.label or ""] + r.aliases if n):
                return r
        for r in rooms:
            if norm(r.name) == t or norm(r.pt) == t:
                return r
        for r in rooms:
            if r.label and (t in norm(r.label) or norm(r.label) in t):
                return r
        typ = ROOM_WORDS.get(t)
        if typ is None:                                    # longest known word inside the text ("a sala de tv")
            hits = [w for w in ROOM_WORDS if norm(w) in t]
            typ = ROOM_WORDS[max(hits, key=len)] if hits else None
        if typ is None and t in R.ROOM_TYPES:
            typ = t
        cands = [r for r in rooms if r.type == typ]
        return cands[0] if cands else None

    def object_class(self, text):
        t = norm(text)
        for w in sorted(OBJECT_WORDS, key=len, reverse=True):
            if norm(w) == t:
                return OBJECT_WORDS[w]
        for w in sorted(OBJECT_WORDS, key=len, reverse=True):
            if norm(w) in t:
                return OBJECT_WORDS[w]
        return text                                        # already English / free text for CLIP

    def resolve(self, stack, room=None, obj=None, place=None, robot_xy=None):
        """-> (Goal or None, message)"""
        floor = stack.slam.floor
        robot_xy = robot_xy if robot_xy is not None else stack.pose()[:2]
        if place:
            p = self.places.get(norm(place))
            if p is None:
                return None, f"não conheço o lugar '{place}'"
            return Goal(p.xy, p.heading, p.floor, p.name, "place"), ""
        r = self.find_room(room, floor) if room else None
        if room and r is None:
            return None, f"não encontrei '{room}' no mapa"
        if obj:
            cls = self.object_class(obj)
            hits = stack.memory.query(cls, room=r.name if r else None) if stack.memory else []
            if not hits and r is not None:
                hits = stack.memory.query(cls) if stack.memory else []
                hits = [(s, o) for s, o in hits if self.room_of(o.centroid[:2], o.floor) is r] or []
            if not hits:
                where = f" em {r.label or r.pt or r.name}" if r else ""
                return None, f"não vi nenhum(a) {obj}{where}"
            o = hits[0][1]
            xy, heading = self.approach(stack, o, robot_xy)
            if xy is None:
                return None, f"não acho um lugar livre perto do(a) {obj}"
            return Goal(xy, heading, o.floor, f"{obj} ({o.cls} #{o.id})", "object",
                        dict(object=o.id, cls=o.cls, room=o.room)), ""
        if r is not None:
            xy = self.room_point(r)
            return Goal(xy, None, r.floor, r.label or r.name, "room", dict(room=r.name)), ""
        return None, "não entendi para onde ir"

    def room_point(self, r):
        """The free cell of the room farthest from obstacles near its centroid."""
        lab, g = self.regions[r.floor], self.grids[r.floor]
        m = lab == r.id
        dist = ndimage.distance_transform_edt(m) * g.res
        ii, jj = np.nonzero(m)
        c = np.array(r.centroid)
        xy = g.xy(ii, jj)
        score = np.minimum(dist[ii, jj], 1.2) - 0.15 * np.linalg.norm(xy - c, axis=1)
        k = int(np.argmax(score))
        return (float(xy[k, 0]), float(xy[k, 1]))

    def approach(self, stack, o, robot_xy, gap=0.55):
        """A reachable point next to the object (gap metres from its footprint), facing it, near the robot."""
        cm = stack.costmap(o.floor)
        cost = cm.cost(robot_xy)
        half = 0.5 * np.maximum(o.hi[:2] - o.lo[:2], 0.2)
        c = o.centroid[:2]
        best, bs = None, math.inf
        for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            d = np.array([math.cos(a), math.sin(a)])
            r = float(np.abs(d * half).sum()) + gap + cm.robot_r
            p = c + d * r
            i, j = cm.ij(p)
            if not cm.g.inside(i, j) or not np.isfinite(cost[i, j]):
                continue
            s = float(np.linalg.norm(p - np.asarray(robot_xy))) * 0.3 + float(cost[i, j])
            if s < bs:
                best, bs = p, s
        if best is None:
            return None, None
        heading = math.atan2(c[1] - best[1], c[0] - best[0])
        return (float(best[0]), float(best[1])), heading

    # ------------------------------------------------------------------------------------------------ text / io
    def to_text(self, stack=None, max_objects=8):
        lines = []
        for fid, rs in self.rooms.items():
            z = stack.slam.floors[fid].z if stack is not None else 0.0
            lines.append(f"andar {fid} (z {z:+.1f} m):")
            for r in sorted(rs, key=lambda r: -r.area):
                objs = []
                if stack is not None and stack.memory is not None:
                    from collections import Counter
                    cnt = Counter(stack.memory.objects[i].cls for i in r.objects if i in stack.memory.objects)
                    objs = [f"{k}" + (f" x{v}" if v > 1 else "") for k, v in cnt.most_common(max_objects)]
                name = f"{r.label} ({r.pt})" if r.label else r.pt
                lines.append(f"  - {name}: {r.area:.0f} m2" + (f"; {', '.join(objs)}" if objs else ""))
        if self.places:
            lines.append("lugares: " + ", ".join(p.name for p in self.places.values()))
        return "\n".join(lines)

    def to_dict(self):
        return dict(rooms={f: [r.to_dict() for r in rs] for f, rs in self.rooms.items()},
                    labels={f: [[list(xy), n] for xy, n in v] for f, v in self.labels.items()},
                    places={k: dict(name=p.name, xy=list(p.xy), heading=p.heading, floor=p.floor)
                            for k, p in self.places.items()})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
