"""The test worlds of the home sim: the generated CMA house and the realistic apartments (sim/apartments).

    w = get_world("cma")                 # or "kujiale_0026", "kujiale_0003", ...
    w.scene                              # SkillWorld scene ("house" or the generated MJCF path)
    w.gt, w.gt_dir                       # ground truth dict (house_gt.json / gt.json) and its folder (caches)
    w.room_at(x, y, z)                   # room name or None (rectangles or polygons)
    w.room_mask(room, X, Y)              # bool mask of a room on a grid of points
    w.bounds(floor)                      # (floor_z, (x0, x1, y0, y1)) for the ground-truth grid
"""
import json
from pathlib import Path

import numpy as np

PRIMUS = Path(__file__).resolve().parents[2]
MODELS = PRIMUS / "sim" / "models"

CMA_DOORS = ("kitchen_door_w", "kitchen_door_e", "service_entry", "cellar_pivot", "lavabo_door", "staffbath_door",
             "pantry_cellar_door", "master_door", "suite3_door", "suite2_door", "master_bath_door",
             "suite2_bath_door", "suite3_bath_door", "upper_service_door", "study_door")
CMA_BOUNDS = {
    "social": (0.0, (-10.5, 22.0, -2.0, 17.0)),
    "upper": (3.6, (-2.6, 27.8, -0.2, 18.8)),
    "sea_deck": (-3.4, (-27.5, -13.0, -2.0, 17.5)),
}


class World:
    def __init__(self, name):
        self.name = name
        if name == "cma":
            self.scene = "house"
            self.gt_path = MODELS / "house" / "house_gt.json"
            self.doors = CMA_DOORS
            self.house_runtime = True
            self.default_floor = "social"
            self.usd = PRIMUS / "sim" / "isaac" / "usd" / "house" / "cma_house.usd"
        else:
            self.scene = str(MODELS / f"cena_{name}.xml")
            self.gt_path = MODELS / "apartments" / name / "gt.json"
            if not self.gt_path.exists():
                raise FileNotFoundError(f"{self.gt_path}: run sim/apartments/convert.py {name}")
            self.doors = ()
            self.house_runtime = False
            self.default_floor = "main"
            self.usd = None
        self.gt = json.loads(self.gt_path.read_text())
        if self.usd is None:
            self.usd = PRIMUS / self.gt["usd"]
        self.gt_dir = self.gt_path.parent
        self._paths = {}

    # ------------------------------------------------------------------------------------------------ rooms
    def _poly(self, r):
        from matplotlib.path import Path as MPath
        if r["name"] not in self._paths:
            self._paths[r["name"]] = MPath(np.asarray(r["polygon"])) if "polygon" in r else None
        return self._paths[r["name"]]

    def room_at(self, x, y, z=0.0, tol_z=1.2):
        for r in self.gt["rooms"]:
            if abs(z - self.gt["floors"][r["floor"]]) > tol_z:
                continue
            if "polygon" in r:
                if self._poly(r).contains_point((x, y)):
                    return r["name"]
            else:
                for x0, x1, y0, y1 in r["rects"]:
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        return r["name"]
        return None

    def room_mask(self, r, X, Y):
        if isinstance(r, str):
            r = next(q for q in self.gt["rooms"] if q["name"] == r)
        if "polygon" in r:
            return self._poly(r).contains_points(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)
        m = np.zeros(X.shape, bool)
        for x0, x1, y0, y1 in r["rects"]:
            m |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        return m

    def bounds(self, floor):
        if self.name == "cma":
            return CMA_BOUNDS[floor]
        lo, hi = self.gt["bounds"]
        return self.gt["floors"][floor], (lo[0], hi[0], lo[1], hi[1])

    def spawn_names(self):
        return list(self.gt.get("spawns", {}))


_CACHE = {}


def get_world(name="cma"):
    if name not in _CACHE:
        _CACHE[name] = World(name)
    return _CACHE[name]
