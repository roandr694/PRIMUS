"""Ground truth of the CMA house for the home-autonomy stack (brain/): rooms and labelled objects.

    rooms    hand-drawn from structure.py (axis-aligned rectangles per floor level, plan frame X east, Y north)
    objects  recorded automatically while interiors.py builds: every call of a labelled furniture builder
             (OBJECT_CLASSES) becomes one object with its class, world bounding box and room; kitchen-run modules
             (fridge, sink, oven, dishwasher, cooktop) are expanded from the builder arguments.

build.py writes it to sim/models/house/house_gt.json (generated, not in Git). It scores localisation, room
segmentation and naming, object recall and navigation goals in sim/home_sim; the robot never reads it.
"""
import json
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# floors (z of the walkable surface) and rooms
# ---------------------------------------------------------------------------------------------------------------
FLOORS = {"sea_deck": -3.4, "social": 0.0, "upper": 3.6, "parking": 13.2}

# name, Portuguese names the owner may use, floor, rectangles (x0, x1, y0, y1), indoor
ROOMS = [
    # social level (z 0)
    ("varanda", ["varanda", "cozinha gourmet", "área gourmet"], "social", [(-0.1, 6.25, 0.1, 16.4)], True),
    ("living room", ["sala", "sala de estar", "sala de tv", "sala de jantar", "living"], "social",
     [(6.25, 16.1, 4.2, 16.4)], True),
    ("kitchen", ["cozinha"], "social", [(6.3, 14.95, 0.1, 4.2)], True),
    ("service area", ["área de serviço", "lavanderia"], "social", [(14.95, 21.1, 0.1, 3.55)], True),
    ("service hall", ["corredor de serviço"], "social", [(15.0, 21.1, 3.55, 4.85)], True),
    ("powder room", ["lavabo"], "social", [(16.2, 18.15, 4.9, 7.45)], True),
    ("staff bathroom", ["banheiro de serviço"], "social", [(19.05, 21.0, 4.9, 7.45)], True),
    ("pantry", ["despensa"], "social", [(16.2, 21.0, 7.45, 10.08)], True),
    ("wine cellar", ["adega"], "social", [(16.2, 21.1, 10.08, 16.4)], True),
    ("lawn", ["gramado", "jardim do gramado"], "social", [(-7.4, -0.1, -1.7, 16.6)], False),
    ("pool deck", ["deck da piscina", "piscina"], "social", [(-10.0, -7.4, -1.7, 15.0)], False),
    # upper level (z 3.6)
    ("south suite", ["suíte sul", "suíte 3", "quarto 3"], "upper", [(-2.3, 6.06, 0.15, 5.01)], True),
    ("middle suite", ["suíte do meio", "suíte 2", "quarto 2"], "upper", [(-2.3, 6.06, 5.01, 10.2)], True),
    ("master suite", ["suíte master", "quarto principal", "suíte 1"], "upper", [(-2.3, 7.75, 10.2, 16.48)], True),
    ("bedroom corridor", ["corredor dos quartos", "corredor"], "upper", [(6.06, 7.75, 0.15, 10.2)], True),
    ("upper lounge", ["sala de cima", "lounge", "sala íntima"], "upper", [(7.8, 17.72, 0.12, 4.1)], True),
    ("upper service", ["serviço de cima"], "upper", [(17.72, 20.8, 0.12, 4.1)], True),
    ("study", ["escritório"], "upper", [(20.8, 25.7, 0.12, 4.06)], True),
    ("east strip", ["corredor leste"], "upper", [(15.9, 17.75, 4.1, 16.55)], True),
    ("entrance garden", ["jardim", "jardim de entrada"], "upper", [(17.75, 25.7, 4.1, 13.0)], False),
    ("terrace", ["terraço"], "upper", [(17.75, 27.5, 13.0, 18.5)], False),
    # sea deck level (z -3.4)
    ("pavilion", ["pavilhão", "sala de jogos"], "sea_deck", [(-17.5, -13.4, 0.0, 14.0)], True),
    ("sea deck", ["deck", "deck do mar"], "sea_deck", [(-27.0, -17.5, -1.7, 17.2)], False),
]

# furniture builder -> object class (None: not an object the robot should know by name)
OBJECT_CLASSES = {
    "sofa_sectional": "sofa", "sofa_builtin": "sofa", "armchair_woven": "armchair", "armchair_lounge": "armchair",
    "deck_lounge_chair": "lounge chair", "chair_wood": "chair", "chair_windsor": "chair", "bar_stool": "stool",
    "stool_ceramic": "stool", "lounger": "sun lounger", "parasol": "parasol", "hammock": "hammock",
    "coffee_table_log": "coffee table", "coffee_table_round": "coffee table", "side_table_glass": "side table",
    "side_table_round": "side table", "dining_table": "dining table", "desk": "desk", "office_chair": "office chair",
    "bench_shelf": "tv bench", "tv": "tv", "bookcase": "bookcase", "sideboard": "sideboard",
    "wardrobe": "wardrobe", "wine_rack": "wine rack", "floor_lamp": "floor lamp", "plant_pot": "potted plant",
    "bed": "bed", "nightstand": "nightstand", "bench_end": "bench", "vanity": "bathroom sink", "toilet": "toilet",
    "shower": "shower", "bathtub": "bathtub", "island": "kitchen island", "washer": "washing machine",
    "foosball": "foosball table", "massage_table": "massage table", "sauna": "sauna", "car": "car",
    "generator": "generator", "boat": "boat",
}
KITCHEN_MODULES = {"fridge": "fridge", "sink": "kitchen sink", "oven_tall": "oven", "oven": "oven",
                   "dishwasher": "dishwasher", "cooktop": "cooktop"}

# Portuguese names, for the agent's tests (the robot learns names from its own detector + LLM)
CLASS_PT = {
    "sofa": "sofá", "armchair": "poltrona", "lounge chair": "espreguiçadeira", "chair": "cadeira", "stool": "banqueta",
    "sun lounger": "espreguiçadeira", "parasol": "guarda-sol", "hammock": "rede", "coffee table": "mesa de centro",
    "side table": "mesa lateral", "dining table": "mesa de jantar", "desk": "escrivaninha",
    "office chair": "cadeira de escritório", "tv bench": "rack da tv", "tv": "televisão", "bookcase": "estante",
    "sideboard": "aparador", "wardrobe": "guarda-roupa", "wine rack": "adega de vinhos", "floor lamp": "luminária",
    "potted plant": "planta", "bed": "cama", "nightstand": "criado-mudo", "bench": "banco",
    "bathroom sink": "pia do banheiro", "toilet": "vaso sanitário", "shower": "chuveiro", "bathtub": "banheira",
    "kitchen island": "ilha da cozinha", "washing machine": "máquina de lavar", "foosball table": "pebolim",
    "massage table": "maca", "sauna": "sauna", "car": "carro", "generator": "gerador", "boat": "barco",
    "fridge": "geladeira", "kitchen sink": "pia da cozinha", "oven": "forno", "dishwasher": "lava-louças",
    "cooktop": "fogão",
}


def room_at(x, y, z, tol_z=1.2):
    """Name of the room containing (x, y) on the floor nearest to z (None outside every room)."""
    for name, _, floor, rects, _ in ROOMS:
        if abs(z - FLOORS[floor]) > tol_z:
            continue
        for x0, x1, y0, y1 in rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
    return None


def floor_of(z):
    return min(FLOORS, key=lambda f: abs(FLOORS[f] - z))


# ---------------------------------------------------------------------------------------------------------------
# object recorder
# ---------------------------------------------------------------------------------------------------------------
class Recorder:
    """Wraps furniture builders and Scene.add_mesh while the house is built; each labelled call accumulates the
    world vertices of the meshes it adds (body-local meshes through the body frame chain)."""

    def __init__(self, scene):
        self.s, self.objects, self.stack = scene, [], []

    def _world(self, V):
        b = self.s.target
        while b is not None:                                    # body-local -> world through the parents
            V = V @ b.frame.R.T + b.frame.t
            b = b.parent
        return V

    @contextmanager
    def active(self):
        import furniture as FU
        from geom import Scene
        orig_mesh = Scene.add_mesh
        rec = self

        def add_mesh(scene, mat, v, n, uv_m, f, frame=None, **kw):
            if frame is None:
                from geom import IDENT
                frame = IDENT
            if rec.stack:
                Vw = np.asarray(v, float) @ frame.R.T + frame.t
                rec.stack[-1]["pts"].append(rec._world(Vw))
            return orig_mesh(scene, mat, v, n, uv_m, f, frame=frame, **kw)

        originals = {}
        for fn_name, cls in OBJECT_CLASSES.items():
            if hasattr(FU, fn_name):
                originals[fn_name] = getattr(FU, fn_name)
                setattr(FU, fn_name, self._wrap(originals[fn_name], cls))
        originals["kitchen_run"] = FU.kitchen_run
        FU.kitchen_run = self._wrap_kitchen(FU.kitchen_run)
        Scene.add_mesh = add_mesh
        try:
            yield self
        finally:
            Scene.add_mesh = orig_mesh
            for k, f in originals.items():
                setattr(FU, k, f)

    def _wrap(self, fn, cls):
        def wrapped(s, *a, **kw):
            self.stack.append(dict(cls=cls, pts=[]))
            try:
                return fn(s, *a, **kw)
            finally:
                rec = self.stack.pop()
                if rec["pts"]:
                    self._add(cls, np.concatenate(rec["pts"]))
        return wrapped

    def _wrap_kitchen(self, fn):
        def wrapped(s, fr, modules, D=0.62, H=0.9, *a, **kw):
            x = 0.0
            for kind, w in modules:
                cls = KITCHEN_MODULES.get(kind)
                if cls:
                    tall = kind in ("fridge", "tall", "oven_tall")
                    hh = 2.4 if tall else H
                    lo, hi = np.array([x, 0.0, 0.0]), np.array([x + w, D, hh])
                    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (0, hh)])
                    self._add(cls, corners @ fr.R.T + fr.t)
                x += w
            self.stack.append(dict(cls="kitchen counter", pts=[]))
            try:
                return fn(s, fr, modules, D, H, *a, **kw)
            finally:
                rec = self.stack.pop()
                if rec["pts"]:
                    self._add("kitchen counter", np.concatenate(rec["pts"]))
        return wrapped

    def _add(self, cls, P):
        lo, hi = P.min(0), P.max(0)
        c = (lo + hi) / 2
        z_floor = FLOORS[floor_of(lo[2])]
        self.objects.append(dict(id=f"{cls.replace(' ', '_')}_{len(self.objects)}", cls=cls,
                                 name_pt=CLASS_PT.get(cls, cls), center=[round(float(v), 3) for v in c],
                                 bbox=[[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
                                 floor=floor_of(lo[2]), room=room_at(c[0], c[1], z_floor)))


def ground_truth(scene, recorder):
    rooms = [dict(name=n, names_pt=pt, floor=f, z=FLOORS[f], rects=[list(r) for r in rects], indoor=ind)
             for n, pt, f, rects, ind in ROOMS]
    return dict(frame="world = plan frame of structure.py (X east, Y north, z up)", floors=FLOORS, rooms=rooms,
                objects=recorder.objects, spawns={k: list(v) for k, v in scene.spawns.items()})


def write(path, scene, recorder):
    gt = ground_truth(scene, recorder)
    path.write_text(json.dumps(gt, indent=1, ensure_ascii=False))
    n_room = sum(o["room"] is not None for o in gt["objects"])
    print(f"ground truth: {len(gt['rooms'])} rooms, {len(gt['objects'])} objects ({n_room} inside a room) -> {path}")
    return gt
