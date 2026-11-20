"""P3 test: the semantic tour (lidar SLAM + RGB-D detections) and the scene graph, scored against house_gt.json.

    MUJOCO_GL=egl python3 sim/home_sim/eval_semantic.py --floor social --out sim/models/house/robot_map

Objects: a ground-truth object (of a class the detector knows) is found when a confirmed memory object of the same
class lies within 1.0 m of its centre (precision: confirmed objects with a same-class GT object within 1.0 m).
Rooms: every GT room gets the predicted room covering most of its area; a room is named right when that predicted
room's type is an acceptable type for it (varanda -> living / kitchen / dining, ...).
The map, memory and scene graph are saved for the agent tests (load with brain.persist.load_stack).
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from brain import persist  # noqa: E402
from brain.semantic.memory import SemanticMemory  # noqa: E402
from brain.semantic.people import PeopleTracker  # noqa: E402
from brain.semantic.scene_graph import SceneGraph  # noqa: E402
from brain.stack import HomeStack  # noqa: E402
from runner import Scenario, run_stack  # noqa: E402
from tour import TourDriver  # noqa: E402
from world import GT_PATH, HomeSim  # noqa: E402

GT_TO_DET = {"tv": "television", "bookcase": "bookshelf", "fridge": "refrigerator", "cooktop": "stove",
             "tv bench": None, "kitchen counter": None, "wine rack": None, "parasol": None, "hammock": None,
             "lounge chair": "sun lounger", "kitchen counter": "kitchen counter"}
ACCEPT = {"varanda": {"living room", "kitchen", "dining room"}, "living room": {"living room", "dining room"},
          "kitchen": {"kitchen"}, "service area": {"laundry"}, "service hall": {"hallway", "room", "laundry"},
          "powder room": {"bathroom"}, "staff bathroom": {"bathroom"}, "pantry": {"room", "hallway"},
          "wine cellar": {"room"}, "south suite": {"bedroom"}, "middle suite": {"bedroom"}, "master suite": {"bedroom"},
          "bedroom corridor": {"hallway"}, "upper lounge": {"living room", "dining room"}, "study": {"office"},
          "upper service": {"room", "laundry"}, "east strip": {"hallway"}}


class TourTask:
    """TourDriver adapted to run_stack (the pilot drives; the stack only perceives)."""

    def __init__(self, sim, floor, rooms=None):
        self.tour = TourDriver(sim, floor, order=rooms)
        self.message = ""

    def start(self, sim, stack):
        self.tour.start(sim, stack.brain)

    def step(self, now, stack, ev):
        stack.pilot_override = self.tour.step(now, stack.brain, ev)
        self.message = self.tour.message

    def done(self):
        return self.tour.done()

    def extra(self):
        return self.tour.extra()


def score_objects(mem, gt, floor):
    gobj = [o for o in gt["objects"] if o["floor"] == floor and o["room"] is not None]
    pred = mem.confirmed()
    det_cls = set(mem.classes)
    found, considered = 0, 0
    per_cls = Counter()
    per_cls_found = Counter()
    for o in gobj:
        c = GT_TO_DET.get(o["cls"], o["cls"])
        if c is None or c not in det_cls:
            continue
        considered += 1
        per_cls[c] += 1
        ok = any(p.cls == c and np.linalg.norm(p.centroid[:2] - np.array(o["center"][:2])) < 1.0 for p in pred)
        found += ok
        per_cls_found[c] += ok
    good = 0
    for p in pred:
        good += any(GT_TO_DET.get(o["cls"], o["cls"]) == p.cls and
                    np.linalg.norm(p.centroid[:2] - np.array(o["center"][:2])) < 1.0 for o in gobj)
    return found, considered, good, len(pred), {k: f"{per_cls_found[k]}/{v}" for k, v in per_cls.most_common()}


def score_rooms(sg, stack, gt, floor):
    fid = stack.slam.floor
    lab, g = sg.regions[fid], sg.grids[fid]
    rooms = {r.id: r for r in sg.rooms[fid]}
    out = []
    for r in gt["rooms"]:
        if r["floor"] != floor or not r["indoor"]:
            continue
        x0, x1, y0, y1 = r["rects"][0]
        xs = np.arange(x0 + 0.1, x1, 0.1)
        ys = np.arange(y0 + 0.1, y1, 0.1)
        X, Y = np.meshgrid(xs, ys)
        i, j = g.ij(np.stack([X.ravel(), Y.ravel()], 1))
        ok = g.inside(i, j)
        ids = lab[i[ok], j[ok]]
        ids = ids[ids > 0]
        if len(ids) == 0:
            out.append((r["name"], None, False, 0.0))
            continue
        rid, cnt = Counter(ids.tolist()).most_common(1)[0]
        pr = rooms.get(rid)
        right = pr is not None and pr.type in ACCEPT.get(r["name"], set())
        out.append((r["name"], pr.name if pr else None, right, cnt / len(ids)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="social")
    ap.add_argument("--spawn", default="home")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--rgbd-hz", type=float, default=2.0)
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/models/house/robot_map"))
    ap.add_argument("--timeout", type=float, default=1500)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    sim = HomeSim(spawn=a.spawn, rgbd_hz=a.rgbd_hz)
    stack = HomeStack()
    stack.memory = SemanticMemory()
    stack.people = PeopleTracker()
    task = TourTask(sim, a.floor, a.rooms)
    res, sim = run_stack(Scenario("semantic tour", spawn=a.spawn, timeout=a.timeout, rgbd_hz=a.rgbd_hz), stack, task,
                         sim=sim, video=a.video, verbose=False)
    print(res.line(), res.extra.get("visited"), "failed:", res.extra.get("failed"))
    mem = stack.memory
    tm = mem.timing
    print(f"memory: {len(mem.confirmed())} confirmed objects ({len(mem.objects)} incl. tentative), "
          f"{len(mem.keyframes)} keyframes; detector {1000 * tm['det'] / max(tm['n'], 1):.0f} ms, "
          f"CLIP {1000 * tm['emb'] / max(tm['n'], 1):.0f} ms per frame; classes {mem.summary()}")
    stack.scene = SceneGraph().rebuild(stack)
    gt = json.loads(GT_PATH.read_text())
    f, n, good, npred, per = score_objects(mem, gt, a.floor)
    print(f"objects: recall {f}/{n} = {100 * f / max(n, 1):.0f} %, precision {good}/{npred} = "
          f"{100 * good / max(npred, 1):.0f} %; per class {per}")
    rr = score_rooms(stack.scene, stack, gt, a.floor)
    print(f"rooms: {sum(r[2] for r in rr)}/{len(rr)} named acceptably; predicted rooms: "
          f"{[(r.name, round(r.area)) for r in stack.scene.rooms[stack.slam.floor]]}")
    for name, pred, right, cover in rr:
        print(f"   {name:18s} -> {str(pred):18s} {'OK ' if right else '-- '} ({cover * 100:.0f} % of its area)")
    print(stack.scene.to_text(stack))
    persist.save_stack(stack, a.out)
    print("saved", a.out)
    sim.close()


if __name__ == "__main__":
    main()
