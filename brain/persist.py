"""Save / load what the robot learnt about the house: floor maps (SLAM), object memory, scene graph labels and
places. One directory per house; the robot loads it at boot and localises on it.

    save_stack(stack, "maps/casa"); stack = load_stack("maps/casa")
"""
import json
import pickle
from pathlib import Path

import numpy as np


def save_stack(stack, directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stack.slam.save(d)
    if stack.memory is not None:
        objs = []
        for o in stack.memory.objects.values():
            objs.append(dict(id=o.id, cls=o.cls, floor=o.floor, centroid=o.centroid, lo=o.lo, hi=o.hi, emb=o.emb,
                             score=o.score, hits=o.hits, misses=o.misses, first_seen=o.first_seen,
                             last_seen=o.last_seen, room=o.room, crop=o.crop, votes=o.votes))
        with open(d / "memory.pkl", "wb") as f:
            pickle.dump(dict(objects=objs, history=stack.memory.history, next_id=stack.memory.next_id,
                             keyframes=stack.memory.keyframes), f)
        (d / "memory.json").write_text(json.dumps(stack.memory.to_dict(), indent=1, ensure_ascii=False))
    if stack.scene is not None:
        stack.scene.save(d / "scene_graph.json")
    for fid, cm in stack.costmaps.items():              # camera obstacles of the tour (what the lidar misses)
        np.savez_compressed(d / f"camera_{fid}.npz", t_edge=cm.t_edge, terrain=cm.terrain, seen=cm.terrain_seen)


def load_stack(directory, memory=True, guard=True):
    from .semantic.memory import Obj, SemanticMemory
    from .semantic.people import PeopleTracker
    from .semantic.scene_graph import Place, SceneGraph
    from .stack import HomeStack
    d = Path(directory)
    stack = HomeStack.load(d, guard=guard)
    for fid in stack.slam.floors:
        f = d / f"camera_{fid}.npz"
        if f.exists():
            z = np.load(f)
            cm = stack.costmap(fid)
            cm.t_edge[:], cm.terrain[:], cm.terrain_seen[:] = z["t_edge"], z["terrain"], z["seen"]
    if memory and (d / "memory.pkl").exists():
        mem = SemanticMemory()
        with open(d / "memory.pkl", "rb") as f:
            m = pickle.load(f)
        for o in m["objects"]:
            ob = Obj(o["id"], o["cls"], o["floor"], np.asarray(o["centroid"]), np.asarray(o["lo"]), np.asarray(o["hi"]),
                     np.asarray(o["emb"]), o["score"], o["hits"], o["misses"], o["first_seen"], o["last_seen"],
                     o["room"], o["crop"], 0.0, dict(o["votes"]))
            mem.objects[ob.id] = ob
        mem.history, mem.next_id = m["history"], m["next_id"]
        mem.keyframes = m.get("keyframes", [])
        stack.memory = mem
        stack.people = PeopleTracker()
    sg = SceneGraph()
    if (d / "scene_graph.json").exists():
        js = json.loads((d / "scene_graph.json").read_text())
        sg.labels = {f: [(tuple(xy), n) for xy, n in v] for f, v in js.get("labels", {}).items()}
        for k, p in js.get("places", {}).items():
            sg.places[k] = Place(p["name"], tuple(p["xy"]), p["heading"], p["floor"])
    stack.scene = sg.rebuild(stack)
    return stack
