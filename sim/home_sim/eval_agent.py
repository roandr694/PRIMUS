"""P4/P5 test: Portuguese requests to the agent, executed in the house, judged on the truth.

    MUJOCO_GL=egl python3 sim/home_sim/eval_agent.py --map sim/models/house/robot_map [--llm Qwen/Qwen3-4B-Instruct-2507]

Each case: a sentence, then a check on the final true state (in a GT room, within reach of a GT object of a class,
leg height, carry flag, ...). Cases run one after another from where the previous one ended, like a real day.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "sim" / "house"))
from brain import persist  # noqa: E402
from brain.agent.agent import Agent  # noqa: E402
from runner import CollisionMonitor  # noqa: E402
from semantics import room_at  # noqa: E402
from world import GT_PATH, HomeSim  # noqa: E402

GT = json.loads(GT_PATH.read_text())


def in_room(names):
    names = [names] if isinstance(names, str) else names
    return lambda sim, st: room_at(*sim.truth().p[:2], 0.0) in names, f"in {names}"


def near(cls, dist=1.3, room=None):
    def f(sim, st):
        xy = sim.truth().xy
        for o in GT["objects"]:
            if o["cls"] == cls and (room is None or o["room"] == room):
                lo, hi = np.array(o["bbox"][0][:2]), np.array(o["bbox"][1][:2])
                d = np.linalg.norm(np.maximum(0, np.maximum(lo - xy, xy - hi)))
                if d < dist:
                    return True
        return False
    return f, f"near {cls}" + (f" in {room}" if room else "")


def leg(lo, hi):
    return lambda sim, st: lo <= st.brain.state.leg_length <= hi, f"leg in [{lo}, {hi}]"


def flag(name, val):
    return lambda sim, st: getattr(st, name) == val, f"{name} == {val}"


def reply_has(*words):
    return lambda sim, st, reply="": True, f"reply mentions {words}"


CASES = [
    ("vai para a cozinha", in_room("kitchen"), 120),
    ("vai para a sala e fica do lado do sofá", near("sofa", room="living room"), 120),
    ("fica mais baixo", leg(0.17, 0.24), 10),
    ("fica na altura normal", leg(0.26, 0.34), 10),
    ("vai até a geladeira", near("fridge", 1.4), 120),
    ("vai para a varanda", in_room("varanda"), 120),
    ("vai para perto da mesa de jantar", near("dining table", 1.4), 120),
    ("vou botar esse copo em cima de você", flag("carrying", True), 5),
    ("vai para a lavanderia", in_room(["service area", "service hall"]), 150),
    ("já tirei o copo", flag("carrying", False), 5),
    ("fica mais alto", leg(0.40, 0.52), 10),
    ("onde você está?", (lambda sim, st: True, "answers"), 3),
    ("vai para a televisão", near("tv", 2.0), 120),
    ("lembra deste lugar como cantinho da TV", (lambda sim, st: "cantinho da tv" in st.scene.places, "place saved"), 3),
    ("vai para a cozinha", in_room("kitchen"), 120),
    ("volta para o cantinho da TV", near("tv", 2.2), 150),
]
LLM_CASES = [
    ("Primus, estou com fome, me espera perto da geladeira", near("fridge", 1.4), 150),
    ("dá uma olhada na varanda pra mim", in_room("varanda"), 150),
    ("quero assistir televisão, fica pertinho do sofá grande da sala", near("sofa", 1.4, room="living room"), 150),
    ("você consegue abrir a porta da cozinha?", (lambda sim, st: True, "explains"), 5),
    ("se abaixa que vai passar embaixo da mesa", leg(0.17, 0.26), 15),
]


def run_case(sim, stack, agent, text, check, timeout, coll, verbose=True):
    t0 = sim.t
    w0 = time.time()
    reply = agent.handle(text)
    llm_ms = getattr(agent.llm, "last_ms", 0.0) if agent.log and agent.log[-1][1] == "llm" else 0.0
    cmd = patch = None
    events = []
    fell = False
    while sim.t - t0 < timeout:
        ev = sim.step(cmd, patch)
        stack.on_state(ev.state)
        if ev.depth is not None:
            stack.on_depth(ev.depth)
        if ev.scan is not None:
            stack.on_scan(ev.scan)
        if ev.rgbd is not None:
            stack.on_rgbd(ev.rgbd)
        coll.update(sim.d, sim.t)
        cmd, patch = stack.step(sim.t)
        m = agent.poll()
        if m:
            events.append(m)
        if ev.fallen:
            fell = True
            break
        if stack.behaviour is None and sim.t - t0 > 1.0:
            break
    fn, desc = check
    ok = (not fell) and bool(fn(sim, stack))
    src = agent.log[-1][1] if agent.log else "?"
    calls = agent.log[-1][2] if agent.log else []
    if verbose:
        print(f"  [{'OK ' if ok else 'BAD'}] {text!r:60s} ({src}{f' {llm_ms:.0f} ms' if llm_ms else ''}) -> {reply!r}"
              f" | {'; '.join(events)} | check: {desc} | t={sim.t - t0:.1f}s{' FELL' if fell else ''}")
    return dict(text=text, ok=ok, source=src, calls=calls, reply=reply, events=events, sim_time=round(sim.t - t0, 1),
                wall=round(time.time() - w0, 1), fell=fell, llm_ms=round(llm_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE.parents[1] / "sim/models/house/robot_map"))
    ap.add_argument("--spawn", default="home")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--only-llm", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sim = HomeSim(spawn=a.spawn, rgbd_hz=2.0)
    stack = persist.load_stack(a.map)
    llm = None
    if a.llm:
        from brain.agent.llm import TransformersLLM
        llm = TransformersLLM(a.llm)
    agent = Agent(stack, llm)
    coll = CollisionMonitor(sim.m)
    # stand up first
    cmd = patch = None
    while sim.t < 4.0:
        ev = sim.step(cmd, patch)
        stack.on_state(ev.state)
        cmd, patch = stack.step(sim.t)
    cases = ([] if a.only_llm else CASES) + (LLM_CASES if llm is not None else [])
    res = [run_case(sim, stack, agent, t, c, to, coll) for t, c, to in cases]
    n = sum(r["ok"] for r in res)
    print(f"agent: {n}/{len(res)} requests OK; collisions {coll.count}; falls {sum(r['fell'] for r in res)}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    sim.close()


if __name__ == "__main__":
    main()
