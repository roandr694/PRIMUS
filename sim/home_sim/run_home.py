"""PRIMUS at home, live: the CMA house in MuJoCo (real time) with the whole Jetson stack, the agent and the owner's
web page. Talk to the robot from a browser (phone on the same Wi-Fi works too).

    MUJOCO_GL=egl python3 sim/home_sim/run_home.py --map sim/models/house/robot_map [--llm Qwen/Qwen3-4B-Instruct-2507]
    then open http://localhost:8090

Needs a saved house (eval_semantic.py writes one). --person adds a person walking through the living room, to
try "me segue". --viewer also opens the MuJoCo viewer.
"""
import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from aiohttp import web  # noqa: E402

from brain import persist  # noqa: E402
from brain.agent.agent import Agent  # noqa: E402
from brain.ui.server import make_app, render_map, to_jpeg  # noqa: E402
from world import HomeSim  # noqa: E402


class Hub:
    def __init__(self, sim, stack, agent):
        self.sim, self.stack, self.agent = sim, stack, agent
        self.lock = threading.Lock()
        self.events, self.cam = [], np.zeros((360, 640, 3), np.uint8)
        self._map, self._map_t = b"", 0.0

    def handle(self, text):
        with self.lock:
            return self.agent.handle(text)

    def state(self):
        with self.lock:
            st = self.stack.status()
            ev, self.events = self.events, []
        r = self.stack.scene.room_of(self.stack.pose()[:2], self.stack.floor) if self.stack.scene else None
        line = (f"{r.label or r.pt if r else '?'} · {st['nav']} · perna {self.stack.leg * 100:.0f} cm"
                + (" · carga" if self.stack.carrying else "") + (f" · {st['message']}" if st.get("message") else ""))
        return dict(line=line, events=ev)

    def map_png(self):
        if time.time() - self._map_t > 0.6:
            with self.lock:
                self._map = render_map(self.stack)
            self._map_t = time.time()
        return self._map

    def camera_jpg(self):
        return to_jpeg(self.cam)


def sim_loop(hub, realtime=True, viewer=None):
    sim, stack, agent = hub.sim, hub.stack, hub.agent
    cmd = patch = None
    while True:
        t0 = time.time()
        with hub.lock:
            ev = sim.step(cmd, patch)
            stack.on_state(ev.state)
            if ev.depth is not None:
                stack.on_depth(ev.depth)
            if ev.scan is not None:
                stack.on_scan(ev.scan)
            if ev.rgbd is not None:
                stack.on_rgbd(ev.rgbd)
                hub.cam = ev.rgbd.rgb
            cmd, patch = stack.step(sim.t)
            msg = agent.poll()
            if msg:
                hub.events.append(msg)
            if ev.fallen:
                hub.events.append("caí! (aperte o botão de levantar)")
        if viewer is not None:
            viewer.sync()
        if realtime:
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(HERE.parents[1] / "sim/models/house/robot_map"))
    ap.add_argument("--spawn", default="home")
    ap.add_argument("--llm", default=None, help="local LLM (transformers name); default: rules only")
    ap.add_argument("--person", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()
    sim = HomeSim(spawn=a.spawn, rgbd_hz=4.0, person=a.person)
    if a.person:
        from person import Walker
        sim.walker = Walker(sim, [(11.5, 10.0), (8.5, 6.5), (3.0, 9.0), (3.0, 13.0), (11.0, 14.5), (12.5, 8.0)] * 5,
                            start_delay=8.0, pause=4.0)
    stack = persist.load_stack(a.map)
    llm = None
    if a.llm:
        from brain.agent.llm import TransformersLLM
        llm = TransformersLLM(a.llm)
    agent = Agent(stack, llm)
    hub = Hub(sim, stack, agent)
    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)
    threading.Thread(target=sim_loop, args=(hub, True, viewer), daemon=True).start()
    print(f"PRIMUS at home: http://localhost:{a.port}")
    web.run_app(make_app(hub), port=a.port, print=None)


if __name__ == "__main__":
    main()
