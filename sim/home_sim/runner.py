"""Closed loop for one scenario: HomeSim (physics + Pi) <-> Brain (Jetson stack) <-> a driver (scripted pilot,
navigator, agent), with the metrics every phase is judged on.

    r = run(Scenario(...), driver)      -> Result(success, reason, time, falls, collisions, path_length, ...)

Collisions: contacts between a robot geom other than the wheels and any house geom (a wheel touching a step riser
during a skill is not a collision; the chassis touching a sofa is). Falls: the fall detector of HomeSim.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import mujoco

from world import HomeSim, STUDENT_V1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brain.runtime import Brain  # noqa: E402


@dataclass
class Scenario:
    name: str
    spawn: str = "home"
    timeout: float = 60.0
    seed: int = 0
    rgbd_hz: float = 0.0
    odom_noise: bool = True
    setup: object = None                 # callable(sim) -> None: move objects, close doors, ... before the start
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    name: str
    success: bool
    reason: str
    time: float
    falls: int
    collisions: int
    path_length: float
    final_xy: tuple
    wall_time: float
    log: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def line(self):
        return (f"{self.name:32s} {'OK  ' if self.success else 'FAIL'} {self.reason:28s} t={self.time:6.1f}s "
                f"falls={self.falls} coll={self.collisions:3d} path={self.path_length:5.1f}m wall={self.wall_time:5.0f}s")


class CollisionMonitor:
    def __init__(self, m):
        self.m = m
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        root = m.body_rootid[base]
        self.robot_body = np.array([m.body_rootid[b] == root for b in range(m.nbody)])
        wheel_bodies = set()
        for b in range(m.nbody):
            if self.robot_body[b] and "wheel" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""):
                wheel_bodies.add(b)
        gear_bodies = {b for b in range(m.nbody) if self.robot_body[b]
                       and "gear" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")}
        self.ignore = wheel_bodies | gear_bodies          # the landing gear touches the floor when parking
        self.count, self.active, self.events = 0, set(), []

    def update(self, d, t):
        m = self.m
        now = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = self.robot_body[b1], self.robot_body[b2]
            if r1 == r2:
                continue
            rb, other = (b1, c.geom2) if r1 else (b2, c.geom1)
            if rb in self.ignore:
                continue
            now.add(other)
        new = now - self.active
        if new:
            self.count += len(new)
            for g in new:
                self.events.append((round(t, 2), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"))
        self.active = now


def run(sc: Scenario, driver, policy_dir=STUDENT_V1, sim=None, brain=None, video=None, verbose=True):
    t0 = time.time()
    if sim is None:
        sim = HomeSim(spawn=sc.spawn, policy_dir=policy_dir, seed=sc.seed, rgbd_hz=sc.rgbd_hz,
                      odom_noise=sc.odom_noise)
    else:
        sim.reset(sc.spawn)
    if sc.setup is not None:
        sc.setup(sim)
        mujoco.mj_forward(sim.m, sim.d)
    brain = brain or Brain()
    coll = CollisionMonitor(sim.m)
    rec = VideoRecorder(sim, video) if video else None
    cmd = patch = None
    path, last, log = 0.0, sim.truth().xy, []
    falls, success, reason = 0, False, "timeout"
    driver.start(sim, brain)
    last_msg = None
    while sim.t < sc.timeout:
        ev = sim.step(cmd, patch)
        brain.on_state(ev.state)
        if ev.depth is not None:
            brain.on_depth(ev.depth)
        if ev.scan is not None:
            brain.on_scan(ev.scan)
        if ev.rgbd is not None and hasattr(driver, "on_rgbd"):
            driver.on_rgbd(ev.rgbd)
        coll.update(sim.d, sim.t)
        xy = sim.truth().xy
        path += float(np.linalg.norm(xy - last))
        last = xy
        if ev.fallen:
            falls, reason = 1, "fell"
            break
        pilot = driver.step(sim.t, brain, ev)
        cmd, patch = brain.step(sim.t, pilot)
        msg = getattr(driver, "message", "") or brain.status()["message"]
        if msg != last_msg:
            log.append((round(sim.t, 2), msg))
            if verbose and msg:
                print(f"  t={sim.t:6.2f} {msg}")
            last_msg = msg
        if rec is not None:
            rec.frame(ev, brain, driver)
        done = driver.done()
        if done is not None:
            success, reason = done
            break
    if rec is not None:
        rec.close()
    res = Result(sc.name, success and falls == 0, reason, sim.t, falls, coll.count, path, tuple(np.round(last, 2)),
                 time.time() - t0, log, dict(collisions=coll.events[:20], drift=sim.odom_to_world()))
    if hasattr(driver, "extra"):
        res.extra.update(driver.extra())
    return res, sim


class ScriptedDriver:
    """A list of (until_time, speed, yaw_rate, leg_length) segments; ends successfully after the last one."""

    def __init__(self, segments, press_a_on_offer=False):
        self.segments, self.press, self.message = segments, press_a_on_offer, ""

    def start(self, sim, brain):
        self.t = 0.0

    def step(self, t, brain, ev):
        from brain.runtime import PilotInput
        self.t = t
        for until, v, w, L in self.segments:
            if t < until:
                if self.press and brain.status().get("offer"):
                    brain.request_skill()
                return PilotInput(speed=v if ev.state.mode == 2 else 0.0, yaw_rate=w, leg_length=L, source="pilot")
        return PilotInput(source="pilot")

    def done(self):
        return (True, "script finished") if self.t >= self.segments[-1][0] else None


class VideoRecorder:
    """Chase camera + first-person colour view, written with imageio (mp4)."""

    def __init__(self, sim, path, size=(640, 480), fps=25):
        import imageio.v2 as iio
        self.sim, self.fps, self.every = sim, fps, max(1, int(round(1 / (fps * 0.02))))
        w, h = size
        m = sim.m
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, h)
        self.r = mujoco.Renderer(m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.sens.base
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.2, -28.0, 90.0
        self.w = iio.get_writer(str(path), fps=fps, macro_block_size=8)
        self.n = 0

    def frame(self, ev, brain, driver):
        self.n += 1
        if self.n % self.every:
            return
        self.cam.azimuth = math.degrees(self.sim.truth().yaw) + 90.0          # behind the robot, looking forward
        self.r.update_scene(self.sim.d, self.cam)
        img = self.r.render().copy()
        overlay = getattr(driver, "overlay", None)
        if overlay is not None:
            img = overlay(img)
        self.w.append_data(img)

    def close(self):
        self.w.close()
        self.r.close()
