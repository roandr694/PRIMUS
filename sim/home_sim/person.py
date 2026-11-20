"""A person walking in the house (sim only), for "me segue" / "vem aqui" and people-aware navigation.

A mocap body (legs, torso, arms, head; clothes and skin colours) in geom group 0, so the D435i renders it and the
lidar sees it; it does not collide (contype 0) so it cannot be knocked over or push the robot. It walks a list of
waypoints at a human pace along A* paths on the ground-truth grid, with small sway, stopping at each waypoint.

    sim = HomeSim(..., person=True); walker = Walker(sim, [(x, y), ...])
    each sim step: walker.step(sim.t)
"""
import math

import numpy as np
import mujoco


def add_person(spec, name="person", start=(0.0, 0.0, 0.0)):
    b = spec.worldbody.add_body(name=name, pos=list(start), mocap=True)
    jeans, shirt, skin, hair, shoe = (0.18, 0.25, 0.45, 1), (0.75, 0.15, 0.12, 1), (0.85, 0.66, 0.52, 1), \
        (0.12, 0.08, 0.05, 1), (0.1, 0.1, 0.1, 1)

    def geom(gtype, size, pos, rgba, fromto=None):
        kw = dict(type=gtype, size=size, rgba=rgba, contype=0, conaffinity=0, group=0)
        if fromto is not None:
            kw["fromto"] = fromto
        else:
            kw["pos"] = pos
        b.add_geom(**kw)
    C, S, B, E = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX, \
        mujoco.mjtGeom.mjGEOM_ELLIPSOID
    for sx in (-1, 1):
        geom(C, [0.065, 0, 0], None, jeans, fromto=[sx * 0.1, 0, 0.1, sx * 0.1, 0, 0.88])
        geom(B, [0.05, 0.12, 0.04], [sx * 0.1, 0.04, 0.04], shoe)
        geom(C, [0.045, 0, 0], None, shirt, fromto=[sx * 0.23, 0, 1.45, sx * 0.26, 0.02, 1.1])
        geom(C, [0.04, 0, 0], None, skin, fromto=[sx * 0.26, 0.02, 1.1, sx * 0.27, 0.06, 0.82])
    geom(E, [0.2, 0.12, 0.3], [0, 0, 1.2], shirt)
    geom(B, [0.17, 0.1, 0.08], [0, 0, 0.92], jeans)
    geom(C, [0.05, 0, 0], None, skin, fromto=[0, 0, 1.47, 0, 0, 1.56])
    geom(E, [0.095, 0.11, 0.125], [0, 0, 1.66], skin)
    geom(E, [0.1, 0.115, 0.07], [0, -0.01, 1.74], hair)
    return b


class Walker:
    def __init__(self, sim, waypoints, speed=0.8, pause=2.0, start_delay=3.0, floor="social"):
        import json
        import gt_grid
        from world import GT_PATH
        from tour import cost_from_free
        sys_path_brain()
        from brain.nav.astar import astar, simplify
        self.sim, self.speed, self.pause, self.start_delay = sim, speed, pause, start_delay
        self.m, self.d = sim.m, sim.d
        self.mocap = self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "person")]
        self.grid = gt_grid.cached(self.m, self.d, floor, GT_PATH.parent)
        cost = cost_from_free(self.grid.free, self.grid.res, robot_r=0.25, soft=0.5)
        self.path = [np.asarray(waypoints[0], float)]
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            p = astar(cost, self.grid.world_to_ij(np.asarray(a)), self.grid.world_to_ij(np.asarray(b)))
            if p is None:
                pts = [np.asarray(b, float)]
            else:
                p = simplify(p, cost)
                pts = list(self.grid.ij_to_world(np.array([q[0] for q in p]), np.array([q[1] for q in p]))[1:])
            self.path += pts + [None]                          # None = pause at a waypoint
        self.k, self.pos, self.t_pause = 1, self.path[0].copy(), None
        self.heading = 0.0
        self.z = self.grid.floor_z
        self.done = False
        self._place()

    def _place(self):
        self.d.mocap_pos[self.mocap] = [self.pos[0], self.pos[1], self.z]
        a = self.heading - math.pi / 2                         # the body faces its local +Y
        self.d.mocap_quat[self.mocap] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]

    def xy(self):
        return self.pos.copy()

    def step(self, t, dt=0.02):
        if t < self.start_delay or self.done:
            self._place()
            return
        while self.k < len(self.path) and self.path[self.k] is None:
            if self.t_pause is None:
                self.t_pause = t
            if t - self.t_pause < self.pause:
                self._place()
                return
            self.t_pause = None
            self.k += 1
        if self.k >= len(self.path):
            self.done = True
            return
        tgt = self.path[self.k]
        d = tgt - self.pos
        n = float(np.linalg.norm(d))
        step = self.speed * dt
        if n <= step:
            self.pos = tgt.copy()
            self.k += 1
        else:
            self.pos = self.pos + d / n * step
            self.heading = math.atan2(d[1], d[0])
        self._place()


def sys_path_brain():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parents[2])
    if p not in sys.path:
        sys.path.insert(0, p)
