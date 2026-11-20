"""Runtime automation of the CMA house scene, called once per control step by the viewers / scripts.

Inclined lift ("plano inclinado", photo 6): glass cabin on a 39 deg rail between the upper-level terrace
(z = 3.6) and the parking (z = 13.2), driven by an integrated-velocity servo, with sliding glass gates at both ends.

    * the robot drives into the cabin and stops (< 5 cm/s for 2 s) -> the gate closes and the cabin travels to
      the other station, where the other gate opens;
    * approaching the station where the cabin is not -> the cabin is called;
    * after the robot has left, the cabin returns to the lower station so the rail pit is covered.
    * `call("up" | "down")` forces a trip (keyboard / gamepad shortcut in the viewers).

Usage:
    rt = HouseRuntime(mujoco, model); rt.reset(data)
    each step: rt.step(data)              (after the controller, before mj_step)
"""
import math

import numpy as np

GATE_OPEN = 1.45
SPEED = 0.6                 # [m/s] cabin speed along the rail
DWELL = 2.0                 # [s] robot standing still inside before departing
RETURN_AFTER = 8.0          # [s] without the robot nearby before the cabin returns to the lower station


class HouseRuntime:
    def __init__(self, mujoco, model, robot_body="base"):
        self.mj = mujoco
        m = model
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.a_lift, self.a_up, self.a_down = aid("lift"), aid("lift_gate_up"), aid("lift_gate_down")
        self.enabled = self.a_lift >= 0
        if not self.enabled:
            return
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "lift_cabin_j")
        self.qadr, self.dadr = m.jnt_qposadr[j], m.jnt_dofadr[j]
        self.travel = float(m.jnt_range[j][1])
        self.axis = m.jnt_axis[j].copy()
        self.cabin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lift_cabin")
        self.robot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        self.dt = m.opt.timestep
        self.actadr = int(m.actuator_actadr[self.a_lift])
        self.reset(None)

    # ------------------------------------------------------------------
    def reset(self, data):
        self.state = "bottom"
        self.timer = 0.0
        self.away = 0.0
        self.pending = None
        self.armed = False          # departs only after the robot has been outside the cabin since the last arrival
        if data is not None and self.enabled:
            s = float(data.qpos[self.qadr])
            self.state = "top" if s > self.travel / 2 else "bottom"
            if self.actadr >= 0:
                data.act[self.actadr] = s

    def call(self, where):
        """Manual request: 'up' or 'down'."""
        if self.enabled:
            self.pending = where

    # ------------------------------------------------------------------
    def _robot_in_cabin(self, data):
        if self.robot < 0:
            return False, 0.0
        p = data.xpos[self.robot] - data.xpos[self.cabin]
        v = float(np.linalg.norm(data.qvel[0:2] - data.qvel[self.dadr] * self.axis[:2]))
        inside = abs(p[0]) < 0.8 and abs(p[1]) < 0.62 and -0.1 < p[2] < 1.5
        return inside, v

    def _near_station(self, data, top):
        if self.robot < 0:
            return False
        p = data.xpos[self.robot]
        c0 = data.xpos[self.cabin] - self.axis * float(data.qpos[self.qadr])          # lower station floor
        c = c0 + self.axis * (self.travel if top else 0.0)
        d = p - c
        return abs(d[2]) < 1.2 and math.hypot(d[0], d[1]) < 3.0

    def step(self, data):
        if not self.enabled:
            return
        dt = self.dt
        s = float(data.qpos[self.qadr])
        inside, v = self._robot_in_cabin(data)
        ctrl = data.ctrl
        st = self.state
        if st in ("bottom", "top"):
            ctrl[self.a_lift] = 0.0
            at_top = st == "top"
            ctrl[self.a_up] = GATE_OPEN if at_top else 0.0
            ctrl[self.a_down] = 0.0 if at_top else GATE_OPEN
            go = None
            if not inside:
                self.armed = True
            if inside and v < 0.05 and self.armed:
                self.timer += dt
                if self.timer > DWELL:
                    go = "down" if at_top else "up"
            else:
                self.timer = 0.0
            if not inside and self._near_station(data, top=not at_top):
                go = "down" if at_top else "up"                     # called from the other station
            if at_top and not inside and not self._near_station(data, top=True):
                self.away += dt
                if self.away > RETURN_AFTER:
                    go = "down"
            else:
                self.away = 0.0
            if self.pending:
                go, self.pending = self.pending, None
                if (go == "up") == at_top:
                    go = None
            if go:
                self.state = "closing_" + go
                self.timer = 0.0
        elif st.startswith("closing_"):
            ctrl[self.a_up] = 0.0
            ctrl[self.a_down] = 0.0
            self.timer += dt
            if self.timer > 1.2:
                self.state = "moving_" + st.split("_")[1]
        elif st.startswith("moving_"):
            up = st.endswith("up")
            ctrl[self.a_lift] = SPEED if up else -SPEED
            target = self.travel if up else 0.0
            if abs(s - target) < 0.005 and abs(float(data.qvel[self.dadr])) < 0.02:
                self.state = "top" if up else "bottom"
                self.timer = 0.0
                self.away = 0.0
                self.armed = False
        self.position = s

    def status(self):
        if not self.enabled:
            return ""
        names = {"bottom": "lift at terrace", "top": "lift at parking", "closing_up": "lift: closing gate",
                 "closing_down": "lift: closing gate", "moving_up": "lift going up", "moving_down": "lift going down"}
        return names.get(self.state, self.state)
