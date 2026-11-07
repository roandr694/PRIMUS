"""Pilot inputs -> RobotCommand, shared by every input device (Xbox gamepad, Meta Quest, keyboard).

Each device fills a dict of logical inputs:
    speed, yaw   sticks in [-1, 1]; + = forward / turn right
    up, down     [0, 1] (triggers, D-pad, keys): leg length rate; the height latches when released
    brake        hard brake (snap the speed target to 0)      stop   soft stop (zero speed and turn)
    park         toggles land <-> stand up (on press)          reset  restart the simulation (on press)
    faster, slower   responsiveness level up / down (on press); or `level` = 0, 1, 2 to pick one directly
"""
import math

import _paths  # noqa: F401  (control/ on sys.path)
from repo_control_policy import MAX_LEG_LENGTH, MIN_LEG_LENGTH, NOMINAL_LEG_LENGTH, RobotCommand


# responsiveness: how fast the speed target ramps to the stick value (0 -> 1.5 m/s in ~2.7 / 1.2 / 0.7 s)
LEVELS = (("smooth", 0.5), ("normal", 1.5), ("sport", 5.0))   # (name, max acceleration [m/s^2])


class Pilot:
    def __init__(self, max_speed=1.5, max_yaw_rate=1.8, height_rate=0.12, deadzone=0.12):
        self.max_speed, self.max_yaw_rate = max_speed, max_yaw_rate
        self.height_rate, self.deadzone = height_rate, deadzone
        self.leg_length, self.park, self.level = NOMINAL_LEG_LENGTH, False, 1
        self._pressed = {}

    def _dz(self, v):
        v = float(v or 0.0)
        return 0.0 if abs(v) < self.deadzone else math.copysign((abs(v) - self.deadzone) / (1 - self.deadzone), v)

    def _press(self, inp, key):
        now = bool(inp.get(key))
        hit = now and not self._pressed.get(key, False)
        self._pressed[key] = now
        return hit

    def command(self, inp, dt, stale=False):
        """-> (RobotCommand, reset requested). `stale`: no input for a while -> stop, keep height and park."""
        accel = LEVELS[self.level][1]
        if stale:
            return RobotCommand(leg_length=self.leg_length, park=self.park, max_accel=accel), False
        rate = min(1.0, max(-1.0, float(inp.get("up", 0)) - float(inp.get("down", 0))))
        self.leg_length = min(MAX_LEG_LENGTH, max(MIN_LEG_LENGTH, self.leg_length + rate * self.height_rate * dt))
        if self._press(inp, "park"):
            self.park = not self.park
        if inp.get("level") is not None:
            self.level = min(len(LEVELS) - 1, max(0, int(inp["level"])))
        self.level = min(len(LEVELS) - 1, self.level + self._press(inp, "faster")) - self._press(inp, "slower")
        self.level = max(0, self.level)
        accel = LEVELS[self.level][1]
        reset = self._press(inp, "reset")
        if reset:
            self.leg_length, self.park = NOMINAL_LEG_LENGTH, False
        brake = bool(inp.get("brake"))
        halt = brake or bool(inp.get("stop"))
        return RobotCommand(speed_cmd=0.0 if halt else self._dz(inp.get("speed")) * self.max_speed,
                            yaw_speed_cmd=0.0 if halt else -self._dz(inp.get("yaw")) * self.max_yaw_rate,
                            leg_length=self.leg_length, brake=brake, park=self.park, max_accel=accel), reset

    @property
    def level_name(self):
        return LEVELS[self.level][0]
