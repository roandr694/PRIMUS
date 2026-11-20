"""What the Raspberry Pi 5 runs at 50 Hz on top of the 1 kHz controller: the safety supervisor and the RL policy.

    Command (Jetson, link.py) ─┐
    gamepad (paired to the Pi) ─┼─► supervisor ─► RobotCommand ─► student policy residual ─► RepoControlPolicy
    ElevationPatch (Jetson) ────┘                                  (height / ceiling scans from the patch)

Rules (docs/home_autonomy_architecture.md §2):
  * the pilot always wins: a stick above the dead band, or B, takes over at once;
  * Jetson command older than 150 ms -> speed target 0 (the controller ramps down and holds); older than 2 s -> park;
  * elevation patch older than 300 ms -> the policy sees unknown ground (known = 0) and the speed is capped at 0.3 m/s;
  * a fall (pitch or leg angle beyond 60 deg, as the controller's own torque cut) is latched until stand-up.
The simulator (sim/home_sim) calls `tick` with the same measurements the robot's state estimator provides.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import link  # noqa: E402
from repo_control_policy import RobotCommand  # noqa: E402

CMD_STALE, CMD_PARK, PATCH_STALE = 0.15, 2.0, 0.30
PAD_DEADBAND, PAD_MAX_SPEED, PAD_MAX_YAW = 0.12, 1.2, 1.5


class PiRuntime:
    def __init__(self, policy_dir=None):
        self.policy = None
        if policy_dir:
            from skill_policy import SkillPolicy
            self.policy = SkillPolicy(str(policy_dir))
        self.reset()

    def reset(self):
        self.cmd, self.patch, self.fallen, self.faults = None, None, False, 0
        self.source = "none"
        self.last_rc = RobotCommand(park=False)
        if self.policy is not None:
            self.policy.reset()

    # -------------------------------------------------------------------------------- inputs from the Jetson
    def on_command(self, cmd: link.Command):
        self.cmd = cmd

    def on_patch(self, patch: link.ElevationPatch):
        self.patch = patch

    # -------------------------------------------------------------------------------- 50 Hz
    def tick(self, now, out, meas, pad=None):
        """out: the controller's last output; meas: dict(ctl, R, gyro, wheel_speeds, base_z, xy, heading,
        contact_z) in the odom frame; pad: dict(speed, yaw, buttons) in [-1, 1] or None. -> RobotCommand"""
        self.faults = 0
        if abs(out["pitch"]) > math.pi / 3 or abs(out["theta"]) > math.pi / 3:
            self.fallen = True
        if self.fallen:
            self.faults |= link.FAULTS["fall"]
            self.source = "safety"
            self.last_rc = RobotCommand(brake=True)
            return self.last_rc

        pad = pad or {}
        pilot = abs(pad.get("speed", 0.0)) > PAD_DEADBAND or abs(pad.get("yaw", 0.0)) > PAD_DEADBAND \
            or pad.get("buttons", 0) & link.BUTTONS["b"]
        c = self.cmd
        age = now - c.stamp if c is not None else math.inf
        skill = False
        if pilot:
            self.source = "pilot"
            rc = RobotCommand(speed_cmd=pad.get("speed", 0.0) * PAD_MAX_SPEED, yaw_speed_cmd=pad.get("yaw", 0.0) * PAD_MAX_YAW,
                              leg_length=self.last_rc.leg_length, brake=bool(pad.get("buttons", 0) & link.BUTTONS["b"]))
        elif age > CMD_PARK:
            self.faults |= link.FAULTS["stale_command"]
            self.source = "safety"
            rc = RobotCommand(park=c is not None)          # never commanded yet: stay as we are (parked at boot)
            if c is None:
                rc = RobotCommand(park=True)
        elif age > CMD_STALE:
            self.faults |= link.FAULTS["stale_command"]
            self.source = "safety"
            rc = replace(self.last_rc, speed_cmd=0.0, yaw_speed_cmd=0.0, residual=None)
        else:
            self.source = link.SOURCES[c.source] if 0 <= c.source < len(link.SOURCES) else "none"
            rc = RobotCommand(speed_cmd=c.speed, yaw_speed_cmd=c.yaw_rate, leg_length=c.leg_length,
                              pitch_ref=c.pitch_ref, brake=c.brake, park=c.park, max_accel=c.max_accel)
            skill = c.skill

        patch_ok = self.patch is not None and now - self.patch.stamp <= PATCH_STALE
        if not patch_ok:
            self.faults |= link.FAULTS["stale_patch"]
            rc = replace(rc, speed_cmd=float(np.clip(rc.speed_cmd, -0.3, 0.3)))
        if self.policy is not None and out["mode"] == "operating":
            p = self.patch if patch_ok else None
            gfn = (lambda xy: p.ground_at(xy)) if p is not None else (lambda xy: np.full(len(xy), np.nan))
            cfn = (lambda xy: p.ceiling_at(xy)) if p is not None else (lambda xy: np.full(len(xy), np.inf))
            kw = dict(skill=skill) if self.policy.always_on else {}
            priv = dict(v_body=np.zeros(3), base_height=0.4, torque_limit=20.0, friction=1.0, mass_scale=1.0,
                        grounded=out["touching_ground"])           # the student does not read them
            res = self.policy.act(meas["ctl"], meas["R"], meas["gyro"], meas["wheel_speeds"], rc, meas["base_z"],
                                  gfn, cfn, meas["xy"], meas["heading"], priv, contact_z=meas["contact_z"], **kw)
            rc = replace(rc, residual=res)
        self.skill = skill
        self.last_rc = rc
        return rc

    def state(self, now, out, odom, battery=1.0, pad=None):
        """-> link.State for the Jetson. odom: dict(x, y, z, yaw, speed, yaw_rate)."""
        pad = pad or {}
        mode = {"parked": 0, "standing_up": 1, "operating": 2, "landing": 3}.get(str(out["mode"]), 2)
        if self.fallen:
            mode = 4
        L = 0.5 * (out["left_leg_length"] + out["right_leg_length"])
        return link.State(stamp=now, x=odom["x"], y=odom["y"], z=odom["z"], yaw=odom["yaw"], pitch=out["pitch"],
                          roll=out["roll"], speed=odom["speed"], yaw_rate=odom["yaw_rate"], leg_length=L, mode=mode,
                          gear_down=bool(out["gear_down"]), contact=bool(out["touching_ground"]), battery=battery,
                          skill_active=bool(getattr(self, "skill", False)), faults=self.faults,
                          pad_speed=pad.get("speed", 0.0), pad_yaw=pad.get("yaw", 0.0), pad_buttons=pad.get("buttons", 0))
