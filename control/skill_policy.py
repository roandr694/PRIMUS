"""Runs an exported skill policy (sim/warp_rl/export_policy.py) on top of RepoControlPolicy, at 50 Hz.

The observation must match sim/warp_rl/skill_env.py (_compute_obs), in this order:
  history (3 frames, oldest first) of: gravity in the base frame (3), gyro (3), leg lengths / angles and their
      rates (8: L_xp, L_xm, th_xp, th_xm, dL.., dth..), wheel speeds x 0.1 (2), odometry speed (1),
      operating (1), commands (speed, yaw rate, leg length) (3), last action (9; 8 for v1 exports),
      jump phase / 2 (1; obs_version >= 2)
  privileged (teacher; from the simulator, or estimates): base velocity in the base frame (3), base height
      above the ground (1), (torque limit - 20) / 5 (1), friction (1), mass scale (1), wheels on the ground (1)
  height scan (155): base z - ground - 0.4, clipped to [-1, 1], at (lateral, forward) points of the heading
      frame: forward -0.30..1.20 every 5 cm (outer loop), lateral -0.30..0.30 every 15 cm (inner loop)
  ceiling scan (45): lowest overhead surface - base z, clipped to [-0.5, 1.0] (open = 1.0), forward
      -0.2..1.2 every 10 cm, lateral -0.2, 0, 0.2
Student (obs_version 3, sim/warp_rl/student_env.py): 10 frames of the same history, then the height scan from the
elevation map (unknown cells = the ground under the wheels), its known mask (155) and the ceiling scan; no
privileged values.
The action (in [-1, 1]) x residual_scale + residual_offset is RobotCommand.residual (9th value: jump trigger;
10th-12th: jump extension, push, tuck length).

v12 (obs_version 4, control/policy_interface.py): one always-on policy. The frame is policy_interface.FRAME_NAMES
(all commands: speed, yaw rate, leg length, body pitch, responsiveness / 5, the ramped speed target, the skill
flag; the last action in the pair basis). `act(..., skill=True)` while a skill executes; otherwise the assist
regime (reduced authority, jump trigger off). The action pairs are turned into the per-side residual by
policy_interface.to_sides. Do not reset the history when a skill starts.
"""
import json
import math
from pathlib import Path

import numpy as np

import policy_interface as PI

SIDES = ("xp", "xm")


def scan_points():
    f = np.arange(-0.30, 1.2001, 0.05)
    lat = np.array([-0.30, -0.15, 0.0, 0.15, 0.30])
    hs = np.stack(np.meshgrid(lat, f, indexing="xy"), -1).reshape(-1, 2)
    f = np.arange(-0.2, 1.2001, 0.1)
    lat = np.array([-0.2, 0.0, 0.2])
    cs = np.stack(np.meshgrid(lat, f, indexing="xy"), -1).reshape(-1, 2)
    return hs, cs


def heading_points(pts, xy, heading):
    """(lateral, forward) points in the heading frame -> world xy"""
    c, s = math.cos(heading), math.sin(heading)
    return np.stack([xy[0] + c * pts[:, 0] - s * pts[:, 1], xy[1] + s * pts[:, 0] + c * pts[:, 1]], -1)


class SkillPolicy:
    def __init__(self, exported_dir):
        import torch
        d = Path(exported_dir)
        self.meta = json.loads((d / "policy.json").read_text())
        self.net = torch.jit.load(str(d / "policy.pt"), map_location="cpu").eval()
        self.torch = torch
        self.scale = np.array(self.meta["residual_scale"], np.float32)
        self.offset = np.array(self.meta.get("residual_offset") or np.zeros(len(self.scale)), np.float32)
        self.history = self.meta["history"]
        self.hscan, self.cscan = scan_points()
        self.reset()

    @property
    def always_on(self):
        return self.meta.get("obs_version", 1) >= 4

    def reset(self):
        self.hist = None
        self.ramp = 0.0
        self.action = np.zeros(len(self.scale), np.float32)
        if hasattr(self.net, "reset"):
            self.net.reset()                      # recurrent student: clear the memory at the start of a skill

    def frame(self, ctl, R, gyro, wheel_speeds, cmd, skill=False, dt=0.02):
        legs = ctl.last_leg
        leg = [legs["xp"].length, legs["xm"].length, legs["xp"].theta, legs["xm"].theta,
               legs["xp"].d_length, legs["xm"].d_length, legs["xp"].d_theta, legs["xm"].d_theta]
        if self.always_on:
            a = PI.ramp_accel(cmd.max_accel) * dt                  # the speed target of the controller's ramp
            self.ramp += max(-a, min(a, cmd.speed_cmd - self.ramp))
            cmds = [cmd.speed_cmd, cmd.yaw_speed_cmd, cmd.leg_length, cmd.pitch_ref, cmd.max_accel / PI.ACCEL_NORM,
                    self.ramp, 1.0 if skill else 0.0]
            return np.concatenate([-R[2, :], gyro, leg, 0.1 * np.asarray(wheel_speeds), [ctl.dx],
                                   [1.0 if ctl.mode == "operating" else 0.0], cmds, self.action,
                                   [0.5 * ctl.jump_phase]]).astype(np.float32)
        return np.concatenate([-R[2, :], gyro, leg, 0.1 * np.asarray(wheel_speeds), [ctl.dx],
                               [1.0 if ctl.mode == "operating" else 0.0],
                               [cmd.speed_cmd, cmd.yaw_speed_cmd, cmd.leg_length], self.action,
                               [0.5 * ctl.jump_phase] if self.meta.get("obs_version", 1) >= 2 else []]).astype(np.float32)

    def act(self, ctl, R, gyro, wheel_speeds, cmd, base_z, ground_fn, ceiling_fn, xy, heading, priv, contact_z=None,
            skill=False):
        """-> residual tuple for RobotCommand. ground_fn / ceiling_fn: world xy [N, 2] -> heights [N] (ground nan
        where unknown, filled with contact_z = the ground under the wheels; ceiling +inf where open).
        priv (teacher only): dict(v_body, base_height, torque_limit, friction, mass_scale, grounded).
        skill (v12): True while a skill executes (the planner's EXECUTE), else the assist regime."""
        f = self.frame(ctl, R, gyro, wheel_speeds, cmd, skill)
        if self.hist is None:
            self.hist = np.zeros((self.history, len(f)), np.float32)
        self.hist = np.concatenate([self.hist[1:], f[None]], 0)
        h = ground_fn(heading_points(self.hscan, xy, heading))
        known = np.isfinite(h)
        fill = contact_z if contact_z is not None else base_z - priv["base_height"]
        scan = np.clip(base_z - np.where(known, h, fill) - 0.4, -1.0, 1.0)
        c = np.minimum(ceiling_fn(heading_points(self.cscan, xy, heading)), 10.0)
        clear = np.clip(c - base_z, -0.5, 1.0)
        if self.meta.get("obs_version", 1) == 3:                    # student: map + mask, no privileged values
            obs = np.concatenate([self.hist.ravel(), scan, known, clear]).astype(np.float32)
        else:
            p = [*priv["v_body"], priv["base_height"], (priv["torque_limit"] - 20.0) / 5.0, priv["friction"],
                 priv["mass_scale"], 1.0 if priv["grounded"] else 0.0]
            obs = np.concatenate([self.hist.ravel(), p, scan, clear]).astype(np.float32)
        assert obs.shape[0] == self.meta["obs_dim"], (obs.shape, self.meta["obs_dim"])
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(obs)[None])[0].numpy()
        self.action = np.clip(a, -1.0, 1.0).astype(np.float32)
        if self.always_on:
            return tuple(float(v) for v in PI.to_sides(self.action, np.float32(1.0 if skill else 0.0), np))
        return tuple(float(v) for v in self.action * self.scale + self.offset)
