"""Runs an exported stand-up policy (sim/warp_rl/recovery_env.py) when the robot is down: 50 Hz policy, 1 kHz PD
on the hips (the balance controller is off). Ends at the parked pose; RepoControlPolicy.reset() then finds the
gear down and the legs at 0.18 m (PARKED) and the normal stand-up follows.

Observation (must match recovery_env._compute_obs), 3 frames, oldest first: gravity in the base frame (3), gyro (3),
leg lengths and leg angles relative to the body (4), hip speeds x 0.1 (4: Yp_xp, Ym_xp, Yp_xm, Ym_xm), wheel
speeds x 0.05 (2), gear up (1), settling (1, always 0 here), last action (7).
Action -> targets: leg length 0.30 + 0.15 a, leg angle 1.4 a (both hips of the leg), wheel torque 4 a, gear down if > 0.
"""
import json
from pathlib import Path

import numpy as np

import five_bar

KP, KD = 60.0, 1.5


class RecoveryPolicy:
    def __init__(self, exported_dir):
        import torch
        d = Path(exported_dir)
        self.meta = json.loads((d / "policy.json").read_text())
        self.net = torch.jit.load(str(d / "policy.pt"), map_location="cpu").eval()
        self.torch = torch
        Ls = np.linspace(0.17, 0.54, 75)
        self.tab_L, self.tab_a = Ls, np.array([five_bar.symmetric_angle_for_length("xp", L) for L in Ls])
        self.reset()

    def reset(self):
        self.hist, self.action = None, np.zeros(7, np.float32)
        self.targets = None

    def act(self, R, gyro, legs, hip_speeds, wheel_speeds, gear_up):
        """legs: (L_xp, L_xm, angle_xp, angle_xm); hip_speeds (Yp_xp, Ym_xp, Yp_xm, Ym_xm). -> joint targets"""
        f = np.concatenate([-R[2, :], gyro, legs, 0.1 * np.asarray(hip_speeds), 0.05 * np.asarray(wheel_speeds),
                            [1.0 if gear_up else 0.0], [0.0], self.action]).astype(np.float32)
        if self.hist is None:
            self.hist = np.zeros((3, len(f)), np.float32)
        self.hist = np.concatenate([self.hist[1:], f[None]], 0)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(self.hist.ravel())[None])[0].numpy()
        self.action = a = np.clip(a, -1.0, 1.0).astype(np.float32)
        L = 0.30 + 0.15 * a[0:2]
        half = np.interp(L, self.tab_L, self.tab_a)
        phi = 1.4 * a[2:4]
        self.targets = dict(hips=np.array([half[0] + phi[0], -half[0] + phi[0], half[1] + phi[1], -half[1] + phi[1]]),
                            wheels=4.0 * a[4:6], gear_down=bool(a[6] > 0.0))
        return self.targets

    @staticmethod
    def pd(targets, hip_q, hip_qd):
        """1 kHz: hip torques [N.m at the joint] toward the targets (clamp to the drive limit outside)."""
        return KP * (targets["hips"] - np.asarray(hip_q)) - KD * np.asarray(hip_qd)
