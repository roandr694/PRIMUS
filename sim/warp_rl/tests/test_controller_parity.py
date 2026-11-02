"""Compares the Warp (GPU) controller with the numpy controller from control/ on CPU MuJoCo.

1. Open loop: at every MuJoCo step (driven by numpy) both controllers get the same state (mj_forward first,
   so the numpy IMU sensordata is not one step behind the qpos Warp reads) and the torques are compared.
   Wheels must match within 1%. Hips within 3%: the analytic leg length differs ~0.03 mm from the simulated
   one (MuJoCo's loop closure is slightly compliant) and the leg-length PI integral settles slightly
   differently (~0.1 N.m).
2. Closed loop: the Warp controller drives CPU MuJoCo through parked -> stand up -> drive -> max/min height
   -> land; checks the modes and that it does not fall.

    ~/test/isaac-rl/bin/python sim/warp_rl/tests/test_controller_parity.py
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import controller as C  # noqa: E402
import robot_model as rm  # noqa: E402
from repo_control_policy import RepoControlPolicy, RobotCommand  # noqa: E402

PLAN = [(0.0, dict(park=True)), (1.0, {}), (5.0, dict(speed_cmd=0.6)), (8.0, dict(leg_length=0.52)),
        (11.0, dict(leg_length=0.52, speed_cmd=0.5, yaw_speed_cmd=1.0)), (14.0, dict(leg_length=0.18)),
        (17.0, dict(leg_length=0.18, speed_cmd=-0.5)), (19.0, dict(leg_length=0.40)), (21.0, dict(park=True))]
T_END = 26.0


def command_at(t):
    kw = [c for ts, c in PLAN if t >= ts][-1]
    return RobotCommand(**kw)


class WarpOnCpuData:
    """Warp controller with nworld=1, reading/writing a CPU MjData."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.ctl = C.BatchedController(model, 1, device)
        z = lambda n: wp.zeros((1, n), dtype=float, device=device)
        self.qpos, self.qvel, self.sd, self.qfrc, self.ctrl = z(model.nq), z(model.nv), z(model.nsensordata), z(model.nv), z(model.nu)

    def _push(self, data):
        for arr, src in ((self.qpos, data.qpos), (self.qvel, data.qvel), (self.sd, data.sensordata), (self.qfrc, data.qfrc_actuator)):
            arr.assign(np.asarray(src, dtype=np.float32)[None])

    def reset(self, data):
        self._push(data)
        self.ctl.mask.fill_(1)
        wp.launch(C.reset_kernel, dim=1, inputs=[self.ctl.prm, self.ctl.mask, self.qpos, self.ctl.cs], device=self.device)

    def step(self, data, cmd):
        self._push(data)
        self.ctl.cmd.assign(np.array([[cmd.speed_cmd, cmd.yaw_speed_cmd, cmd.leg_length, cmd.roll_angle, float(cmd.park)]], np.float32))
        wp.launch(C.control_kernel, dim=1, inputs=[self.ctl.prm, self.ctl.coeffs, self.qpos, self.qvel, self.sd, self.qfrc,
                                                   self.ctl.cmd, self.ctl.res, self.ctl.cs, self.ctl.out, self.ctrl], device=self.device)
        return self.ctrl.numpy()[0].astype(np.float64), int(self.ctl.cs.numpy()[0, C.S["mode"]])


def open_loop(device):
    _, model, data, _ = rm.load(flat=True)
    ref = RepoControlPolicy(mujoco, model)
    ref.reset(data)
    wctl = WarpOnCpuData(model, device)
    wctl.reset(data)
    diffs, mags, mode_mismatch = [], [], 0
    for k in range(int(T_END / model.opt.timestep)):
        cmd = command_at(k * model.opt.timestep)
        mujoco.mj_forward(model, data)
        u_w, mode_w = wctl.step(data, cmd)
        ref.apply(data, cmd)
        u_r = data.ctrl.copy()
        mode_mismatch += C.MODE_NAMES[mode_w] != ref.mode
        diffs.append((u_w - u_r) * 9.0)
        mags.append(u_r * 9.0)
        mujoco.mj_step(model, data)
    diffs, mags = np.array(diffs)[:, :6], np.array(mags)[:, :6]
    rms_d = np.sqrt((diffs ** 2).mean(0)); rms_m = np.sqrt((mags ** 2).mean(0))
    print("open loop: torque RMS [N.m at the joint] (numpy reference | warp - numpy)")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(6)]
    for n, a, b in zip(names, rms_m, rms_d):
        print(f"  {n:10s} {a:7.3f} | {b:7.4f}")
    print(f"  steps with a different mode: {mode_mismatch} of {len(diffs)}")
    rel = rms_d / np.maximum(rms_m, 0.5)
    return float(rel[:4].max()), float(rel[4:].max()), mode_mismatch


def closed_loop(device):
    _, model, data, _ = rm.load(flat=True)
    wctl = WarpOnCpuData(model, device)
    wctl.reset(data)
    last, worst, fell = None, 0.0, False
    transitions = []
    for k in range(int(T_END / model.opt.timestep)):
        t = k * model.opt.timestep
        u, mode = wctl.step(data, command_at(t))
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        R = data.xmat[1].reshape(3, 3)
        pitch = math.asin(max(-1.0, min(1.0, R[2, 1])))
        if mode != C.PARKED:
            worst = max(worst, abs(pitch))
        if mode != last:
            transitions.append((round(t, 2), C.MODE_NAMES[mode]))
            last = mode
        if abs(pitch) > 1.0 or data.qpos[2] < 0.15:
            fell = True
            break
    print("closed loop (warp drives CPU MuJoCo):", transitions)
    print(f"  worst |pitch| while balancing {math.degrees(worst):.1f} deg; fell: {fell}")
    return transitions, fell


if __name__ == "__main__":
    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    rel_hip, rel_wheel, mism = open_loop(device)
    trans, fell = closed_loop(device)
    modes = [m for _, m in trans]
    ok = (rel_hip < 0.03 and rel_wheel < 0.01 and mism < 50 and not fell
          and modes == ["parked", "standing_up", "operating", "landing", "parked"])
    print(f"\nworst relative difference: hips {rel_hip:.2%}, wheels {rel_wheel:.2%}  ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
