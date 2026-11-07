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
        (11.0, dict(leg_length=0.52, speed_cmd=1.5, yaw_speed_cmd=1.8)), (14.0, dict(leg_length=0.18)),
        (17.0, dict(leg_length=0.18, speed_cmd=-0.5)), (19.0, dict(leg_length=0.40)),
        # policy action (dL, dth, dT, dF per leg): crouch + lean + one leg ahead, then a leg-force push
        (20.0, dict(leg_length=0.40, speed_cmd=0.3, residual=(-0.06, -0.10, 0.05, -0.05, 0.5, -0.5, 0.0, 0.0, 0.0))),
        (21.0, dict(leg_length=0.30, residual=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0, 0.0))),
        (22.0, dict(leg_length=0.30, residual=(0.0,) * 8 + (1.0,))),           # jump primitive
        (22.05, dict(leg_length=0.30)),
        (24.0, dict(park=True))]
T_END = 29.0


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
        self.ctl.cmd.assign(np.array([[cmd.speed_cmd, cmd.yaw_speed_cmd, cmd.leg_length, cmd.roll_angle, float(cmd.park),
                                       cmd.max_accel]], np.float32))
        self.ctl.res.assign(np.array([cmd.residual or (0.0,) * C.NRES], np.float32))
        wp.launch(C.control_kernel, dim=1, inputs=[self.ctl.prm, self.ctl.coeffs, self.qpos, self.qvel, self.sd, self.qfrc,
                                                   self.ctl.cmd, self.ctl.res, self.ctl.drv, self.ctl.cs, self.ctl.out, self.ctrl], device=self.device)
        cs = self.ctl.cs.numpy()[0]
        return self.ctrl.numpy()[0].astype(np.float64), int(cs[C.S["mode"]]), int(cs[C.S["jump_phase"]])


def open_loop(device):
    _, model, data, _ = rm.load(flat=True)
    ref = RepoControlPolicy(mujoco, model)
    ref.reset(data)
    wctl = WarpOnCpuData(model, device)
    wctl.reset(data)
    diffs, mags, mode_mismatch, jump_mismatch, jumping = [], [], 0, 0, []
    for k in range(int(T_END / model.opt.timestep)):
        cmd = command_at(k * model.opt.timestep)
        mujoco.mj_forward(model, data)
        u_w, mode_w, jump_w = wctl.step(data, cmd)
        out = ref.apply(data, cmd)
        u_r = data.ctrl.copy()
        mode_mismatch += C.MODE_NAMES[mode_w] != ref.mode
        jump_mismatch += jump_w != out["jump_phase"]
        jumping.append(jump_w or out["jump_phase"])
        diffs.append((u_w - u_r) * 9.0)
        mags.append(u_r * 9.0)
        mujoco.mj_step(model, data)
    # a jump saturates the torques: a phase switch one step apart is a large but brief difference, so the
    # torque comparison excludes the jump window (+0.3 s of landing); the phases are compared separately
    jw = np.convolve(np.array(jumping, float), np.ones(300), mode="full")[:len(jumping)] > 0
    print(f"  jump: {int(np.sum(np.array(jumping) > 0))} steps jumping, phase differs on {jump_mismatch}")
    diffs, mags = np.array(diffs)[~jw, :6], np.array(mags)[~jw, :6]
    rms_d = np.sqrt((diffs ** 2).mean(0)); rms_m = np.sqrt((mags ** 2).mean(0))
    print("open loop: torque RMS [N.m at the joint] (numpy reference | warp - numpy)")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(6)]
    for n, a, b in zip(names, rms_m, rms_d):
        print(f"  {n:10s} {a:7.3f} | {b:7.4f}")
    print(f"  steps with a different mode: {mode_mismatch} of {len(diffs)}")
    rel = rms_d / np.maximum(rms_m, 0.5)
    return float(rel[:4].max()), float(rel[4:].max()), mode_mismatch, jump_mismatch


def closed_loop(device):
    _, model, data, _ = rm.load(flat=True)
    wctl = WarpOnCpuData(model, device)
    wctl.reset(data)
    last, worst, fell = None, 0.0, False
    transitions = []
    for k in range(int(T_END / model.opt.timestep)):
        t = k * model.opt.timestep
        u, mode, _ = wctl.step(data, command_at(t))
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
    rel_hip, rel_wheel, mism, jmism = open_loop(device)
    trans, fell = closed_loop(device)
    modes = [m for _, m in trans]
    ok = (rel_hip < 0.03 and rel_wheel < 0.01 and mism < 50 and jmism < 10 and not fell
          and modes == ["parked", "standing_up", "operating", "landing", "parked"])
    print(f"\nworst relative difference: hips {rel_hip:.2%}, wheels {rel_wheel:.2%}  ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
