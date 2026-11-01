"""Compara o controlador Warp (GPU) com o controlador numpy de control/ no MuJoCo de CPU.

1. Malha aberta: a cada passo do MuJoCo (dirigido pelo numpy), os dois controladores recebem o mesmo
   estado (mj_forward antes, para o sensordata do IMU do numpy nao ficar um passo atrasado em relacao
   ao qpos que o Warp le); compara os torques. Rodas devem bater em < 1%. Quadris em < 3%: o comprimento
   da perna analitico difere ~0.03 mm do simulado (o laco fechado do MuJoCo e levemente elastico) e o
   integrador do PID de comprimento assenta num valor um pouco diferente (~0.1 N.m).
2. Malha fechada: o controlador Warp dirige o MuJoCo de CPU no ciclo estacionado -> levanta -> anda ->
   altura maxima/minima -> pousa; confere os modos e que nao cai.

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
    """Controlador Warp com nworld=1, lendo/escrevendo um MjData de CPU."""

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
    print("malha aberta: RMS do torque [N.m na junta] (referencia numpy | diferenca warp-numpy)")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(6)]
    for n, a, b in zip(names, rms_m, rms_d):
        print(f"  {n:10s} {a:7.3f} | {b:7.4f}")
    print(f"  passos com modo diferente: {mode_mismatch} de {len(diffs)}")
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
    print("malha fechada (warp dirige o MuJoCo de CPU):", transitions)
    print(f"  pior |pitch| equilibrando {math.degrees(worst):.1f} graus; caiu: {fell}")
    return transitions, fell


if __name__ == "__main__":
    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    rel_hip, rel_wheel, mism = open_loop(device)
    trans, fell = closed_loop(device)
    modes = [m for _, m in trans]
    ok = (rel_hip < 0.03 and rel_wheel < 0.01 and mism < 50 and not fell
          and modes == ["parked", "standing_up", "operating", "landing", "parked"])
    print(f"\npior diferenca relativa: quadris {rel_hip:.2%}, rodas {rel_wheel:.2%}  ->  {'PASSOU' if ok else 'FALHOU'}")
    sys.exit(0 if ok else 1)
