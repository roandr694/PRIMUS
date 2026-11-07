"""Stand-up-after-a-fall policy: from lying on the ground (fallen forward, backward or on a side) back to the
parked pose (upright on the landing gear, legs at 0.18 m), where the automatic stand-up of the controller takes
over. Triggered by the pilot (a button) when the robot is down; the balance controller is off meanwhile.

The model-based controller cuts the torques when the robot falls, so here the policy drives the joints directly,
through a PD kernel at 1 kHz (recovery_kernel), in leg space:
    [L_xp, L_xm]      leg length targets  (0.30 +- 0.15 m; the parked 0.18 m is inside the range)
    [phi_xp, phi_xm]  leg angle targets   (+- 1.4 rad: both hips of a leg turn together)
    [T_xp, T_xm]      wheel torques       (+- 4 N.m)
    [gear]            landing gear down (> 0) / up
Flat ground, collision="full" (chassis and legs touch the ground), the motor torque-speed droop, torque limit
18-25 N.m, friction and mass randomised. Each episode starts with the robot dropped in a random fallen pose (tilted
75-180 deg, 30% on a side) and 0.6 s with the motors off, so it settles as it would really lie.
"""
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from tensordict import TensorDict

import controller as C
import model as M
from env import _shared_stream
from rsl_rl.env import VecEnv
from skill_env import W0

NREC = 7


@wp.kernel
def recovery_kernel(prm: C.Params, qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                    tgt: wp.array2d(dtype=float), drv: wp.array2d(dtype=float), on: wp.array(dtype=int),
                    kp: float, kd: float, out: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    """PD on the 4 hips toward tgt[0:4], wheel torques tgt[4:6], gear tgt[6]; out = leg length / angle."""
    w = wp.tid()
    lim = drv[w, 0]
    w0 = drv[w, 1]
    for s in range(2):
        hp = qpos[w, prm.qp_hipYp[s]]
        hm = qpos[w, prm.qp_hipYm[s]]
        v = C.foot(prm, C.knee_p(prm, hp), C.knee_m(prm, hm)) - prm.hip_mid
        out[w, s] = wp.length(v)
        out[w, 2 + s] = wp.atan2(-v[0], -v[1])          # leg angle relative to the body
        t_p = kp * (tgt[w, 2 * s] - hp) - kd * qvel[w, prm.qv_hipYp[s]]
        t_m = kp * (tgt[w, 2 * s + 1] - hm) - kd * qvel[w, prm.qv_hipYm[s]]
        t_w = tgt[w, 4 + s]
        if on[w] == 0:
            t_p = 0.0
            t_m = 0.0
            t_w = 0.0
        ctrl[w, prm.u_hipYp[s]] = C.drive_clamp(t_p, qvel[w, prm.qv_hipYp[s]], lim, w0) / prm.gear_ratio
        ctrl[w, prm.u_hipYm[s]] = C.drive_clamp(t_m, qvel[w, prm.qv_hipYm[s]], lim, w0) / prm.gear_ratio
        ctrl[w, prm.u_wheel[s]] = C.drive_clamp(t_w, qvel[w, prm.qv_wheel[s]], lim, w0) / prm.gear_ratio
    for g in range(2):
        if tgt[w, 6] > 0.0:
            ctrl[w, prm.u_servo[g]] = prm.gear_down_angle[g]
        else:
            ctrl[w, prm.u_servo[g]] = prm.gear_up_angle[g]


@dataclass
class RecoveryCfg:
    num_envs: int = 4096
    decimation: int = 20
    episode_s: float = 8.0
    settle_s: float = 0.6
    history: int = 3
    kp: float = 60.0
    kd: float = 1.5
    torque_range: tuple = (18.0, 25.0)
    friction_range: tuple = (0.5, 1.25)
    base_mass_scale: tuple = (0.85, 1.15)
    p_side: float = 0.3                      # fallen on a side (else forward / backward)
    tilt_range: tuple = (75.0, 180.0)        # [deg] from upright; 180 = upside down
    success_hold_s: float = 0.5
    rew: dict = field(default_factory=lambda: dict(
        upright=2.0, height=1.0, pose=1.0, pose_coarse=1.0, gear=0.5, still=0.5, joint_vel=-2e-3, torque=-2e-4,
        action_rate=-0.05,
        success=20.0))


class RecoveryEnv(VecEnv):
    def __init__(self, cfg: RecoveryCfg = RecoveryCfg(), device="cuda:0", seed=0):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = N = cfg.num_envs
        self.num_actions = NREC
        self.dt = 0.001 * cfg.decimation
        self.max_episode_length = int(cfg.episode_s / self.dt)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.mjm, _ = M.build(None, collision="full")
        self.mjm.actuator_ctrlrange[:6] = [-cfg.torque_range[1] / 9.0, cfg.torque_range[1] / 9.0]
        mjd = mujoco.MjData(self.mjm)
        mjd.qpos[:] = M.parked_qpos()
        mujoco.mj_forward(self.mjm, mjd)
        wp.init()
        self.stream = _shared_stream(str(self.device))
        rng = np.random.default_rng(seed + 99)
        with wp.ScopedStream(self.stream):
            self.mw = mjw.put_model(self.mjm)
            self.dw = mjw.put_data(self.mjm, mjd, nworld=N, nconmax=48, njmax=200)
            m = self.mjm
            fr = np.tile(m.geom_friction[None], (N, 1, 1)).astype(np.float32)
            mu = rng.uniform(*cfg.friction_range, N).astype(np.float32)
            fr[:, :, 0] = mu[:, None]
            self.mw.geom_friction = wp.array(fr, dtype=wp.vec3)
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
            ms = rng.uniform(*cfg.base_mass_scale, N).astype(np.float32)
            mass = np.tile(m.body_mass[None], (N, 1)).astype(np.float32)
            inertia = np.tile(m.body_inertia[None], (N, 1, 1)).astype(np.float32)
            mass[:, b] *= ms
            inertia[:, b] *= ms[:, None]
            self.mw.body_mass = wp.array(mass, dtype=float)
            self.mw.body_inertia = wp.array(inertia, dtype=wp.vec3)
            self.prm = C.make_params(self.mjm)
            self.tgt_w = wp.zeros((N, NREC), dtype=float)
            self.drv_w = wp.array(np.stack([rng.uniform(*cfg.torque_range, N), W0 * rng.uniform(0.9, 1.1, N)], 1)
                                  .astype(np.float32))
            self.on_w = wp.zeros(N, dtype=int)
            self.out_w = wp.zeros((N, 4), dtype=float)
        tt = wp.to_torch
        self.qpos, self.qvel, self.ctrl = tt(self.dw.qpos), tt(self.dw.qvel), tt(self.dw.ctrl)
        self.tgt, self.drv, self.on, self.legs = tt(self.tgt_w), tt(self.drv_w), tt(self.on_w), tt(self.out_w)
        self.reset_buffers = [tt(getattr(self.dw, n)) for n in ("qacc_warmstart", "qacc", "qfrc_actuator", "act",
                                                                  "qfrc_applied", "xfrc_applied")
                              if getattr(self.dw, n, None) is not None and getattr(self.dw, n).shape[0] == N]
        self.dr_friction, self.dr_mass = torch.tensor(mu, device=self.device), torch.tensor(ms, device=self.device)
        self._tables()

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.actions, self.last_actions = z(N, NREC), z(N, NREC)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.hold = z(N)
        self.frame_dim = 18 + NREC          # gravity, gyro, legs (4), hip speeds (4), wheels (2), gear, settling, action
        self.hist = z(N, cfg.history, self.frame_dim)
        self.ep_sums = {k: z(N) for k in cfg.rew}
        self.extras = {"log": {}}
        self._reset(torch.arange(N, device=self.device))
        with wp.ScopedStream(self.stream):
            self._substeps()
            with wp.ScopedCapture(stream=self.stream) as cap:
                self._substeps()
            self.graph = cap.graph
        self._compute_obs()

    def _substeps(self):
        for _ in range(self.cfg.decimation):
            wp.launch(recovery_kernel, dim=self.num_envs,
                      inputs=[self.prm, self.dw.qpos, self.dw.qvel, self.tgt_w, self.drv_w, self.on_w,
                              self.cfg.kp, self.cfg.kd, self.out_w, self.dw.ctrl])
            mjw.step(self.mw, self.dw)

    def _tables(self):
        m = self.mjm
        jq = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        jv = lambda n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        self.q_hips = [jq(n) for n in ("hip_Yp_xp", "hip_Ym_xp", "hip_Yp_xm", "hip_Ym_xm")]
        self.v_hips = [jv(n) for n in ("hip_Yp_xp", "hip_Ym_xp", "hip_Yp_xm", "hip_Ym_xm")]
        self.v_wheels = [jv("wheel_xp"), jv("wheel_xm")]
        self.q_gear = [jq("gear_front"), jq("gear_rear")]
        self.leg_joints = {s: [jq(f"hip_Yp_{s}"), jq(f"knee_Yp_{s}"), jq(f"hip_Ym_{s}"), jq(f"knee_Ym_{s}")]
                           for s in ("xp", "xm")}
        Ls = np.linspace(0.17, 0.54, 75)
        self.tab_L = torch.tensor(Ls, dtype=torch.float32, device=self.device)
        self.tab_a = torch.tensor([C.five_bar.symmetric_angle_for_length("xp", L) for L in Ls], dtype=torch.float32,
                                  device=self.device)
        rows = []
        for L in Ls:
            q, zb = C.five_bar.nominal_pose(L, wheel_radius=0.10751, clearance=0.004)
            rows.append((q["hip_Yp_xp"], q["knee_Yp_xp"], q["knee_Ym_xp"]))
        self.pose_tab = torch.tensor(rows, dtype=torch.float32, device=self.device)
        self.parked = torch.tensor(M.parked_qpos(), dtype=torch.float32, device=self.device)
        self.parked_z = float(M.parked_qpos()[2])

    def _interp(self, tab, L):
        k = ((L - self.tab_L[0]) / (self.tab_L[1] - self.tab_L[0])).clamp(0, len(self.tab_L) - 1.001)
        i0 = k.long()
        f = k - i0
        if tab.dim() == 1:
            return tab[i0] * (1 - f) + tab[i0 + 1] * f
        return tab[i0] * (1 - f[:, None]) + tab[i0 + 1] * f[:, None]

    def _rand(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.device)

    # ------------------------------------------------------------------------------------
    def _reset(self, ids):
        n = len(ids)
        if n == 0:
            return
        q = self.parked.repeat(n, 1)
        L = self._rand(n, 0.18, 0.45)
        row = self._interp(self.pose_tab, L)
        for s in ("xp", "xm"):
            hp, kp_, hm, km_ = self.leg_joints[s]
            q[:, hp], q[:, kp_], q[:, hm], q[:, km_] = row[:, 0], row[:, 1], -row[:, 0], row[:, 2]
        gear_up = torch.rand(n, generator=self.gen, device=self.device) < 0.7
        q[gear_up, self.q_gear[0]], q[gear_up, self.q_gear[1]] = math.radians(-100.0), math.radians(100.0)
        # fallen orientation: tilted 75-180 deg forward / backward / sideways (up to upside down), any heading
        side = torch.rand(n, generator=self.gen, device=self.device) < self.cfg.p_side
        tilt = self._rand(n, *(math.radians(v) for v in self.cfg.tilt_range)) * torch.where(
            torch.rand(n, generator=self.gen, device=self.device) < 0.5, -1.0, 1.0)
        yaw = self._rand(n, -math.pi, math.pi)
        ax = torch.where(side[:, None], torch.tensor([0.0, 1.0, 0.0], device=self.device),
                         torch.tensor([1.0, 0.0, 0.0], device=self.device))        # pitch about base x, roll about y
        qt = torch.cat([torch.cos(tilt / 2)[:, None], ax * torch.sin(tilt / 2)[:, None]], 1)
        qy = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
        q[:, 3:7] = self._qmul(qy, qt)
        q[:, 0:2] = 0.0
        q[:, 2] = 0.45
        self.qpos[ids] = q
        self.qvel[ids] = 0.0
        for buf in self.reset_buffers:
            buf[ids] = 0.0
        self.ctrl[ids] = 0.0
        self.tgt[ids] = 0.0
        self.on[ids] = 0
        self.actions[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.hist[ids] = 0.0
        self.hold[ids] = 0.0
        self.episode_length_buf[ids] = 0
        for v in self.ep_sums.values():
            v[ids] = 0.0

    @staticmethod
    def _qmul(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

    @staticmethod
    def _R(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)

    # ------------------------------------------------------------------------------------
    def _targets(self):
        a = self.actions
        L = 0.30 + 0.15 * a[:, 0:2]
        phi = 1.4 * a[:, 2:4]
        half = self._interp(self.tab_a, L.reshape(-1)).view(-1, 2)        # symmetric hip angle for each length
        self.tgt[:, 0], self.tgt[:, 1] = half[:, 0] + phi[:, 0], -half[:, 0] + phi[:, 0]
        self.tgt[:, 2], self.tgt[:, 3] = half[:, 1] + phi[:, 1], -half[:, 1] + phi[:, 1]
        self.tgt[:, 4:6] = 4.0 * a[:, 4:6]
        self.tgt[:, 6] = a[:, 6]

    def _compute_obs(self):
        N = self.num_envs
        R = self._R(self.qpos[:, 3:7])
        self.grav_b = -R[:, 2, :]
        gear_up = (self.qpos[:, self.q_gear[1]:self.q_gear[1] + 1] > 1.0).float()       # servo position
        frame = torch.cat([self.grav_b, self.qvel[:, 3:6], self.legs, self.qvel[:, self.v_hips] * 0.1,
                           self.qvel[:, self.v_wheels] * 0.05, gear_up,
                           (self.episode_length_buf[:, None] * self.dt < self.cfg.settle_s).float(),
                           self.actions], -1)
        noise = 0.02 * torch.randn(frame.shape, generator=self.gen, device=self.device)
        noise[:, -NREC:] = 0.0
        self.hist = torch.cat([self.hist[:, 1:], (frame + noise)[:, None]], 1)
        critic = torch.cat([frame, self.qpos[:, 2:3], self.qvel[:, 0:3], self.dr_friction[:, None],
                            self.dr_mass[:, None], (self.drv[:, 0:1] - 20.0) / 5.0], -1)
        self.obs = TensorDict({"policy": self.hist.reshape(N, -1), "critic": critic}, batch_size=[N])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        cfg = self.cfg
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1, 1))
        self.on[:] = (self.episode_length_buf.float() * self.dt >= cfg.settle_s).int()
        self._targets()
        wp.capture_launch(self.graph, stream=self.stream)
        bad = ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1))
        if bad.any():
            self.qpos[bad] = self.parked
            self.qvel[bad] = 0.0
        self.episode_length_buf += 1
        self._compute_obs()
        rew, success, timeout = self._rewards()
        dones = success | timeout | bad
        ids = dones.nonzero().flatten()
        log = self.extras["log"] = {}
        if len(ids):
            for k, v in self.ep_sums.items():
                log[f"Episode_Reward/{k}"] = (v[ids] / cfg.episode_s).mean()
            log["Metrics/success"] = success[ids].float().mean()
            log["Metrics/episode_s"] = (self.episode_length_buf[ids].float() * self.dt).mean()
        self._reset(ids)
        if len(ids):
            self._compute_obs()
        self.extras["time_outs"] = timeout & ~success
        return self.obs, rew, dones, self.extras

    def _rewards(self):
        cfg, w = self.cfg, self.cfg.rew
        up = 0.5 - 0.5 * self.grav_b[:, 2].clamp(-1, 1)                  # 1 upright .. 0 upside down
        upright = self.grav_b[:, 2] < -math.cos(math.radians(15.0))
        L = self.legs[:, 0:2]
        pose_err = (L - M_PARK_LEN).abs().amax(-1)
        gear_down = self.tgt[:, 6] > 0
        still = (self.qvel[:, 0:3].norm(dim=-1) < 0.2) & (self.qvel[:, 3:6].norm(dim=-1) < 0.5)
        good = upright & (pose_err < 0.03) & gear_down & still & (self.on > 0)
        self.hold = torch.where(good, self.hold + self.dt, torch.zeros_like(self.hold))
        success = self.hold >= cfg.success_hold_s
        on = (self.on > 0).float()
        terms = dict(
            upright=up * on,
            height=torch.exp(-(self.qpos[:, 2] - self.parked_z).abs() / 0.05) * upright * on,
            pose=torch.exp(-pose_err.square() / 0.03 ** 2) * (up > 0.9) * on,
            pose_coarse=torch.exp(-pose_err / 0.08) * (up > 0.9) * on,       # gradient from far away too
            gear=gear_down.float() * (up > 0.9) * on,
            still=still.float() * upright * on,
            joint_vel=self.qvel[:, self.v_hips].square().sum(-1) * on,
            torque=(self.ctrl[:, 0:6] * 9.0).square().sum(-1) * on,
            action_rate=(self.actions - self.last_actions).square().sum(-1) * on,
        )
        timeout = self.episode_length_buf >= self.max_episode_length
        terms["success"] = success.float()
        rew = torch.zeros(self.num_envs, device=self.device)
        for k, v in terms.items():
            r = w[k] * v * (self.dt if k != "success" else 1.0)
            rew += r
            self.ep_sums[k] += r
        return rew, success, timeout


M_PARK_LEN = 0.18
