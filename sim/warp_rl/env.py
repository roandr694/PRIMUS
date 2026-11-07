"""Residual RL environment on the GPU: MuJoCo Warp + the LQR/VMC controller (Warp kernel) + PPO (RSL-RL).

Each robot is a MuJoCo Warp world; all of them run on the same large terrain (terrain.py). At every policy step
(50 Hz) the CUDA graph runs `decimation` 1 ms physics steps, and the base controller runs at every physics step
(1 kHz), as on CPU MuJoCo. The policy only adds a residual (controller.py, RESIDUAL) and only in OPERATING mode;
with a zero action the robot is exactly the model-based controller.

Actor: only what the real robot measures (IMU, hip/wheel encoders, odometry) with noise, plus the commands and a
short history. Critic: also the true velocity, the height above ground, a height map around the robot and the
randomised parameters (friction, mass).
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
import terrain as T
from rsl_rl.env import VecEnv


@dataclass
class EnvCfg:
    num_envs: int = 4096
    decimation: int = 20                 # physics steps (1 ms) per policy step -> 50 Hz
    episode_s: float = 20.0
    # residual per unit action: [dL_xp, dL_xm] m, [dth_xp, dth_xm] rad, [dT_xp, dT_xm] N.m
    residual_scale: tuple = (0.08, 0.08, 0.25, 0.25, 3.0, 3.0)
    history: int = 5                     # stacked actor frames
    # commands
    cmd_interval_s: tuple = (4.0, 8.0)
    speed_range: tuple = (-1.0, 1.5)
    yaw_range: tuple = (-1.5, 1.5)
    leg_range: tuple = (0.20, 0.48)
    p_zero_speed: float = 0.15
    p_zero_yaw: float = 0.3
    # episode start
    p_parked_start: float = 0.15
    start_leg_range: tuple = (0.22, 0.45)
    spawn_offset: float = 1.0            # [m] random start offset around the tile centre
    # pushes (change the base velocity)
    push_interval_s: tuple = (3.0, 7.0)
    push_speed: float = 0.6
    # domain randomisation
    friction_range: tuple = (0.5, 1.25)
    base_mass_scale: tuple = (0.85, 1.15)
    com_offset: float = 0.02
    obs_noise: bool = True
    # terrain and curriculum
    terrain_levels: int = 6
    init_max_level: int = 1
    curriculum: bool = True              # False (evaluation): each robot stays on its self.level/self.ttype tile
    random_tile_type: bool = True        # draw the terrain type at every reset
    # rewards (per second; multiplied by the policy dt)
    rew: dict = field(default_factory=lambda: dict(
        track_speed=3.0, track_yaw=1.5, track_leg=0.5, upright=-5.0, ang_vel_xy=-0.05, lin_vel_z=-1.0,
        action=-0.02, action_rate=-0.1, torque=-2e-4, gear_contact=-2.0, termination=-10.0))


_STREAMS = {}


def _shared_stream(device):
    """A dedicated stream (torch's default stream cannot be captured in a CUDA graph), shared by torch and Warp
    so they need no synchronisation. One per process: if it died with an environment, torch would be left
    with an invalid stream."""
    if device not in _STREAMS:
        _STREAMS[device] = wp.Stream(device)
        torch.cuda.set_stream(wp.stream_to_torch(_STREAMS[device]))
    return _STREAMS[device]


class PrimusWarpEnv(VecEnv):
    collision = "minimal"                # model.robot_spec collision set
    nconmax, njmax = 24, 100             # MuJoCo Warp contact / constraint buffers

    def __init__(self, cfg: EnvCfg = EnvCfg(), device="cuda:0", seed=0, terrain=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = N = cfg.num_envs
        self.num_actions = len(cfg.residual_scale)
        self.dt = 0.001 * cfg.decimation
        self.max_episode_length = int(cfg.episode_s / self.dt)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # ---------------- model and terrain ----------------
        self.terrain = terrain or self._make_terrain(seed)
        self.mjm, self.spec = M.build(self.terrain, collision=self.collision)
        self._patch_model(self.mjm)
        mjd = mujoco.MjData(self.mjm)
        mjd.qpos[:] = M.parked_qpos()
        mjd.qpos[0:2] = self.terrain["centers"][0, 0]      # flat tile of level 0
        mujoco.mj_forward(self.mjm, mjd)
        wp.init()
        self.stream = _shared_stream(str(self.device))
        with wp.ScopedStream(self.stream):
            self.mw = mjw.put_model(self.mjm)
            self.dw = mjw.put_data(self.mjm, mjd, nworld=N, nconmax=self.nconmax, njmax=self.njmax)
            self._setup_randomization()
            self.ctl = C.BatchedController(self.mjm, N, str(self.device))
        tt = lambda a: wp.to_torch(a)
        self.qpos, self.qvel, self.ctrl = tt(self.dw.qpos), tt(self.dw.qvel), tt(self.dw.ctrl)
        self.cs, self.out, self.cmd, self.res = tt(self.ctl.cs), tt(self.ctl.out), tt(self.ctl.cmd), tt(self.ctl.res)
        self.mask = tt(self.ctl.mask)
        self.con_geom, self.con_world, self.nacon = tt(self.dw.contact.geom), tt(self.dw.contact.worldid), tt(self.dw.nacon)
        self.qacc_warmstart = tt(self.dw.qacc_warmstart)
        # per-world buffers the next step reads before recomputing them
        self.reset_buffers = [tt(getattr(self.dw, n)) for n in ("qacc_warmstart", "qacc", "qfrc_actuator", "sensordata",
                                                                  "act", "qfrc_applied", "xfrc_applied")
                              if getattr(self.dw, n, None) is not None and getattr(self.dw, n).shape[0] == N]
        # every per-world MuJoCo Warp array (solver ones included): a world that blew up keeps corrupted hidden
        # state and blows up again after each reset, so it gets a copy of a healthy world
        self.world_arrays = self._per_world_arrays(self.dw, N)
        self.H = torch.tensor(self.terrain["heights"], device=self.device)
        self.tile_centers = torch.tensor(self.terrain["centers"], device=self.device)   # [levels, types, 2]
        self._ids()
        self._start_pose_tables()

        # ---------------- buffers ----------------
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.actions, self.last_actions = z(N, self.num_actions), z(N, self.num_actions)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.cmd_timer, self.push_timer = z(N), z(N)
        self.level = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ttype = torch.zeros(N, dtype=torch.long, device=self.device)
        self.spawn_xy = z(N, 2)
        self.ep_speed_err, self.ep_steps_oper = z(N), z(N)
        self.ep_sums = {k: z(N) for k in cfg.rew}
        self.frame_dim = 21 + self.num_actions
        self.hist = z(N, cfg.history, self.frame_dim)
        self.scan_xy = torch.tensor([(x, y) for x in (-0.3, 0.0, 0.3) for y in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2)],
                                    device=self.device)
        self.res_scale = torch.tensor(cfg.residual_scale, device=self.device)
        self.res_offset = torch.tensor(getattr(cfg, "residual_offset", None) or [0.0] * len(cfg.residual_scale),
                                       device=self.device)
        self.extras = {"log": {}}
        self.bad = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._init_buffers()

        self.level[:] = torch.randint(0, cfg.init_max_level + 1, (N,), generator=self.gen, device=self.device)
        self._reset(torch.arange(N, device=self.device))
        self._capture()
        self._compute_obs()

    # ---- hooks for derived environments (skill_env.py) ----
    def _make_terrain(self, seed):
        return T.make_terrain(n_levels=self.cfg.terrain_levels, seed=seed)

    def _patch_model(self, mjm):
        pass

    def _init_buffers(self):
        pass

    def _pre_physics(self):
        pass

    def _draw_types(self, ids):
        if self.cfg.random_tile_type:
            self.ttype[ids] = torch.randint(0, len(self.terrain["types"]), (len(ids),), generator=self.gen,
                                            device=self.device)

    def _spawn(self, ids):
        """Start position and heading: around the tile centre."""
        n = len(ids)
        center = self.tile_centers[self.level[ids], self.ttype[ids]]
        xy = center + self._rand(n * 2, -self.cfg.spawn_offset, self.cfg.spawn_offset).view(n, 2)
        return xy, self._rand(n, -math.pi, math.pi)

    # ------------------------------------------------------------------------------------
    def _ids(self):
        m = self.mjm
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        jq = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        jv = lambda n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        self.g_ground = gid("ground")
        self.g_gear = torch.tensor([gid("gear_front_tip"), gid("gear_rear_tip")], device=self.device)
        self.qv_wheel = [jv("wheel_xp"), jv("wheel_xm")]
        self.leg_joints = {s: [jq(f"hip_Yp_{s}"), jq(f"knee_Yp_{s}"), jq(f"hip_Ym_{s}"), jq(f"knee_Ym_{s}")] for s in ("xp", "xm")}
        self.q_gear = [jq("gear_front"), jq("gear_rear")]
        self.parked = torch.tensor(M.parked_qpos(), dtype=torch.float32, device=self.device)

    @staticmethod
    def _per_world_arrays(obj, N, seen=None):
        seen = set() if seen is None else seen
        out = []
        for name in dir(obj):
            if name.startswith("_"):
                continue
            a = getattr(obj, name, None)
            if isinstance(a, wp.array):
                if a.ptr is not None and a.ptr not in seen and a.shape and a.shape[0] == N:
                    seen.add(a.ptr)
                    try:
                        out.append(wp.to_torch(a))
                    except Exception:
                        pass
            elif hasattr(a, "__dataclass_fields__") and not isinstance(a, type):
                out += PrimusWarpEnv._per_world_arrays(a, N, seen)
        return out

    def _start_pose_tables(self):
        """Table L0 -> (hip angle, knee Yp, knee Ym, base height) to spawn operating at any leg length."""
        Ls = np.linspace(0.18, 0.52, 69)
        rows = []
        for L in Ls:
            q, z = C.five_bar.nominal_pose(L, wheel_radius=0.10751, clearance=0.004)
            rows.append((q["hip_Yp_xp"], q["knee_Yp_xp"], q["knee_Ym_xp"], z))
        self.pose_L = torch.tensor(Ls, dtype=torch.float32, device=self.device)
        self.pose_tab = torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _setup_randomization(self):
        cfg, N, m = self.cfg, self.num_envs, self.mjm
        rng = np.random.default_rng(1234)
        fr = np.tile(m.geom_friction[None], (N, 1, 1)).astype(np.float32)
        mu = rng.uniform(*cfg.friction_range, N).astype(np.float32)
        tires = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{s}_tire") for s in ("xp", "xm")]
        world = [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0]
        for i in tires + world:        # MuJoCo uses the larger friction of the pair: draw both equal
            fr[:, i, 0] = mu
        self.mw.geom_friction = wp.array(fr, dtype=wp.vec3)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        ms = rng.uniform(*cfg.base_mass_scale, N).astype(np.float32)
        mass = np.tile(m.body_mass[None], (N, 1)).astype(np.float32)
        inertia = np.tile(m.body_inertia[None], (N, 1, 1)).astype(np.float32)
        ipos = np.tile(m.body_ipos[None], (N, 1, 1)).astype(np.float32)
        mass[:, b] *= ms
        inertia[:, b] *= ms[:, None]
        ipos[:, b] += rng.uniform(-cfg.com_offset, cfg.com_offset, (N, 3)).astype(np.float32)
        self.mw.body_mass = wp.array(mass, dtype=float)
        self.mw.body_inertia = wp.array(inertia, dtype=wp.vec3)
        self.mw.body_ipos = wp.array(ipos, dtype=wp.vec3)
        self.dr_friction = torch.tensor(mu, device=self.device)
        self.dr_mass = torch.tensor(ms, device=self.device)

    def _capture(self):
        with wp.ScopedStream(self.stream):
            self._substeps()                      # compile/load the kernels before capturing
            with wp.ScopedCapture(stream=self.stream) as cap:
                self._substeps()
            self.graph = cap.graph

    def _substeps(self):
        for _ in range(self.cfg.decimation):
            self.ctl.step(self.dw)
            mjw.step(self.mw, self.dw)

    # ------------------------------------------------------------------------------------
    def _rand(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.device)

    def _ground(self, xy, radius=0.35):
        """Highest terrain point on a disc around xy (to spawn without penetrating)."""
        offs = torch.tensor([(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)], device=self.device)
        return T.sample_heights(self.H, self.terrain["size_x"], self.terrain["size_y"], xy[:, None, :] + offs).amax(-1)

    def _resample_commands(self, ids):
        n, cfg = len(ids), self.cfg
        speed = self._rand(n, *cfg.speed_range)
        yaw = self._rand(n, *cfg.yaw_range)
        speed[torch.rand(n, generator=self.gen, device=self.device) < cfg.p_zero_speed] = 0.0
        yaw[torch.rand(n, generator=self.gen, device=self.device) < cfg.p_zero_yaw] = 0.0
        self.cmd[ids, 0], self.cmd[ids, 1] = speed, yaw
        self.cmd[ids, 2] = self._rand(n, *cfg.leg_range)
        self.cmd[ids, 3] = 0.0
        self.cmd[ids, 4] = 0.0
        self.cmd[ids, 5] = 1.5                      # responsiveness: the original speed ramp
        self.cmd_timer[ids] = self._rand(n, *cfg.cmd_interval_s)

    def _reset(self, ids):
        if len(ids) == 0:
            return
        n, cfg = len(ids), self.cfg
        self._draw_types(ids)
        xy, yaw = self._spawn(ids)
        ground = self._ground(xy)
        q = self.parked.repeat(n, 1)
        parked = torch.rand(n, generator=self.gen, device=self.device) < cfg.p_parked_start
        # spawn operating: any leg length, gear retracted
        L = self._rand(n, *cfg.start_leg_range)
        k = ((L - self.pose_L[0]) / (self.pose_L[1] - self.pose_L[0])).clamp(0, len(self.pose_L) - 1.001)
        i0 = k.long(); fr = (k - i0)[:, None]
        row = self.pose_tab[i0] * (1 - fr) + self.pose_tab[i0 + 1] * fr
        oper = ~parked
        for s in ("xp", "xm"):
            hp, kp_, hm, km_ = self.leg_joints[s]
            q[oper, hp], q[oper, kp_], q[oper, hm], q[oper, km_] = row[oper, 0], row[oper, 1], -row[oper, 0], row[oper, 2]
        q[oper, self.q_gear[0]], q[oper, self.q_gear[1]] = math.radians(-100.0), math.radians(100.0)
        q[:, 2] = ground + torch.where(parked, self.parked[2], row[:, 3])
        q[:, 0:2] = xy
        q[:, 3] = torch.cos(yaw / 2); q[:, 4] = 0; q[:, 5] = 0; q[:, 6] = torch.sin(yaw / 2)
        self.qpos[ids] = q
        self.qvel[ids] = 0.0
        for buf in self.reset_buffers:
            buf[ids] = 0.0
        self.ctrl[ids] = 0.0
        self.res[ids] = 0.0
        self.out[ids] = 0.0
        self.spawn_xy[ids] = xy
        self._resample_commands(ids)
        self.push_timer[ids] = self._rand(n, *cfg.push_interval_s)
        self.mask.zero_()
        self.mask[ids] = 1
        with wp.ScopedStream(self.stream):
            wp.launch(C.reset_kernel, dim=self.num_envs, inputs=[self.ctl.prm, self.ctl.mask, self.dw.qpos, self.ctl.cs])
        self.actions[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.hist[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.ep_speed_err[ids] = 0.0
        self.ep_steps_oper[ids] = 0.0
        for v in self.ep_sums.values():
            v[ids] = 0.0

    # ------------------------------------------------------------------------------------
    def _base_frame(self):
        q = self.qpos[:, 3:7]
        w, x, y, z = q.unbind(-1)
        R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
        return R

    def _compute_obs(self):
        cfg, o, N = self.cfg, self.out, self.num_envs
        R = self._base_frame()
        grav = -R[:, 2, :]                                   # unit gravity in the base frame
        gyro = self.qvel[:, 3:6]
        legs = torch.stack([o[:, C.O["L_xp"]], o[:, C.O["L_xm"]], o[:, C.O["th_xp"]], o[:, C.O["th_xm"]],
                            o[:, C.O["dL_xp"]], o[:, C.O["dL_xm"]], o[:, C.O["dth_xp"]], o[:, C.O["dth_xm"]]], -1)
        wheels = self.qvel[:, self.qv_wheel] * 0.1
        dx = o[:, C.O["dx"]:C.O["dx"] + 1]
        oper = (self.cs[:, C.S["mode"]] == C.OPERATING).float()[:, None]
        if cfg.obs_noise:
            nz = lambda t, s: t + s * torch.randn(t.shape, generator=self.gen, device=self.device)
            grav, gyro, legs, wheels, dx = nz(grav, 0.02), nz(gyro, 0.05), nz(legs, 0.01), nz(wheels, 0.05), nz(dx, 0.03)
        frame = torch.cat([grav, gyro, legs, wheels, dx, oper, self.cmd[:, 0:3], self.actions], -1)
        self.hist = torch.cat([self.hist[:, 1:], frame[:, None]], 1)
        # critic: true state + height map in the heading frame + randomised parameters
        v_body = torch.einsum("nji,nj->ni", R, self.qvel[:, 0:3])
        heading = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        c, s = torch.cos(heading), torch.sin(heading)
        sx, sy = self.scan_xy[:, 0], self.scan_xy[:, 1]
        px = self.qpos[:, 0:1] + c[:, None] * sx - s[:, None] * sy
        py = self.qpos[:, 1:2] + s[:, None] * sx + c[:, None] * sy
        hs = T.sample_heights(self.H, self.terrain["size_x"], self.terrain["size_y"], torch.stack([px, py], -1))
        ground = T.sample_heights(self.H, self.terrain["size_x"], self.terrain["size_y"], self.qpos[:, 0:2])
        self.base_height = self.qpos[:, 2] - ground
        scan = (self.qpos[:, 2:3] - hs - 0.4).clamp(-1, 1)
        critic = torch.cat([frame, v_body, self.base_height[:, None], scan, self.dr_friction[:, None],
                            self.dr_mass[:, None], o[:, C.O["grounded"]:C.O["grounded"] + 1]], -1)
        self.obs = TensorDict({"policy": self.hist.reshape(N, -1), "critic": critic}, batch_size=[N])
        self.R, self.v_body = R, v_body

    def get_observations(self):
        return self.obs

    # ------------------------------------------------------------------------------------
    def step(self, actions):
        cfg, N = self.cfg, self.num_envs
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1, 1))
        self.res[:, :self.num_actions] = self.actions * self.res_scale + self.res_offset
        # pushes: change the horizontal base velocity
        self.push_timer -= self.dt
        pushed = (self.push_timer <= 0).nonzero().flatten()
        if len(pushed):
            self.qvel[pushed, 0:2] += self._rand(len(pushed) * 2, -cfg.push_speed, cfg.push_speed).view(-1, 2)
            self.push_timer[pushed] = self._rand(len(pushed), *cfg.push_interval_s)
        self._pre_physics()
        wp.capture_launch(self.graph, stream=self.stream)
        # a world that blew up numerically (rare; e.g. EPA failure in a heightfield contact) counts as a fall:
        # replace its state by a finite one before any computation and reset it at the end of the step
        self.bad = ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1)
                     & torch.isfinite(self.out).all(-1))
        n_bad = int(self.bad.sum())
        if n_bad:
            bad = self.bad.nonzero().flatten()
            good = int((~self.bad).nonzero()[0, 0]) if n_bad < self.num_envs else 0
            for t in self.world_arrays:
                t[bad] = t[good].clone()
            self.qpos[bad] = self.parked
            self.qpos[bad, 0:2] = self.spawn_xy[bad]
            self.qvel[bad] = 0.0
            self.out[bad] = 0.0
        self.episode_length_buf += 1
        self.cmd_timer -= self.dt
        self._compute_obs()

        rew, terminated, timeout = self._rewards_and_dones()
        dones = terminated | timeout
        ids = dones.nonzero().flatten()
        self._log_and_curriculum(ids, terminated, timeout)
        self._reset(ids)
        resample = (self.cmd_timer <= 0).nonzero().flatten()
        if len(resample):
            self._resample_commands(resample)
        if len(ids):
            self._compute_obs()
        self.extras["time_outs"] = timeout
        return self.obs, rew, dones, self.extras

    def _rewards_and_dones(self):
        cfg, o, R = self.cfg, self.out, self.R
        w = cfg.rew
        oper = self.cs[:, C.S["mode"]] == C.OPERATING
        heading = torch.stack([R[:, 0, 1], R[:, 1, 1]], -1)
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_fwd = (self.qvel[:, 0:2] * heading).sum(-1)
        yaw_rate = (R @ self.qvel[:, 3:6, None])[:, 2, 0]
        speed_err = (v_fwd - self.cmd[:, 0]).square()
        yaw_err = (yaw_rate - self.cmd[:, 1]).square()
        leg = 0.5 * (o[:, C.O["L_xp"]] + o[:, C.O["L_xm"]])
        grav_b = -R[:, 2, :]
        # landing gear touching the ground while operating (the "sit on the gear" exploit of the Isaac run)
        n = min(int(self.nacon[0].item()), self.con_geom.shape[0])
        gear_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if n:
            g, wid = self.con_geom[:n], self.con_world[:n].long()
            other = torch.where(g[:, 0] == self.g_ground, g[:, 1], g[:, 0])
            gear_hit[wid[torch.isin(other, self.g_gear)]] = True
        gear_hit &= oper
        terms = dict(
            track_speed=torch.exp(-speed_err / 0.1) * oper,
            track_yaw=torch.exp(-yaw_err / 0.2) * oper,
            track_leg=torch.exp(-(leg - self.cmd[:, 2]).square() / 0.03 ** 2) * oper,
            upright=grav_b[:, 0].square() + grav_b[:, 1].square(),
            ang_vel_xy=self.qvel[:, 3:5].square().sum(-1),
            lin_vel_z=self.qvel[:, 2].square(),
            action=self.actions.square().sum(-1),
            action_rate=(self.actions - self.last_actions).square().sum(-1),
            torque=self.out[:, C.O["tau_hYp_xp"]:C.O["tau_w_xm"] + 1].square().sum(-1),
            gear_contact=gear_hit.float(),
        )
        # falls from the geometry (only tires and gear tips collide with the terrain): tilted too much, chassis
        # below the minimum operating height, or the controller protection cut the torques
        tilt = grav_b[:, 2] > -math.cos(math.radians(50.0))
        fell = tilt | (self.base_height < 0.2) | (o[:, C.O["protect"]] > 0.5) | self.bad
        half = torch.tensor([self.terrain["size_x"], self.terrain["size_y"]], device=self.device) / 2 - 1.0
        off_map = (self.qpos[:, 0:2].abs() > half).any(-1)
        timeout = (self.episode_length_buf >= self.max_episode_length) | off_map
        terms["termination"] = fell.float()
        rew = torch.zeros(self.num_envs, device=self.device)
        for k, v in terms.items():
            r = w[k] * v * (self.dt if k != "termination" else 1.0)
            rew += r
            self.ep_sums[k] += r
        self.ep_speed_err += speed_err.sqrt() * oper
        self.ep_steps_oper += oper.float()
        return rew, fell, timeout & ~fell

    def _log_and_curriculum(self, ids, terminated, timeout):
        log = self.extras["log"] = {}
        if len(ids) == 0:
            return
        for k, v in self.ep_sums.items():
            log[f"Episode_Reward/{k}"] = (v[ids] / self.cfg.episode_s).mean()
        err = self.ep_speed_err[ids] / self.ep_steps_oper[ids].clamp_min(1)
        log["Metrics/speed_error_mps"] = err.mean()
        log["Metrics/fell"] = terminated[ids].float().mean()
        log["Metrics/episode_s"] = (self.episode_length_buf[ids].float() * self.dt).mean()
        log["Metrics/nonfinite_resets"] = self.bad[ids].float().sum()
        if not self.cfg.curriculum:
            return
        # curriculum: level up after finishing the episode tracking the command; level down after a fall
        up = timeout[ids] & (err < 0.25)
        down = terminated[ids]
        lv = self.level[ids] + up.long() - down.long()
        top = self.cfg.terrain_levels - 1
        lv = torch.where(lv > top, torch.randint(0, top + 1, lv.shape, generator=self.gen, device=self.device), lv)
        self.level[ids] = lv.clamp(0, top)
        log["Curriculum/level_mean"] = self.level.float().mean()
        for t, name in enumerate(self.terrain["types"]):
            sel = self.ttype[ids] == t
            if sel.any():
                log[f"Fell_by_terrain/{name}"] = terminated[ids][sel].float().mean()
