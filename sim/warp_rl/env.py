"""Ambiente de RL residual em GPU: MuJoCo Warp + controlador LQR/VMC (kernel Warp) + politica PPO (RSL-RL).

Cada robo e um mundo do MuJoCo Warp; todos rodam no mesmo terreno grande (terrain.py). A cada passo da
politica (50 Hz) o grafo CUDA roda `decimation` passos de fisica de 1 ms, e o controlador base roda a
cada passo de fisica (1 kHz), como no MuJoCo de CPU. A politica so soma um residual (controller.py,
RESIDUAL) e so no modo OPERATING; com acao zero o robo e exatamente o controlador model-based.

Actor: so o que o robo real mede (IMU, encoders dos quadris/rodas, odometria), com ruido, mais os comandos
e o historico curto. Critic: tambem a velocidade real, a altura sobre o chao, um mapa de alturas em volta
e os parametros sorteados (atrito, massa).
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
    decimation: int = 20                 # passos de fisica (1 ms) por passo da politica -> 50 Hz
    episode_s: float = 20.0
    # residual por unidade de acao: [dL_xp, dL_xm] m, [dth_xp, dth_xm] rad, [dT_xp, dT_xm] N.m
    residual_scale: tuple = (0.08, 0.08, 0.25, 0.25, 3.0, 3.0)
    history: int = 5                     # quadros do actor empilhados
    # comandos
    cmd_interval_s: tuple = (4.0, 8.0)
    speed_range: tuple = (-1.0, 1.5)
    yaw_range: tuple = (-1.5, 1.5)
    leg_range: tuple = (0.20, 0.48)
    p_zero_speed: float = 0.15
    p_zero_yaw: float = 0.3
    # inicio do episodio
    p_parked_start: float = 0.15
    start_leg_range: tuple = (0.22, 0.45)
    spawn_offset: float = 1.0            # [m] sorteio da posicao inicial em volta do centro do ladrilho
    # empurroes (muda a velocidade da base)
    push_interval_s: tuple = (3.0, 7.0)
    push_speed: float = 0.6
    # randomizacao de dominio
    friction_range: tuple = (0.5, 1.25)
    base_mass_scale: tuple = (0.85, 1.15)
    com_offset: float = 0.02
    obs_noise: bool = True
    # terreno e curriculum
    terrain_levels: int = 6
    init_max_level: int = 1
    curriculum: bool = True              # False (avaliacao): cada robo fica no ladrilho de self.level/self.ttype
    random_tile_type: bool = True        # sorteia o tipo de terreno a cada reset
    # recompensas (por segundo; multiplicadas por dt da politica)
    rew: dict = field(default_factory=lambda: dict(
        track_speed=3.0, track_yaw=1.5, track_leg=0.5, upright=-5.0, ang_vel_xy=-0.05, lin_vel_z=-1.0,
        action=-0.02, action_rate=-0.1, torque=-2e-4, gear_contact=-2.0, termination=-10.0))


_STREAMS = {}


def _shared_stream(device):
    """Stream proprio (o stream padrao do torch nao pode ser capturado num grafo CUDA), compartilhado pelo
    torch e pelo Warp para nao precisar sincronizar entre eles. Um por processo: se fosse destruido junto
    com um ambiente, o torch ficaria com um stream invalido."""
    if device not in _STREAMS:
        _STREAMS[device] = wp.Stream(device)
        torch.cuda.set_stream(wp.stream_to_torch(_STREAMS[device]))
    return _STREAMS[device]


class PrimusWarpEnv(VecEnv):
    def __init__(self, cfg: EnvCfg = EnvCfg(), device="cuda:0", seed=0, terrain=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = N = cfg.num_envs
        self.num_actions = C.NRES
        self.dt = 0.001 * cfg.decimation
        self.max_episode_length = int(cfg.episode_s / self.dt)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # ---------------- modelo e terreno ----------------
        self.terrain = terrain or T.make_terrain(n_levels=cfg.terrain_levels, seed=seed)
        self.mjm, self.spec = M.build(self.terrain)
        mjd = mujoco.MjData(self.mjm)
        mjd.qpos[:] = M.parked_qpos()
        mjd.qpos[0:2] = self.terrain["centers"][0, 0]      # ladrilho plano do nivel 0
        mujoco.mj_forward(self.mjm, mjd)
        wp.init()
        self.stream = _shared_stream(str(self.device))
        with wp.ScopedStream(self.stream):
            self.mw = mjw.put_model(self.mjm)
            self.dw = mjw.put_data(self.mjm, mjd, nworld=N, nconmax=24, njmax=100)
            self._setup_randomization()
            self.ctl = C.BatchedController(self.mjm, N, str(self.device))
        tt = lambda a: wp.to_torch(a)
        self.qpos, self.qvel, self.ctrl = tt(self.dw.qpos), tt(self.dw.qvel), tt(self.dw.ctrl)
        self.cs, self.out, self.cmd, self.res = tt(self.ctl.cs), tt(self.ctl.out), tt(self.ctl.cmd), tt(self.ctl.res)
        self.mask = tt(self.ctl.mask)
        self.con_geom, self.con_world, self.nacon = tt(self.dw.contact.geom), tt(self.dw.contact.worldid), tt(self.dw.nacon)
        self.qacc_warmstart = tt(self.dw.qacc_warmstart)
        # buffers por mundo que o passo seguinte le antes de recalcular
        self.reset_buffers = [tt(getattr(self.dw, n)) for n in ("qacc_warmstart", "qacc", "qfrc_actuator", "sensordata",
                                                                  "act", "qfrc_applied", "xfrc_applied")
                              if getattr(self.dw, n, None) is not None and getattr(self.dw, n).shape[0] == N]
        # todos os arrays por mundo do MuJoCo Warp (inclusive os do solver): um mundo que explodiu fica com estado
        # escondido corrompido e volta a explodir a cada reset; ele recebe a copia de um mundo saudavel
        self.world_arrays = self._per_world_arrays(self.dw, N)
        self.H = torch.tensor(self.terrain["heights"], device=self.device)
        self.tile_centers = torch.tensor(self.terrain["centers"], device=self.device)   # [levels, types, 2]
        self._ids()
        self._start_pose_tables()

        # ---------------- buffers ----------------
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.actions, self.last_actions = z(N, 6), z(N, 6)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.cmd_timer, self.push_timer = z(N), z(N)
        self.level = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ttype = torch.zeros(N, dtype=torch.long, device=self.device)
        self.spawn_xy = z(N, 2)
        self.ep_speed_err, self.ep_steps_oper = z(N), z(N)
        self.ep_sums = {k: z(N) for k in cfg.rew}
        self.frame_dim = 27
        self.hist = z(N, cfg.history, self.frame_dim)
        self.scan_xy = torch.tensor([(x, y) for x in (-0.3, 0.0, 0.3) for y in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2)],
                                    device=self.device)
        self.res_scale = torch.tensor(cfg.residual_scale, device=self.device)
        self.extras = {"log": {}}
        self.bad = torch.zeros(N, dtype=torch.bool, device=self.device)

        self.level[:] = torch.randint(0, cfg.init_max_level + 1, (N,), generator=self.gen, device=self.device)
        self._reset(torch.arange(N, device=self.device))
        self._capture()
        self._compute_obs()

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
        """Tabelas L0 -> (angulo do quadril, joelhos, altura do eixo) para nascer operando numa altura qualquer."""
        Ls = np.linspace(0.18, 0.52, 69)
        rows = []
        for L in Ls:
            a = C.five_bar.symmetric_angle_for_length("xp", L)
            kp, km, f = C.five_bar.closure("xp", a, -a)
            rows.append((a, kp, km, f[1]))
        self.pose_L = torch.tensor(Ls, dtype=torch.float32, device=self.device)
        self.pose_tab = torch.tensor(rows, dtype=torch.float32, device=self.device)   # [a, knee_p, knee_m, foot_z]

    def _setup_randomization(self):
        cfg, N, m = self.cfg, self.num_envs, self.mjm
        rng = np.random.default_rng(1234)
        fr = np.tile(m.geom_friction[None], (N, 1, 1)).astype(np.float32)
        mu = rng.uniform(*cfg.friction_range, N).astype(np.float32)
        tires = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{s}_tire") for s in ("xp", "xm")]
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        for i in tires + [g]:          # o MuJoCo usa o maior atrito do par: sorteia os dois iguais
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
            self._substeps()                      # compila/carrega os kernels antes da captura
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
        """Maior altura do terreno num disco em volta de xy (para nascer sem penetrar)."""
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
        self.cmd_timer[ids] = self._rand(n, *cfg.cmd_interval_s)

    def _reset(self, ids):
        if len(ids) == 0:
            return
        n, cfg = len(ids), self.cfg
        if cfg.random_tile_type:
            self.ttype[ids] = torch.randint(0, len(self.terrain["types"]), (n,), generator=self.gen, device=self.device)
        center = self.tile_centers[self.level[ids], self.ttype[ids]]
        xy = center + self._rand(n * 2, -cfg.spawn_offset, cfg.spawn_offset).view(n, 2)
        yaw = self._rand(n, -math.pi, math.pi)
        ground = self._ground(xy)
        q = self.parked.repeat(n, 1)
        parked = torch.rand(n, generator=self.gen, device=self.device) < cfg.p_parked_start
        # nasce operando: perna numa altura qualquer, trem recolhido
        L = self._rand(n, *cfg.start_leg_range)
        k = ((L - self.pose_L[0]) / (self.pose_L[1] - self.pose_L[0])).clamp(0, len(self.pose_L) - 1.001)
        i0 = k.long(); fr = (k - i0)[:, None]
        row = self.pose_tab[i0] * (1 - fr) + self.pose_tab[i0 + 1] * fr
        oper = ~parked
        for s in ("xp", "xm"):
            hp, kp_, hm, km_ = self.leg_joints[s]
            q[oper, hp], q[oper, kp_], q[oper, hm], q[oper, km_] = row[oper, 0], row[oper, 1], -row[oper, 0], row[oper, 2]
        q[oper, self.q_gear[0]], q[oper, self.q_gear[1]] = math.radians(-100.0), math.radians(100.0)
        z_oper = -row[:, 3] + 0.10751 + 0.004
        q[:, 2] = ground + torch.where(parked, self.parked[2], z_oper)
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
        grav = -R[:, 2, :]                                   # gravidade no frame da base (unitaria)
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
        # critic: estado real + mapa de alturas no frame de heading + parametros sorteados
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
        self.res.copy_(self.actions * self.res_scale)
        # empurroes: muda a velocidade horizontal da base
        self.push_timer -= self.dt
        pushed = (self.push_timer <= 0).nonzero().flatten()
        if len(pushed):
            self.qvel[pushed, 0:2] += self._rand(len(pushed) * 2, -cfg.push_speed, cfg.push_speed).view(-1, 2)
            self.push_timer[pushed] = self._rand(len(pushed), *cfg.push_interval_s)
        wp.capture_launch(self.graph, stream=self.stream)
        # um mundo que explodiu numericamente (raro; ex.: falha do EPA no contato com o heightfield) vira
        # queda: troca o estado por um finito antes de qualquer conta e reinicia no fim do passo
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
        # trem de pouso tocando o chao com o robo operando (o "sentar no trem" do treino anterior)
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
        # queda pela geometria (so pneus e pontas do trem colidem com o terreno): inclinou demais, o chassi
        # desceu abaixo da altura minima de operacao, ou a protecao do controlador cortou os torques
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
        # curriculum: sobe de nivel se aguentou o episodio seguindo o comando; desce se caiu
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
