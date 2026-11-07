"""Teacher environment for the autonomous skills: stairs up/down, curbs, gaps, crouching under bars.

Same structure as env.py (MuJoCo Warp worlds, the base controller in the CUDA graph, residual action, RSL-RL),
with what the skills need:
  * terrain: skill_terrain.py (box stairs/curbs/islands/bars with sharp edges), collision="full" (chassis box and
    leg capsules collide too, so hitting a step edge or a bar is physical);
  * action: 12 values with more authority (leg length over the whole range, leg angles, wheel torques), a force
    along each leg, a jump trigger (> 0.5 starts the controller's jump primitive: push, then tuck the legs) and
    the jump's shape (extension 0.40-0.52 m, push 150-300 N, tuck length 0.18-0.32 m);
  * drive: joint torque limit drawn per robot (18-25 N.m) with the motor torque-speed droop;
  * commands: forward speed + a heading (the yaw-rate command comes from the heading error, as the skill
    planner will do on the robot) + a preferred leg length that only matters where nothing is overhead;
  * observations (teacher = privileged): proprioception + history, true velocity, torque limit, friction, mass,
    a perfect height scan and a ceiling scan around the robot. The student (later) replaces the scans by the
    D435i elevation map and the privileged values by a longer history;
  * curriculum: per robot level 0-5, up after crossing the tile, down after a fall or getting stuck.
"""
import math
from dataclasses import dataclass, field

import mujoco
import torch
import warp as wp
from tensordict import TensorDict

import controller as C
import skill_terrain as ST
import terrain as T
from env import EnvCfg, PrimusWarpEnv

W0 = 24 * 130 * 2 * math.pi / 60 / 9      # joint no-load speed [rad/s]: 130 rpm/V at 24 V through 9:1
EASY = ("flat", "rough", "slope")


@dataclass
class SkillCfg(EnvCfg):
    episode_s: float = 20.0               # a whole flight of stairs fits
    # residual = action x scale + offset: [dL_xp, dL_xm] m, [dth_xp, dth_xm] rad, [dT_xp, dT_xm] N.m,
    # [dF_xp, dF_xm] N, jump trigger, jump extension [m], push [N], tuck length [m]
    residual_scale: tuple = (0.17, 0.17, 0.5, 0.5, 4.0, 4.0, 150.0, 150.0, 1.0, 0.06, 75.0, 0.07)
    residual_offset: tuple = (0.0,) * 9 + (0.46, 225.0, 0.25)
    p_climb: float = 0.7                  # stairs tiles: share of robots starting at the foot (else on top)
    history: int = 3
    # commands: forward speed, heading (-> yaw rate), preferred leg length
    speed_range: tuple = (0.3, 1.2)
    back_range: tuple = (-0.8, -0.2)
    p_back: float = 0.15                  # backwards only on the easy tiles
    p_zero_speed: float = 0.1
    p_turn: float = 0.3                   # chance that a resample turns the heading (easy tiles only)
    heading_gain: float = 1.5
    max_yaw: float = 1.5
    leg_range: tuple = (0.22, 0.45)
    p_parked_start: float = 0.1
    push_speed: float = 0.3
    # tile type probabilities, in skill_terrain.TYPES order
    type_weights: tuple = (0.08, 0.08, 0.09, 0.25, 0.17, 0.17, 0.16)
    torque_range: tuple = (18.0, 25.0)    # joint torque limit [N.m]
    w0_scale: tuple = (0.9, 1.1)          # x W0
    level_up_distance: float = 2.5        # [m] from the spawn point at the end of the episode
    rew: dict = field(default_factory=lambda: dict(
        track_speed=3.0, track_yaw=1.0, track_leg=0.3, upright=-2.0, ang_vel_xy=-0.05, lin_vel_z=-0.2,
        action=-0.01, action_rate=-0.05, torque=-1e-4, gear_contact=-2.0, body_contact=-3.0, impact=-1.0,
        stand_still=-1.0, climb=40.0, termination=-10.0))


class SkillEnv(PrimusWarpEnv):
    collision = "full"
    nconmax, njmax = 48, 200

    def __init__(self, cfg: SkillCfg = SkillCfg(), device="cuda:0", seed=0, terrain=None):
        super().__init__(cfg, device=device, seed=seed, terrain=terrain)
        # drive limits per robot (the graph reads the same array)
        lo, hi = cfg.torque_range
        g = torch.Generator(device=self.device).manual_seed(seed + 7)
        u = lambda a, b: a + (b - a) * torch.rand(self.num_envs, generator=g, device=self.device)
        self.drv[:, 0] = u(lo, hi)
        self.drv[:, 1] = W0 * u(*cfg.w0_scale)
        self._compute_obs()

    # ---------------- setup ----------------
    def _make_terrain(self, seed):
        return ST.make_terrain(n_levels=self.cfg.terrain_levels, seed=seed)

    def _patch_model(self, mjm):
        # actuator range = the largest torque drawn; the kernel clamps each robot to its own limit
        mjm.actuator_ctrlrange[:6] = [-self.cfg.torque_range[1] / 9.0, self.cfg.torque_range[1] / 9.0]

    def _init_buffers(self):
        N, dev, m = self.num_envs, self.device, self.mjm
        z = lambda *s: torch.zeros(*s, device=dev)
        ter = self.terrain
        self.G = torch.tensor(ter["ground"], device=dev)
        self.Cg = torch.tensor(ter["ceiling"], device=dev).nan_to_num(posinf=10.0).clamp(max=10.0)
        self.type_w = torch.tensor(self.cfg.type_weights, device=dev)
        self.goal_heading, self.last_vz, self.ep_cmd_dist = z(N), z(N), z(N)
        self.ground_best, self.ground0 = z(N), z(N)       # highest ground reached on the wheels, ground at spawn
        self.drv = None
        self.frame_dim += 1                               # + jump phase
        self.hist = z(N, self.cfg.history, self.frame_dim)
        # height scan: 31 forward x 5 lateral points (5 cm forward, 15 cm lateral); ceiling: 15 x 3 (10 cm)
        f = torch.arange(-0.30, 1.2001, 0.05, device=dev)
        l = torch.tensor([-0.30, -0.15, 0.0, 0.15, 0.30], device=dev)
        self.hscan = torch.stack(torch.meshgrid(l, f, indexing="xy"), -1).reshape(-1, 2)    # (lateral, forward)
        f = torch.arange(-0.2, 1.2001, 0.1, device=dev)
        l = torch.tensor([-0.2, 0.0, 0.2], device=dev)
        self.cscan = torch.stack(torch.meshgrid(l, f, indexing="xy"), -1).reshape(-1, 2)
        # contact category of every geom: 0 world, 1 tire, 2 gear tip, 3 chassis, 4 leg
        cat = torch.zeros(m.ngeom, dtype=torch.long, device=dev)
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if m.geom_bodyid[i] == 0:
                continue
            cat[i] = 1 if name.endswith("_tire") else 2 if name.endswith("_tip") else 3 if name == "base_box" else 4
        self.geom_cat = cat
        self.world_geom = torch.tensor([m.geom_bodyid[i] == 0 for i in range(m.ngeom)], device=dev)

    # ---------------- terrain queries ----------------
    def _sample_ground(self, xy):
        return T.sample_heights(self.G, self.terrain["size_x"], self.terrain["size_y"], xy)

    def _sample_ceiling(self, xy):
        ny, nx = self.Cg.shape
        sx, sy = self.terrain["size_x"], self.terrain["size_y"]
        j = ((xy[..., 0] + sx / 2) / sx * (nx - 1)).round().long().clamp(0, nx - 1)
        i = ((xy[..., 1] + sy / 2) / sy * (ny - 1)).round().long().clamp(0, ny - 1)
        return self.Cg[i, j]

    def _disc(self, xy, radius=0.35):
        a = torch.arange(8, device=self.device) * (math.pi / 4)
        offs = torch.cat([torch.zeros(1, 2, device=self.device),
                          radius * torch.stack([torch.cos(a), torch.sin(a)], -1)])
        return self._sample_ground(xy[..., None, :] + offs)

    def _ground(self, xy, radius=0.35):
        return self._disc(xy, radius).amax(-1)

    @staticmethod
    def _heading_of(d):
        """heading angle whose forward direction (-sin h, cos h) is the vector d"""
        return torch.atan2(-d[..., 0], d[..., 1])

    def _world_points(self, pts):
        """(lateral, forward) points in the heading frame -> world xy [N, P, 2]"""
        h = self.heading
        c, s = torch.cos(h)[:, None], torch.sin(h)[:, None]
        px = self.qpos[:, 0:1] + c * pts[:, 0] - s * pts[:, 1]
        py = self.qpos[:, 1:2] + s * pts[:, 0] + c * pts[:, 1]
        return torch.stack([px, py], -1)

    # ---------------- episode start ----------------
    def _draw_types(self, ids):
        if self.cfg.random_tile_type:
            self.ttype[ids] = torch.multinomial(self.type_w, len(ids), replacement=True, generator=self.gen)

    def _spawn(self, ids, K=6):
        """Start points by tile type (K candidates each; the first one on flat ground is used)."""
        n, dev = len(ids), self.device
        types = self.terrain["types"]
        center = self.tile_centers[self.level[ids], self.ttype[ids]]
        r = lambda lo, hi: self._rand(n * K, lo, hi).view(n, K)
        local = torch.stack([r(-1.0, 1.0), r(-1.0, 1.0)], -1)
        yaw = r(-math.pi, math.pi)
        tt = self.ttype[ids][:, None].expand(n, K)
        # curbs: ring between the centre platform and the outer platforms
        rad, ang = r(1.1, 1.5), r(-math.pi, math.pi)
        ring = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
        local = torch.where((tt == types.index("curbs"))[..., None], ring, local)
        # gaps: on the centre island
        local = torch.where((tt == types.index("gaps"))[..., None], 0.5 * local, local)
        # stairs: half on the top platform (descend), half at the foot facing the stairs (climb)
        stairs = tt == types.index("stairs")
        side = torch.randint(0, 4, (n, K), generator=self.gen, device=dev).float() * (math.pi / 2)
        d, lat = r(ST.STAIR_HALF + 0.25, 3.75), r(-2.0, 2.0)
        cs_, sn = torch.cos(side), torch.sin(side)
        foot = torch.stack([d * cs_ - lat * sn, d * sn + lat * cs_], -1)
        climb = stairs & (r(0.0, 1.0) < self.cfg.p_climb)
        local = torch.where(climb[..., None], foot, local)
        yaw = torch.where(climb, self._heading_of(-torch.stack([cs_, sn], -1)) + r(-0.3, 0.3), yaw)
        xy = center[:, None, :] + local
        disc = self._disc(xy)
        flat = (disc.amax(-1) - disc.amin(-1)) < 0.01
        k = flat.float().argmax(-1)            # first flat candidate (0 if none)
        a = torch.arange(n, device=dev)
        return xy[a, k], yaw[a, k]

    def _reset(self, ids):
        if len(ids) == 0:
            return
        super()._reset(ids)
        q = self.qpos[ids, 3:7]
        R = self._quat_R(q)
        self.goal_heading[ids] = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        self.last_vz[ids] = 0.0
        self.ep_cmd_dist[ids] = 0.0
        g0 = self._sample_ground(self.qpos[ids, 0:2])
        self.ground0[ids], self.ground_best[ids] = g0, g0

    @staticmethod
    def _quat_R(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)

    def _resample_commands(self, ids):
        n, cfg, dev = len(ids), self.cfg, self.device
        easy = torch.zeros(n, dtype=torch.bool, device=dev)
        for name in EASY:
            easy |= self.ttype[ids] == self.terrain["types"].index(name)
        u = lambda: torch.rand(n, generator=self.gen, device=dev)
        speed = self._rand(n, *cfg.speed_range)
        speed = torch.where(easy & (u() < cfg.p_back), self._rand(n, *cfg.back_range), speed)
        speed[u() < cfg.p_zero_speed] = 0.0
        self.cmd[ids, 0] = speed
        self.cmd[ids, 2] = self._rand(n, *cfg.leg_range)
        self.cmd[ids, 3] = 0.0
        self.cmd[ids, 4] = 0.0
        self.cmd[ids, 5] = 1.5
        turn = easy & (u() < cfg.p_turn) & (self.episode_length_buf[ids] > 0)
        self.goal_heading[ids] += torch.where(turn, self._rand(n, -1.5, 1.5), torch.zeros(n, device=dev))
        self.cmd_timer[ids] = self._rand(n, *cfg.cmd_interval_s)

    def _pre_physics(self):
        R = self._quat_R(self.qpos[:, 3:7])
        heading = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        err = torch.atan2(torch.sin(self.goal_heading - heading), torch.cos(self.goal_heading - heading))
        self.cmd[:, 1] = (self.cfg.heading_gain * err).clamp(-self.cfg.max_yaw, self.cfg.max_yaw)

    # ---------------- observations ----------------
    def _compute_obs(self):
        if self.drv is None:                           # base constructor, before the drive array exists
            self.drv = wp.to_torch(self.ctl.drv)
        cfg, o, N = self.cfg, self.out, self.num_envs
        R = self._quat_R(self.qpos[:, 3:7])
        self.R = R
        self.heading = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        grav = -R[:, 2, :]
        gyro = self.qvel[:, 3:6]
        legs = torch.stack([o[:, C.O["L_xp"]], o[:, C.O["L_xm"]], o[:, C.O["th_xp"]], o[:, C.O["th_xm"]],
                            o[:, C.O["dL_xp"]], o[:, C.O["dL_xm"]], o[:, C.O["dth_xp"]], o[:, C.O["dth_xm"]]], -1)
        wheels = self.qvel[:, self.qv_wheel] * 0.1
        dx = o[:, C.O["dx"]:C.O["dx"] + 1]
        oper = (self.cs[:, C.S["mode"]] == C.OPERATING).float()[:, None]
        if cfg.obs_noise:
            nz = lambda t, s: t + s * torch.randn(t.shape, generator=self.gen, device=self.device)
            grav, gyro, legs, wheels, dx = nz(grav, 0.02), nz(gyro, 0.05), nz(legs, 0.01), nz(wheels, 0.05), nz(dx, 0.03)
        jump = o[:, C.O["jump_phase"]:C.O["jump_phase"] + 1] * 0.5
        frame = torch.cat([grav, gyro, legs, wheels, dx, oper, self.cmd[:, 0:3], self.actions, jump], -1)
        self.hist = torch.cat([self.hist[:, 1:], frame[:, None]], 1)
        # privileged
        self.v_body = torch.einsum("nji,nj->ni", R, self.qvel[:, 0:3])
        z = self.qpos[:, 2:3]
        self.base_height = self.qpos[:, 2] - self._sample_ground(self.qpos[:, 0:2])
        hs = self._sample_ground(self._world_points(self.hscan))
        scan = (z - hs - 0.4).clamp(-1, 1)
        self.clear = (self._sample_ceiling(self._world_points(self.cscan)) - z).clamp(-0.5, 1.0)
        priv = torch.cat([self.v_body, self.base_height[:, None], (self.drv[:, 0:1] - 20.0) / 5.0,
                          self.dr_friction[:, None], self.dr_mass[:, None],
                          o[:, C.O["grounded"]:C.O["grounded"] + 1]], -1)
        x = torch.cat([self.hist.reshape(N, -1), priv, scan, self.clear], -1)
        self.obs = TensorDict({"policy": x, "critic": x}, batch_size=[N])

    # ---------------- rewards ----------------
    def _contacts(self):
        """Per robot: (gear tip, chassis, leg) touching the world."""
        N = self.num_envs
        flags = torch.zeros(N, 5, dtype=torch.bool, device=self.device)
        n = min(int(self.nacon[0].item()), self.con_geom.shape[0])
        if n:
            g, wid = self.con_geom[:n].long(), self.con_world[:n].long()
            w0, w1 = self.world_geom[g[:, 0]], self.world_geom[g[:, 1]]
            robot = torch.where(w0, g[:, 1], g[:, 0])
            cat = self.geom_cat[robot]
            ok = w0 ^ w1
            flags[wid[ok], cat[ok]] = True
        return flags[:, 2], flags[:, 3], flags[:, 4]

    def _rewards_and_dones(self):
        cfg, o, R = self.cfg, self.out, self.R
        w = cfg.rew
        oper = self.cs[:, C.S["mode"]] == C.OPERATING
        heading = torch.stack([R[:, 0, 1], R[:, 1, 1]], -1)
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_fwd = (self.qvel[:, 0:2] * heading).sum(-1)
        yaw_rate = (R @ self.qvel[:, 3:6, None])[:, 2, 0]
        speed_err = (v_fwd - self.cmd[:, 0]).square()
        yaw_err = (yaw_rate - self.cmd[:, 1].clamp(-o[:, C.O["yaw_limit"]], o[:, C.O["yaw_limit"]])).square()
        leg = 0.5 * (o[:, C.O["L_xp"]] + o[:, C.O["L_xm"]])
        grav_b = -R[:, 2, :]
        gear_hit, base_hit, leg_hit = self._contacts()
        open_above = self.clear.view(self.num_envs, -1).amin(-1) >= 0.99
        vz = self.qvel[:, 2]
        impact = (vz - self.last_vz - 0.3).clamp_min(0).square()     # sudden upward velocity change = landing
        # climb: each new height of the ground under the wheels (on the ground: jumping in place earns nothing)
        g = self._sample_ground(self.qpos[:, 0:2])
        on_ground = (o[:, C.O["grounded"]] > 0.5) & oper
        gain = torch.where(on_ground, (g - self.ground_best).clamp(0.0, 0.25), torch.zeros_like(g))
        self.ground_best = torch.where(on_ground, torch.maximum(self.ground_best, g), self.ground_best)
        self.last_vz = vz.clone()
        terms = dict(
            track_speed=torch.exp(-speed_err / 0.1) * oper,
            track_yaw=torch.exp(-yaw_err / 0.2) * oper,
            track_leg=torch.exp(-(leg - self.cmd[:, 2]).square() / 0.04 ** 2) * oper * open_above,
            upright=grav_b[:, 0].square() + grav_b[:, 1].square(),
            ang_vel_xy=self.qvel[:, 3:5].square().sum(-1),
            lin_vel_z=vz.square(),
            action=self.actions.square().sum(-1),
            action_rate=(self.actions - self.last_actions).square().sum(-1),
            torque=self.out[:, C.O["tau_hYp_xp"]:C.O["tau_w_xm"] + 1].square().sum(-1),
            gear_contact=(gear_hit & oper).float(),
            body_contact=leg_hit.float(),
            impact=impact,
            stand_still=v_fwd.abs() * (self.cmd[:, 0] == 0) * oper,      # hold still when told to (no rocking)
            climb=gain / self.dt,                                         # x dt below = metres gained
        )
        # falls: tilted, chassis touching a box feature or too low over the heightfield (it does not collide
        # with it), down a gap, or the controller protection cut the torques
        tilt = grav_b[:, 2] > -math.cos(math.radians(50.0))
        low = self.qpos[:, 2] - T.sample_heights(self.H, self.terrain["size_x"], self.terrain["size_y"],
                                                 self.qpos[:, 0:2]) < 0.2
        in_pit = self.qpos[:, 2] < 0.12                     # every walkable surface is at z >= -0.03
        fell = tilt | base_hit | low | in_pit | (o[:, C.O["protect"]] > 0.5) | self.bad
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
        self.ep_cmd_dist += self.cmd[:, 0].abs() * self.dt
        return rew, fell, timeout & ~fell

    def _log_and_curriculum(self, ids, terminated, timeout):
        log = self.extras["log"] = {}
        if len(ids) == 0:
            return
        for k, v in self.ep_sums.items():
            log[f"Episode_Reward/{k}"] = (v[ids] / self.cfg.episode_s).mean()
        dist = (self.qpos[ids, 0:2] - self.spawn_xy[ids]).norm(dim=-1)
        err = self.ep_speed_err[ids] / self.ep_steps_oper[ids].clamp_min(1)
        log["Metrics/speed_error_mps"] = err.mean()
        log["Metrics/fell"] = terminated[ids].float().mean()
        log["Metrics/distance_m"] = dist.mean()
        log["Metrics/episode_s"] = (self.episode_length_buf[ids].float() * self.dt).mean()
        log["Metrics/nonfinite_resets"] = self.bad[ids].float().sum()
        if self.cfg.curriculum:
            # up after crossing the tile; down after a fall or when it moved less than 30% of the commanded way
            up = timeout[ids] & (dist > self.cfg.level_up_distance)
            rise = torch.tensor(ST.STAIR_RISE, device=self.device)[self.level[ids]]
            stairs = self.ttype[ids] == self.terrain["types"].index("stairs")
            up |= stairs & ~terminated[ids] & (self.ground_best[ids] - self.ground0[ids] >= 3 * rise - 0.01)  # 3 steps
            down = terminated[ids] | (dist < 0.3 * self.ep_cmd_dist[ids])
            lv = self.level[ids] + up.long() - (down & ~up).long()
            top = self.cfg.terrain_levels - 1
            lv = torch.where(lv > top, torch.randint(0, top + 1, lv.shape, generator=self.gen, device=self.device), lv)
            self.level[ids] = lv.clamp(0, top)
            log["Curriculum/level_mean"] = self.level.float().mean()
        for t, name in enumerate(self.terrain["types"]):
            sel = self.ttype[ids] == t
            if sel.any():
                log[f"Fell_by_terrain/{name}"] = terminated[ids][sel].float().mean()
                log[f"Level_by_terrain/{name}"] = self.level[self.ttype == t].float().mean()
