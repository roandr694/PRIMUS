"""v12 environment: one always-on residual policy over the LQR, with an assist and a skill regime.

Same structure as env.py (MuJoCo Warp worlds, the base controller in the CUDA graph, residual action, RSL-RL).
The interface (action pairs, observation frame, skill table, remote envelope) is control/policy_interface.py.

Regimes (the `skill` flag is in the observation; docs/v12_plan.md):
  assist   everyday driving with the remote: full command envelope (speed +-1.8 m/s, raw turn rate +-1.8 rad/s,
           height, body pitch, the three responsiveness levels), strong pushes, slippery ground. The trigger is
           ignored by the environment (as the robot-side wrapper does) and the continuous actions are scaled down
           (ASSIST_AUTHORITY). Rewards ask it to ride at least as well as the LQR (height, pitch rate, power,
           residual size, symmetric stance) and to fall less.
  skill    what the planner does after the pilot presses A: forward speed and a heading across the feature.
           Episodes on feature tiles start in skill (or hold in assist for 0.5-2.5 s first), switch back to assist
           when an open flight is climbed, and a share is cancelled half way (flag off, speed 0: hold on the step).

Terrain: skill_terrain.py (box stairs/curbs/islands/bars with sharp edges, variants per tile, walled stairwells,
`bumps` for the assist), collision="full" (chassis and legs collide too). Episodes on flat / rough / slope /
bumps are assist episodes; the others are skill episodes.

Observation (teacher = privileged): 3 frames of policy_interface.FRAME_NAMES, then privileged values (8), a
perfect height scan (155) and a ceiling scan (45). Mirror symmetry: mirror.py.

Plant and sensor model per robot: torque limit with the motor torque-speed droop, torque lag, battery power
limit, action delay, gearbox friction, attitude-estimate bias and hip encoder offsets (read by the controller).
Curriculum: a level per robot and per terrain type; a share of episodes at a random level.
"""
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import torch
import warp as wp
from tensordict import TensorDict

import controller as C
import skill_terrain as ST
import terrain as T
from env import EnvCfg, PrimusWarpEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "control"))
import policy_interface as PI  # noqa: E402

W0 = 24 * 130 * 2 * math.pi / 60 / 9      # joint no-load speed [rad/s]: 130 rpm/V at 24 V through 9:1
ASSIST_TYPES = ("flat", "rough", "slope", "bumps")
STAIRS = ("stairs", "stairs_wide", "stairs_long", "stairs_narrow")
SKILL_OF_TYPE = {"stairs": "stairs_up", "stairs_wide": "stairs_up", "stairs_long": "stairs_up",
                 "stairs_narrow": "stairs_up", "curbs": "curb_up", "gaps": "gap", "gaps_wide": "gap_wide",
                 "hurdles": "hurdle", "overhang": "overhang"}


@dataclass
class SkillCfg(EnvCfg):
    episode_s: float = 20.0               # a whole flight of stairs fits
    residual_scale: tuple = PI.SKILL_SCALE
    residual_offset: tuple = PI.OFFSET
    p_climb: float = 0.5                  # stairs tiles: share of robots starting at the foot (else on top)
    history: int = 3
    # assist commands (the remote): policy_interface.TRAIN_ENVELOPE
    assist_speed: tuple = PI.TRAIN_ENVELOPE["speed"]
    assist_yaw: tuple = PI.TRAIN_ENVELOPE["yaw_rate"]
    p_zero_speed: float = 0.2
    p_zero_yaw: float = 0.35
    p_pitch: float = 0.3                  # assist: share of commands with a body pitch (the planner looking down)
    cmd_interval_s: tuple = (2.0, 6.0)
    leg_range: tuple = PI.TRAIN_ENVELOPE["leg"]
    # skill commands (the planner): speed from policy_interface.SKILLS per feature, heading hold
    eval_speed: tuple = None              # evaluation: one speed range for every feature instead of the table
    skill_leg_range: tuple = (0.22, 0.45)
    p_skill_pitch: float = 0.5
    heading_gain: float = 1.5
    max_yaw: float = 1.5
    p_flag_late: float = 0.35             # skill tiles: hold in assist first, then the skill starts
    flag_delay_s: tuple = (0.5, 2.5)
    p_cancel: float = 0.12                # skill tiles: the pilot presses B half way (hold where it is)
    cancel_after_s: tuple = (1.5, 9.0)
    hold_after_finish_s: float = 3.0      # an open flight climbed: hold on top in assist, then the episode ends
    p_parked_start: float = 0.05
    # pushes
    push_speed: float = 1.0               # assist: +-1.0 m/s per axis
    push_speed_skill: float = 0.4
    push_interval_s: tuple = (2.0, 6.0)
    # friction drawn per episode (tyres and ground)
    friction_assist: tuple = (0.3, 1.25)
    friction_skill: tuple = (0.5, 1.25)
    # tile type probabilities, in skill_terrain.TYPES order (assist tiles: 0.5 in total)
    type_weights: tuple = (0.12, 0.12, 0.10, 0.07, 0.05, 0.06, 0.04, 0.03, 0.06, 0.05, 0.05, 0.09, 0.16)
    terrain_variants: bool = True
    rise_jitter: float = 0.015            # +- per step [m]: 4/7/10... cm levels overlap, no untrained band (eval: 0.01)
    # refusing: stalled with the flag on and a speed target (v12: stood against 6-7 cm risers, no hop)
    stall_speed: float = 0.1              # [m/s] forward speed below which the robot counts as stalled
    stall_grace_s: float = 0.5            # stalled this long before the cost starts
    stall_end_s: float = 5.0              # stalled this long: the episode ends as a refusal
    # curriculum: a level per robot and type
    init_max_level: int = 5
    p_random_level: float = 0.2
    level_up_distance: float = 2.5        # [m] from the spawn point at the end of the episode
    # v13 (2026-11-18 analysis): up to v12b a robot drove on past its tile for the whole 20 s episode, so about half
    # of every gap/curb/overhang/hurdle episode and most of its falls were on the neighbouring tiles (another type or
    # level), which drove the type's curriculum and made stopping look safe. Leaving the own tile now truncates the
    # episode (a time-out, bootstrapped: no bonus, no penalty) and counts as crossing it.
    tile_truncate: bool = True
    # levels beyond the plant (experiments.md §7, v13 analysis): the curriculum does not go above these
    max_level_by_type: tuple = (("gaps_wide", 3), ("hurdles", 4))
    # overhang tiles: the policy's height authority is -0.17 m, so a taller command cannot crouch under the 0.42 m bar
    overhang_leg_max: float = 0.36
    # plant and sensor model
    torque_range: tuple = (18.0, 25.0)    # joint torque limit [N.m]
    w0_scale: tuple = (0.8, 1.05)         # x W0 (battery sag under load lowers the no-load speed)
    torque_lag_s: tuple = (0.001, 0.004)  # current loop + CAN-FD
    power_limit_w: tuple = (500.0, 900.0) # battery (24 V 3 Ah drill pack) for all motors together
    action_delay_steps: tuple = (0, 10)   # policy inference + communication [ms]
    joint_friction: tuple = (0.05, 0.3)   # gearbox friction at the joint [N.m] (hips and wheels)
    attitude_bias: float = math.radians(1.0)     # +- per robot, pitch and roll, controller and observation
    hip_offset: float = math.radians(0.5)        # +- per hip encoder
    extension_speed_free: float = 4.5     # [m/s] leg extension speed with no cost (the motor model limits the rest)
    leg_force_free: float = 400.0         # [N] estimated contact force per leg with no cost (landings)
    # rewards: skill regime (rew) and assist regime (rew_assist); per second, except the events (EVENTS)
    rew: dict = field(default_factory=lambda: dict(
        track_speed=3.0, track_yaw=1.0, track_leg=0.3, leg_error=0.0, upright=-2.0, ang_vel_xy=-0.05,
        lin_vel_z=-0.2, action=-0.01, action_diff=-0.05, action_rate=-0.05, torque=-1e-4, gear_contact=-2.0,
        body_contact=-3.0, impact=-1.0, stand_still=-1.0, climb=40.0, hip_limit=-20.0, leg_split=-5.0,
        power=-0.002, extension_speed=-0.5, action_excess=-0.2, wheel_levels=-1.0, wheel_slip=-0.5,
        leg_load=-0.05, bottom_out=-1.0, stall=-2.0,
        termination=-15.0, fall_height=-20.0, jump_start=-0.5, finish=10.0, refuse=-6.0))
    rew_assist: dict = field(default_factory=lambda: dict(
        track_speed=3.0, track_yaw=1.5, track_leg=1.0, leg_error=-200.0, upright=-5.0, ang_vel_xy=-1.0,
        lin_vel_z=-1.0, action=-0.3, action_diff=-0.5, action_rate=-0.2, torque=-1e-4, gear_contact=-2.0,
        body_contact=-3.0, impact=-1.0, stand_still=-2.0, climb=0.0, hip_limit=-20.0, leg_split=-20.0,
        power=-0.01, extension_speed=-0.5, action_excess=-0.2, wheel_levels=0.0, wheel_slip=-1.0,
        leg_load=-0.05, bottom_out=-1.0, stall=0.0,
        termination=-25.0, fall_height=-20.0, jump_start=0.0, finish=0.0, refuse=0.0))


EVENTS = ("termination", "fall_height", "jump_start", "finish", "refuse")       # not multiplied by dt


class SkillEnv(PrimusWarpEnv):
    collision = "full"
    nconmax, njmax = 48, 200

    def __init__(self, cfg: SkillCfg = SkillCfg(), device="cuda:0", seed=0, terrain=None):
        super().__init__(cfg, device=device, seed=seed, terrain=terrain)
        # plant and sensor model per robot (the graph reads the same array)
        lo, hi = cfg.torque_range
        g = torch.Generator(device=self.device).manual_seed(seed + 7)
        u = lambda a, b, n=self.num_envs: a + (b - a) * torch.rand(n, generator=g, device=self.device)
        self.drv[:, 0] = u(lo, hi)
        self.drv[:, 1] = W0 * u(*cfg.w0_scale)
        self.drv[:, 2] = u(*cfg.torque_lag_s)
        self.drv[:, 3] = u(*cfg.power_limit_w)
        self.drv[:, 4] = torch.randint(cfg.action_delay_steps[0], cfg.action_delay_steps[1] + 1, (self.num_envs,),
                                       generator=g, device=self.device).float()
        for c in (5, 6):
            self.drv[:, c] = u(-cfg.attitude_bias, cfg.attitude_bias)
        for c in range(7, 11):
            self.drv[:, c] = u(-cfg.hip_offset, cfg.hip_offset)
        self._compute_obs()

    # ---------------- setup ----------------
    def _make_terrain(self, seed):
        return ST.make_terrain(n_levels=self.cfg.terrain_levels, seed=seed, variants=self.cfg.terrain_variants,
                               rise_jitter=self.cfg.rise_jitter)

    def _setup_randomization(self):
        super()._setup_randomization()
        import numpy as np
        m, N = self.mjm, self.num_envs
        rng = np.random.default_rng(4321)
        fl = np.tile(m.dof_frictionloss[None], (N, 1)).astype(np.float32)
        dofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{s}")])
                for j in ("hip_Yp", "hip_Ym", "wheel") for s in ("xp", "xm")]
        fl[:, dofs] = rng.uniform(*self.cfg.joint_friction, (N, len(dofs))).astype(np.float32)
        self.mw.dof_frictionloss = wp.array(fl, dtype=float)
        jid = lambda j, s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{s}")
        hips = [(j, s) for j in ("hip_Yp", "hip_Ym") for s in ("xp", "xm")]
        self.hip_q = [int(m.jnt_qposadr[jid(*h)]) for h in hips]
        self.hip_v = [int(m.jnt_dofadr[jid(*h)]) for h in hips]
        self.hip_lo = torch.tensor([m.jnt_range[jid(*h)][0] for h in hips], device=self.device)
        self.hip_hi = torch.tensor([m.jnt_range[jid(*h)][1] for h in hips], device=self.device)
        # per-episode friction: the tyres and every world geom (MuJoCo uses the larger of the pair)
        tires = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{s}_tire") for s in ("xp", "xm")]
        self.fr_geoms = torch.tensor(tires + [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0], device=self.device)
        self.geom_friction = wp.to_torch(self.mw.geom_friction)

    def _patch_model(self, mjm):
        # actuator range = the largest torque drawn; the kernel clamps each robot to its own limit
        mjm.actuator_ctrlrange[:6] = [-self.cfg.torque_range[1] / 9.0, self.cfg.torque_range[1] / 9.0]

    def _init_buffers(self):
        N, dev, m, cfg = self.num_envs, self.device, self.mjm, self.cfg
        z = lambda *s: torch.zeros(*s, device=dev)
        ter = self.terrain
        types = ter["types"]
        self.G = torch.tensor(ter["floor"], device=dev)            # walkable surface (no walls)
        self.Gs = torch.tensor(ter["ground"], device=dev)          # what a camera sees (walls included)
        self.Cg = torch.tensor(ter["ceiling"], device=dev).nan_to_num(posinf=10.0).clamp(max=10.0)
        self.type_w = torch.tensor(cfg.type_weights, device=dev)
        assert len(cfg.type_weights) == len(types), (len(cfg.type_weights), types)
        is_t = lambda names: torch.tensor([t in names for t in types], device=dev)
        self.assist_type, self.stairs_type = is_t(ASSIST_TYPES), is_t(STAIRS)
        lo = [PI.SKILLS[SKILL_OF_TYPE[t]]["train_speed"][0] if t in SKILL_OF_TYPE else 0.0 for t in types]
        hi = [PI.SKILLS[SKILL_OF_TYPE[t]]["train_speed"][1] if t in SKILL_OF_TYPE else 0.0 for t in types]
        self.speed_lo, self.speed_hi = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)
        self.flight = z(N)                                         # stairs_narrow: +1 climbing, -1 descending, 0 other
        self.goal_heading, self.last_vz, self.ep_cmd_dist = z(N), z(N), z(N)
        self.stall_t = z(N)
        self.refused_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self.since_jump = z(N) + 10.0                     # seconds since the jump primitive was last active
        self.jphase_prev = z(N)
        self.ground_best, self.ground0 = z(N), z(N)       # highest ground reached on the wheels, ground at spawn
        # regime: skill flag and its schedule (seconds into the episode)
        self.skill = z(N)
        self.skill_tile = torch.zeros(N, dtype=torch.bool, device=dev)
        self.t_on, self.t_cancel, self.t_finish = z(N), z(N) + 1e9, z(N)
        self.cancelled, self.finished = (torch.zeros(N, dtype=torch.bool, device=dev) for _ in range(2))
        self.skill_speed, self.ramp = z(N), z(N)
        self.ep_skill_steps, self.ep_jumps_assist, self.ep_split = z(N), z(N), z(N)
        # curriculum: a level per robot and per type
        self.lvl_tab = torch.randint(0, cfg.init_max_level + 1, (N, len(types)), generator=self.gen, device=dev)
        top = torch.full((len(types),), cfg.terrain_levels - 1, dtype=torch.long, device=dev)
        for name, lv in cfg.max_level_by_type:
            if name in types:
                top[types.index(name)] = min(lv, cfg.terrain_levels - 1)
        self.type_top = top
        self.lvl_tab = torch.minimum(self.lvl_tab, top[None])
        self.rand_lv = torch.zeros(N, dtype=torch.bool, device=dev)
        self.drv = None
        self.frame_dim = PI.FRAME_DIM
        self.hist = z(N, cfg.history, self.frame_dim)
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
        self.xpos = wp.to_torch(self.dw.xpos)             # body positions [N, nbody, 3]
        self.wheel_body = [int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_{s}")])
                           for s in ("xp", "xm")]
        self.excess = z(N)                                # how far the raw action went beyond +-1 (squared)
        self.fell_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self.min_len = float(C.make_params(m).min_len)

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

    def wheel_grounds(self):
        """Ground height under each wheel [N, 2]."""
        wxy = self.xpos[:, self.wheel_body, 0:2]
        wxy = torch.where(torch.isfinite(wxy), wxy, self.qpos[:, None, 0:2].expand_as(wxy))
        return self._sample_ground(wxy)

    def wheel_ground(self):
        """Ground height under the lower of the two wheels (what the robot has really climbed)."""
        return self.wheel_grounds().amin(-1)

    # ---------------- episode start ----------------
    def _draw_types(self, ids):
        if self.cfg.random_tile_type:
            self.ttype[ids] = torch.multinomial(self.type_w, len(ids), replacement=True, generator=self.gen)
        if self.cfg.curriculum:
            n, top = len(ids), self.cfg.terrain_levels - 1
            rnd = torch.rand(n, generator=self.gen, device=self.device) < self.cfg.p_random_level
            tt = self.type_top[self.ttype[ids]]
            lv = torch.where(rnd, (torch.rand(n, generator=self.gen, device=self.device) * (tt + 1)).long().clamp(max=tt),
                             self.lvl_tab[ids, self.ttype[ids]])
            self.level[ids], self.rand_lv[ids] = lv, rnd

    def _spawn(self, ids, K=6):
        """Start points by tile type (K candidates each; the first one on flat ground is used)."""
        n, dev = len(ids), self.device
        types = self.terrain["types"]
        center = self.tile_centers[self.level[ids], self.ttype[ids]]
        r = lambda lo, hi: self._rand(n * K, lo, hi).view(n, K)
        yaw_tol = math.radians(PI.APPROACH_DEG)
        local = torch.stack([r(-1.0, 1.0), r(-1.0, 1.0)], -1)
        yaw = r(-math.pi, math.pi)
        tt = self.ttype[ids][:, None].expand(n, K)
        # curbs: ring between the centre platform and the outer platforms
        rad, ang = r(1.1, 1.5), r(-math.pi, math.pi)
        ring = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
        local = torch.where((tt == types.index("curbs"))[..., None], ring, local)
        # gaps: on the centre island
        on_island = (tt == types.index("gaps")) | (tt == types.index("gaps_wide")) | (tt == types.index("hurdles"))
        local = torch.where(on_island[..., None], 0.5 * local, local)
        # stairs: half on the top platform (descend), half at the foot facing the stairs (climb)
        stairs = (tt == types.index("stairs")) | (tt == types.index("stairs_wide")) | (tt == types.index("stairs_long"))
        small_top = (tt == types.index("stairs_wide")) | (tt == types.index("stairs_long"))
        local = torch.where(small_top[..., None], 0.4 * local, local)
        side = torch.randint(0, 4, (n, K), generator=self.gen, device=dev).float() * (math.pi / 2)
        d, lat = r(ST.STAIR_HALF + 0.25, 3.75), r(-2.0, 2.0)
        cs_, sn = torch.cos(side), torch.sin(side)
        foot = torch.stack([d * cs_ - lat * sn, d * sn + lat * cs_], -1)
        climb = stairs & (r(0.0, 1.0) < self.cfg.p_climb)
        local = torch.where(climb[..., None], foot, local)
        yaw = torch.where(climb, self._heading_of(-torch.stack([cs_, sn], -1)) + r(-yaw_tol, yaw_tol), yaw)
        # stairs_narrow: in one of the flights, at the foot facing up (+y) or on the top platform facing down
        narrow = tt == types.index("stairs_narrow")
        fl = torch.tensor(ST.NARROW_FLIGHTS, device=dev)[torch.randint(0, len(ST.NARROW_FLIGHTS), (n, K),
                                                                       generator=self.gen, device=dev)]
        up = r(0.0, 1.0) < self.cfg.p_climb
        nx_ = fl[..., 0] + r(-1.0, 1.0) * (fl[..., 1] / 2 - 0.40)
        ny_ = torch.where(up, r(-3.4, -2.95), r(2.1, 3.2))
        local = torch.where(narrow[..., None], torch.stack([nx_, ny_], -1), local)
        yaw = torch.where(narrow, torch.where(up, 0.0, math.pi) + r(-yaw_tol, yaw_tol), yaw)
        xy = center[:, None, :] + local
        disc = self._disc(xy)
        flat = (disc.amax(-1) - disc.amin(-1)) < 0.01
        k = flat.float().argmax(-1)            # first flat candidate (0 if none)
        a = torch.arange(n, device=dev)
        self.flight[ids] = torch.where(narrow[a, k], torch.where(up[a, k], 1.0, -1.0), 0.0)
        return xy[a, k], yaw[a, k]

    def _reset(self, ids):
        if len(ids) == 0:
            return
        n, cfg, dev = len(ids), self.cfg, self.device
        super()._reset(ids)                     # -> _draw_types, _spawn, _resample_commands
        q = self.qpos[ids, 3:7]
        R = self._quat_R(q)
        self.goal_heading[ids] = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        fl = self.flight[ids]                           # in a stairwell the command is along the flight
        self.goal_heading[ids] = torch.where(fl > 0, 0.0, torch.where(fl < 0, math.pi, self.goal_heading[ids]))
        self.last_vz[ids] = 0.0
        self.since_jump[ids] = 10.0
        self.jphase_prev[ids] = 0.0
        self.ep_cmd_dist[ids] = 0.0
        self.stall_t[ids] = 0.0
        g0 = self._sample_ground(self.qpos[ids, 0:2])
        self.ground0[ids], self.ground_best[ids] = g0, g0
        # regime schedule
        u = lambda: torch.rand(n, generator=self.gen, device=dev)
        st = ~self.assist_type[self.ttype[ids]]
        self.skill_tile[ids] = st
        late = st & (u() < cfg.p_flag_late)
        self.t_on[ids] = torch.where(late, self._rand(n, *cfg.flag_delay_s), torch.zeros(n, device=dev))
        cancel = st & (u() < cfg.p_cancel)
        self.t_cancel[ids] = torch.where(cancel, self.t_on[ids] + self._rand(n, *cfg.cancel_after_s),
                                         torch.full((n,), 1e9, device=dev))
        self.skill[ids] = (st & ~late).float()
        self.cancelled[ids] = False
        self.finished[ids] = False
        self.ramp[ids] = 0.0
        self.ep_skill_steps[ids] = 0.0
        self.ep_jumps_assist[ids] = 0.0
        self.ep_split[ids] = 0.0
        # friction per episode
        fr = torch.where(st, self._rand(n, *cfg.friction_skill), self._rand(n, *cfg.friction_assist))
        self.geom_friction[ids[:, None], self.fr_geoms[None, :], 0] = fr[:, None]
        self.dr_friction[ids] = fr
        self._schedule()

    @staticmethod
    def _quat_R(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)

    def _resample_commands(self, ids):
        n, cfg, dev = len(ids), self.cfg, self.device
        u = lambda: torch.rand(n, generator=self.gen, device=dev)
        st = ~self.assist_type[self.ttype[ids]]
        # assist: the remote's envelope
        speed = self._rand(n, *cfg.assist_speed)
        speed = torch.where(u() < cfg.p_zero_speed, torch.zeros_like(speed), speed)
        yaw = self._rand(n, *cfg.assist_yaw)
        yaw = torch.where(u() < cfg.p_zero_yaw, torch.zeros_like(yaw), yaw)
        pitch = torch.where(u() < cfg.p_pitch, self._rand(n, *PI.TRAIN_ENVELOPE["pitch"]), torch.zeros(n, device=dev))
        leg = self._rand(n, *cfg.leg_range)
        # skill: the planner's speed for the feature (applied while the flag is on), heading hold, maybe looking down
        tt = self.ttype[ids]
        lo, hi = (cfg.eval_speed if cfg.eval_speed else (self.speed_lo[tt], self.speed_hi[tt]))
        self.skill_speed[ids] = torch.where(st, lo + (hi - lo) * u(), self.skill_speed[ids])
        spitch = torch.where(u() < cfg.p_skill_pitch, self._rand(n, *PI.TRAIN_ENVELOPE["pitch"]),
                             torch.zeros(n, device=dev))
        levels = torch.tensor(PI.TRAIN_ENVELOPE["accel_levels"], device=dev)
        accel = levels[torch.randint(0, len(levels), (n,), generator=self.gen, device=dev)]
        self.cmd[ids, 0] = torch.where(st, self.cmd[ids, 0], speed)
        self.cmd[ids, 1] = torch.where(st, self.cmd[ids, 1], yaw)
        sleg = self._rand(n, *cfg.skill_leg_range)
        over = tt == self.terrain["types"].index("overhang")
        sleg = torch.where(over, sleg.clamp(max=cfg.overhang_leg_max), sleg)
        self.cmd[ids, 2] = torch.where(st, sleg, leg)
        self.cmd[ids, 3] = 0.0                        # roll: not commanded by the remote
        self.cmd[ids, 4] = 0.0
        self.cmd[ids, 5] = accel
        self.cmd[ids, 6] = torch.where(st, spitch, pitch)
        self.cmd_timer[ids] = self._rand(n, *cfg.cmd_interval_s)

    def _schedule(self):
        """Skill tiles: flag on at t_on, off when cancelled or after an open flight is climbed; the planner's
        speed while on, hold (speed 0, keep the heading) while off. Then the heading -> yaw-rate command."""
        t = self.episode_length_buf.float() * self.dt
        st = self.skill_tile
        self.cancelled |= st & (t >= self.t_cancel)
        on = st & (t >= self.t_on) & ~self.cancelled & ~self.finished
        self.skill = on.float()
        self.cmd[:, 0] = torch.where(st, torch.where(on, self.skill_speed, torch.zeros_like(self.skill_speed)),
                                     self.cmd[:, 0])
        R = self._quat_R(self.qpos[:, 3:7])
        heading = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        err = torch.atan2(torch.sin(self.goal_heading - heading), torch.cos(self.goal_heading - heading))
        hold = (self.cfg.heading_gain * err).clamp(-self.cfg.max_yaw, self.cfg.max_yaw)
        self.cmd[:, 1] = torch.where(st, hold, self.cmd[:, 1])

    def _residual(self, actions):
        return PI.to_sides(actions, self.skill, torch)

    def _push_speed(self, ids):
        return torch.where(self.skill_tile[ids], self.cfg.push_speed_skill, self.cfg.push_speed)

    def step(self, actions):
        # the environment clamps the action to +-1: without a cost on the part beyond it a policy output can drift
        # without limit (v9: the jump trigger reached -50 and could never come back)
        self.excess = (actions.abs() - 1.0).clamp_min(0).square().sum(-1).clamp(max=100.0)
        return super().step(actions)

    def _pre_physics(self):
        self._schedule()
        # the speed target the controller's ramp follows (responsiveness): what the speed reward compares with
        a = PI.ramp_accel(self.cmd[:, 5]) * self.dt
        self.ramp = self.ramp + torch.maximum(torch.minimum(self.cmd[:, 0] - self.ramp, a), -a)

    # ---------------- observations ----------------
    def _compute_obs(self):
        if self.drv is None:                           # base constructor, before the drive array exists
            self.drv = wp.to_torch(self.ctl.drv)
        cfg, o, N = self.cfg, self.out, self.num_envs
        R = self._quat_R(self.qpos[:, 3:7])
        self.R = R
        self.heading = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        grav = -R[:, 2, :]
        grav = grav + torch.stack([self.drv[:, 6], -self.drv[:, 5], torch.zeros_like(self.drv[:, 5])], -1)
        gyro = self.qvel[:, 3:6]
        legs = torch.stack([o[:, C.O["L_xp"]], o[:, C.O["L_xm"]], o[:, C.O["th_xp"]], o[:, C.O["th_xm"]],
                            o[:, C.O["dL_xp"]], o[:, C.O["dL_xm"]], o[:, C.O["dth_xp"]], o[:, C.O["dth_xm"]]], -1)
        wheels = self.qvel[:, self.qv_wheel] * 0.1
        dx = o[:, C.O["dx"]:C.O["dx"] + 1]
        oper = (self.cs[:, C.S["mode"]] == C.OPERATING).float()[:, None]
        if cfg.obs_noise:
            nz = lambda t, s: t + s * torch.randn(t.shape, generator=self.gen, device=self.device)
            grav, gyro, legs, wheels, dx = nz(grav, 0.02), nz(gyro, 0.05), nz(legs, 0.01), nz(wheels, 0.05), nz(dx, 0.03)
        cmds = torch.stack([self.cmd[:, 0], self.cmd[:, 1], self.cmd[:, 2], self.cmd[:, 6],
                            self.cmd[:, 5] / PI.ACCEL_NORM, self.ramp, self.skill], -1)
        jump = o[:, C.O["jump_phase"]:C.O["jump_phase"] + 1] * 0.5
        frame = torch.cat([grav, gyro, legs, wheels, dx, oper, cmds, self.actions, jump], -1)
        self.hist = torch.cat([self.hist[:, 1:], frame[:, None]], 1)
        # privileged
        self.v_body = torch.einsum("nji,nj->ni", R, self.qvel[:, 0:3])
        z = self.qpos[:, 2:3]
        self.base_height = self.qpos[:, 2] - self._sample_ground(self.qpos[:, 0:2])
        hs = T.sample_heights(self.Gs, self.terrain["size_x"], self.terrain["size_y"], self._world_points(self.hscan))
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
        self.contact_fill = int(self.nacon[0].item()) / self.con_geom.shape[0]     # S7: buffer use (1 = overflow)
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
        oper = self.cs[:, C.S["mode"]] == C.OPERATING
        skill = self.skill > 0.5
        heading = torch.stack([R[:, 0, 1], R[:, 1, 1]], -1)
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_fwd = (self.qvel[:, 0:2] * heading).sum(-1)
        yaw_rate = (R @ self.qvel[:, 3:6, None])[:, 2, 0]
        speed_err = (v_fwd - self.ramp).square()
        yaw_err = (yaw_rate - self.cmd[:, 1].clamp(-o[:, C.O["yaw_limit"]], o[:, C.O["yaw_limit"]])).square()
        leg = 0.5 * (o[:, C.O["L_xp"]] + o[:, C.O["L_xm"]])
        grav_b = -R[:, 2, :]
        gear_hit, base_hit, leg_hit = self._contacts()
        open_above = self.clear.view(self.num_envs, -1).amin(-1) >= 0.99
        vz = self.qvel[:, 2]
        impact = (vz - self.last_vz - 0.3).clamp_min(0).square()     # sudden upward velocity change = landing
        grounded = (o[:, C.O["grounded"]] > 0.5) & oper
        # climb: each new height of the ground under both wheels, only on the ground, with the skill flag on and
        # driving forward (jumping in place or climbing against the command earns nothing)
        gw = self.wheel_grounds()
        g = gw.amin(-1)
        gain = torch.where(grounded, (g - self.ground_best).clamp(0.0, 0.25), torch.zeros_like(g))
        self.ground_best = torch.where(grounded, torch.maximum(self.ground_best, g), self.ground_best)
        gain = gain * (skill & (self.cmd[:, 0] > 0.05))
        self.last_vz = vz.clone()
        # the jump primitive (and the 0.3 s after it): the plant model already limits torque, speed and power, so
        # its cost is not punished, only what comes after it (falls, impacts) and a small price per jump
        ph = o[:, C.O["jump_phase"]]
        jumping = ph > 0.5
        jump_start = jumping & (self.jphase_prev < 0.5)
        self.jphase_prev = ph.clone()
        self.since_jump = torch.where(jumping, torch.zeros_like(self.since_jump), self.since_jump + self.dt)
        free = (self.since_jump > 0.3).float()
        a = self.actions
        split = o[:, C.O["th_xp"]] - o[:, C.O["th_xm"]]
        tf = torch.stack([o[:, C.O["tf_xp"]], o[:, C.O["tf_xm"]]], -1)
        Ls = torch.stack([o[:, C.O["L_xp"]], o[:, C.O["L_xm"]]], -1)
        # finishing a feature: the walled flight (episode ends) or an open flight climbed (flag off, hold on top)
        rise = torch.tensor(ST.STAIR_RISE, device=self.device)[self.level]
        need = torch.where(self.ttype == self.terrain["types"].index("stairs_long"), 6.0, 3.0) * rise - 0.01
        y_loc = self.qpos[:, 1] - self.tile_centers[self.level, self.ttype][:, 1]
        sin_p = torch.sin(self.cmd[:, 6])
        terms = dict(
            track_speed=torch.exp(-speed_err / 0.1) * oper,
            track_yaw=torch.exp(-yaw_err / 0.2) * oper,
            track_leg=torch.exp(-(leg - self.cmd[:, 2]).square() / 0.04 ** 2) * oper * open_above,
            leg_error=(leg - self.cmd[:, 2]).square() * oper * open_above,
            upright=grav_b[:, 0].square() + (grav_b[:, 1] + sin_p).square(),     # relative to the pitch command
            ang_vel_xy=self.qvel[:, 3:5].square().sum(-1),
            lin_vel_z=vz.square() * free,
            action=a[:, :8].square().sum(-1),
            action_diff=a[:, list(PI.DIFF_ACTIONS)].square().sum(-1),
            action_rate=(a - self.last_actions).square().sum(-1),
            torque=self.out[:, C.O["tau_hYp_xp"]:C.O["tau_w_xm"] + 1].square().sum(-1),
            gear_contact=(gear_hit & oper).float(),
            body_contact=leg_hit.float(),
            impact=impact,
            stand_still=v_fwd.abs() * (self.cmd[:, 0] == 0) * (self.ramp.abs() < 0.05) * oper,
            climb=gain / self.dt,                                         # x dt below = metres gained
            # stay away from the hip limits (the 5-bar folding through itself) by 10 deg
            hip_limit=((self.hip_lo + 0.175 - self.qpos[:, self.hip_q]).clamp_min(0).square()
                       + (self.qpos[:, self.hip_q] - self.hip_hi + 0.175).clamp_min(0).square()).sum(-1),
            leg_split=split.square(),                                      # no free zone (v7-v11 staggered stance)
            power=(self.out[:, C.O["tau_hYp_xp"]:C.O["tau_w_xm"] + 1]
                   * self.qvel[:, [self.hip_v[0], self.hip_v[2], self.hip_v[1], self.hip_v[3]] + self.qv_wheel]
                   ).clamp_min(0).sum(-1) * free,
            extension_speed=(torch.stack([o[:, C.O["dL_xp"]], o[:, C.O["dL_xm"]]], -1).abs()
                             - cfg.extension_speed_free).clamp_min(0).square().sum(-1) * free,
            action_excess=self.excess,
            # one wheel on a different level than the other (landing square on the tread)
            wheel_levels=((gw[:, 0] - gw[:, 1]).abs() > 0.03).float() * grounded,
            # wheel surface speed (odometry) vs the true speed: spin, and what corrupts the map
            wheel_slip=(o[:, C.O["dx"]] - v_fwd).square().clamp(max=4.0) * grounded,
            leg_load=((tf - cfg.leg_force_free) / 100.0).clamp(0.0, 5.0).square().sum(-1),
            bottom_out=((Ls < self.min_len + 0.01).any(-1) & grounded).float(),
            jump_start=jump_start.float(),
        )
        # stalled with the flag on and a speed target, short of the goal (a hop in progress does not count)
        stalled = skill & (self.ramp > 0.2) & (v_fwd < cfg.stall_speed) & ~self.finished & (self.since_jump > 0.5)
        self.stall_t = torch.where(stalled, self.stall_t + self.dt, torch.zeros_like(self.stall_t))
        terms["stall"] = (self.stall_t > cfg.stall_grace_s).float()
        # falls: tilted, chassis touching a box feature or too low over the heightfield (it does not collide
        # with it), down a gap, or the controller protection cut the torques
        tilt = grav_b[:, 2] > -math.cos(math.radians(50.0))
        low = self.qpos[:, 2] - T.sample_heights(self.H, self.terrain["size_x"], self.terrain["size_y"],
                                                 self.qpos[:, 0:2]) < 0.2
        in_pit = self.qpos[:, 2] < 0.12                     # every walkable surface is at z >= -0.03
        fell = tilt | base_hit | low | in_pit | (o[:, C.O["protect"]] > 0.5) | self.bad
        self.fall_cause = torch.stack([tilt, base_hit, low, in_pit, o[:, C.O["protect"]] > 0.5, self.bad], -1)
        half = torch.tensor([self.terrain["size_x"], self.terrain["size_y"]], device=self.device) / 2 - 1.0
        off_map = (self.qpos[:, 0:2].abs() > half).any(-1)
        # stairs_narrow: the flight is done on the top platform (climbing) or past the foot (descending)
        self.arrived = grounded & (((self.flight > 0) & (y_loc > ST.NARROW_TOP_Y)) |
                                   ((self.flight < 0) & (y_loc < ST.NARROW_FOOT_Y))) & ~fell
        climbed = (self.flight == 0) & self.stairs_type[self.ttype] & skill & ~fell & \
            (self.ground_best - self.ground0 >= need)
        done_now = (self.arrived | climbed) & ~self.finished
        self.finished |= climbed
        t = self.episode_length_buf.float() * self.dt
        self.t_finish = torch.where(done_now, t, self.t_finish)
        terms["finish"] = done_now.float()
        # time limits are bootstrapped by RSL-RL (reward += gamma * V(s_t)); a success end is a true terminal state:
        # bootstrapping it turned the finish bonus into a value fixed point of r / (1 - gamma) (~3600 at gamma
        # 0.995), the critic diverged and the hop died (first v12 run, 2026-11-17)
        success = self.arrived | (self.finished & (t - self.t_finish > cfg.hold_after_finish_s))
        timeout = (self.episode_length_buf >= self.max_episode_length) | off_map
        if cfg.tile_truncate:
            c = self.tile_centers[self.level, self.ttype]
            self.left_tile = ((self.qpos[:, 0:2] - c).abs() > self.terrain["tile"] / 2 - 0.1).any(-1) & ~fell
            timeout = timeout | self.left_tile
        refused = (self.stall_t > cfg.stall_end_s) & ~fell & ~success
        self.fell_now, self.refused_now = fell, refused
        terms["refuse"] = refused.float()
        terms["termination"] = fell.float()
        terms["fall_height"] = fell.float() * g.clamp(0.0, 3.0)        # a fall from higher up costs more
        rew = torch.zeros(self.num_envs, device=self.device)
        ws, wa = cfg.rew, cfg.rew_assist
        for k, v in terms.items():
            w = torch.where(skill, ws[k], wa[k])
            r = w * v * (1.0 if k in EVENTS else self.dt)
            rew += r
            self.ep_sums[k] += r
        self.ep_speed_err += speed_err.sqrt() * oper
        self.ep_steps_oper += oper.float()
        self.ep_cmd_dist += self.cmd[:, 0].abs() * self.dt
        self.ep_skill_steps += skill.float()
        self.ep_jumps_assist += (jump_start & ~skill).float()
        self.ep_split += split.abs() * oper * ~skill
        return rew, fell | success | refused, timeout & ~fell & ~success & ~refused

    def _log_and_curriculum(self, ids, terminated, timeout):
        terminated = self.fell_now                     # falls only (a success also ends the episode, not bootstrapped)
        log = self.extras["log"] = {}
        if len(ids) == 0:
            return
        for k, v in self.ep_sums.items():
            log[f"Episode_Reward/{k}"] = (v[ids] / self.cfg.episode_s).mean()
        dist = (self.qpos[ids, 0:2] - self.spawn_xy[ids]).norm(dim=-1)
        err = self.ep_speed_err[ids] / self.ep_steps_oper[ids].clamp_min(1)
        st = self.skill_tile[ids]
        log["Metrics/speed_error_mps"] = err.mean()
        log["Metrics/fell"] = terminated[ids].float().mean()
        if (~st).any():
            log["Assist/fell"] = terminated[ids][~st].float().mean()
            log["Assist/speed_error_mps"] = err[~st].mean()
            assist_steps = (self.episode_length_buf[ids] - self.ep_skill_steps[ids]).clamp_min(1).float()
            log["Assist/leg_split_deg"] = (self.ep_split[ids] / assist_steps)[~st].mean() * 57.3
        if st.any():
            log["Skill/fell"] = terminated[ids][st].float().mean()
            log["Skill/refused"] = self.refused_now[ids][st].float().mean()
            log["Skill/cancelled_fell"] = (terminated[ids] & self.cancelled[ids]).float().sum() / \
                self.cancelled[ids].float().sum().clamp_min(1)
        log["Assist/jump_starts"] = self.ep_jumps_assist[ids].sum()
        log["Metrics/distance_m"] = dist.mean()
        log["Metrics/episode_s"] = (self.episode_length_buf[ids].float() * self.dt).mean()
        log["Metrics/nonfinite_resets"] = self.bad[ids].float().sum()
        log["Metrics/contact_buffer_used"] = torch.tensor(self.contact_fill)
        if self.cfg.curriculum:
            T_ = self.terrain["types"]
            assist = self.assist_type[self.ttype[ids]]
            # skill tiles: up after crossing the tile (stairs: climbing enough steps; stairwells: the whole flight);
            # down after a fall or when it moved less than 30% of the commanded way. Assist tiles: up after an
            # episode without a fall that tracked the speed, down after a fall.
            up = timeout[ids] & (dist > self.cfg.level_up_distance)
            rise = torch.tensor(ST.STAIR_RISE, device=self.device)[self.level[ids]]
            stairs = (self.ttype[ids] == T_.index("stairs")) | (self.ttype[ids] == T_.index("stairs_wide"))
            long_ = self.ttype[ids] == T_.index("stairs_long")
            need = torch.where(long_, 6.0, 3.0) * rise - 0.01
            up |= (stairs | long_) & ~terminated[ids] & (self.ground_best[ids] - self.ground0[ids] >= need)
            narrow = self.flight[ids] != 0
            up = torch.where(narrow, self.arrived[ids], up)
            if narrow.any():
                log["Metrics/flights_done"] = self.arrived[ids][narrow].float().mean()
            down = terminated[ids] | self.refused_now[ids] | (dist < 0.3 * self.ep_cmd_dist[ids])
            up = torch.where(assist, timeout[ids] & (err < 0.35), up)
            down = torch.where(assist, terminated[ids], down)
            cancelled = self.cancelled[ids]                      # a cancelled skill says nothing about the level
            lv = self.level[ids] + (up & ~cancelled).long() - (down & ~up).long()
            top = self.type_top[self.ttype[ids]]
            rnd_lv = (torch.rand(lv.shape, generator=self.gen, device=self.device) * (top + 1)).long().clamp(max=top)
            lv = torch.where(lv > top, rnd_lv, lv)
            lv = torch.minimum(lv.clamp_min(0), top)
            keep = ~self.rand_lv[ids]
            self.lvl_tab[ids[keep], self.ttype[ids][keep]] = lv[keep]
            log["Curriculum/level_mean"] = self.lvl_tab.float().mean()
        for t, name in enumerate(self.terrain["types"]):
            sel = self.ttype[ids] == t
            if sel.any():
                log[f"Fell_by_terrain/{name}"] = terminated[ids][sel].float().mean()
                log[f"Level_by_terrain/{name}"] = self.lvl_tab[:, t].float().mean()
