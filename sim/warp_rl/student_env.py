"""Student environment: the skill policy with what the robot will have. The student gets the D435i elevation map
instead of the perfect scans, and a longer proprioceptive history instead of the privileged values. It is distilled
from the teacher (skill_env.py) by DAgger (distill.py): the student drives, the teacher labels the actions.

The map is simulated per robot on the GPU, like perception/elevation_map.py on the robot: a window of 4 cm cells
around the robot, stored as a ring buffer in the robot's *odometry* frame. Every `camera_every` policy steps
(~17 Hz), the cells the camera sees are measured:
  - visibility: inside the D435i field of view and range (87 x 58 deg, 0.28-3 m, perception/d435i.py), from the
    camera pose `latency` steps ago, and not occluded by the terrain (ray marched). The body pitch (the planner's
    look down / up) moves the camera like on the robot.
  - noise: grows with depth^2; random holes (per-episode rate, P5).
  - odometry drift (P3): the robot's pose estimate drifts as a random walk per metre driven (x, y, yaw, height),
    with a per-episode scale error and a jump at every hop landing. A cell is written where the robot *believed* it
    was when it measured it, and read where it believes it is now. So old cells shift by the drift since they were
    seen, like on the robot. Heights are relative to the robot's own height estimate, which drifts too.
  - memory: cells keep their (filtered) last value. Cells never seen are "unknown": a mask goes to the policy, and
    their height is the ground under the wheels. Ceilings are seen the same way (unknown = open).
  - camera blackout (P6): on assist tiles, some episodes lose the camera for a while (the map stops updating and is
    cleared). Assist must stay safe with an empty map. Skill tiles keep the camera: the teacher's skill labels
    need the terrain.
  - peek at the start (the planner's ALIGN): at reset the camera takes frames from the spawn pose looking level
    and down, as the robot does before a skill.

Observations:
  teacher   the teacher's observation (skill_env.py) for the labels. With `teacher_legacy_history`, its history
            is the one it was trained with ([t-1, t, t], SkillCfg.history_fix).
  student   `student_history` frames of the proprioceptive frame (oldest first, one per policy step, zeros after
            a reset), then the height scan from the map (155, same points and scaling as the teacher), its known
            mask (155) and the ceiling scan (45). No privileged values.
"""
import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

import controller as C
import terrain as T
from skill_env import SkillCfg, SkillEnv

TILT = math.radians(0.0)                  # perception/d435i.py (CAD: level)
CAM_POS = (-0.0292, 0.1643, -0.0003)      # base frame (x right, y forward, z up)
LOOK_DOWN = -0.32                         # policy_interface.REMOTE["look_down"]


@dataclass
class StudentCfg(SkillCfg):
    student_history: int = 50             # 1 s
    teacher_legacy_history: bool = True   # the teachers up to v13c were trained with [t-1, t, t]
    map_cells: int = 72                   # 2.88 m window (the scan reaches 1.24 m from the base)
    map_res: float = 0.04
    camera_every: int = 3                 # policy steps between depth frames
    latency: tuple = (1, 3)               # camera delay [policy steps], per episode
    ray_samples: int = 8
    camera_chunk: int = 1024              # robots per camera batch (GPU memory)
    depth_noise: float = 0.0036           # sigma = depth_noise * depth^2 (perception/d435i.py)
    hole_rate: tuple = (0.0, 0.25)        # share of the visible cells lost per frame, per episode
    # odometry drift: random walk per sqrt(metre), a per-episode scale error, a jump per hop
    drift_xy: float = 0.03                # [m / sqrt(m)]
    drift_yaw: float = 0.015              # [rad / sqrt(m)]
    drift_z: float = 0.01                 # [m / sqrt(m)]
    drift_scale: float = 0.03             # odometry scale error, +- per episode
    hop_xy: float = 0.03                  # [m] per landing
    hop_yaw: float = 0.02                 # [rad] per landing
    hop_z: float = 0.01                   # [m] per landing
    # camera blackout on assist tiles (P6)
    p_blackout: float = 0.15
    blackout_s: tuple = (1.0, 8.0)
    peek: bool = True
    # what the teacher v13c was trained with (more cancels on the stairs)
    p_cancel_by_type: tuple = (("stairs", 0.35), ("stairs_narrow", 0.35), ("stairs_long", 0.35), ("stairs_wide", 0.35))


def _rot(v, a):
    """rotate 2-D vectors v [..., 2] by angles a (broadcast)"""
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


class StudentEnv(SkillEnv):
    def _init_buffers(self):
        super()._init_buffers()
        cfg, N, dev = self.cfg, self.num_envs, self.device
        n = cfg.map_cells
        self.map_h = torch.zeros(N, n, n, device=dev)
        self.map_c = torch.full((N, n, n), 10.0, device=dev)
        self.map_known = torch.zeros(N, n, n, dtype=torch.bool, device=dev)
        self.map_tag = torch.full((N, n, n, 2), -10 ** 6, dtype=torch.long, device=dev)
        o = torch.arange(n, device=dev) - n // 2
        self.win = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1)        # [n, n, 2] cell offsets (x, y)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.e_xy, self.e_yaw, self.e_z = z(N, 2), z(N), z(N)                   # pose estimate error
        self.scale_err = z(N)
        self.prev_xy = z(N, 2)
        self.was_air = torch.zeros(N, dtype=torch.bool, device=dev)
        self.hole = z(N)
        self.delay = torch.ones(N, dtype=torch.long, device=dev)
        D = cfg.latency[1] + 1
        self.pose_buf = z(N, D, 3)                                              # base position, last D steps
        self.R_buf = torch.eye(3, device=dev).repeat(N, D, 1, 1)
        self.black_t0, self.black_t1 = z(N) + 1e9, z(N) + 1e9
        self.shist = z(N, cfg.student_history, self.frame_dim)
        self._shist_tick = -1
        self.cam_step = 0
        self.tan_h, self.tan_v = math.tan(math.radians(43.5)), math.tan(math.radians(29.0))
        self.cam_off = torch.tensor(CAM_POS, device=dev)

    def _reset(self, ids):
        if len(ids) == 0:
            return
        super()._reset(ids)
        if not hasattr(self, "map_h"):
            return
        cfg, n = self.cfg, len(ids)
        self.map_known[ids] = False
        self.map_c[ids] = 10.0
        self.map_tag[ids] = -10 ** 6
        self.shist[ids] = 0.0
        self.e_xy[ids], self.e_yaw[ids], self.e_z[ids] = 0.0, 0.0, 0.0
        self.scale_err[ids] = self._rand(n, -cfg.drift_scale, cfg.drift_scale)
        self.prev_xy[ids] = self.qpos[ids, 0:2]
        self.was_air[ids] = False
        self.hole[ids] = self._rand(n, *cfg.hole_rate)
        lo, hi = cfg.latency
        self.delay[ids] = torch.randint(lo, hi + 1, (n,), generator=self.gen, device=self.device)
        R = self._quat_R(self.qpos[ids, 3:7])
        self.pose_buf[ids] = self.qpos[ids, None, 0:3]
        self.R_buf[ids] = R[:, None]
        black = self.assist_type[self.ttype[ids]] & (self._rand(n, 0.0, 1.0) < cfg.p_blackout)
        t0 = self._rand(n, 0.0, max(0.0, cfg.episode_s - cfg.blackout_s[0]))
        self.black_t0[ids] = torch.where(black, t0, torch.full_like(t0, 1e9))
        self.black_t1[ids] = self.black_t0[ids] + self._rand(n, *cfg.blackout_s)
        if cfg.peek:                                    # the planner's ALIGN: look level and down before going
            for ch in torch.arange(n, device=self.device).split(cfg.camera_chunk):
                for pitch in (0.0, LOOK_DOWN, LOOK_DOWN):
                    self._camera_update(ids[ch], self.qpos[ids[ch], 0:3], R[ch], extra_pitch=pitch, holes=False)

    # ---------------- odometry drift ----------------
    def _drift(self):
        cfg, N = self.cfg, self.num_envs
        xy = self.qpos[:, 0:2]
        step = xy - self.prev_xy
        self.prev_xy = xy.clone()
        d = step.norm(dim=-1).clamp(max=0.2)            # (a reset jumps: the new episode starts with e = 0)
        r = lambda k: torch.randn(N, k, generator=self.gen, device=self.device)
        sq = d.sqrt()
        self.e_xy += cfg.drift_xy * sq[:, None] * r(2) + self.scale_err[:, None] * step * (d > 0)[:, None]
        self.e_yaw += cfg.drift_yaw * sq * r(1)[:, 0]
        self.e_z += cfg.drift_z * sq * r(1)[:, 0]
        air = self.out[:, C.O["grounded"]] < 0.5
        land = self.was_air & ~air
        self.was_air = air
        if bool(land.any()):
            lf = land.float()
            self.e_xy += cfg.hop_xy * lf[:, None] * r(2)
            self.e_yaw += cfg.hop_yaw * lf * r(1)[:, 0]
            self.e_z += cfg.hop_z * lf * r(1)[:, 0]

    def _to_est(self, x, ids=None):
        """true world points [n, P, 2] -> the robot's odometry frame (where it believes they are)"""
        sl = slice(None) if ids is None else ids
        p = self.qpos[sl, None, 0:2]
        return p + self.e_xy[sl, None] + _rot(x - p, self.e_yaw[sl, None])

    # ---------------- simulated D435i map ----------------
    def _sample_seen(self, xy):
        """What the camera sees: the ground with the walls (as the teacher's scan), not the walkable floor."""
        return T.sample_heights(self.Gs, self.terrain["size_x"], self.terrain["size_y"], xy)

    def _camera_update(self, ids, base, R, extra_pitch=0.0, holes=True):
        """One depth frame for robots `ids` from base position [n, 3] and attitude R [n, 3, 3] (true)."""
        cfg, n, res = self.cfg, self.cfg.map_cells, self.cfg.map_res
        m = len(ids)
        dev = self.device
        if extra_pitch:                                 # look down / up: rotate about the base x axis
            c, s = math.cos(extra_pitch), math.sin(extra_pitch)
            Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=dev, dtype=R.dtype)
            R = R @ Rx
        cam = base + torch.einsum("nij,j->ni", R, self.cam_off)
        f = torch.einsum("nij,j->ni", R, torch.tensor([0.0, math.cos(TILT), -math.sin(TILT)], device=dev))
        right = R[:, :, 0]
        down = torch.cross(f, right, dim=-1)
        # the window in the odometry frame, centred on the estimated position
        p_est = self.qpos[ids, 0:2] + self.e_xy[ids]
        gc = torch.floor(p_est / res).long()
        g = gc[:, None, None, :] + self.win[None]                               # [m, n, n, 2] cells (odometry)
        cxy = (g.float() + 0.5) * res
        # where each odometry cell really is (inverse of _to_est)
        p = self.qpos[ids, None, None, 0:2]
        wxy = p + _rot(cxy - p - self.e_xy[ids, None, None], -self.e_yaw[ids, None, None])
        si, sj = g[..., 0] % n, g[..., 1] % n
        bi = ids[:, None, None]
        h_true = self._sample_seen(wxy)
        v = torch.cat([wxy, h_true[..., None]], -1) - cam[:, None, None, :]
        depth = (v * f[:, None, None, :]).sum(-1)
        x = (v * right[:, None, None, :]).sum(-1)
        y = (v * down[:, None, None, :]).sum(-1)
        vis = (depth > 0.28) & (depth < 3.0) & (x.abs() < self.tan_h * depth) & (y.abs() < self.tan_v * depth)
        for t in torch.linspace(0.15, 0.95, cfg.ray_samples):                    # occlusion by the terrain
            q = cam[:, None, None, :] + t * v
            vis &= q[..., 2] >= self._sample_seen(q[..., 0:2]) - 0.01
        if holes:
            vis &= torch.rand(vis.shape, generator=self.gen, device=dev) >= self.hole[ids, None, None]
        meas = h_true + self.e_z[ids, None, None] + cfg.depth_noise * depth.square() * torch.randn(
            h_true.shape, generator=self.gen, device=dev)
        tag_ok = (self.map_tag[bi, si, sj] == g).all(-1)
        known = self.map_known[bi, si, sj] & tag_ok
        old = self.map_h[bi, si, sj]
        self.map_h[bi, si, sj] = torch.where(vis, torch.where(known, 0.6 * old + 0.4 * meas, meas), old)
        self.map_known[bi, si, sj] = (known | vis)
        self.map_tag[bi, si, sj] = torch.where(vis[..., None], g, self.map_tag[bi, si, sj])
        # ceiling: a finite one is measured when its underside is in view; a seen floor with nothing above = open
        c_true = self._sample_ceiling(wxy)
        vc = torch.cat([wxy, c_true[..., None]], -1) - cam[:, None, None, :]
        dc = (vc * f[:, None, None, :]).sum(-1)
        xc, yc = (vc * right[:, None, None, :]).sum(-1), (vc * down[:, None, None, :]).sum(-1)
        vis_c = (c_true < 9.0) & (dc > 0.28) & (dc < 3.0) & (xc.abs() < self.tan_h * dc) & (yc.abs() < self.tan_v * dc)
        old_c = torch.where(tag_ok, self.map_c[bi, si, sj], torch.full_like(c_true, 10.0))
        self.map_c[bi, si, sj] = torch.where(vis_c, c_true + self.e_z[ids, None, None],
                                             torch.where(vis & (c_true >= 9.0), torch.full_like(c_true, 10.0), old_c))

    def _map_lookup(self, xy):
        """map height (nan where unknown) and ceiling at true world points [N, P, 2], seen through the drift"""
        n, res = self.cfg.map_cells, self.cfg.map_res
        g = torch.floor(self._to_est(xy) / res).long()
        si, sj = g[..., 0] % n, g[..., 1] % n
        bi = torch.arange(self.num_envs, device=self.device)[:, None]
        tag = (self.map_tag[bi, si, sj] == g).all(-1)
        ok = self.map_known[bi, si, sj] & tag
        h = torch.where(ok, self.map_h[bi, si, sj], torch.full(ok.shape, float("nan"), device=self.device))
        c = torch.where(tag, self.map_c[bi, si, sj], torch.full(ok.shape, 10.0, device=self.device))
        return h, c

    # ---------------- observations ----------------
    def _compute_obs(self):
        super()._compute_obs()                          # teacher observation (+ self.hist, R, heading)
        if not hasattr(self, "map_h"):
            return
        cfg, N = self.cfg, self.num_envs
        first = self._shist_tick != self._tick          # once per policy step (the base computes twice on resets)
        self._shist_tick = self._tick
        frame = self.hist[:, -1]
        if first:
            self.shist = torch.cat([self.shist[:, 1:], frame[:, None]], 1)
            self._drift()
            self.pose_buf = torch.cat([self.pose_buf[:, 1:], self.qpos[:, None, 0:3]], 1)
            self.R_buf = torch.cat([self.R_buf[:, 1:], self.R[:, None]], 1)
            if self.cam_step % cfg.camera_every == 0:
                t = self.episode_length_buf.float() * self.dt
                black = (t >= self.black_t0) & (t < self.black_t1)
                if bool(black.any()):                   # the camera is off: the map is lost
                    self.map_known[black] = False
                    self.map_c[black] = 10.0
                ids_all = torch.nonzero(~black).flatten()
                for ids in ids_all.split(cfg.camera_chunk):         # bounded memory for the [n, cells] tensors
                    k = self.pose_buf.shape[1] - 1 - self.delay[ids]
                    self._camera_update(ids, self.pose_buf[ids, k], self.R_buf[ids, k])
            self.cam_step += 1
        else:
            self.shist[:, -1] = frame                   # reset robots: their first frame
        z = self.qpos[:, 2:3]
        z_est = z + self.e_z[:, None]
        h, c = self._map_lookup(self._world_points(self.hscan))
        known = torch.isfinite(h)
        base_h = self.base_height[:, None]              # known on the robot: legs + attitude
        scan = torch.where(known, z_est - h, base_h).sub(0.4).clamp(-1, 1)
        _, cc = self._map_lookup(self._world_points(self.cscan))
        clear = (cc - z_est).clamp(-0.5, 1.0)
        student = torch.cat([self.shist.reshape(N, -1), scan, known.float(), clear], -1)
        teacher = self.obs["policy"]
        if cfg.teacher_legacy_history and cfg.history >= 2:
            H, F = cfg.history, self.frame_dim
            idx = list(range(1, H)) + [H - 1]            # [t-1, t, t] for history 3
            teacher = torch.cat([self.hist[:, idx].reshape(N, -1), teacher[:, H * F:]], -1)
        self.obs = TensorDict({"teacher": teacher, "student": student, "policy": teacher,
                               "critic": self.obs["critic"]}, batch_size=[N])
