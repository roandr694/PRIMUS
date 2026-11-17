"""Student environment: the skill policy with what the robot will have — the D435i elevation map instead of the
perfect scans, and a longer proprioceptive history instead of the privileged values. Distilled from the teacher
(skill_env.py) with RSL-RL's Distillation (the student drives, the teacher labels the actions: DAgger).

The map is simulated per robot on the GPU, like perception/elevation_map.py on the robot: a 64 x 64 window of
4 cm cells around the robot, world aligned, stored as a ring buffer. Every 3 policy steps (~17 Hz) the cells the
camera sees are measured: inside the D435i field of view and range (87 x 58 deg, 0.28-3 m, mount and tilt of
perception/d435i.py), not occluded by the terrain (ray marched), with noise growing with depth^2, a per-episode
height bias and a per-episode map offset (odometry drift). Cells keep their last value (memory); cells never seen
are "unknown" (a mask goes to the policy, their height is the ground under the wheels). Ceilings are seen the
same way (unknown = open). The camera sees the walls of the stairwells (as the teacher's scan does).

Observations:
  teacher   the teacher's (skill_env.py), for the labels
  student   10 frames of the teacher's proprioceptive frame (no privileged values), the height scan from the map
            (155, same points and scaling as the teacher), its known mask (155), and the ceiling scan (45)
"""
import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

import terrain as T
from skill_env import SkillCfg, SkillEnv

TILT = math.radians(0.0)                  # perception/d435i.py (CAD: level)
CAM_POS = (-0.0292, 0.1643, -0.0003)


@dataclass
class StudentCfg(SkillCfg):
    student_history: int = 10
    map_cells: int = 64
    map_res: float = 0.04
    camera_every: int = 3                 # policy steps between depth frames
    ray_samples: int = 8
    depth_noise: float = 0.0036           # sigma = depth_noise * depth^2 (perception/d435i.py)
    height_bias: float = 0.015            # per episode [m]
    map_offset: float = 0.02              # per episode [m] (odometry drift)


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
        self.bias = torch.zeros(N, device=dev)
        self.offset = torch.zeros(N, 2, device=dev)
        self.shist = torch.zeros(N, cfg.student_history, self.frame_dim, device=dev)
        self.cam_step = torch.zeros(N, dtype=torch.long, device=dev)
        self.tan_h, self.tan_v = math.tan(math.radians(43.5)), math.tan(math.radians(29.0))

    def _reset(self, ids):
        if len(ids) == 0:
            return
        super()._reset(ids)
        n = len(ids)
        self.map_known[ids] = False
        self.map_c[ids] = 10.0
        self.shist[ids] = 0.0
        self.bias[ids] = self._rand(n, -self.cfg.height_bias, self.cfg.height_bias)
        self.offset[ids] = self._rand(2 * n, -self.cfg.map_offset, self.cfg.map_offset).view(n, 2)
        self.cam_step[ids] = 0

    # ---------------- simulated D435i map ----------------
    def _sample_seen(self, xy):
        """What the camera sees: the ground with the walls (as the teacher's scan), not the walkable floor."""
        return T.sample_heights(self.Gs, self.terrain["size_x"], self.terrain["size_y"], xy)

    def _camera_update(self):
        cfg, N, n, res = self.cfg, self.num_envs, self.cfg.map_cells, self.cfg.map_res
        R = self.R
        cam = self.qpos[:, 0:3] + torch.einsum("nij,j->ni", R, torch.tensor(CAM_POS, device=self.device))
        f = torch.einsum("nij,j->ni", R, torch.tensor([0.0, math.cos(TILT), -math.sin(TILT)], device=self.device))
        right = R[:, :, 0]
        down = torch.cross(f, right, dim=-1)
        gc = torch.floor(self.qpos[:, 0:2] / res).long()
        g = gc[:, None, None, :] + self.win[None]                               # [N, n, n, 2] global cells
        wxy = (g.float() + 0.5) * res
        si, sj = g[..., 0] % n, g[..., 1] % n
        bi = torch.arange(N, device=self.device)[:, None, None]
        h_true = self._sample_seen(wxy)
        v = torch.cat([wxy, h_true[..., None]], -1) - cam[:, None, None, :]
        depth = (v * f[:, None, None, :]).sum(-1)
        x = (v * right[:, None, None, :]).sum(-1)
        y = (v * down[:, None, None, :]).sum(-1)
        vis = (depth > 0.28) & (depth < 3.0) & (x.abs() < self.tan_h * depth) & (y.abs() < self.tan_v * depth)
        for t in torch.linspace(0.15, 0.95, cfg.ray_samples):                    # occlusion by the terrain
            q = cam[:, None, None, :] + t * v
            vis &= q[..., 2] >= self._sample_seen(q[..., 0:2]) - 0.01
        meas = h_true + self.bias[:, None, None] + cfg.depth_noise * depth.square() * torch.randn(
            h_true.shape, generator=self.gen, device=self.device)
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
        self.map_c[bi, si, sj] = torch.where(vis_c, c_true + self.bias[:, None, None],
                                             torch.where(vis & (c_true >= 9.0), torch.full_like(c_true, 10.0), old_c))

    def _map_lookup(self, xy):
        """map height (nan where unknown) and ceiling at world points [N, P, 2], with the episode drift"""
        n, res = self.cfg.map_cells, self.cfg.map_res
        g = torch.floor((xy + self.offset[:, None, :]) / res).long()
        si, sj = g[..., 0] % n, g[..., 1] % n
        bi = torch.arange(self.num_envs, device=self.device)[:, None]
        ok = self.map_known[bi, si, sj] & (self.map_tag[bi, si, sj] == g).all(-1)
        return torch.where(ok, self.map_h[bi, si, sj], torch.full_like(ok, float("nan"), dtype=torch.float)), \
            torch.where((self.map_tag[bi, si, sj] == g).all(-1), self.map_c[bi, si, sj], torch.full(ok.shape, 10.0,
                                                                                                  device=self.device))

    # ---------------- observations ----------------
    def _compute_obs(self):
        super()._compute_obs()                          # teacher observation (+ self.hist, R, heading, clear)
        if not hasattr(self, "map_h"):
            return
        N = self.num_envs
        due = self.cam_step % self.cfg.camera_every == 0
        if bool(due.any()):
            self._camera_update()                       # all robots (cheaper than masking); same rate for all
        self.cam_step += 1
        frame = self.hist[:, -1]
        self.shist = torch.cat([self.shist[:, 1:], frame[:, None]], 1)
        z = self.qpos[:, 2:3]
        h, c = self._map_lookup(self._world_points(self.hscan))
        known = torch.isfinite(h)
        ground = self._sample_ground(self.qpos[:, 0:2])[:, None]                # under the wheels
        scan = (z - torch.where(known, h, ground) - 0.4).clamp(-1, 1)
        _, cc = self._map_lookup(self._world_points(self.cscan))
        clear = (cc - z).clamp(-0.5, 1.0)
        student = torch.cat([self.shist.reshape(N, -1), scan, known.float(), clear], -1)
        self.obs = TensorDict({"teacher": self.obs["policy"], "student": student, "policy": self.obs["policy"],
                               "critic": self.obs["critic"]}, batch_size=[N])
