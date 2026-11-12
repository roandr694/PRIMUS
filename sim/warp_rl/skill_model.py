"""Teacher networks for the skills: separate encoders for the terrain and for the privileged values, then an MLP.

Observation layout (skill_env.py): [proprioceptive history | privileged (8) | height scan (155) | ceiling (45)].
  scan encoder        155 -> 128 -> 32   terrain in front, as features that generalise across shapes
  ceiling encoder      45 -> 32 -> 8
  privileged encoder    8 -> 32 -> 8     the "hidden" physics (velocity, height, torque limit, friction, mass,
                                         contact) squeezed into a latent the student will later learn to estimate
                                         from its history (RMA-style distillation)
  MLP                 history + 32 + 8 + 8 -> 512 -> 256 -> 128 -> actions (or value)
The encoders run after the observation normaliser. as_jit / as_onnx export the whole thing.
"""
import copy

import torch
from torch import nn

from rsl_rl.models import MLPModel


def _enc(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, o), nn.ELU())


class SkillTeacherModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=True, distribution_cfg=None, layout=(102, 8, 155, 45), scan_latent=32,
                 priv_latent=8, ceil_latent=8, assist=None):
        # plain ints before nn.Module.__init__ (the base class builds the MLP from _get_latent_dim)
        self.layout, self.lat = tuple(layout), (scan_latent, priv_latent, ceil_latent)
        self.assist, self._calls = assist, 0
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         distribution_cfg)
        h, p, s, c = self.layout
        assert sum(self.layout) == self.obs_dim, (self.layout, self.obs_dim)
        self.priv_enc = _enc(p, 32, priv_latent)
        self.scan_enc = _enc(s, 128, scan_latent)
        self.ceil_enc = _enc(c, 32, ceil_latent)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        out = super().forward(obs, masks, hidden_state, stochastic_output)
        if stochastic_output and self.assist is not None and not torch.is_grad_enabled():   # rollout only
            out = self._inject_hops(obs, out)
        return out

    @torch.no_grad()
    def _inject_hops(self, obs, a):
        """Exploration of the jump the plain Gaussian never finds (the trigger sits ~7 sigma from its threshold):
        where the scan shows a step up / a wall (0.15-0.45 m ahead) or a trench (0.2-0.6 m ahead) and the robot
        stands on level ground, a few % of the steps replace the jump actions (trigger, extension, push, tuck)
        by a hop trigger (the shape comes from the policy's own sample). The injected trigger is what the rollout
        stores as the action taken, so PPO credits the outcome to it (v9 first try: also injecting the shape
        put 12 sigma outliers in the batch and the policy went NaN). Probability decays from p0 to p1 over `anneal_calls` steps."""
        c = self.assist
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        h, p, s = self.layout[0], self.layout[1], self.layout[2]
        scan = x[:, h + p:h + p + s].view(-1, 31, 5)[:, :, 2]          # centre column, forward -0.30 .. 1.20 m
        rel = -(scan - scan[:, 6:7])                                    # ground height ahead relative to here
        fwd = lambda lo, hi: slice(int(round((lo + 0.30) / 0.05)), int(round((hi + 0.30) / 0.05)) + 1)
        step = rel[:, 1:] - rel[:, :-1]                                 # change between neighbouring 5 cm cells
        e_lo, e_hi = fwd(0.15, 0.45), fwd(0.20, 0.60)
        edge_up = step[:, e_lo.start:e_lo.stop].amax(-1) > 0.04                # a step edge, not a slope / rough ground
        edge_dn = step[:, e_hi.start:e_hi.stop].amin(-1) < -0.20               # a trench edge
        rise = rel[:, e_lo].amax(-1)
        level_now = rel[:, fwd(-0.10, 0.05)].abs().amax(-1) < 0.04
        up = edge_up & (rise > 0.05) & (rise < 0.30) & level_now
        near = up | (edge_dn & level_now)
        mult = torch.where(up, 1.0, c.get("trench_mult", 0.5))
        k = min(self._calls / c["anneal_calls"], 1.0)
        pr = c["p0"] + (c["p1"] - c["p0"]) * k
        self._calls += 1
        fire = near & (torch.rand(a.shape[0], device=a.device) < pr * mult)
        if fire.any():                      # only the trigger is replaced (a bit above the 0.5 threshold); the shape
            a = a.clone()                   # actions keep the policy's own samples, so nothing is far off-policy
            a[fire, 8] = c.get("trigger_value", 0.65)
        return a

    def _get_latent_dim(self):
        return self.layout[0] + sum(self.lat)

    def encode(self, x):
        h, p, s, c = self.layout
        return torch.cat([x[:, :h], self.priv_enc(x[:, h:h + p]), self.scan_enc(x[:, h + p:h + p + s]),
                          self.ceil_enc(x[:, h + p + s:])], -1)

    def privileged_latent(self, obs):
        """The latent the student will learn to estimate."""
        x = self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
        h, p = self.layout[0], self.layout[1]
        return self.priv_enc(x[:, h:h + p])

    def get_latent(self, obs, masks=None, hidden_state=None):
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        return self.encode(self.obs_normalizer(x))

    def as_jit(self):
        return _Export(self)

    def as_onnx(self, verbose=False):
        return _Export(self)


class _Export(nn.Module):
    """TorchScript / ONNX export: normaliser -> encoders -> MLP -> deterministic output."""

    is_recurrent: bool = False

    def __init__(self, model: SkillTeacherModel):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.priv_enc, self.scan_enc, self.ceil_enc = (copy.deepcopy(m) for m in
                                                       (model.priv_enc, model.scan_enc, model.ceil_enc))
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (model.distribution.as_deterministic_output_module()
                                     if model.distribution is not None else nn.Identity())
        self.h, self.p, self.s = model.layout[0], model.layout[1], model.layout[2]
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        h, p, s = self.h, self.p, self.s
        z = torch.cat([x[:, :h], self.priv_enc(x[:, h:h + p]), self.scan_enc(x[:, h + p:h + p + s]),
                       self.ceil_enc(x[:, h + p + s:])], -1)
        return self.deterministic_output(self.mlp(z))

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self):
        return ["obs"]

    @property
    def output_names(self):
        return ["actions"]
