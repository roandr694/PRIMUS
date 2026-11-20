"""DAgger distillation of the camera student from the skill teacher (RSL-RL Distillation, extended).

What RSL-RL's Distillation does: the student drives (stochastic), the teacher labels every visited state, and the
student regresses the teacher's actions (MSE), one optimiser step per `gradient_length` time steps.
What this adds:
  - teacher-driven start (beta): early on a robot is driven by the teacher with probability beta (per episode).
    So the student first sees the states the skills go through (stairs half way, a gap in flight) instead of only
    falling at the spawn. beta goes linearly from `beta0` to 0 over `beta_iters` iterations, then it is pure DAgger.
  - action weights: the jump trigger (pair index 8) decides a hop at 0.5, so its errors cost more.
  - learning-rate decay: linear from learning_rate to `lr_min` over `lr_decay_iters` iterations.
  - logs: beta, lr, the per-action error, and the share of steps where student and teacher disagree on the trigger.
"""
from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation


class DAggerDistillation(Distillation):
    def __init__(self, *args, beta0: float = 1.0, beta_iters: int = 300, action_weights=None, lr_min: float = 1e-4,
                 lr_decay_iters: int = 3000, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta0, self.beta_iters = beta0, beta_iters
        self.lr0, self.lr_min, self.lr_decay_iters = self.learning_rate, lr_min, lr_decay_iters
        self.w = None if action_weights is None else torch.tensor(action_weights, device=self.device)
        self.use_teacher = None
        self.trigger_disagree, self.n_act = 0.0, 0

    @property
    def beta(self):
        return max(0.0, self.beta0 * (1.0 - self.num_updates / max(1, self.beta_iters)))

    def act(self, obs):
        a_s = self.student(obs, stochastic_output=True).detach()
        a_t = self.teacher(obs).detach()
        if self.use_teacher is None:
            self.use_teacher = torch.rand(a_s.shape[0], device=self.device) < self.beta
        self.transition.actions = a_s
        self.transition.privileged_actions = a_t
        self.transition.observations = obs
        self.trigger_disagree += float(((a_s[:, 8] > 0.5) != (a_t[:, 8] > 0.5)).float().mean())
        self.n_act += 1
        return torch.where(self.use_teacher[:, None], a_t, a_s)

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        d = dones.view(-1).bool()
        if self.use_teacher is not None and bool(d.any()):
            self.use_teacher[d] = torch.rand(int(d.sum()), device=self.device) < self.beta

    def _loss(self, a, target):
        e = (a - target).square()
        if self.w is not None:
            e = e * self.w
        return e.mean()

    def update(self):
        self.loss_fn = self._loss
        frac = min(1.0, self.num_updates / max(1, self.lr_decay_iters))
        lr = self.lr0 + (self.lr_min - self.lr0) * frac
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        # per-action error on this rollout (before the update)
        with torch.no_grad():
            s = self.storage
            err = None
            if not self.student.is_recurrent:
                n = s.num_transitions_per_env
                err = sum((self.student(s.observations[i]) - s.privileged_actions[i]).square().mean(0)
                          for i in range(0, n, 4)) / len(range(0, n, 4))
        out = super().update()
        out.update(beta=self.beta, lr=lr, trigger_disagree=self.trigger_disagree / max(1, self.n_act))
        if err is not None:
            for k, v in enumerate(err.tolist()):
                out[f"err_a{k}"] = v
        self.trigger_disagree, self.n_act = 0.0, 0
        return out
