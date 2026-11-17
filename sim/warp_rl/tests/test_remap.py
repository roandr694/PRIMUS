"""Checks mirror.init_from_v11: the remapped v12 network must output what the v11 network did, in the pair basis.

v12 observations come from a short rollout of the v12 environment; the matching v11 observation is rebuilt from
them (same proprioception, the last action converted back to v11's per-side basis, no new commands). The v11
network is evaluated directly from its state dict.

    ~/test/isaac-rl/bin/python sim/warp_rl/tests/test_remap.py <v11>/logs/model_17997.pt
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mirror  # noqa: E402
import policy_interface as PI  # noqa: E402


def v11_forward(sd, x):
    x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
    h = 102
    enc = lambda p, v: Fn.elu(Fn.linear(Fn.elu(Fn.linear(v, sd[p + ".0.weight"], sd[p + ".0.bias"])),
                                        sd[p + ".2.weight"], sd[p + ".2.bias"]))
    z = torch.cat([x[:, :h], enc("priv_enc", x[:, h:h + 8]), enc("scan_enc", x[:, h + 8:h + 163]),
                   enc("ceil_enc", x[:, h + 163:])], -1)
    for i in (0, 2, 4):
        z = Fn.elu(Fn.linear(z, sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]))
    return Fn.linear(z, sd["mlp.6.weight"], sd["mlp.6.bias"])


def main():
    ckpt = sys.argv[1]
    from rsl_rl.runners import OnPolicyRunner
    from skill_env import SkillCfg, SkillEnv
    from train import skill_ppo
    env = SkillEnv(SkillCfg(num_envs=256), device="cuda:0", seed=3)
    runner = OnPolicyRunner(env, skill_ppo(env, "encoders", symmetry=False), log_dir=None, device="cuda:0")
    mirror.init_from_v11(runner, ckpt, env.cfg.history)
    pol = runner.get_inference_policy(device="cuda:0")
    sd = {k: v.cuda() for k, v in torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"].items()}
    A, B = (m.cuda() for m in mirror._action_maps())
    obs = env.get_observations()
    worst = 0.0
    with torch.inference_mode():
        for step in range(40):
            x = obs["policy"]
            F = PI.FRAME_DIM
            parts = []
            for h in range(3):
                f = x[:, h * F:(h + 1) * F]
                a_new = f[:, PI.FRAME_NAMES.index("a_height"):PI.FRAME_NAMES.index("a_height") + 12]
                parts += [f[:, :21], a_new @ A.T, f[:, -1:]]
            x_old = torch.cat(parts + [x[:, 3 * F:]], -1)
            y_old = v11_forward(sd, x_old) @ B.T
            y_new = pol(obs)
            worst = max(worst, float((y_new - y_old).abs().max()))
            obs, _, _, _ = env.step(y_new.clamp(-1, 1))
    print(f"max |v12 remapped - v11| over 40 steps x 256 robots: {worst:.2e} -> {'PASS' if worst < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
