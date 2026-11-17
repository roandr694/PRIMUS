"""Left-right mirror symmetry of the v12 policy (RSL-RL symmetry extension) and the v11 -> v12 weight remap.

The robot is (nearly) symmetric about its sagittal plane, the plane normal to the body x axis (the wheel axle).
Mirroring a state swaps the two legs and flips the lateral quantities; the layout of every entry is in
control/policy_interface.py (FRAME_MIRROR, PRIV_MIRROR, scan_mirror, ACTION_SIGN). PPO is trained on both the
rollout and its mirror image (data augmentation) and a mirror loss asks the policy for mirrored actions in mirrored
states (Mittal et al., ICRA 2024). Without it every teacher v7-v11 settled into one arbitrary staggered stance.

Check before training (tests/test_mirror.py): two robots in mirrored states driven by mirrored actions must keep
mirrored observations.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "control"))
import policy_interface as PI  # noqa: E402

N_REST = 8 + PI.SCAN_FWD * PI.SCAN_LAT + PI.CEIL_FWD * PI.CEIL_LAT     # privileged + scan + ceiling
_CACHE = {}


def obs_permutation(dim):
    """(index, sign) such that mirrored_obs = obs[:, index] * sign, for the teacher observation of size dim."""
    hist = (dim - N_REST) // PI.FRAME_DIM
    assert hist * PI.FRAME_DIM + N_REST == dim, f"not a v12 teacher observation: {dim}"
    idx, sign = [], []
    fi, fs = PI.FRAME_MIRROR
    for h in range(hist):
        idx += [h * PI.FRAME_DIM + i for i in fi]
        sign += list(fs)
    base = hist * PI.FRAME_DIM
    pi_, ps = PI.PRIV_MIRROR
    idx += [base + i for i in pi_]
    sign += list(ps)
    base += 8
    n = PI.SCAN_FWD * PI.SCAN_LAT
    idx += [base + i for i in PI.scan_mirror(PI.SCAN_FWD, PI.SCAN_LAT)]
    sign += [1.0] * n
    base += n
    idx += [base + i for i in PI.scan_mirror(PI.CEIL_FWD, PI.CEIL_LAT)]
    sign += [1.0] * (PI.CEIL_FWD * PI.CEIL_LAT)
    return torch.tensor(idx), torch.tensor(sign)


def mirror_obs(x):
    key = (x.shape[-1], x.device)
    if key not in _CACHE:
        i, s = obs_permutation(x.shape[-1])
        _CACHE[key] = (i.to(x.device), s.to(x.device, x.dtype))
    i, s = _CACHE[key]
    return x[..., i] * s


def mirror_actions(a):
    return a * torch.tensor(PI.ACTION_SIGN, device=a.device, dtype=a.dtype)


def augment(env=None, obs=None, actions=None):
    """RSL-RL data_augmentation_func: returns ([obs; mirrored obs], [actions; mirrored actions])."""
    out_obs = out_act = None
    if obs is not None:
        m = obs.clone()
        for k in obs.keys():
            m[k] = mirror_obs(obs[k])
        out_obs = torch.cat([obs, m], dim=0)
    if actions is not None:
        out_act = torch.cat([actions, mirror_actions(actions)], dim=0)
    return out_obs, out_act


def symmetry_cfg(coeff=0.5):
    return {"use_data_augmentation": True, "use_mirror_loss": True, "mirror_loss_coeff": coeff,
            "data_augmentation_func": augment}


# ---------------------------------------------------------------------------------------------------------------
V11_SCALE = (0.17, 0.17, 0.35, 0.35, 4.0, 4.0, 150.0, 150.0, 1.0, 0.06, 60.0, 0.07)
V11_FRAME = 34             # 21 proprioceptive + commands (speed, yaw rate, leg), 12 actions, jump phase


def _action_maps():
    """A: v11 per-side action <- v12 pair action; B: v12 <- v11 (skill regime, before clipping)."""
    A, B = torch.zeros(12, 12), torch.zeros(12, 12)
    for c, d in ((0, 1), (2, 3), (4, 5), (6, 7)):
        so, sc, sd = V11_SCALE[c], PI.SKILL_SCALE[c], PI.SKILL_SCALE[d]
        A[c, c], A[c, d] = sc / so, -0.5 * sd / so          # xp side
        A[d, c], A[d, d] = sc / so, 0.5 * sd / so           # xm side
        B[c, c], B[c, d] = 0.5 * so / sc, 0.5 * so / sc
        B[d, c], B[d, d] = -so / sd, so / sd
    for i in range(8, 12):
        A[i, i] = B[i, i] = V11_SCALE[i] / PI.SKILL_SCALE[i]
    return A, B


def init_from_v11(runner, path, history=3, std=0.2):
    """Loads a v11 teacher (obs 3 x 34 + 208, per-side actions) into the v12 networks so that, in the skill regime
    and before clipping, the v12 network outputs exactly what v11 did: the history columns are moved to the new
    frame layout, the action inputs and the action outputs are changed to the pair basis (the normaliser of the
    action inputs is folded into the weights), and the new inputs (pitch command, responsiveness, speed target,
    skill flag) start with zero weight. The value network gets the same input mapping. A fresh optimiser."""
    old = torch.load(path, map_location="cpu", weights_only=False)
    A, B = _action_maps()
    Fn = PI.FRAME_DIM
    a_new0 = PI.FRAME_NAMES.index("a_height")
    col = {}                                          # old history column -> new column (non-action entries)
    for h in range(history):
        for c in range(V11_FRAME):
            if c < 21:
                col[h * V11_FRAME + c] = h * Fn + c
            elif c == 33:
                col[h * V11_FRAME + c] = h * Fn + Fn - 1
    n_old_rest_start = history * V11_FRAME
    for net, key in ((runner.alg.actor, "actor_state_dict"), (runner.alg.critic, "critic_state_dict")):
        sd, osd = net.state_dict(), old[key]
        dev = sd["mlp.0.weight"].device
        mean, std_o = osd["obs_normalizer._mean"][0], osd["obs_normalizer._std"][0]
        new_mean = torch.zeros_like(sd["obs_normalizer._mean"][0].cpu())
        new_var = torch.ones_like(new_mean)
        W_old, b_old = osd["mlp.0.weight"], osd["mlp.0.bias"].clone()
        W = torch.zeros(sd["mlp.0.weight"].shape)
        h_new = history * Fn
        for o, n in col.items():
            W[:, n] = W_old[:, o]
            new_mean[n], new_var[n] = mean[o], osd["obs_normalizer._var"][0][o]
        for h in range(history):
            oa = [h * V11_FRAME + 21 + i for i in range(12)]
            Wn = W_old[:, oa] / (std_o[oa] + 1e-2)        # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
            b_old -= (Wn * mean[oa]).sum(-1)
            W[:, [h * Fn + a_new0 + j for j in range(12)]] = (Wn @ A) * (1.0 + 1e-2)   # new columns: mean 0, std 1
        W[:, h_new:] = W_old[:, n_old_rest_start:]        # encoder latents
        rest_old = slice(n_old_rest_start, mean.shape[0])
        new_mean[h_new:], new_var[h_new:] = mean[rest_old], osd["obs_normalizer._var"][0][rest_old]
        for name, t in osd.items():
            if name in ("mlp.0.weight", "mlp.0.bias") or name.startswith("obs_normalizer"):
                continue
            if key == "actor_state_dict" and name == "mlp.6.weight":
                t = B @ t
            elif key == "actor_state_dict" and name == "mlp.6.bias":
                t = B @ t
            elif name == "distribution.std_param":
                t = torch.cat([torch.full((8,), std), t[8:]])
            assert sd[name].shape == t.shape, (name, sd[name].shape, t.shape)
            sd[name] = t.to(dev)
        sd["mlp.0.weight"], sd["mlp.0.bias"] = W.to(dev), b_old.to(dev)
        sd["obs_normalizer._mean"] = new_mean[None].to(dev)
        sd["obs_normalizer._var"] = new_var[None].to(dev)
        sd["obs_normalizer._std"] = new_var.sqrt()[None].to(dev)
        sd["obs_normalizer.count"] = osd["obs_normalizer.count"].to(dev)
        net.load_state_dict(sd)
    print(f"initialised from v11 {path} (actions and history remapped to the v12 interface)")
