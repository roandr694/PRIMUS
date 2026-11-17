"""Checks the left-right mirror map of the v12 policy (mirror.py, control/policy_interface.py) before training.

1. Static: the scan permutations flip the lateral coordinate of every scan point; mirroring twice is the identity;
   a mirrored action gives the per-side residual with the sides swapped.
2. Dynamic: two robots start in the same symmetric state (a flat tile, then the foot of a stair face); robot 1 gets
   the mirror image of robot 0's commands and actions (smooth random actions, jump triggers included). If the map
   is right, robot 1's observation stays the mirror image of robot 0's. Randomisation that differs between the
   two worlds is switched off. Per entry, the error of the map is compared with the error of the wrong alternative
   (no mirror where the map swaps or flips the entry, a sign flip where it keeps it) over a short horizon: the
   robot is symmetric about the plane x = -0.0117 m of the base frame (checked: masses, inertias, sites), but the
   simulated contacts of the cylinder tyres are not exactly mirror symmetric, so long rollouts drift apart.

    ~/test/isaac-rl/bin/python sim/warp_rl/tests/test_mirror.py
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mirror  # noqa: E402
import policy_interface as PI  # noqa: E402


def static_checks(env):
    hs, cs = env.hscan, env.cscan
    p = torch.tensor(PI.scan_mirror(PI.SCAN_FWD, PI.SCAN_LAT), device=hs.device)
    assert torch.allclose(hs[p], hs * torch.tensor([-1.0, 1.0], device=hs.device)), "height scan permutation"
    p = torch.tensor(PI.scan_mirror(PI.CEIL_FWD, PI.CEIL_LAT), device=cs.device)
    assert torch.allclose(cs[p], cs * torch.tensor([-1.0, 1.0], device=cs.device)), "ceiling scan permutation"
    x = torch.randn(64, env.obs["policy"].shape[1], device=env.device)
    assert torch.allclose(mirror.mirror_obs(mirror.mirror_obs(x)), x), "obs mirror is not an involution"
    a = torch.rand(64, 12, device=env.device) * 2 - 1
    for skill in (0.0, 1.0):
        k = torch.full((64,), skill, device=env.device)
        r, rm = PI.to_sides(a, k, torch), PI.to_sides(mirror.mirror_actions(a), k, torch)
        sw = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 8, 9, 10, 11], device=env.device)
        assert torch.allclose(rm, r[:, sw], atol=1e-6), "action pairs -> sides"
    print("static checks: OK (scan permutations, involution, action pairs)")


def dynamic(env, tile, xy, yaw, skill, seconds=1.0, seed=0):
    """Per observation entry: error of the mirror map vs the error of the wrong alternative (identity where the
    map swaps or flips the entry, a sign flip where it keeps it). Short horizon: the simulated contacts of the
    cylinder tyres are not exactly mirror symmetric, so long rollouts drift apart chaotically."""
    dev = env.device
    types = env.terrain["types"]
    env.ttype[:], env.level[:] = types.index(tile), 2
    env._spawn = lambda ids: (xy.to(dev).expand(len(ids), 2).clone(), torch.full((len(ids),), yaw, device=dev))
    env._reset(torch.arange(2, device=dev))
    env.drv[1] = env.drv[0]
    env.cmd_timer[:] = 1e9
    env.push_timer[:] = 1e9
    env.t_cancel[:] = 1e9
    env.t_on[:] = 0.0 if skill else 1e9
    env.skill_speed[:] = 0.8
    if not skill:
        env.cmd[:, 0], env.cmd[0, 1], env.cmd[1, 1] = 0.6, 0.5, -0.5
    env.cmd[:, 2], env.cmd[:, 5], env.cmd[:, 6] = 0.32, 1.5, -0.1
    env._compute_obs()
    g = torch.Generator(device=dev).manual_seed(seed)
    ph = torch.rand(12, generator=g, device=dev) * 6.28
    fr = 2.0 + 6.0 * torch.rand(12, generator=g, device=dev)
    amp = torch.tensor([0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.0, 0.5, 0.5, 0.5], device=dev)
    D = env.obs["policy"].shape[1]
    e_map, e_alt = torch.zeros(D, device=dev), torch.zeros(D, device=dev)
    idx, sign = mirror.obs_permutation(D)
    trivial = (idx.to(dev) == torch.arange(D, device=dev)) & (sign.to(dev) > 0)
    jumps = 0
    for k in range(int(seconds / env.dt)):
        t = k * env.dt
        a0 = amp * torch.sin(fr * t + ph)
        a0[8] = 1.0 if (skill and 0.56 < t < 0.62) else -1.0                 # one jump (after the 0.5 s cooldown)
        a0[9] = 1.0
        a = torch.stack([a0, mirror.mirror_actions(a0[None])[0]])
        obs, _, dones, _ = env.step(a)
        jumps += int(env.out[0, __import__("controller").O["jump_phase"]] > 0.5)
        if bool(dones.any()):
            print(f"  {tile}: an episode ended at {t:.2f} s")
            break
        x = obs["policy"]
        m = mirror.mirror_obs(x[0:1])[0]
        alt = torch.where(trivial, -x[0], x[0])
        e_map = torch.maximum(e_map, (x[1] - m).abs())
        e_alt = torch.maximum(e_alt, (x[1] - alt).abs())
    informative = e_alt > 0.02
    informative[env.cfg.history * PI.FRAME_DIM + 8:] = False      # scans: exact permutation, checked statically
    ratio = torch.where(informative, e_map / e_alt.clamp_min(1e-6), torch.zeros_like(e_map))
    names = [f"h{h}.{n}" for h in range(env.cfg.history) for n in PI.FRAME_NAMES] + list(PI.PRIV_NAMES) + \
        [f"scan{i}" for i in range(155)] + [f"ceil{i}" for i in range(45)]
    print(f"  {tile} ({'skill, one jump' if skill else 'assist'}; jump active {jumps} steps), {seconds} s: "
          f"{int(informative.sum())} informative entries of {D}")
    for i in ratio.argsort(descending=True)[:6].tolist():
        print(f"    {names[i]:16s} map error {float(e_map[i]):.4f}   wrong-map error {float(e_alt[i]):.4f}   "
              f"ratio {float(ratio[i]):.3f}")
    return float(ratio.max())


def main():
    from skill_env import SkillCfg, SkillEnv
    cfg = SkillCfg(num_envs=2, curriculum=False, random_tile_type=False, p_parked_start=0.0, obs_noise=False,
                   com_offset=0.0, base_mass_scale=(1.0, 1.0), friction_range=(0.9, 0.9), friction_assist=(0.9, 0.9),
                   friction_skill=(0.9, 0.9), joint_friction=(0.1, 0.1), push_speed=0.0, push_speed_skill=0.0,
                   attitude_bias=0.0, hip_offset=0.0, start_leg_range=(0.32, 0.32), p_flag_late=0.0, p_cancel=0.0,
                   action_delay_steps=(5, 5), episode_s=30.0)
    env = SkillEnv(cfg, device="cuda:0", seed=0)
    static_checks(env)
    print("dynamic checks (robot 1 = mirror of robot 0):")
    c = env.tile_centers[0, env.terrain["types"].index("flat")]
    r_flat = dynamic(env, "flat", c, 0.0, skill=False)
    # stairs: at the foot of the -y face, facing +y (heading 0), on the face's centre line
    lvl = 2
    c = env.tile_centers[lvl, env.terrain["types"].index("stairs")]
    xy = c + torch.tensor([0.0, -(3.3)], device=c.device)
    r_stairs = dynamic(env, "stairs", xy, 0.0, skill=True, seconds=0.55)
    print("  (informative only: the same with a jump; the landing is a contact event and the two robots touch down")
    print("   a step apart, so the error grows, but the map still has to explain the mirror robot better than the")
    print("   wrong alternative, ratio < 1)")
    r_jump = dynamic(env, "stairs", xy, 0.0, skill=True, seconds=1.0)
    worst = max(r_flat, r_stairs)
    assert r_jump < 1.0, "jump: a wrong map"
    verdict = "PASS" if worst < 0.5 else "FAIL"
    print(f"worst map error / wrong-map error: {worst:.3f} -> {verdict} (a wrong sign or swap gives >= 1)")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
