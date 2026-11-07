"""Skill evaluation on the GPU: model-based controller alone vs a skill policy, per terrain type and level.

Each robot stays on its tile (no curriculum), with the spawn rules of skill_env.py (stairs: half climb from the
foot, half descend from the top). One episode per robot; success = no fall and at least `--distance` metres
from the start. Stairs are reported separately for climbing and descending.

    ~/test/isaac-rl/bin/python sim/warp_rl/skill_eval.py --checkpoint <run>/logs/model_3999.pt [--out <run>]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@torch.inference_mode()
def run(env, policy, seconds, distance):
    N, dev = env.num_envs, env.device
    idx = torch.arange(N, device=dev)
    env._reset(idx)
    env._compute_obs()
    start = env.qpos[:, 0:3].clone()
    climb = env.qpos[:, 2] < 0.55                # spawned at the foot (descending robots start on top)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    fell = torch.zeros_like(done)
    dist = torch.zeros(N, device=dev)
    obs = env.get_observations()
    for _ in range(int(seconds / env.dt)):
        a = policy(obs) if policy is not None else torch.zeros(N, env.num_actions, device=dev)
        prev = env.qpos[:, 0:2].clone()
        obs, _, dones, ex = env.step(a)
        if policy is not None and hasattr(policy, "reset"):
            policy.reset(dones)                     # recurrent student: clear the memory of finished episodes
        new = dones & ~done
        fell |= new & ~ex["time_outs"]
        dist = torch.where(new, (prev - start[:, 0:2]).norm(dim=-1), dist)
        done |= dones
    alive = ~done
    dist = torch.where(alive, (env.qpos[:, 0:2] - start[:, 0:2]).norm(dim=-1), dist)
    ok = ~fell & (dist >= distance)
    gained = env.ground_best - env.ground0          # highest ground reached on the wheels
    return ok, fell, dist, climb, gained


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint (omit: controller alone only)")
    ap.add_argument("--num-envs", type=int, default=2520)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--distance", type=float, default=2.5)
    ap.add_argument("--torque", type=float, default=None, help="fixed joint torque limit (default: random 18-25)")
    ap.add_argument("--student", action="store_true", help="the checkpoint is a distilled student (camera map)")
    ap.add_argument("--recurrent", action="store_true", help="the student has a GRU")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.student:
        from student_env import StudentCfg as SkillCfg, StudentEnv as SkillEnv
    else:
        from skill_env import SkillCfg, SkillEnv
    cfg = SkillCfg(num_envs=args.num_envs, curriculum=False, random_tile_type=False, p_parked_start=0.0,
                   episode_s=args.seconds + 1.0, push_speed=0.0, p_back=0.0, p_zero_speed=0.0, p_turn=0.0,
                   speed_range=(0.5, 1.0))
    env = SkillEnv(cfg, device=args.device, seed=123)
    if args.torque:
        env.drv[:, 0] = args.torque
    types, L = env.terrain["types"], cfg.terrain_levels
    idx = torch.arange(env.num_envs, device=env.device)
    tile = idx * (len(types) * L) // env.num_envs
    env.ttype[:], env.level[:] = tile % len(types), tile // len(types)

    policies = [("controller", None)]
    if args.checkpoint:
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner
        from train import DISTILL_CFG, PPO_CFG, recurrent_student
        if args.student:
            dcfg = copy.deepcopy(DISTILL_CFG)
            if args.recurrent:
                dcfg["student"] = recurrent_student()
            runner = DistillationRunner(env, dcfg, log_dir=None, device=args.device)
        else:
            ppo = copy.deepcopy(PPO_CFG)
            ppo["actor"]["distribution_cfg"].update(init_std=0.3, std_range=[0.1, 0.5])
            runner = OnPolicyRunner(env, ppo, log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policies.append(("policy", runner.get_inference_policy(device=args.device)))
    results = {}
    for name, pol in policies:
        ok, fell, dist, climb, gained = run(env, pol, args.seconds, args.distance)
        import skill_terrain as ST
        st = env.ttype == types.index("stairs")
        steps = [float((gained[st & climb & (env.level == lv)] / ST.STAIR_RISE[lv]).mean()) for lv in range(L)]
        print(f"  {name:10s} stairs up, steps climbed (of 5) per level: " + " ".join(f"{v:4.1f}" for v in steps))
        rows = {}
        for t, tname in enumerate(types):
            for sub, mask in (((tname + " up", climb), (tname + " down", ~climb)) if tname == "stairs" else ((tname, None),)):
                vals = []
                for lv in range(L):
                    sel = (env.ttype == t) & (env.level == lv)
                    if mask is not None:
                        sel &= mask
                    vals.append((float(ok[sel].float().mean()), float(fell[sel].float().mean()), int(sel.sum())))
                rows[sub] = vals
        results[name] = rows
    print(f"success (no fall, >= {args.distance} m in {args.seconds} s) / falls, per level 0..{L - 1}")
    for sub in results["controller"]:
        for name in results:
            cells = " ".join(f"{s:4.0%}/{f:3.0%}" for s, f, _ in results[name][sub])
            print(f"  {sub:14s} {name:10s} {cells}")
    if args.out:
        Path(args.out, "skill_eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
