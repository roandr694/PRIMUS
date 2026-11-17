"""Skill evaluation on the GPU: a skill policy (or the model-based controller alone) per terrain type and level.

Each robot stays on its tile (no curriculum), one episode per robot. Per cell it reports
  ok / fell / refused   no fall and at least `--distance` metres; fall; neither (stopped, turned back, stuck)
  +-                    95% interval half-width of `ok` (binomial), so small cells are not over-read
  jumps                 starts of the jump primitive per robot
  steps                 stairs: steps climbed, measured under the lower wheel
  roll                  median of each robot's largest roll [deg]
A cell needs ~150 robots for +-8 points; `--per` sets the robots per (type, level). Stairs are split into robots
that climb and robots that descend (`--p-climb 1` makes every stairs robot climb).

    ~/test/isaac-rl/bin/python sim/warp_rl/skill_eval.py --checkpoint <run>/logs/model_N.pt --arch encoders --out <run>
    ... --types stairs,stairs_long --p-climb 1 --per 200          # stairs only, all climbing
    ... --motors weak | nominal | strong                           # fixed motor draw instead of the random one
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MOTORS = dict(weak=(18.0, 0.8, 0.004, 500.0, 10.0), nominal=(21.5, 0.925, 0.0025, 700.0, 5.0),
              strong=(25.0, 1.05, 0.001, 900.0, 0.0))     # torque, x W0, lag [s], power [W], delay [steps]


@torch.inference_mode()
def run(env, policy, seconds, distance):
    import controller as C
    N, dev = env.num_envs, env.device
    idx = torch.arange(N, device=dev)
    env._reset(idx)
    env._compute_obs()
    start = env.qpos[:, 0:3].clone()
    climb = env.qpos[:, 2] < 0.55                # spawned at the foot (descending robots start on top)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    fell = torch.zeros_like(done)
    dist, jumps, phase0, roll = (torch.zeros(N, device=dev) for _ in range(4))
    g0 = env._sample_ground(env.qpos[:, 0:2])
    best = g0.clone()
    obs = env.get_observations()
    for _ in range(int(seconds / env.dt)):
        a = policy(obs) if policy is not None else torch.zeros(N, env.num_actions, device=dev)
        prev = env.qpos[:, 0:2].clone()
        obs, _, dones, ex = env.step(a)
        if policy is not None and hasattr(policy, "reset"):
            policy.reset(dones)                     # recurrent student: clear the memory of finished episodes
        live = ~done & ~dones
        ph = env.out[:, C.O["jump_phase"]]
        jumps += ((ph > 0.5) & (phase0 < 0.5) & live).float()
        phase0 = torch.where(dones, torch.zeros_like(ph), ph)
        roll = torch.where(live, torch.maximum(roll, env.out[:, C.O["roll"]].abs()), roll)
        grounded = env.out[:, C.O["grounded"]] > 0.5
        best = torch.where(live & grounded, torch.maximum(best, env.wheel_ground()), best)
        new = dones & ~done
        fell |= new & ~ex["time_outs"]
        dist = torch.where(new, (prev - start[:, 0:2]).norm(dim=-1), dist)
        done |= dones
    dist = torch.where(~done, (env.qpos[:, 0:2] - start[:, 0:2]).norm(dim=-1), dist)
    ok = ~fell & (dist >= distance)
    return dict(ok=ok, fell=fell, refused=~fell & ~ok, climb=climb, gained=best - g0, jumps=jumps, roll=roll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint (omit: controller alone only)")
    ap.add_argument("--per", type=int, default=150, help="robots per (terrain type, level)")
    ap.add_argument("--types", default=None, help="comma-separated terrain types (default: all)")
    ap.add_argument("--p-climb", type=float, default=0.5, help="share of the stairs robots that climb")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--distance", type=float, default=2.5)
    ap.add_argument("--motors", choices=("random", *MOTORS), default="random", help="motor draw (default: as in training)")
    ap.add_argument("--controller", action="store_true", help="also evaluate the model-based controller alone")
    ap.add_argument("--student", action="store_true", help="the checkpoint is a distilled student (camera map)")
    ap.add_argument("--recurrent", action="store_true", help="the student has a GRU")
    ap.add_argument("--arch", choices=("mlp", "encoders"), default="mlp", help="teacher network (train.py --arch)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None, help="run directory: writes skill_eval[_<tag>].json there")
    ap.add_argument("--tag", default=None, help="suffix of the JSON file (default: the motor set)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import copy
    import skill_terrain as ST
    if args.student:
        from student_env import StudentCfg as SkillCfg, StudentEnv as SkillEnv
    else:
        from skill_env import SkillCfg, SkillEnv
    from skill_env import W0
    all_types = ST.TYPES
    use = [all_types.index(t) for t in args.types.split(",")] if args.types else list(range(len(all_types)))
    L = 6
    N = len(use) * L * args.per
    cfg = SkillCfg(num_envs=N, curriculum=False, random_tile_type=False, p_parked_start=0.0,
                   episode_s=args.seconds + 1.0, push_speed=0.0, p_back=0.0, p_zero_speed=0.0, p_turn=0.0,
                   speed_range=(0.5, 1.0), fast_speed=(0.5, 1.0), p_climb=args.p_climb)
    env = SkillEnv(cfg, device=args.device, seed=args.seed)
    if args.motors != "random":
        tq, w0, lag, pw, dl = MOTORS[args.motors]
        env.drv[:, 0], env.drv[:, 1], env.drv[:, 2], env.drv[:, 3], env.drv[:, 4] = tq, w0 * W0, lag, pw, dl
    idx = torch.arange(N, device=env.device)
    tile = idx * (len(use) * L) // N
    env.ttype[:], env.level[:] = torch.tensor(use, device=env.device)[tile % len(use)], tile // len(use)

    policies = [("controller", None)] if args.controller or not args.checkpoint else []
    if args.checkpoint:
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner
        from train import DISTILL_CFG, recurrent_student, skill_ppo
        if args.student:
            dcfg = copy.deepcopy(DISTILL_CFG)
            if args.recurrent:
                dcfg["student"] = recurrent_student()
            runner = DistillationRunner(env, dcfg, log_dir=None, device=args.device)
        else:
            runner = OnPolicyRunner(env, skill_ppo(env, args.arch), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policies.append(("policy", runner.get_inference_policy(device=args.device)))
    results = {}
    print(f"{args.seconds:.0f} s per robot, motors: {args.motors}. Per level 0..{L - 1}: ok/fell/refused %; "
          f"+- = widest 95% interval of ok; then jump-primitive starts per robot, steps climbed, roll [deg]")
    for name, pol in policies:
        r = run(env, pol, args.seconds, args.distance)
        rows = {}
        for t in use:
            tn = all_types[t]
            subs = ((tn + " up", r["climb"]), (tn + " down", ~r["climb"])) if tn.startswith("stairs") else ((tn, None),)
            for sub, mask in subs:
                cells = []
                for lv in range(L):
                    sel = (env.ttype == t) & (env.level == lv)
                    if mask is not None:
                        sel &= mask
                    n = int(sel.sum())
                    if n == 0:
                        cells.append(None)
                        continue
                    f = lambda x: float(x[sel].float().mean())
                    ok = f(r["ok"])
                    cells.append(dict(n=n, ok=ok, fell=f(r["fell"]), refused=f(r["refused"]),
                                      ci=1.96 * math.sqrt(max(ok * (1 - ok), 1e-9) / n), jumps=f(r["jumps"]),
                                      steps=f(r["gained"]) / ST.STAIR_RISE[lv] if tn.startswith("stairs") else None,
                                      roll=math.degrees(float(r["roll"][sel].median()))))
                have = [c for c in cells if c]
                if not have or max(c["n"] for c in have) < 10:
                    continue
                rows[sub] = cells
                cell = lambda c: "     -     " if c is None else f"{c['ok'] * 100:3.0f}/{c['fell'] * 100:3.0f}/{c['refused'] * 100:3.0f}"
                print(f"  {sub:17s} {name:10s} " + " ".join(cell(c) for c in cells)
                      + f"   +-{max(c['ci'] for c in have) * 100:2.0f}  n~{sum(c['n'] for c in have) // len(have)}")
                extra = "jumps " + " ".join("  - " if c is None else f"{c['jumps']:4.1f}" for c in cells)
                if tn.startswith("stairs") and sub.endswith("up"):
                    extra += "   steps " + " ".join("  - " if c is None else f"{c['steps']:4.1f}" for c in cells)
                print(f"  {'':17s} {'':10s} {extra}   roll " + " ".join(" - " if c is None else f"{c['roll']:3.0f}" for c in cells))
        results[name] = rows
    if args.out:
        tag = args.tag or ("" if args.motors == "random" else args.motors)
        Path(args.out, f"skill_eval{'_' + tag if tag else ''}.json").write_text(json.dumps(
            dict(args=vars(args), results=results), indent=1))


if __name__ == "__main__":
    main()
