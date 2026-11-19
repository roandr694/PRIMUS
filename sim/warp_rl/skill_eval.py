"""Evaluation of the v12 policy (or of the model-based controller alone) on the GPU. Three suites:

  skill   per feature type and level; every robot in the skill regime from the start, on its tile (no
          curriculum), one episode each. Per cell:
            ok / fell / refused   no fall and at least `--distance` metres (stairwells: the flight done); fall; neither
            +-                    widest 95% interval of `ok` (binomial); ~150 robots per cell give +-8 points
            jumps                 jump-primitive starts per robot;  steps: stairs climbed under the lower wheel
            roll                  median of each robot's largest roll [deg]
            split                 mean |leg split| [deg] (symmetry; the LQR alone holds ~0)
            1-wheel               share of grounded time with the wheels on different levels [%]
          stairs_narrow up also gives the falls per step and what that means over 50 steps.
  assist  everyday driving: flat / rough / slope / bumps, the remote's command envelope (as in training), pushes
          of +-`--push` m/s per axis every 2-6 s. Per terrain: falls per minute, fell %, speed error against the
          ramped target, turn-rate error, height error, |roll|, |pitch rate|, power, torque rms, leg split, jumps.
  cancel  stairs and stairwells, climbing; the skill is cancelled 1.5-6 s after the start (flag off, speed 0):
          the robot must hold where it is. Per level: fell %.

The terrain is built from `--seed` (default 123): the stair treads, rises, nosings and hurdle thicknesses are
drawn from it, so the evaluation terrain is geometry the policy never trained on (training uses seed 0).

    ~/test/isaac-rl/bin/python sim/warp_rl/skill_eval.py --checkpoint <run>/logs/model_N.pt --out <run>
    ... --suite assist --controller --push 1.0                    # the policy against the LQR alone
    ... --suite skill --types stairs,stairs_narrow --p-climb 1 --per 200
    ... --v11 <v11>/logs/model_17997.pt                            # a v11 teacher, remapped to the v12 interface
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
ASSIST = ("flat", "rough", "slope", "bumps")


@torch.inference_mode()
def run(env, policy, seconds, distance, tile_only=False):
    import controller as C
    N, dev, O = env.num_envs, env.device, C.O
    idx = torch.arange(N, device=dev)
    env._reset(idx)
    env._compute_obs()
    start = env.qpos[:, 0:3].clone()
    climb = env.qpos[:, 2] < 0.55                # spawned at the foot (descending robots start on top)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    fell, arrived, finished = torch.zeros_like(done), torch.zeros_like(done), torch.zeros_like(done)
    left = torch.zeros_like(done)
    z = lambda: torch.zeros(N, device=dev)
    dist, jumps, phase0, roll, t_live, t_ground, one_wheel = (z() for _ in range(7))
    s = {k: z() for k in ("verr", "yerr", "lerr", "roll_mean", "pitchr", "power", "torque", "split")}
    hv = [env.hip_v[0], env.hip_v[2], env.hip_v[1], env.hip_v[3]] + list(env.qv_wheel)
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
        lf = live.float()
        o, R = env.out, env.R
        ph = o[:, O["jump_phase"]]
        jumps += ((ph > 0.5) & (phase0 < 0.5) & live).float()
        phase0 = torch.where(dones, torch.zeros_like(ph), ph)
        roll = torch.where(live, torch.maximum(roll, o[:, O["roll"]].abs()), roll)
        grounded = o[:, O["grounded"]] > 0.5
        best = torch.where(live & grounded, torch.maximum(best, env.wheel_ground()), best)
        gw = env.wheel_grounds()
        t_live += lf * env.dt
        t_ground += (live & grounded).float() * env.dt
        one_wheel += (live & grounded & ((gw[:, 0] - gw[:, 1]).abs() > 0.03)).float() * env.dt
        hd = torch.stack([R[:, 0, 1], R[:, 1, 1]], -1)
        hd = hd / hd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v = (env.qvel[:, 0:2] * hd).sum(-1)
        yr = (R @ env.qvel[:, 3:6, None])[:, 2, 0]
        tau = o[:, O["tau_hYp_xp"]:O["tau_w_xm"] + 1]
        s["verr"] += (v - env.ramp).abs() * lf
        s["yerr"] += (yr - env.cmd[:, 1].clamp(-o[:, O["yaw_limit"]], o[:, O["yaw_limit"]])).abs() * lf
        s["lerr"] += (0.5 * (o[:, O["L_xp"]] + o[:, O["L_xm"]]) - env.cmd[:, 2]).abs() * lf
        s["roll_mean"] += o[:, O["roll"]].abs() * lf
        s["pitchr"] += o[:, O["pitch_rate"]].abs() * lf
        s["power"] += (tau * env.qvel[:, hv]).clamp_min(0).sum(-1) * lf
        s["torque"] += tau.square().sum(-1) * lf
        s["split"] += (o[:, O["th_xp"]] - o[:, O["th_xm"]]).abs() * lf
        new = dones & ~done
        fell |= new & env.fell_now                  # a success also ends an episode (not a time-out)
        arrived |= new & env.arrived
        finished |= live & env.finished              # open flight climbed: the flag goes off, it holds on top
        dist = torch.where(new, (prev - start[:, 0:2]).norm(dim=-1), dist)
        done |= dones
        if tile_only:       # leaving the own tile ends the robot's run (what lies beyond is another type or level)
            c = env.tile_centers[env.level, env.ttype]
            out_ = ((env.qpos[:, 0:2] - c).abs() > env.terrain["tile"] / 2 - 0.1).any(-1) & ~done
            left |= out_
            dist = torch.where(out_, (env.qpos[:, 0:2] - start[:, 0:2]).norm(dim=-1), dist)
            done |= out_
    dist = torch.where(~done, (env.qpos[:, 0:2] - start[:, 0:2]).norm(dim=-1), dist)
    narrow = env.flight != 0
    ok = ~fell & torch.where(narrow, arrived, (dist >= distance) | finished | left)
    steps_alive = (t_live / env.dt).clamp_min(1)
    avg = {k: v / steps_alive for k, v in s.items()}
    return dict(ok=ok, fell=fell, refused=~fell & ~ok, climb=climb, gained=best - g0, jumps=jumps, roll=roll,
                t_live=t_live, one_wheel=one_wheel / t_ground.clamp_min(1e-3), **avg)


def load_policy(env, args):
    if not (args.checkpoint or args.v11):
        return None
    from rsl_rl.runners import OnPolicyRunner
    from train import skill_ppo
    runner = OnPolicyRunner(env, skill_ppo(env, "encoders", symmetry=False), log_dir=None, device=args.device)
    if args.v11:
        import mirror
        mirror.init_from_v11(runner, args.v11, env.cfg.history)
    else:
        runner.load(args.checkpoint)
    return runner.get_inference_policy(device=args.device)


def make_env(args, types, L, suite):
    import skill_terrain as ST
    from skill_env import SkillCfg, SkillEnv
    use = [ST.TYPES.index(t) for t in types]
    N = len(use) * L * args.per
    common = dict(num_envs=N, curriculum=False, random_tile_type=False, p_parked_start=0.0,
                  episode_s=args.seconds + 1.0, p_climb=args.p_climb, terrain_variants=not args.v11_terrain,
                  rise_jitter=0.01, stall_end_s=1e9, tile_truncate=False, max_level_by_type=(),
                  overhang_leg_max=getattr(args, "overhang_leg_max", 1.0))    # the v12 evaluation terrain; refusals counted, not ended
    if suite == "assist":
        cfg = SkillCfg(**common, push_speed=args.push, push_interval_s=(2.0, 6.0))
    else:
        speed = None if args.speed == "table" else tuple(float(v) for v in args.speed.split(","))
        cfg = SkillCfg(**common, push_speed=0.0, push_speed_skill=0.0, eval_speed=speed, p_flag_late=0.0,
                       p_cancel=1.0 if suite == "cancel" else 0.0, cancel_after_s=(1.5, 6.0))
    env = SkillEnv(cfg, device=args.device, seed=args.seed)
    if args.motors != "random":
        from skill_env import W0
        tq, w0, lag, pw, dl = MOTORS[args.motors]
        env.drv[:, 0], env.drv[:, 1], env.drv[:, 2], env.drv[:, 3], env.drv[:, 4] = tq, w0 * W0, lag, pw, dl
    idx = torch.arange(N, device=env.device)
    tile = idx * (len(use) * L) // N
    env.ttype[:], env.level[:] = torch.tensor(use, device=env.device)[tile % len(use)], tile // len(use)
    return env, use


def suite_skill(args, policies, env, use, L):
    import skill_terrain as ST
    all_types = ST.TYPES
    results = {}
    print(f"{args.seconds:.0f} s per robot, motors: {args.motors}, terrain seed {args.seed}. Per level 0..{L - 1}: "
          f"ok/fell/refused %; +- = widest 95% interval of ok; jump starts per robot, steps, roll / split [deg], "
          f"1-wheel [%]")
    for name, pol in policies:
        r = run(env, pol, args.seconds, args.distance, args.tile_only)
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
                    rise = ST.STAIR_RISE[lv]
                    steps = float(r["gained"][sel].clamp_min(0).sum()) / rise if tn.startswith("stairs") else None
                    cells.append(dict(n=n, ok=ok, fell=f(r["fell"]), refused=f(r["refused"]),
                                      ci=1.96 * math.sqrt(max(ok * (1 - ok), 1e-9) / n), jumps=f(r["jumps"]),
                                      steps=steps / n if steps is not None else None,
                                      falls_per_step=(float(r["fell"][sel].sum()) / max(steps, 1.0)) if steps else None,
                                      roll=math.degrees(float(r["roll"][sel].median())),
                                      split=math.degrees(f(r["split"])), one_wheel=100 * f(r["one_wheel"])))
                have = [c for c in cells if c]
                if not have or max(c["n"] for c in have) < 10:
                    continue
                rows[sub] = cells
                cell = lambda c: "     -     " if c is None else f"{c['ok'] * 100:3.0f}/{c['fell'] * 100:3.0f}/{c['refused'] * 100:3.0f}"
                print(f"  {sub:17s} {name:10s} " + " ".join(cell(c) for c in cells)
                      + f"   +-{max(c['ci'] for c in have) * 100:2.0f}  n~{sum(c['n'] for c in have) // len(have)}")
                col = lambda k, fmt: " ".join("  - " if c is None else format(c[k], fmt) for c in cells)
                extra = "jumps " + col("jumps", "4.1f")
                if tn.startswith("stairs") and sub.endswith("up"):
                    extra += "   steps " + col("steps", "4.1f")
                print(f"  {'':17s} {'':10s} {extra}   roll " + col("roll", "3.0f") + "   split " + col("split", "3.0f")
                      + "   1-wheel " + col("one_wheel", "3.0f"))
                if tn == "stairs_narrow" and sub.endswith("up"):
                    fps = [None if c is None else c["falls_per_step"] for c in cells]
                    print(f"  {'':17s} {'':10s} falls per step % "
                          + " ".join("  -  " if v is None else f"{100 * v:5.2f}" for v in fps)
                          + "   -> done over 50 steps % "
                          + " ".join("  - " if v is None else f"{100 * (1 - v) ** 50:4.0f}" for v in fps))
        results[name] = rows
    return results


def suite_assist(args, policies, env, use, L):
    import skill_terrain as ST
    results = {}
    print(f"assist, {args.seconds:.0f} s per robot, pushes +-{args.push} m/s per axis every 2-6 s, terrain seed "
          f"{args.seed}")
    print(f"  {'':10s} {'terrain':7s} falls/min fell% | speed err | yaw err | leg err cm | roll deg | pitch rate | "
          f"power W | torque rms | split deg | jumps")
    for name, pol in policies:
        r = run(env, pol, args.seconds, args.distance, args.tile_only)
        rows = {}
        for t in use:
            s = env.ttype == t
            mins = float(r["t_live"][s].sum()) / 60
            m = lambda k: float(r[k][s].float().mean())
            row = dict(falls_per_min=float(r["fell"][s].sum()) / max(mins, 1e-6), fell=m("fell"), speed_err=m("verr"),
                       yaw_err=m("yerr"), leg_err_cm=100 * m("lerr"), roll_deg=math.degrees(m("roll_mean")),
                       pitch_rate=m("pitchr"), power_w=m("power"), torque_rms=math.sqrt(m("torque") / 6),
                       split_deg=math.degrees(m("split")), jumps=m("jumps"))
            rows[ST.TYPES[t]] = row
            print(f"  {name:10s} {ST.TYPES[t]:7s} {row['falls_per_min']:5.2f} {100 * row['fell']:5.1f} | "
                  f"{row['speed_err']:.3f} | {row['yaw_err']:.3f} | {row['leg_err_cm']:4.1f} | {row['roll_deg']:4.2f} | "
                  f"{row['pitch_rate']:.3f} | {row['power_w']:5.1f} | {row['torque_rms']:.2f} | {row['split_deg']:4.1f} | "
                  f"{row['jumps']:.2f}")
        results[name] = rows
    return results


def suite_cancel(args, policies, env, use, L):
    import skill_terrain as ST
    results = {}
    print("cancel half way (flag off, speed 0 after 1.5-6 s), climbing robots: fell % per level")
    for name, pol in policies:
        r = run(env, pol, args.seconds, args.distance, args.tile_only)
        rows = {}
        for t in use:
            sel0 = (env.ttype == t) & r["climb"]
            cells = [float(r["fell"][sel0 & (env.level == lv)].float().mean()) if bool((sel0 & (env.level == lv)).any())
                     else None for lv in range(L)]
            rows[ST.TYPES[t]] = cells
            print(f"  {ST.TYPES[t]:15s} {name:10s} " + " ".join("  - " if c is None else f"{100 * c:4.1f}" for c in cells))
        results[name] = rows
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="v12 RSL-RL checkpoint (omit: controller alone only)")
    ap.add_argument("--v11", default=None, help="a v11 teacher checkpoint, remapped to the v12 interface")
    ap.add_argument("--suite", choices=("skill", "assist", "cancel"), default="skill")
    ap.add_argument("--per", type=int, default=150, help="robots per (terrain type, level)")
    ap.add_argument("--types", default=None, help="comma-separated terrain types (default: all of the suite)")
    ap.add_argument("--p-climb", type=float, default=0.5, help="share of the stairs robots that climb")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--tile-only", action="store_true",
                    help="a robot's run ends when it leaves its own tile (otherwise it drives on into the neighbours "
                         "for the whole time and their falls count; the protocol up to v12b)")
    ap.add_argument("--overhang-leg-max", type=float, default=1.0,
                    help="cap of the commanded leg on overhang tiles (training from v13: 0.36; 1.0 = no cap)")
    ap.add_argument("--speed", default="0.5,1.0",
                    help="skill suites: commanded speed range 'lo,hi' for every feature, or 'table' for each "
                         "feature's training range (policy_interface.SKILLS train_speed)")
    ap.add_argument("--distance", type=float, default=2.5)
    ap.add_argument("--push", type=float, default=1.0, help="assist suite: push speed per axis [m/s]")
    ap.add_argument("--motors", choices=("random", *MOTORS), default="random", help="motor draw (default: as in training)")
    ap.add_argument("--controller", action="store_true", help="also evaluate the model-based controller alone")
    ap.add_argument("--v11-terrain", action="store_true", help="the v11 geometry (no variants)")
    ap.add_argument("--seed", type=int, default=123, help="terrain and randomisation seed (training uses 0)")
    ap.add_argument("--out", default=None, help="run directory: writes eval_<suite>[_<tag>].json there")
    ap.add_argument("--tag", default=None, help="suffix of the JSON file (default: the motor set)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import skill_terrain as ST
    L = 6
    if args.suite == "assist":
        types = args.types.split(",") if args.types else list(ASSIST)
    elif args.suite == "cancel":
        args.p_climb = 1.0
        types = args.types.split(",") if args.types else ["stairs", "stairs_narrow"]
    else:
        types = args.types.split(",") if args.types else [t for t in ST.TYPES if t not in ASSIST]
    env, use = make_env(args, types, L, args.suite)
    pol = load_policy(env, args)
    policies = ([("controller", None)] if args.controller or pol is None else []) + \
        ([("policy", pol)] if pol is not None else [])
    results = dict(skill=suite_skill, assist=suite_assist, cancel=suite_cancel)[args.suite](args, policies, env, use, L)
    if args.out:
        tag = args.tag or ("" if args.motors == "random" else args.motors)
        Path(args.out, f"eval_{args.suite}{'_' + tag if tag else ''}.json").write_text(json.dumps(
            dict(args=vars(args), results=results), indent=1))


if __name__ == "__main__":
    main()
