"""Bounded deterministic checkpoint evaluation, separate from training metrics."""
import argparse
import json
from pathlib import Path
from workspace import CACHE, RUN, receipt_path, physics_fingerprint
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda:0'])
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--output', type=Path, default=RUN/'balance-evaluation.json')
    args = parser.parse_args()
    if args.steps < 1 or not args.checkpoint.is_file():
        parser.error('Provide an existing checkpoint and at least one evaluation step')
    from preflight import inspect_runtime
    errors = inspect_runtime(args.device)['errors']
    if errors:
        raise SystemExit('\n'.join(errors))
    from isaaclab.app import AppLauncher
    cache = CACHE/'evaluation-kit'
    cache.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(headless=True, device=args.device, kit_args=f'--portable-root {cache}')
    env = None
    try:
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from primus_task.env import PrimusBalanceEnv
        from primus_task.env_cfg import PrimusBalanceEnvCfg
        from primus_task.ppo_cfg import PrimusPPORunnerCfg
        cfg = PrimusBalanceEnvCfg()
        cfg.seed = 43
        cfg.scene.num_envs = 2
        cfg.sim.device = args.device
        env = PrimusBalanceEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        agent = handle_deprecated_rsl_rl_cfg(PrimusPPORunnerCfg(), '5.4.1')
        agent.device = args.device
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        obs = wrapped.get_observations()
        elapsed = torch.zeros(2, device=args.device)
        lengths = []
        survived = 0
        max_gap = 0.0
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, reward, dones, extras = wrapped.step(policy(obs))
                assert torch.isfinite(reward).all() and bool(extras['log']['Physics/finite'])
                elapsed += cfg.decimation*cfg.sim.dt
                max_gap = max(max_gap, float(extras['log']['Loop/max_gap_m']))
                for index in dones.nonzero().flatten().tolist():
                    lengths.append(float(elapsed[index]))
                    survived += int(extras['time_outs'][index])
                    elapsed[index] = 0
                policy.reset(dones)
        result = dict(checkpoint=str(args.checkpoint.resolve()), device=args.device,
                      steps=args.steps, num_envs=2, seed=43, completed_episodes=len(lengths),
                      full_10s_episodes=survived,
                      mean_completed_episode_s=sum(lengths)/len(lengths) if lengths else None,
                      max_loop_gap_m=max_gap, finite_physics=True,
                      scope='Deterministic identical flat-ground resets; not a robustness benchmark')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        failed = sys.exc_info()[0] is not None
        if failed:
            traceback.print_exc()
        if env is not None:
            env.close()
        launcher.app.close(exit_code=int(failed))


if __name__ == '__main__':
    main()
