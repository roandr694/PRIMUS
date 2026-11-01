"""Evaluate fixed terrain starts across seeds; retain rollout poses for honest replay."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from workspace import CACHE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trajectory',type=Path)
    args=p.parse_args()
    if not args.checkpoint.is_file() or args.steps<1: p.error('Existing checkpoint and steps >= 1 required')
    from preflight import inspect_runtime
    errors=inspect_runtime(args.device)['errors']
    if errors: raise SystemExit('\n'.join(errors))
    from isaaclab.app import AppLauncher
    cache=CACHE/'terrain-evaluation-kit';cache.mkdir(parents=True,exist_ok=True)
    launcher=AppLauncher(headless=True,device=args.device,kit_args=f'--portable-root {cache}')
    env=None
    try:
        import numpy as np
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from primus_task.terrain_cfg import PrimusTerrainEnvCfg, PrimusTerrainPPORunnerCfg
        from primus_task.terrain_env import PrimusTerrainEnv
        cfg=PrimusTerrainEnvCfg();cfg.scene.num_envs=2;cfg.sim.device=args.device
        cfg.curriculum=False;cfg.seed=args.seeds[0]
        env=PrimusTerrainEnv(cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=1.0)
        agent=handle_deprecated_rsl_rl_cfg(PrimusTerrainPPORunnerCfg(),'5.4.1');agent.device=args.device
        runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
        runner.load(str(args.checkpoint));policy=runner.get_inference_policy(device=args.device)
        trials=[];poses=[];joints=[];resets=[];labels=[]
        for stage in range(4):
            cfg.evaluation_stage=stage
            for seed in args.seeds:
                env.reset(seed=seed);obs=wrapped.get_observations()
                policy.reset(torch.ones(2,dtype=torch.bool,device=args.device))
                elapsed=torch.zeros(2,device=args.device)
                episodes=[];survived=0;max_gap=0.0;tracking=[];progress=0.0
                with torch.inference_mode():
                    for step in range(args.steps):
                        action=policy(obs)
                        obs,reward,dones,extras=wrapped.step(action)
                        if not torch.isfinite(reward).all() or not bool(extras['log']['Physics/finite']):
                            raise RuntimeError('Nonfinite physical state/reward')
                        max_gap=max(max_gap,float(extras['log']['Loop/max_gap_m']))
                        elapsed+=env.step_dt
                        tracking.append(float(extras['log']['Tracking/forward_error_mps']))
                        progress=max(progress,float(extras['log']['Terrain/progress_m']))
                        for index in dones.nonzero().flatten().tolist():
                            episodes.append(float(elapsed[index]));elapsed[index]=0
                            survived+=int(extras['time_outs'][index])
                        policy.reset(dones)
                        if args.trajectory and seed==args.seeds[0] and step%4==0:
                            pose=env.robot.data.root_link_pose_w.torch[0].cpu().numpy().copy()
                            pose[:3]-=env.scene.env_origins[0].cpu().numpy()
                            poses.append(pose);joints.append(env.robot.data.joint_pos.torch[0].cpu().numpy().copy())
                            resets.append(bool(dones[0]));labels.append(stage)
                trials.append(dict(stage=stage,terrain=['flat','ramp','stairs','uneven'][stage],seed=seed,
                    completed_episodes=len(episodes),timeouts_or_course_completions=survived,
                    mean_completed_episode_s=sum(episodes)/len(episodes) if episodes else None,
                    mean_forward_error_mps=sum(tracking)/len(tracking),max_progress_m=progress,
                    max_loop_gap_m=max_gap,finite_physics=True,
                    unfinished_episode_s=elapsed.cpu().tolist()))
        result=dict(checkpoint=str(args.checkpoint.resolve()),device=args.device,steps_per_trial=args.steps,
            num_envs=2,trials=trials,
            scope='Fixed flat pads before each course section; 3 seeds by default. No sim-to-real claim.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        if args.trajectory:
            args.trajectory.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.trajectory,poses=poses,joints=joints,resets=resets,stages=labels,
                joint_names=np.array(env.robot.joint_names),dt=env.step_dt*4,
                checkpoint=str(args.checkpoint.resolve()),device=args.device)
        print(json.dumps(result,indent=2),flush=True)
    finally:
        failed=sys.exc_info()[0] is not None
        if failed: traceback.print_exc()
        if env is not None: env.close()
        launcher.app.close(exit_code=int(failed))


if __name__=='__main__': main()
