"""Physical gate: clone/reset isolation, finite tensors and closed-loop gap < 2 mm.

No claim about balance learning is made by this test. Zero torque can fall.
"""
import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from workspace import CACHE, RUN, receipt_path, physics_fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['balance','terrain'],default='balance')
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--num_envs',type=int,default=2)
    parser.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if args.output is None: args.output=receipt_path(args.task)
    args.output.unlink(missing_ok=True)
    if args.steps<1 or args.num_envs<2: parser.error('Use steps >= 1 and num_envs >= 2 for reset isolation')
    from preflight import inspect_runtime
    errors=inspect_runtime(args.device)['errors']
    if errors: raise SystemExit('Preflight failed:\n- '+'\n- '.join(errors))
    from isaaclab.app import AppLauncher
    cache=CACHE/'smoke-kit'
    cache.mkdir(parents=True,exist_ok=True)
    launcher=AppLauncher(headless=True,device=args.device,kit_args=f'--portable-root {cache}')
    env=None
    try:
        import torch
        from primus_task.env_cfg import PrimusBalanceEnvCfg
        from primus_task.env import PrimusBalanceEnv
        if args.task=='terrain':
            from primus_task.terrain_cfg import PrimusTerrainEnvCfg as PrimusBalanceEnvCfg
            from primus_task.terrain_env import PrimusTerrainEnv as PrimusBalanceEnv
        cfg=PrimusBalanceEnvCfg(); cfg.seed=42; cfg.scene.num_envs=args.num_envs; cfg.sim.device=args.device
        if args.task=='terrain': cfg.curriculum=False
        env=PrimusBalanceEnv(cfg)
        obs,_=env.reset(seed=42)
        assert obs['policy'].shape==(args.num_envs,cfg.observation_space)
        assert obs['critic'].shape==(args.num_envs,cfg.state_space)
        max_gap=env.loop_gaps().max().item()
        assert math.isfinite(max_gap) and max_gap<=cfg.max_loop_gap, ('reset loop gap',max_gap)
        before=env.robot.data.root_link_pose_w.torch[1:].clone()
        env._reset_idx(torch.tensor([0],device=env.device,dtype=torch.long))
        assert torch.allclose(before,env.robot.data.root_link_pose_w.torch[1:]), 'Partial reset changed another environment'
        # Check the inference-to-reset transition used between evaluation seeds.
        with torch.inference_mode():
            env.step(torch.zeros((args.num_envs,6),device=env.device))
        env.reset(seed=42)
        for step in range(args.steps):
            if args.task=='terrain' and step % max(1,args.steps//4)==0:
                cfg.evaluation_stage=min(3,step//max(1,args.steps//4))
                env.reset(seed=42+cfg.evaluation_stage)
            actions=torch.zeros((args.num_envs,6),device=env.device) if step<args.steps//2 else 0.1*torch.rand((args.num_envs,6),device=env.device)-0.05
            obs,reward,terminated,truncated,extras=env.step(actions)
            assert all(torch.isfinite(x).all() for x in [*obs.values(),reward]), 'Nonfinite observation/reward'
            gap=float(extras['log']['Loop/max_gap_m'])
            assert bool(extras['log']['Physics/finite']), 'Nonfinite physical state before automatic reset'
            assert math.isfinite(gap) and gap<=cfg.max_loop_gap, ('loop gap',gap)
            max_gap=max(max_gap,gap)
        result={'task':args.task,'fingerprint':physics_fingerprint(args.task),'passed':True,'device':args.device,'num_envs':args.num_envs,'steps':args.steps,'max_loop_gap_m':max_gap,'seed':42}
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        failed = sys.exc_info()[0] is not None
        if failed: traceback.print_exc()
        if env is not None: env.close()
        launcher.app.close(exit_code=int(failed))


if __name__=='__main__': main()
