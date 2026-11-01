"""Use the official Isaac Lab train/play CLI with the external PRIMUS task."""
import os
from workspace import CACHE, RUN, receipt_path, physics_fingerprint
import json
import argparse
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in ('train','play') else 'train'
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task',default='Primus-Balance-Direct')
    parser.add_argument('--device',default='cuda:0')
    selected,_ = parser.parse_known_args(args)
    if '--help' not in args and '-h' not in args:
        from preflight import inspect_runtime
        result = inspect_runtime(selected.device, video='--video' in args)
        if result['errors']:
            raise SystemExit('Preflight failed:\n- '+'\n- '.join(result['errors']))
        if not (Path(__file__).resolve().parent/'usd/robo_bipedal.usd').is_file():
            raise SystemExit('Convert the URDF and pass smoke_test.py before training.')
        task = 'terrain' if selected.task=='Primus-Terrain-Direct' else 'balance'
        receipt = receipt_path(task)
        if not receipt.is_file() or not json.loads(receipt.read_text()).get('passed'):
            raise SystemExit('Run smoke_test.py successfully before training or playback.')
        if json.loads(receipt.read_text()).get('fingerprint') != physics_fingerprint(task):
            raise SystemExit('Physics/source changed: repeat smoke_test.py for this task.')
        if json.loads(receipt.read_text()).get('device') != selected.device:
            raise SystemExit('Repeat smoke_test.py on the requested training/playback device first.')
    import primus_task
    from isaaclab_rl.entrypoints import run_train_cli, run_play_cli
    if '--task' not in args: args = ['--task','Primus-Balance-Direct',*args]
    # The official non-distributed CLI overrides physics and agent devices separately.
    if not any(arg.startswith('agent.device=') for arg in args):
        args.append('agent.device='+selected.device)
    if not any(arg.startswith('--kit_args') for arg in args):
        cache = CACHE/'train-kit'
        cache.mkdir(parents=True,exist_ok=True)
        args.extend(['--kit_args','--portable-root '+str(cache)])
    if '--checkpoint' in args:
        index=args.index('--checkpoint')+1
        if index<len(args): args[index]=str(Path(args[index]).expanduser().resolve())
    RUN.mkdir(parents=True,exist_ok=True)
    os.chdir(RUN)
    return (run_train_cli if mode=='train' else run_play_cli)(args)


if __name__=='__main__': raise SystemExit(main())
