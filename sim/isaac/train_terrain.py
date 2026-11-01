"""Run a tracked terrain experiment. No CPU fallback for requested GPU runs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from workspace import REPO, EXPERIMENTS, WORKSPACE

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cuda:0','cpu'],default='cuda:0')
    p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-video',action='store_true')
    args = p.parse_args()
    if args.iterations<1 or args.num_envs<4: p.error('Use iterations >= 1 and num-envs >= 4')
    if args.device=='cpu' and not args.no_video:
        p.error('CPU checks require --no-video; Kit video rendering needs an NVIDIA GPU')
    name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'-terrain-'+args.device.replace(':','-')
    run = EXPERIMENTS/name; run.mkdir(parents=True)
    env = os.environ.copy(); env['PRIMUS_RUN_DIR'] = str(run)
    # Reuse an installed encoder if it is not on PATH; never install into shared Sim.
    encoder = shutil.which('ffmpeg')
    local_encoder = Path.home()/'anaconda3/envs/audio/bin/ffmpeg'
    if not encoder and local_encoder.is_file(): encoder=str(local_encoder)
    if encoder: env.setdefault('IMAGEIO_FFMPEG_EXE',encoder)
    manifest = dict(name=name,started_utc=datetime.now(timezone.utc).isoformat(),
                    options=vars(args),status='running',steps=[],
                    git_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                    git_status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True))
    (run/'tracked-changes.patch').write_text(subprocess.check_output(
        ['git','-C',str(REPO),'diff','--binary','HEAD'],text=True))
    manifest['source_snapshot_scope']='Isaac scripts, URDF and actuator profile; CAD meshes remain in the Git repository'
    # Preserve the exact source used even when a run precedes a commit.
    shutil.copytree(HERE,run/'source',ignore=shutil.ignore_patterns('usd','__pycache__','runtime.local.json','logs','runs'))
    shutil.copy2(REPO/'hardware/actuators.json',run/'actuators.json')
    def save(): (run/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def stage(label, script, *arguments):
        command=[sys.executable,str(HERE/'launch.py'),script,*map(str,arguments)]
        entry=dict(stage=label,command=command);manifest['steps'].append(entry);save()
        print(f'[{label}] Log: {run/(label+".log")}',flush=True)
        with (run/(label+'.log')).open('w') as output:
            with subprocess.Popen(command,cwd=HERE,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                for line in process.stdout:
                    output.write(line);output.flush()
                    print(line,end='',flush=True)
                returncode=process.wait()
        entry['exit_code']=returncode;save()
        if returncode:
            raise RuntimeError(label+' failed; see '+str(run/(label+'.log')))
    try:
        stage('preflight','preflight.py','--device',args.device,'--output',run/'preflight.json',
              *([] if args.no_video else ['--video']))
        stage('conversion','convert_urdf_to_usd.py')
        stage('smoke','smoke_test.py','--task','terrain','--device',args.device,'--num_envs',4,'--steps',300)
        video=[] if args.no_video else ['--video','--video_length','800','--video_interval','16000']
        stage('train','run_rl.py','train','--task','Primus-Terrain-Direct',
              '--device',args.device,'--num_envs',args.num_envs,'--max_iterations',args.iterations,
              '--seed',args.seed,'--visualizer','none' if args.no_video else 'kit',*video)
        checkpoints=list((run/'logs/rsl_rl/primus_terrain').glob('*/model_*.pt'))
        if not checkpoints: raise RuntimeError('Training returned without a checkpoint')
        checkpoint=max(checkpoints,key=lambda path:int(path.stem.split('_')[-1]))
        manifest['checkpoint']=str(checkpoint);save()
        stage('evaluation','evaluate_terrain.py','--device',args.device,'--checkpoint',checkpoint,
              '--output',run/'evaluation.json','--trajectory',run/'rollout.npz')
        if not args.no_video:
            stage('play','run_rl.py','play','--task','Primus-Terrain-Direct',
                  '--device',args.device,'--num_envs',1,'--checkpoint',checkpoint,
                  '--visualizer','kit','--video','--video_length','2000','--video_interval','0')
            destination=WORKSPACE/'videos'/name;destination.mkdir(parents=True,exist_ok=True)
            for clip in (run/'logs').rglob('*.mp4'):
                relative=clip.relative_to(run/'logs')
                target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(clip,target)
            clips=list(destination.rglob('*.mp4'))
            if not clips: raise RuntimeError('No videos were written; inspect rendering/encoder setup')
            for kind in ('train','play'):
                if not any(kind in clip.relative_to(destination).parts for clip in clips):
                    raise RuntimeError('Missing '+kind+' video; inspect rendering/encoder setup')
            manifest['videos']=[str(clip) for clip in clips]
        manifest['status']='completed'
    except Exception as error:
        manifest['status']='failed';manifest['error']=str(error)
        raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        print('Experiment: '+str(run),flush=True)


if __name__=='__main__': main()
