"""Read-only runtime and model checks; runs without starting Isaac Sim."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def check_assets():
    urdf = HERE/'urdf/robo_bipedal.urdf'
    root = ET.parse(urdf).getroot()
    links = {x.attrib['name'] for x in root.findall('link')}
    joints = {x.attrib['name'] for x in root.findall('joint')}
    errors = []; mass = 0.0
    for mesh in root.findall('.//mesh'):
        path = (urdf.parent/mesh.attrib['filename']).resolve()
        if not path.is_file(): errors.append(f'Missing mesh: {path}')
    for link in root.findall('link'):
        inertial = link.find('inertial'); m = float(inertial.find('mass').attrib['value']); mass += m
        a = {k:float(v) for k,v in inertial.find('inertia').attrib.items()}
        det = a['ixx']*a['iyy']*a['izz']+2*a['ixy']*a['ixz']*a['iyz']-a['ixx']*a['iyz']**2-a['iyy']*a['ixz']**2-a['izz']*a['ixy']**2
        if m<=0 or a['ixx']<=0 or a['ixx']*a['iyy']-a['ixy']**2<=0 or det<=0:
            errors.append('Invalid mass/inertia: '+link.attrib['name'])
    loops = json.loads((HERE.parent/'models/cad/loop_closures.json').read_text())
    if len(loops)!=2: errors.append('Expected two five-bar loop joints')
    for loop in loops:
        if loop['body0'] not in links or loop['body1'] not in links: errors.append('Unknown loop link: '+loop['name'])
        if loop['axis']!='X': errors.append('Unexpected loop axis')
    from kinematics import nominal_pose
    q,z = nominal_pose()
    if set(q)!=joints: errors.append('Nominal pose does not cover exactly the URDF joints')
    if not all(math.isfinite(x) for x in [*q.values(),z]): errors.append('Nonfinite reset pose')
    return {'links':len(links),'joints':len(joints),'loops':len(loops),'mass_kg':mass,'base_height_m':z,'errors':errors}


def version(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None


def inspect_runtime(device='cuda:0', video=False):
    packages = {n:version(n) for n in ['isaacsim','isaaclab','isaaclab-physx','isaaclab-rl','rsl-rl-lib','torch','gymnasium','moviepy','imageio-ffmpeg']}
    report = {'python':sys.executable,'python_version':sys.version.split()[0],'device':device,'packages':packages,'assets':check_assets()}
    lab_root = os.environ.get('ISAACLAB_PATH')
    report['isaaclab_path'] = lab_root
    report['isaaclab_release'] = (Path(lab_root)/'VERSION').read_text().strip() if lab_root and (Path(lab_root)/'VERSION').is_file() else None
    tag = subprocess.run(['git','-C',lab_root,'describe','--tags','--exact-match'],capture_output=True,text=True) if lab_root else None
    report['isaaclab_tag'] = tag.stdout.strip() if tag and not tag.returncode else None
    try:
        smi = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        report['gpu'] = {'exit_code':smi.returncode,'output':(smi.stdout+smi.stderr).strip()}
    except (OSError,subprocess.TimeoutExpired) as e: report['gpu']={'exit_code':1,'output':str(e)}
    try:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
    except ImportError: report['cuda_available'] = False
    errors = list(report['assets']['errors'])
    if not packages['isaacsim'] or not packages['isaacsim'].startswith('6.1.'): errors.append('This integration targets Isaac Sim 6.1')
    if sys.version_info[:2]!=(3,12): errors.append('Python 3.12 required')
    for n in ['isaaclab','isaaclab_physx','isaaclab_rl','gymnasium','rsl_rl','tensordict','hydra','tensorboard','prettytable','rich','h5py']:
        if importlib.util.find_spec(n) is None: errors.append('Missing Python module: '+n)
    if device.startswith('cuda') and not report['cuda_available']: errors.append('CUDA unavailable in this process; check device visibility outside the sandbox, or explicitly select --device cpu')
    if report['isaaclab_tag']!='v3.0.0-EA': errors.append('Set ISAACLAB_PATH to the v3.0.0-EA checkout for Sim 6.1')
    if video and (not packages['moviepy'] or packages['moviepy'].split('.')[0]!='1' or not packages['imageio-ffmpeg']):
        errors.append('Video dependencies missing or incompatible. Run from PRIMUS: python3 -m pip --python '+str(Path.home()/'test/isaac-rl/bin/python')+' install -r sim/isaac/requirements-video.txt')
    report['errors'] = errors
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--assets-only',action='store_true'); p.add_argument('--output',type=Path); p.add_argument('--video',action='store_true'); p.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0']); args=p.parse_args()
    result = check_assets() if args.assets_only else inspect_runtime(args.device, video=args.video)
    output=json.dumps(result,indent=2)
    print(output)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(output+'\n')
    raise SystemExit(bool(result['errors']))
