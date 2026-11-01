"""Select an existing shared runtime without installing or changing packages."""
import os
import json
from pathlib import Path
import subprocess
import sys


def main():
    from workspace import CACHE
    here = Path(__file__).resolve().parent
    local_config = here / 'runtime.local.json'
    settings = json.loads(local_config.read_text()) if local_config.is_file() else {}
    python = Path(os.environ.get('ISAAC_PYTHON', settings.get('python', str(Path.home()/'test/isaac-rl/bin/python'))))
    lab = os.environ.get('ISAACLAB_PATH', settings.get('lab', str(Path.home()/'test/IsaacLab')))
    root = Path(lab).expanduser().resolve()
    tag = subprocess.run(['git','-C',str(root),'describe','--tags','--exact-match'],capture_output=True,text=True)
    if tag.returncode or tag.stdout.strip()!='v3.0.0-EA':
        raise SystemExit('Isaac Sim 6.1 requires tag v3.0.0-EA here; do not mix with Lab 2.2 or 3.0 stable')
    if not python.is_file(): raise SystemExit('Python missing: '+str(python))
    env = os.environ.copy(); env['ISAACLAB_PATH']=str(root)
    env.setdefault('WARP_CACHE_PATH', str(CACHE/'warp-cache'))
    env.setdefault('XDG_CACHE_HOME', str(CACHE/'cache'))
    env.setdefault('PYTHONUNBUFFERED', '1')
    paths = [str(here),*[str(p) for p in sorted((root/'source').iterdir()) if p.is_dir()]]
    if env.get('PYTHONPATH'): paths.append(env['PYTHONPATH'])
    env['PYTHONPATH']=os.pathsep.join(paths)
    args = sys.argv[1:] or ['preflight.py']
    script = Path(args[0])
    if not script.is_file(): script=here/script
    os.execve(str(python),[str(python),str(script),*args[1:]],env)


if __name__=='__main__': main()
