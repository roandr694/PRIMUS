"""Consistent external paths for generated data; supports standalone repo clones."""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get('PRIMUS_WORKSPACE', REPO.parent)).expanduser().resolve()
EXPERIMENTS = WORKSPACE/'experiments'
CACHE_ROOT = WORKSPACE/'scratch/isaac-runtime'
# AppLauncher splits kit_args on whitespace. Use a stable alias for spaced workspace paths.
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
CACHE = CACHE_ROOT
if any(char.isspace() for char in str(CACHE_ROOT)):
    alias = Path.home()/'.cache/primus-kit'/hashlib.sha256(str(WORKSPACE).encode()).hexdigest()[:12]
    alias.parent.mkdir(parents=True,exist_ok=True)
    if alias.is_symlink() and alias.resolve()!=CACHE_ROOT: raise RuntimeError('Unexpected Kit cache alias target')
    if not alias.exists(): alias.symlink_to(CACHE_ROOT,target_is_directory=True)
    CACHE = alias
STATE = EXPERIMENTS/'state'
RUN = Path(os.environ.get('PRIMUS_RUN_DIR', EXPERIMENTS/'manual')).expanduser().resolve()


def receipt_path(task='balance'):
    return STATE/('smoke-'+task+'.json')


def physics_fingerprint(task='balance'):
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve().parent/'usd/robo_bipedal.usd',
             REPO/'hardware/actuators.json',Path(__file__).resolve().parent/'isaaclab_robo_cfg.py']
    paths += sorted((Path(__file__).resolve().parent/'primus_task').glob('*.py'))
    paths += sorted((Path(__file__).resolve().parent/'usd/imported').rglob('*.usd'))
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()
