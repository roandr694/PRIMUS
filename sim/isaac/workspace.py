"""External paths for generated Isaac data (caches and state live beside the repository)."""
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get('PRIMUS_WORKSPACE', REPO.parent)).expanduser().resolve()
STATE = WORKSPACE/'experiments/state'
CACHE = WORKSPACE/'scratch/isaac-runtime'
CACHE.mkdir(parents=True, exist_ok=True)
# Kit splits cache arguments on whitespace: use a stable alias for workspace paths with spaces.
if any(char.isspace() for char in str(CACHE)):
    alias = Path.home()/'.cache/primus-kit'/hashlib.sha256(str(WORKSPACE).encode()).hexdigest()[:12]
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(CACHE, target_is_directory=True)
    CACHE = alias
