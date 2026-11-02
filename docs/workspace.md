# Workspace and experiment conventions

```text
biped_project/
├── PRIMUS/                 # this Git repository: source, models, docs
├── experiments/            # one folder per training run; `latest` -> the current one
│   └── state/              # Isaac physics-check receipts
├── videos/                 # labelled copies of the run videos; `latest` -> the current one
├── reports/                # historical Isaac setup reports (2026-10-31)
└── scratch/isaac-runtime/  # rebuildable Isaac/Kit caches
```

Run Git commands in `PRIMUS/`. Source, model definitions, tests and docs belong in Git; Python environments,
caches, generated USDs, checkpoints, TensorBoard logs and videos stay outside. Set `PRIMUS_WORKSPACE` to put the
generated data elsewhere.

`sim/warp_rl/train.py` creates `experiments/<date>-<name>/` with `experiment.json` (arguments, Git revision and
dirty state, environment and PPO config, stage exit codes, status), `tracked-changes.patch`, `logs/`
(checkpoints + TensorBoard), `evaluation.md/json` and the videos. A finished PPO process is not evidence of a
good policy; the evaluation against the model-based controller is.

Shared runtime on this computer: `~/test/isaac-rl` (Python 3.12, MuJoCo 3.11, MuJoCo Warp, torch, RSL-RL) and
Isaac Lab v3.0.0-EA in `~/test/IsaacLab`. Do not move Python environments; recreate them.

History: on 2026-10-31 the repository moved from `outputs/PRIMUS` to `PRIMUS`, the workspace became
`biped_project`, and failed/diverged runs and the old bootstrap folder were moved to the system trash. Paths
in `reports/` that mention them no longer exist.
