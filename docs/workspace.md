# Workspace and experiment conventions

```text
biped_project/
├── README.md
├── PRIMUS/                 # This Git repository: source and source assets
├── experiments/
│   ├── <timestamp>-terrain-<device>/
│   ├── state/              # Current task/device/asset smoke receipts
│   └── archive/            # Preserved original training runs and duplicate logs
├── videos/                 # Descriptive previews and per-experiment videos
├── reports/                # Human-readable reports and historical setup evidence
└── scratch/
    ├── isaac-runtime/      # Rebuildable simulation caches
    ├── bootstrap/          # Preserved original work folder
    └── workspace-scaffolding/
```

Run Git commands in `PRIMUS/`. Source, model definitions, tests and documentation
belong in Git. Package caches, Python environments, generated USDs, checkpoints,
TensorBoard logs and videos belong outside source history.

A run's `experiment.json` records arguments, device, source revision/dirty state,
stage commands, exit codes, checkpoint and status. The source snapshot and tracked
patch preserve the code used by runs that start with local changes. Evaluation
metrics describe policy performance; a successful PPO process alone does not.
Use unique run folders, preserve failed runs, and label reference-controller,
preview, training and policy-playback videos accurately.

The organization migration on 2026-10-31 preserved everything. The old repository
moved from `outputs/PRIMUS` to `PRIMUS`; surrounding `outputs` reports moved to
`reports/2026-10-31-bootstrap`. Old run folders moved to
`experiments/archive/2026-10-31-cpu-bootstrap`. Historical files keep their original
paths as provenance; those paths may no longer exist. Do not execute archived
virtual environments after moving them: their scripts can contain absolute paths.
The active shared runtime and Isaac Lab stayed at `~/test/isaac-rl` and
`~/test/IsaacLab`; the Isaac installation stayed in its original Conda environment.
The separate archived `scratch/bootstrap/IsaacLab-6.1` is not the selected runtime.

The workspace folder was subsequently renamed to `biped_project` (underscores).
`experiments/latest` and `videos/latest` point to the latest reviewed GPU run.
The videos folder for that run provides `rollout.mp4` and `training-progress.mp4`
for convenient viewing; the native clips remain in their original subdirectories.
