# First GPU terrain PPO run — 2026-10-31

The full pipeline completed on the NVIDIA GeForce RTX 5070 using the source
committed with this record. All Python files matched the run's source snapshot
at review time. The policy is a useful baseline; it has not mastered the course.

## Run and artifacts

- Run: `2026-10-31_15-16-02-terrain-cuda-0`.
- Task: `Primus-Terrain-Direct`; seed 42; 128 environments; 3,000 updates.
- Experience: 24,576,000 transitions (64 steps per environment per update).
- PPO wall time reported by the trainer: 7,658.93 s (about 2 h 8 min).
- Checkpoint: `logs/rsl_rl/primus_terrain/2026-10-31_15-16-36/model_2999.pt` inside the run folder.
- GPU physics gate: four robots, 300 steps, maximum linkage gap 0.0587 mm.
- Videos: twelve native 8-second training clips and one native 20-second checkpoint playback.
- Review exports: a 24-second early/middle/late training comparison and a convenient rollout link.
- Experiment manifests, checkpoint, videos and raw metrics remain outside Git in the workspace.

Raw evidence: [manifest](../../experiments/2026-10-31_15-16-02-terrain-cuda-0/experiment.json),
[evaluation](../../experiments/2026-10-31_15-16-02-terrain-cuda-0/evaluation.json),
[rollout](../../videos/2026-10-31_15-16-02-terrain-cuda-0/rollout.mp4),
[training comparison](../../videos/2026-10-31_15-16-02-terrain-cuda-0/training-progress.mp4).
These links require the local workspace artifacts; cloning Git alone does not retrieve them.

## Bounded evaluation

Two environments per trial, seeds 43/44/45, 2,000 policy steps (20 seconds) per
trial, with fixed flat starting pads before each section. Automatic resets can
produce multiple episodes in a trial. All twelve trials reported finite physics.
The maximum recorded linkage gap was 0.408 mm, below the 2 mm termination threshold.

| Starting section | Completed episodes | Timeout or course-completion endings | Mean absolute forward-speed error (m/s) |
|---|---:|---:|---:|
| flat | 6 | 4 | 0.152 |
| ramp | 6 | 4 | 0.176 |
| stairs | 8 | 2 | 0.180 |
| uneven | 6 | 4 | 0.122 |

The ending counts combine timeouts with course completion. They are not obstacle
clearance rates or a robustness success rate. Other endings can include falls,
leaving the course, or a linkage-gap violation. Unfinished episodes are retained
in the raw evaluation and excluded from completed-episode counts. Speed errors
are averages across the three trials, including standing commands and resets.

The stair starts produced six early endings among eight completed episodes.
The other starting sections performed better, but videos and metrics do not
establish reliable command following or traversal of every obstacle. Training
clips show sampled behavior during stochastic PPO; checkpoint playback is a
separate policy rollout.

## Validation and next work

All six asset/geometry tests passed after the run; the runtime configuration
regression check and CPU smoke/PPO checks had passed before it. GPU preflight,
conversion, physics smoke, training, evaluation and playback all exited zero.
All thirteen native video containers were probed, and frames from early/late
training and playback were inspected. Checkpoints and caches are excluded from Git.

Next work should focus on stair clearance and speed tracking, with separate
obstacle completion measurements and per-environment progress. Longer training
alone is not established as sufficient. Hardware parameters, perception noise
and sim-to-real validation remain unresolved.
