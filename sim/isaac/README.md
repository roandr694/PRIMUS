# Isaac simulation and terrain learning

Current status (2026-10-31): the first GPU PPO terrain run (raw torques) did not learn to balance
(it sat on the landing gear); see the [post-mortem](../../docs/terrain-run-2026-10-31.md). Current RL work
runs in MuJoCo Warp ([warp_rl](../warp_rl/README.md)); Isaac returns later for large steps and a depth camera.

PRIMUS uses Isaac Sim 6.1.0.0, Isaac Lab **v3.0.0-EA** (commit
`ae37b028ea415c91ea2bc32609efcd759ed2b974`), Python 3.12 and RSL RL 5.4.1.
Do not mix this integration with Lab 2.2 or the Lab 3.0 stable/Sim 6.2 combination.

The shared runtime is selected by ignored `runtime.local.json`, or by
`ISAAC_PYTHON` and `ISAACLAB_PATH`. Defaults on this computer are
`~/test/isaac-rl/bin/python` and `~/test/IsaacLab`. Nothing installs packages into
that runtime automatically. A local Lab CPU allocation patch is preserved in
`patches/isaaclab-v3-ea-cpu-memory.patch`.

## Workspace convention

The repository is `Desktop/biped_project/PRIMUS`. See
[workspace conventions](../../docs/workspace.md). Generated data lives beside it:

- `experiments/<timestamp>-terrain-<device>/`: manifest, source snapshot, tracked
  diff, stage logs, checkpoints, evaluation metrics and optional pose trajectories.
- `experiments/state/`: physics smoke receipts, separated by task and checked
  against a hash of the current task source, actuator profile and USD assets.
- `videos/<experiment>/`: copies of training/play MP4s, retaining their relative
  paths; originals remain with the checkpoints.
- `scratch/isaac-runtime/`: disposable Kit/Warp caches. A small alias under
  `~/.cache/primus-kit/` points here because Kit splits cache arguments on spaces.

Set `PRIMUS_WORKSPACE` to relocate generated data. Set `PRIMUS_RUN_DIR` when using
individual scripts to choose their run folder. USD conversion stays under ignored
`sim/isaac/usd/` because it is an asset cache with relative references.

## GPU terrain experiment

From the repository root, in a terminal with NVIDIA access:

```bash
nvidia-smi
# Install the optional native video dependencies into the selected RL environment.
python3 -m pip --python "$HOME/test/isaac-rl/bin/python" install -r sim/isaac/requirements-video.txt
python3 sim/isaac/train_terrain.py --device cuda:0 --num-envs 128 --iterations 3000
```

The pipeline runs preflight → conversion → terrain physics/reset checks → PPO →
checkpoint evaluation over multiple seeds and course sections → recorded playback.
It records periodic clips **during training** using the native Lab video recorder.
It stops on failure, preserves an `experiment.json` with the failed stage, and
never falls back silently to CPU. A completed pipeline means the software ran;
inspect the evaluation and videos to decide whether the policy learned useful
locomotion. Increasing training iterations is not a guarantee of success.

Isaac may request acceptance of the [NVIDIA Omniverse license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
The owner must accept it. For an unattended run, only after accepting those terms,
set `OMNI_KIT_ACCEPT_EULA=YES` in that terminal. Video capture needs NVIDIA rendering
and an FFmpeg encoder; the pipeline can reuse the existing audio environment's
FFmpeg executable when it is not on PATH.

### Terrain task: `Primus-Terrain-Direct`

- Robot forward axis is **+Y**; each isolated environment has its own 3 m × 14 m
  collision mesh and 16 m spacing. Height observations sample that same triangle
  surface, including the mesh's stair-edge bevels.
- Course: flat pads, an 8% up/down ramp (8 cm peak), three 2.5 cm stair increments
  up/down with 30 cm treads, then a continuous uneven patch (at most ±12 mm).
  The 5 cm mesh spacing bevels stair edges; these are introductory small stairs.
- Physics runs at 500 Hz; the six normalized torque actions run at 100 Hz.
  Torques use the shared actuator profile, already on the joint side of the 9:1
  gearbox. Torque, rotor inertia and backlash still need hardware measurements.
- Actor: 22 proprioceptive observations, forward/yaw commands and 15 ideal local
  height samples (39 total). Critic adds true body linear velocity (42 total).
  The height scan is ideal simulated perception, **not** a validated real sensor.
- Commands: 20% standing, otherwise 0.15–0.45 m/s forward; yaw command is zero on
  this introductory narrow course. Rewards track forward speed, upright posture,
  nominal ground-relative height, smooth actions and limited lateral motion.
- Starts progress from flat to pads before ramp, stairs and uneven terrain after
  non-falling timeouts with >0.5 m forward progress. Early falls demote a stage.
  Episodes end after 20 s, course completion, falling, leaving the course,
  nonfinite physics or linkage gap >2 mm.
- Evaluation disables progression, tests all four starts and seeds 43/44/45,
  and reports survival, command error, progress and linkage gap separately.
  Old balance checkpoints have a different network/observation contract and
  cannot be resumed as terrain policies.

## Validation and original balance task

```bash
python3 sim/isaac/launch.py preflight.py --assets-only
/home/rodrigo/test/isaac-rl/bin/python -m unittest discover -s sim/isaac/tests -v
python3 sim/isaac/launch.py tests/check_configs.py

# Explicit CPU software validation; not a GPU training substitute.
python3 sim/isaac/train_terrain.py --device cpu --num-envs 8 --iterations 20 --no-video

# Original small standing experiment remains available.
PRIMUS_DEVICE=cpu bash sim/isaac/run_first_test.sh
```

The terrain smoke test cycles through all four spawn pads, checks finite
observations/rewards and pre-reset physics, linkage closure, and partial-reset
isolation. The independent MuJoCo tests verify closed leg poses at 0.18/0.27/0.36 m.

For individual commands:

```bash
python3 sim/isaac/launch.py convert_urdf_to_usd.py
python3 sim/isaac/launch.py smoke_test.py --task terrain --device cuda:0 --num_envs 4
python3 sim/isaac/launch.py run_rl.py train --task Primus-Terrain-Direct \
  --device cuda:0 --num_envs 128 --max_iterations 3000 --visualizer kit --video
```

Conversion invalidates all smoke receipts. Training/play requires a successful
receipt for the same task, device and source/asset fingerprint. CPU success does
not authorize GPU physics automatically.

## Videos without re-running physics

`render_rollout.py` uses software OpenGL and MuJoCo only to render poses recorded
by `evaluate_terrain.py`. It does not execute the policy in MuJoCo or claim
cross-simulator validation. Videos carry an on-screen label identifying their
origin. Without a trajectory it produces a stationary course preview:

```bash
/home/rodrigo/test/isaac-rl/bin/python sim/isaac/render_rollout.py \
  --output ../videos/terrain-environment-preview.mp4
```

The original CPU balance results are archived outside Git. They showed a working
PPO pipeline but frequent falls; they are not evidence of terrain learning.
