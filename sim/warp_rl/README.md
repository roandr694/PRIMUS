# Residual RL on the GPU (MuJoCo Warp)

> **TODO — retrain.** The base controller changed on 2026-11-06 (leg-sync gains scaled with leg length; mean leg
> length shifted so both legs stay in [0.18, 0.52] m on uneven ground; `max_accel` command, column 5 of
> `controller.cmd`, fixed at 1.5 m/s² in training; turning load-transfer feed-forward and turn-rate envelope —
> the yaw-tracking reward still compares with the raw command, consider comparing with `out["yaw_limit"]`). The kernel already matches it (parity test passes), but the
> v2 checkpoint and the results below were obtained with the old controller. Retrain with `train.py`, then re-run
> `eval.py`/`render.py` and update the numbers here and in `docs/control_roadmap.md`. A quick check confirms it is
> needed: with the new base controller the v2 policy raises falls (0.033 → 0.108 falls/min, 720 robots × 20 s)
> while the controller alone improved (0.267 → 0.188).

The model-based controller (LQR + VMC + loops, [`control/`](../../control/)) keeps balancing the robot. The RL
policy only adds a **residual** at defined points, and only in `operating` mode; with a zero action the robot is
exactly the original controller. Training runs in **MuJoCo Warp** on the GPU, with the same CAD model the LQR was
validated on.

```text
policy (50 Hz)                             action in [-1, 1]^6 x scale
  [dL_xp, dL_xm]   length of each leg (mean -> height; difference -> body roll)
  [dth_xp, dth_xm] angle of each leg (mean -> shifts the LQR equilibrium; difference -> one leg ahead)
  [dT_xp, dT_xm]   torque added to each wheel
        |
LQR + loops + automatic landing gear   (Warp kernel, 1 kHz, one thread per robot)
        |
VMC: analytic 5-bar Jacobian -> 4 hip torques + 2 wheel torques
        |
MuJoCo Warp (1 kHz), thousands of robots on one terrain
```

## Files

| file | what it does |
|---|---|
| `model.py` | training model: the CAD MJCF with simplified collision (cylinder tires + gear tips) and the terrain as a heightfield |
| `terrain.py` | one large shared terrain, as a grid (rows = difficulty 0–5; columns = flat, rough, slope, waves, steps, blocks) |
| `controller.py` | the controller of `control/repo_control_policy.py` ported to a Warp kernel, plus the residual |
| `env.py` | RSL-RL environment: reset, observations, rewards, termination, curriculum, pushes, randomisation |
| `train.py` | PPO (RSL-RL 5.4.1); creates `experiments/<date>-warp-residual/`; runs `eval.py` and `render.py` at the end |
| `eval.py` | LQR only vs LQR + RL on the same scenarios: falls per minute and speed error per terrain |
| `render.py` | `video_overview.mp4` (many robots on one terrain) and `video_compare.mp4` (LQR on top, LQR + RL below) |
| `tests/test_controller_parity.py` | compares the Warp kernel with the numpy controller, step by step and in closed loop |

## Run

```bash
cd PRIMUS
~/test/isaac-rl/bin/python sim/warp_rl/tests/test_controller_parity.py      # kernel == numpy controller
~/test/isaac-rl/bin/python sim/warp_rl/train.py --num-envs 4096 --iterations 1500
~/test/isaac-rl/bin/python sim/warp_rl/eval.py --checkpoint <run>/logs/model_1499.pt --out <run>
~/test/isaac-rl/bin/python sim/warp_rl/render.py --checkpoint <run>/logs/model_1499.pt --out <run>
```

Uses the shared Python environment `~/test/isaac-rl` (MuJoCo 3.11, MuJoCo Warp 3.11, Warp 1.16, torch cu130,
RSL-RL 5.4.1). On an RTX 5070 with 4096 robots: ~45k policy steps/s (~2.2 s per PPO iteration); 1500 iterations
take about an hour.

## Results (v2 policy, 1500 iterations)

2160 robots × 60 s on a terrain with a different seed, pushes and randomisation on, same scenarios in both columns:

| | model-based only | model-based + RL |
|---|---:|---:|
| falls per robot-minute | 0.265 | **0.069** |
| speed error (m/s) | 0.207 | **0.094** |

Better in all 18 terrain groups; largest gain on hard steps (0.89 → 0.18 falls/min); weakest on the 15° slope.

## Design decisions

- **Residual, not raw torques.** The LQR already balances. Training starts from the model-based controller (the
  actor's last layer is zeroed) and only learns corrections, which avoids the "sit on the landing gear" exploit
  of the Isaac run.
- **Per-leg, leg-space residual.** Legs may be misaligned (one ahead, one shorter) without the policy learning the
  5-bar kinematics.
- **Physics and controller in one CUDA graph.** 20 physics steps of 1 ms per policy step, the controller at every
  step, as on CPU MuJoCo. The kernel costs ~16 µs per step for 4096 robots.
- **Minimal collision.** MuJoCo Warp tests every collision part against every heightfield cell below it, and the
  cost grows quickly, so only the tires and gear tips touch the terrain. Falls are detected geometrically (tilt
  > 50°, chassis < 0.2 m above ground, controller protection). The heightfield has 8 cm cells, so a step edge
  is an 8 cm ramp.
- **Exploration bounded.** In a residual the exploration noise acts directly on the robot: std is limited to
  0.05–0.3 and there is no entropy bonus (an unbounded first attempt diverged: std 0.66, 75% falls).
- **Actor sees only what the robot measures** (IMU, leg length/angle from the hip encoders, wheel speeds,
  odometry, commands, 5-frame history, with noise). The critic also gets the true velocity, height above ground,
  a height map and the randomised parameters.
- **Randomisation:** friction 0.5–1.25, chassis mass ±15%, CoM ±2 cm, pushes up to 0.6 m/s every 3–7 s, 15% of
  episodes start parked. Torque (18 N·m) and mass (15.7 kg) stay at the current model values until measured.
- **Robustness of the batch.** A world that blows up numerically (rare) counts as a fall and receives a copy of a
  healthy world's internal state before its reset.

## Autonomous skills: teacher policy (`skill_env.py`)

Stairs up and down (rises 4–19 cm, treads 30 cm), curbs and drops (4–20 cm), gaps (10–45 cm) and bars to pass
under (free height down to 42 cm), learned on top of the same controller. Driven by the perception stack
([`perception/`](../../perception/README.md)): the pilot presses A when a skill is offered, the planner aligns the
robot and gives a speed and a heading, and the policy crosses.

| file | what it does |
|---|---|
| `skill_terrain.py` | tiles of 8 m (rows = level 0–5, columns = flat, rough, slope, stairs, curbs, gaps, overhang); sharp features are boxes; also a fine ground grid and a ceiling grid for perception |
| `skill_env.py` | teacher environment: collision="full" (chassis and legs touch boxes and bars), 9 actions, heading commands, privileged observations (true velocity, torque limit, friction, mass, a perfect height scan 1.5 m × 0.6 m and a ceiling scan), curriculum per robot (up after crossing 2.5 m, down after a fall or getting stuck) |
| `skill_eval.py` | controller alone vs policy, per terrain and level (stairs split into climb / descend) |
| `export_policy.py` | checkpoint → TorchScript + ONNX + `policy.json`, for `control/skill_policy.py` (CPU, robot) |
| `recovery_env.py` | stand up after a fall: from lying down to the parked pose, joints driven directly (PD at 1 kHz) |

Action (per unit, only while operating): leg lengths ±0.17 m, leg angles ±0.5 rad, wheel torques ±4 N·m, leg
forces ±150 N, and a **jump trigger**: > 0.5 starts the controller's jump primitive (push the legs out at full
force to 0.50 m, then tuck them to 0.20 m; 0.5 s cooldown; in `control/repo_control_policy.py`, mirrored in the
kernel). Alone, the primitive lifts the wheels 23–29 cm at 18–20 N·m and ~35 cm at 25 N·m (with the motor
torque-speed droop), and the robot lands balanced. The drive limit is drawn per robot (18–25 N·m) with that droop.

```bash
~/test/isaac-rl/bin/python sim/warp_rl/train.py --task skills --iterations 4000
~/test/isaac-rl/bin/python sim/warp_rl/skill_eval.py --checkpoint <run>/logs/model_3999.pt
~/test/isaac-rl/bin/python sim/warp_rl/export_policy.py --task skills --checkpoint <run>/logs/model_3999.pt
~/test/isaac-rl/bin/python sim/warp_rl/train.py --task recovery --iterations 1500
```

**v1 (no jump primitive, stopped at 1500 iterations).** Success (no fall and ≥ 2.5 m in 12 s) at iteration 900,
controller alone → policy: stairs down ~80% → 87–100% at every level; gaps up to 31 cm 0–83% → 62–88%; curbs and
bars clearly better; stairs **up** only at 4 cm (100%) and 7 cm (26%): from 10 cm the robot stays at the foot.
The exploration noise had collapsed to the minimum (std 0.07) and a two-wheel jump is a coordinated pulse that
noise never finds, hence v2: the jump primitive, std ≥ 0.1, and a small stand-still reward (v1 rocked when told
to stop). On the CPU simulator the exported v1 descended 13 cm stairs and crossed 17 cm gaps through the whole
perception stack (rendered depth → map → planner).

**v2 (jump primitive), iteration 1500:** gaps 31–45 cm 69–73% (v1: 31–62%), single curbs up to 20 cm 21% (v1
6%), bars down to 52 cm free height 83%, stairs up 7 cm 58% — but still 0% from 10 cm (no falls: the robots stay
at the foot or part-way). Climbing one or two steps earned nothing unless the robot covered 2.5 m, so **v3**
(resumed from v2) adds a dense `climb` reward (each new height of the ground under the wheels, only while on the
ground) and levels a stairs robot up after climbing 3 steps.

**Stand up after a fall (`recovery_env.py`):** 99.1% of 10,979 episodes reach the parked pose, median 2.0 s
(including 0.6 s of settling), with v2 of the environment (v1 never parked: the 0.18 m leg target sat at the
action limit and the pose reward was too narrow to give a gradient). In CPU MuJoCo with the full CAD collision,
through `skill_world.FallRecovery` (the viewer's X button): fallen forward / backward → parked in 0.9–1.2 s →
driving again 2 s later. A robot ending upside down after a side fall was outside the training poses (75–105°),
so v3 trains from 75–180°: 98.4% of 9,621 episodes, median 2.1 s; on CPU front / back falls stand up the same.
Known limit: a few side falls end in a leaning pose (~55°) the policy does not get out of (the 1.6% of
timeouts); press X again or restart. The hand-over waits until the gear servos are really down, so the
controller resets into PARKED and stands up normally.

**Student:** `student_env.py` simulates the D435i map per robot on the GPU (field of view, range, occlusion by
ray marching, noise ∝ depth², per-episode height bias and map offset, memory, unknown mask); map vs truth 7.7 mm
median. `train.py --task student --teacher <ckpt>` distils it (RSL-RL Distillation: the student drives, the
teacher labels); `export_policy.py --task student` exports it for `control/skill_policy.py` (obs_version 3).

**Teacher v3 (final, 4000 iterations):** stairs up 7 cm 71% (4.3 of 5 steps), 10 cm+ still weak (0.7 steps);
stairs down 89–97%, gaps to 45 cm 73–100%, bars 79–100%, curbs 25–96%.

**Student v1 (camera map, distilled from teacher v3, 1500 iterations, CAD camera mount):** close to the teacher on
stairs down up to 13 cm (86–100%), curbs and bars (81–96%), but gaps from 24 cm collapse (0–17% vs 73–88%) and
stairs down 16/19 cm drop to 77/67%: with the level camera the inside of a gap and the treads past an edge are
never seen and the near field is blind (~0.85 m), so the jump timing depends on memory and drift. Through the full
CPU stack (rendered depth → map → planner → student) it descended 13 cm stairs, crossed 17 cm gaps and passed
under a 42 cm bar. Options: mount the camera rotated 90° (87° vertical FOV), more distillation, a recurrent student.

**Teacher v4 (training):** the policy shapes the jump — 12 actions: the jump trigger plus its extension
(0.40–0.52 m), push force (150–300 N) and tuck length (0.18–0.32 m), latched when the jump starts
(`residual = action x scale + offset`, the offset is the default jump); 70% of the stairs robots climb, 20 s
episodes (a whole flight), climb reward doubled. Initialised from v3 (`train.py --init-from`: the v3 weights are
copied into the larger networks with the observation columns remapped; checked exact to 4e-7).
**Student v2 (next):** a GRU student (`train.py --task student --recurrent`), distilled from v4.

**Earlier plan:** train the student — the same policy with the elevation map of the simulated D435i (noise, field of view,
occlusions, memory, drift) instead of the perfect scans, and a longer history instead of the privileged values,
distilled from the teacher; then the robot.

## Known limitations

- The residual policy is blind: obstacles above ~5 cm and real stairs are the skill policy's (above).
- The residual does not act while standing up or landing; those modes are model-based only.
- Tire × heightfield contact gives one point per tire (MuJoCo Warp limitation for cylinder × heightfield).
