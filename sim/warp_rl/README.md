# Residual RL on the GPU (MuJoCo Warp)

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

## Known limitations

- No camera: obstacles above ~5 cm and real stairs are for the next phase (Isaac, elevation map).
- The residual does not act while standing up or landing; those modes are model-based only.
- Tire × heightfield contact gives one point per tire (MuJoCo Warp limitation for cylinder × heightfield).
