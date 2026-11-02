# PRIMUS

**P**arallel-linkage **R**obot with **I**ntegrated **M**obility and **U**nderactuated **S**tabilization

PRIMUS is a wheel-legged biped. Each leg is a closed-chain 5-bar linkage driven by two hip actuators, with a
hub-driven wheel at the foot. The robot balances on its two wheels like an inverted pendulum and uses its legs to
change ride height, level the body and absorb terrain. This repository holds the whole project, from CAD to
simulation to the physical robot.

Demo (simulation, sensor-realistic state): [`docs/demo_arena.mp4`](docs/demo_arena.mp4)

## Getting started

```bash
cd sim
python3 -m pip install -r requirements.txt      # mujoco, numpy, scipy, pygame, aiohttp, pillow
python3 scripts/check_model.py                  # load the model, stand up, balance 5 s, report loop closure
python3 scripts/run_xbox_viewer.py              # drive it with a gamepad (macOS: mjpython ...)
python3 scripts/run_quest_viewer.py             # or pilot it from a Meta Quest 3 over Wi-Fi (FPV + pilot panel)
```

The robot starts parked on its landing gear and stands up by itself. Everything else (scenes, analysis, video,
regenerating from CAD) is in [`sim/README.md`](sim/README.md). GPU reinforcement learning:
[`sim/warp_rl/README.md`](sim/warp_rl/README.md).

## Where things are

| folder | contents | status |
|---|---|---|
| [`control/`](control/) | the controller shared by simulation and robot: LQR scheduled on leg length, VMC through the closed-chain Jacobian, cascaded loops, automatic landing gear; LQR design and 5-bar kinematics | working in sim |
| [`sim/`](sim/) | MuJoCo model generated from the CAD, scenes, pilot and analysis scripts, CAD → mesh pipeline | working |
| [`sim/warp_rl/`](sim/warp_rl/) | residual RL on the GPU (MuJoCo Warp + PPO) on top of the model-based controller | working |
| [`sim/isaac/`](sim/isaac/) | URDF → USD with closed loops for Isaac Sim (for the future perception work) | on hold |
| [`hardware/`](hardware/) | actuator data and assumptions; CAD exports, BOM and wiring to come | partial |
| [`robot/`](robot/) | software for the physical robot (drivers, estimation, sim-to-real) | to be filled |
| [`docs/`](docs/) | [model and controller notes](docs/model_and_control.md), [control roadmap](docs/control_roadmap.md), [Isaac post-mortem](docs/terrain-run-2026-10-31.md), [workspace conventions](docs/workspace.md), history of the earlier prototype | |

## Robot at a glance

| | |
|---|---|
| mass | 15.7 kg (from CAD, with corrected material densities) |
| legs | 2 × closed 5-bar linkage, 2 actuated hips + 2 passive knees each |
| actuators | 6 × 9:1 quasi-direct-drive (4 hip, 2 wheel) |
| wheels | Ø 215 mm |
| leg length | 0.18 – 0.52 m (hip to axle), nominal 0.27 m |
| sensing | IMU (WitMotion), joint encoders, motor current |
| extras | retractable landing gear (2 servos), lidar, Jetson Orin Nano, Raspberry Pi 5 |

## Highlights

- **CAD → simulation pipeline.** Fusion 360 scripts export every body with its mass properties; `sim/tools` turns
  them into visual meshes, convex collision hulls (CoACD) and per-link inertias; MJCF and URDF are generated.
- **Closed kinematic chains** in MuJoCo (equality constraints, loop error < 0.1 mm) and Isaac Sim (extra joints
  excluded from the articulation tree).
- **LQR re-derived from the real geometry**, linearised at each leg length with the actual leg CoM and inertia.
- **Sensor-realistic state**: by default the controller sees only what the real robot measures (encoder
  odometry corrected for leg rotation, IMU, contact estimated from hip torques).
- **Residual RL**: a PPO policy adds per-leg corrections to the LQR; ~4× fewer falls and ~2× lower speed error
  than the LQR alone on mixed terrain with pushes ([results](docs/control_roadmap.md)).

## Status

- [x] CAD of the full robot; robot built
- [x] MuJoCo model from CAD + balance/drive controller; leg range 0.18–0.52 m; automatic landing gear
- [x] Residual RL on GPU (MuJoCo Warp); piloting with gamepad or Meta Quest 3
- [x] Isaac Sim USD conversion with closed loops
- [ ] actuator characterisation (peak torque, speed, rotor inertia); current values are placeholders
- [ ] real-robot drivers and state estimation (`robot/`), sim-to-real of controller and policy
- [ ] larger steps with perception (depth camera / elevation map), back in Isaac

The workspace around this repository (experiments, videos, caches) is described in [docs/workspace.md](docs/workspace.md).
