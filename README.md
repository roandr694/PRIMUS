# PRIMUS

**P**arallel-linkage **R**obot with **I**ntegrated **M**obility and **U**nderactuated **S**tabilization

PRIMUS is a wheel-legged biped. Each leg is a closed-chain 5-bar linkage driven by two hip actuators, with a hub-driven wheel at the foot. The robot balances on its two wheels like an inverted pendulum and uses its legs to change ride height, level the body and absorb terrain.

This repository holds the whole project, from CAD to simulation to the physical robot.

Demo (simulation, sensor-realistic state): [`docs/demo_arena.mp4`](docs/demo_arena.mp4)

## Repository layout

| folder | contents | status |
|---|---|---|
| [`hardware/`](hardware/) | mechanical and electrical design: CAD exports, bill of materials, wiring, assembly notes | to be filled |
| [`sim/`](sim/) | MuJoCo model generated from the CAD (meshes, masses, closed 5-bar legs), test scenes, analysis tools; URDF + Isaac Lab balance and mixed-terrain RL tasks with closed loops | MuJoCo working; first GPU PPO terrain run completed |
| [`control/`](control/) | balance and motion controller shared by simulation and the real robot: LQR gain-scheduled on leg length, virtual model control (VMC) through the closed-chain Jacobian, cascaded PIDs for yaw/roll/height | working in sim |
| [`robot/`](robot/) | software for the physical robot: motor and IMU drivers, communication, state estimation, sim-to-real integration | to be filled |
| [`docs/`](docs/) | technical notes, results, history of the earlier prototype model | |

## Robot at a glance

| | |
|---|---|
| mass | 15.7 kg (from CAD, with corrected material densities) |
| legs | 2 × closed 5-bar linkage, 2 actuated hips + 2 passive knees each |
| actuators | 6 × 9:1 quasi-direct-drive (4 hip, 2 wheel) |
| wheels | Ø 215 mm |
| leg length | 0.18 – 0.36 m (hip to axle), nominal 0.27 m |
| sensing | IMU (WitMotion), joint encoders, motor current |
| extras | retractable landing gear (2 servos), lidar, Jetson Orin Nano, Raspberry Pi 5 |

## Highlights

- **CAD → simulation pipeline.** Fusion 360 scripts export every body with its mass properties. `sim/tools` turns them into visual meshes, convex collision hulls (CoACD) and per-link inertias. MJCF and URDF are generated from that data.
- **Closed kinematic chains in both simulators.** In MuJoCo, through equality constraints (loop error < 0.1 mm). In Isaac Sim, through extra joints excluded from the articulation tree.
- **LQR re-derived from the real geometry.** The planar wheel-leg-body model is linearized at each leg length, using the actual leg center of mass and inertia from the 5-bar pose. The resulting gains are fit as cubic polynomials in leg length.
- **Sensor-realistic state.** By default the controller sees only what the real robot can measure:
  - wheel-encoder odometry, corrected for leg rotation;
  - IMU attitude, either the fused output or Mahony from gyro + accelerometer, with optional noise and bias;
  - ground contact estimated from hip motor torques.

  Ground-truth simulator state stays available for comparison and for asymmetric actor-critic RL.

## Quick start (simulation)

```bash
cd sim
pip install -r requirements.txt
python3 scripts/check_model.py          # load, balance 5 s, report loop closure
mjpython scripts/run_xbox_viewer.py     # drive it with an Xbox controller (macOS: mjpython)
```

See [`sim/README.md`](sim/README.md) for everything else: scenes, analysis tools, the Isaac Sim workflow, and regenerating from CAD.

## Status

- [x] CAD of the full robot
- [x] robot built
- [x] MuJoCo model from CAD + balance/drive controller
- [x] Isaac Sim USD conversion and closed-loop PhysX smoke test validated on CPU and GPU
- [ ] actuator characterization (peak torque, speed, rotor inertia); current values are placeholders
- [ ] real-robot drivers and state estimation (`robot/`)
- [ ] sim-to-real transfer of the controller
- [x] First GPU terrain PPO policy: 3,000 updates, evaluation and training/playback videos ([results](docs/terrain-run-2026-10-31.md))
- [ ] Reliable terrain locomotion: stair performance, command tracking and robustness still need improvement

## Workspace and terrain experiments

The Desktop workspace places this repository directly in `biped_project/PRIMUS`.
Runs, videos, reports and caches live beside it; see [workspace conventions](docs/workspace.md).
The experimental `Primus-Terrain-Direct` task adds flat ground, ramps, small stairs
and uneven terrain, with tracked experiments and periodic training videos.
See [the Isaac workflow](sim/isaac/README.md) for the GPU command, validation gates
and current limitations. The first trained policy is a baseline with incomplete terrain mastery.
