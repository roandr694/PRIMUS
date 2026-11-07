# PRIMUS control roadmap

*Updated 2026-11-06 (autonomous skills: see Part 2, roadmap item 4).*

> **Pending: retrain the residual RL policy.** On 2026-11-06 the base controller changed (leg-sync gains scaled
> with leg length, mean leg length shifted to keep both legs in range, configurable acceleration, turning
> load-transfer feed-forward and turn-rate envelope). The v2 policy was trained on the old controller and now
> hurts it (quick check, 720 robots × 20 s: model-based alone 0.267 → 0.188 falls/min; with the v2 policy
> 0.033 → 0.108). Re-run `sim/warp_rl/train.py` and `eval.py` before relying on it.

Control is layered. The **model-based** controller (LQR + VMC + PIDs) is the base: it balances, drives, turns,
changes height and lands, and it is what will run first on the real robot. **RL** sits on top as a robustness
layer: it adds small corrections for pushes, rough ground and legs that need not stay aligned. Later, with
perception, it should get more authority for manoeuvres the linear model does not cover, such as larger steps.

```text
 commands (speed, turn, height, land)                robot sensors (IMU, encoders, torques)
            │                                                   │
            ▼                                                   ▼
 ┌─ residual RL (50 Hz) ─────────────────────────────────────────────────────┐   current phase: blind
 │  Δlength and Δangle of each leg, Δtorque of each wheel                     │   (proprioceptive)
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
 ┌─ model-based (1 kHz) ─────────────────────────────────────────────────────┐
 │  automatic mode (parked / standing up / operating / landing)               │
 │  LQR K(L0) → wheel torque T and virtual leg torque Tp                      │
 │  loops: leg length, roll, yaw, leg synchronisation                         │
 │  VMC: closed 5-bar Jacobian → 4 hip torques                                │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
                         6 motors 9:1 + 2 landing-gear servos
```

---

## Part 1 — Model-based

### Where we are

| block | state |
|---|---|
| model | MuJoCo model generated from the CAD: masses, inertias and closed 5-bar legs via loop constraints (gap < 0.1 mm) |
| LQR | half-robot planar model (wheel + variable-length leg + body) linearised at each L0 with the real 5-bar geometry. Gains fitted as cubics in L0 over **0.17–0.54 m** (fit error < 0.3%) |
| VMC | force along the leg + virtual torque → hip torques by τ = Jᵀ·F, with the closed-loop Jacobian (analytic in `control/five_bar.py`) |
| cascaded loops | leg length (with weight per leg), roll, yaw (rate + heading correction), leg synchronisation |
| state | only what the real robot measures: IMU, encoder odometry corrected by the shin rotation, contact estimated from hip torques |
| leg range | **0.18–0.52 m**, measured with the collision meshes: the wheel hits the chassis below ~0.17 m; fully extended is 0.565 m |
| landing gear | **automatic**: starts parked (0.18 m, gear down), stands up, retracts the gear when still, operates at any height, lands on request. The user only asks to operate or land |

Validated in MuJoCo (CPU): steady pitch +1.7°, speeds −1 to 1.5 m/s and turn rates up to 1.5 rad/s, 80 N push
recovered in ~1 s, arena with a 13° ramp, height 0.18–0.52 m standing and driving, full landing cycle (stands up
in 1.9 s, lands in 3.3 s), Mahony IMU with noise. The GPU port (`sim/warp_rl/controller.py`) matches the numpy
controller (wheels within 0.45%, hips within 2%). It can be piloted with an Xbox gamepad or a Meta Quest 3.

### Decisions and why

- **LQR scheduled on L0 from the real geometry**: the leg CoM is near the wheel (wheel motor on the shin); a
  "CoM at mid-leg" model gave wrong gains.
- **Leg range from the geometry, not from the fit**: the old 0.36 m limit was only the fitted range.
- **State from real sensors only**: the same code runs on the robot; simulator ground truth is only for
  comparison and for the RL critic.
- **"Still" condition for retracting the gear ignores the absolute pitch**: the equilibrium pitch depends on the
  CoM, which is uncertain on the real robot.

### Next steps

1. **Characterise the actuators** (peak and continuous torque, rotor inertia, backlash, torque constant), then
   regenerate the MJCF and the LQR (`generate_mjcf.py`, `lqr_design.py --write`). Today: 18 N·m at the joint;
   the real value should be higher.
2. **Update the mass** of the built robot (somewhat below the CAD's 15.7 kg).
3. **Port to the robot** (`robot/`): moteus drivers, IMU, the same `robot`-mode estimator and the 1 kHz loop;
   compare real logs with simulation.
4. **Banking into turns** (lean the body inwards) to put more load on the inner wheel and allow tighter fast
   turns; needs a design that respects the leg range at tall heights. (Done 2026-11-06: leg-sync gains scaled with
   leg length, mean leg length shifted on uneven ground, three acceleration levels, load-transfer feed-forward and
   turn-rate envelope.)
5. **Contact estimator and active suspension** (gains currently zero) for rough ground.

---

## Part 2 — Reinforcement learning

### Where we came from

The first run in Isaac gave PPO **raw torques for the 6 motors**. The policy did not learn to balance: it lowered
the landing gear and sat on it (servo target never set, reward paying almost full for a tilted robot, lenient
termination, poor curriculum). Details: [terrain-run-2026-10-31.md](terrain-run-2026-10-31.md); the code is kept
at the Git tag `isaac-raw-torque-2026-10-31`.

**Lesson:** with a working LQR, learning everything from scratch wastes samples, is fragile for sim-to-real
(unmeasured actuators) and invites the policy to cheat.

### Current approach: residual RL on the GPU in MuJoCo Warp ([`sim/warp_rl/`](../sim/warp_rl/README.md))

- **Action, per leg, only in `operating` mode**: Δlength of each leg (mean = height, difference = body roll),
  Δangle of each leg (mean = shifts the equilibrium, difference = one leg ahead of the other), Δtorque of each
  wheel. The actor's last layer starts at zero: training starts exactly from the model-based controller.
- **Physics**: MuJoCo Warp with the same CAD model the LQR was validated on; 4096 robots, physics and controller
  at 1 kHz in one CUDA graph (~45k policy steps/s on an RTX 5070).
- **One large shared terrain** (like Isaac): 6 types (flat, rough, slope up to 15°, waves, steps up to 5 cm,
  blocks) × 6 levels, with a level curriculum.
- **Actor**: only what the robot measures (IMU, leg length/angle, wheel speeds, odometry) + commands + 5 frames
  of history, with noise. **Critic**: also true velocity, a height map and the randomised parameters.
- **Robustness**: friction 0.5–1.25, chassis mass ±15%, CoM ±2 cm, pushes up to 0.6 m/s, 15% of episodes start parked.
- **Reward**: speed, turn and height tracking; penalties for tilt, shaking, residual size and rate, torque, gear
  touching the ground while operating, falls.

**Result (v2, 1500 iterations, ~147M steps).** 2160 robots × 60 s on a terrain with a different seed, same
scenarios in both columns:

| | model-based only | model-based + RL |
|---|---:|---:|
| falls per robot-minute | 0.265 | **0.069** |
| speed error (m/s) | 0.207 | **0.094** |

RL is better in all 18 terrain groups; the largest gain is on hard steps (0.89 → 0.18 falls/min), the weakest
spot is the 15° slope (0.25 falls/min). Training lesson: in a residual, exploration noise acts directly on the
robot and must be bounded (std 0.05–0.3, no entropy bonus); in the first attempt it grew to 0.66 and falls went
from 2% to 75%.

### Current limitations

8 cm heightfield (a step edge is an 8 cm ramp); only tires and gear tips collide with the terrain; the residual
does not act while standing up or landing; provisional torque and mass; not yet tested outside the training
environment (CPU MuJoCo, real robot).

### Roadmap

1. **Consolidate the blind residual**: run the policy in CPU MuJoCo (viewer, Xbox, Quest); export the actor
   (TorchScript) to run at 50 Hz next to the 1 kHz loop.
2. **Prepare sim-to-real**: sensor and actuator latency, measured actuator model, realistic IMU noise/bias;
   randomise torque limit and backlash. Then the robot, with reduced residual authority, increased step by step.
3. **More authority**: larger residual scales through curriculum; later a policy with absolute per-leg
   length/angle targets, distilled from model-based + residual, for manoeuvres beyond the linear model (e.g. a
   two-wheel push-off).
4. **Perception and autonomous skills — in progress (MuJoCo Warp, not Isaac).** Done: the perception stack
   ([`perception/`](../perception/README.md): D435i → elevation map with memory → stairs / curbs / drops / gaps /
   overhangs, RPLIDAR A1 obstacles, edge guard, skill offers, A to start, B to cancel), the skill terrain and the
   teacher environment with a jump primitive in the controller (two-wheel crouch-and-extend push, as predicted:
   the wheels clear 23–35 cm), teacher v1 (stairs down, gaps, curbs, bars; not stairs up), CPU runner and Quest
   integration. Running: teacher v2 (jump primitive) and the stand-up-after-fall policy. Next: the student
   (simulated D435i map, distilled from the teacher), then the robot. **Physical limit:** without hip abduction
   the robot cannot walk one leg at a time; steps above the wheel radius (~10 cm) need the two-wheel jump.
5. **Banking in fast turns** as a learned skill on top of the load-transfer feed-forward (turn commands in the
   skill environment; reward on the inner-wheel load).
6. **Front flip** — kept for later: the jump primitive plus a pitch rotation; needs the measured actuators first.

**Rule:** every new policy is evaluated against the model-based controller alone on the same scenarios
(`eval.py`), and only replaces it where it is better.
