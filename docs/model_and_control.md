# Model and controller notes

## 1. Model

- Source: the Fusion assembly "Robo bipedal - montagem principal". Visual meshes are the decimated CAD meshes. Collision uses exact cylinders for the tires (R = 0.1075 m) and convex hulls (CoACD) for everything else.
- The robot collides only with the ground and obstacles; no self-collision.
- Frames: **X right** (X+ = leg `xp`), **Y forward**, **Z up**. All joints rotate about +X. Angle 0 of every joint is the CAD pose, legs almost straight (L0 = 0.394 m).
- In the scenes the robot spawns rotated −90° about Z, facing world +X, where the arena ramp is.

**Joints and actuators:**

- `hip_Yp_*`, `hip_Ym_*`: hip motors.
- `knee_Yp_*`, `knee_Ym_*`: passive knees.
- `wheel_*`: wheels.
- `gear_front`, `gear_rear`: landing gear, position servos.
- Each leg is a **closed** 5-bar via `equality/connect loop_*` at the wheel axle (between `linkYm_*` and `shin_*`). Measured gap below 0.1 mm.

**Sites used by the controller:** `imu` (WitMotion position), `hip_mid_*` (virtual hip, midway between the two hips), `wheel_center_*` (wheel centre, on the shin), `foot_*` (loop anchor, on link Y−).

### Masses

Total **15.68 kg**:

| body | kg |
|---|---|
| base (chassis, electronics, battery, hip stators) | 7.63 |
| shin, with the wheel motor (each) | 1.35 |
| wheel (each) | 1.31 |
| link Y− (each) | 0.65 |
| arm (each of 4) | 0.33 |

CAD material corrections are in `sim/tools/robot_def.py → MASS_OVERRIDES`: M18 battery, aluminium profiles, Orin, lidar, RPi, and the wheel rim **assumed PETG — to confirm**.

### Actuators: PROVISIONAL values

In `sim/scripts/generate_mjcf.py`: 9:1 reduction, **peak motor torque 2 N·m** (18 N·m at the joint), armature 0.006 kg·m². In the MJCF `ctrl` is the **motor** torque; the controller computes joint torque and divides by 9.

Reference values from the tests:

| situation | hip (joint) |
|---|---|
| standing at L0 = 0.27 m | ~6.5 N·m |
| standing at 0.20 m | ~5 N·m |
| standing at 0.34 m | ~9 N·m |
| fast height change | touches 18 N·m for ~0.2 s |

The wheel rarely exceeds 4 N·m. The arena's 7 cm bumps are at the limit: the wheel must push ~2× the weight.

## 2. Controller (`control/repo_control_policy.py`)

**LQR (balance).** State `[θ, θ̇, x, ẋ, φ, φ̇]`, u = −K(L0)·x:

- θ: leg angle from vertical, positive with the hip ahead of the wheel; φ: pitch, positive nose up.
- T: torque of ONE wheel; Tp: virtual torque of ONE leg.
- K(L0) is a cubic in L0 from `control/lqr_design.py` (a port of `sys_calc.m`). The table covers L0 from 0.17 to 0.54 m (fit error < 0.3%); the pose for each L0 comes from the analytic 5-bar kinematics (`control/five_bar.py`).
- The model is **half a robot**, with parameters taken from the MJCF **at each L0** with the closed 5-bar actually posed. Key facts: the leg CoM is near the wheel (L ≈ 0.115 m at L0 = 0.27) because the wheel motor is on the shin; leg inertia about its CoM ~0.09 kg·m²; wheel inertia includes the rotor armature; the body CoM is 2.1 cm **below** the hip and 1.1 cm behind.
- Weights Q = [10, 4, 40, 80, 600, 10], R = [4, 4], dt = 1 ms.

**Equilibrium.** The off-axis CoM and slopes are compensated by a slow integral of the position error that shifts the θ zero (`theta_zero_ki`, limited to ±0.15 rad).

**VMC.** Converts the force along the leg F and Tp into the 2 hip torques, τ = Jᵀ·F_world, with J the wheel-centre Jacobian w.r.t. the hips, projected through the loop constraint so the passive knees follow.

- F = leg length PI + weight per leg (mass without wheels × g / 2) ± roll.
- Tp = LQR ± leg synchronisation (needed because the yaw differential splays the legs).

**Cascaded loops** (tuned for dt = 1 ms): yaw (rate with feed-forward + slow heading correction), roll (P-D leveller), leg length, angle between the legs.

**Leg range.** Measured by sweeping the hip angles with the collision meshes (symmetric legs):

| | L0 |
|---|---|
| wheel hits the chassis | below ~0.17 m |
| **minimum operating** (`MIN_LEG_LENGTH`) | **0.18 m** (margin to the chassis; with the gear down its tips touch the ground here) |
| **maximum operating** (`MAX_LEG_LENGTH`) | **0.52 m** (angle between the lower bars ~47°, away from the singularity) |
| fully extended (singular) | 0.565 m |

The old 0.36 m limit was only the range the LQR had been fitted on, not the geometry.

**Landing gear: automatic.** The user never moves the gear; they ask to operate or land (`RobotCommand.park`) and the controller goes through the modes (`Mode`):

1. `parked`: legs at 0.18 m, gear down (0°) resting on the ground, no balancing. The wheels are held and the legs kept perpendicular to the body.
2. `standing_up` (`park=False`): the LQR starts, the legs rise to ≥ 0.25 m (`GEAR_SAFE_LENGTH` + 1 cm), and once the robot is still (pitch rate < 0.3 rad/s, |ẋ| < 0.1 m/s for 0.5 s) the gear retracts (±100°, ~0.3 s servo travel; waits 0.6 s).
3. `operating`: gear up, any height from 0.18 to 0.52 m, user speed and turn rate.
4. `landing` (`park=True`): ignores speed and turn, stops, rises to the safe height if lower, lowers the gear, descends quickly to 0.25 m and then at 5 cm/s to 0.18 m → `parked`.

The robot starts parked (`parked` keyframe of the scenes) and stands up by itself. Lowered, the gear tip is 9 cm above the ground at L0 = 0.27 m; retracted, 20 cm. The "still" condition does not use the absolute pitch: the equilibrium pitch depends on the CoM, which is uncertain on the real robot (with the CoM 2 cm off it reaches ~5°).

### State: real-robot sensors (default) or simulator

`state_source="robot"` (default) uses only what the real robot measures; `state_source="sim"` uses the simulator's ground truth.

| signal | `robot` (default) | `sim` |
|---|---|---|
| attitude | `imu_quat` (fused angle, as the WitMotion outputs); or Mahony from gyro + accelerometer with `ImuConfig(mode="mahony")` | same |
| φ̇, roll rate, yaw rate | `imu_gyro` | same |
| x, ẋ | v = −R·(ω_wheel_encoder + ω_shin), ω_shin = IMU X rate + shin angle rate relative to the base (hip + knee, from the 5-bar kinematics). Blind to slip | true wheel-centre velocity along the heading |
| contact | estimated Fn: measured hip torques → leg force on the wheel (inverting τ = Jᵀf) + m_below_hip·(g + vertical wheel acceleration), filtered; on the ground if > 15 N for more than 30 ms | true tire force (`mj_contactForce`) > 5 N |
| L0, leg angle | hip kinematics | same |

Measured (flat ground, driving and turning): encoder speed RMS error ≈ 0.04 m/s (peaks when braking, when the wheel slips); estimated vs true Fn: bias ~1 N, RMS ~5 N. In a hop the estimator flags "airborne" for ~80% of the true air time, with no false positives on the ground. The estimator starts as "on the ground" at `reset`, because it depends on the commanded force.

For RL, give the *actor* only the `robot` observations and, optionally, the `sim` ones to the *critic*. The GPU residual training is in [`sim/warp_rl/`](../sim/warp_rl/README.md).

Optional `ImuConfig` noise: `gyro_noise`, `gyro_bias`, `angle_noise`, `acc_noise`. `--imu-noise` in `sim/scripts/analyze_balance.py` uses 0.005 rad/s, 0.002 rad/s, 0.002 rad and 0.05 m/s².

## 3. Results (headless)

| test | result |
|---|---|
| standing, flat ground | pitch steady at +1.7°, peak-to-peak < 0.2°; L0 = 0.270 m |
| speed 0.5 / 1.5 / −1.0 m/s | tracks, 10–20% overshoot at start; hard brake stops in ~1 s |
| yaw 1.0 rad/s standing and at 0.5 m/s | 1.00 / 0.99 rad/s |
| 1 m/s + yaw ±1.5 rad/s with height changes | no fall; roll ~1.5° out of the turn |
| 80 N push for 0.1 s | recovers in ~1 s |
| height 0.18 ↔ 0.52 m, standing and at 0.5 m/s turning 1 rad/s | tracks, no fall |
| parked → stand up → operate → 80 N push → land | stands up and retracts the gear in 1.9 s; lands in 3.3 s; max pitch 4.7° |
| arena: 13° ramp (up, platform, down), floor patches, terrain | no fall |
| Mahony IMU with noise and bias | no fall; pitch peak-to-peak 0.5° |
| all of the above | in `robot` mode (encoders + IMU + estimated contact) and in `sim` mode |

Known limit: 0.52 m leg + 1.5 m/s + 1.8 rad/s turn at the same time makes it fall (each alone is fine).

## 4. To do with real data

1. Peak torque and maximum speed of the motors: `HIP_MOTOR_TAU`, `WHEEL_MOTOR_TAU` in `sim/scripts/generate_mjcf.py`.
2. Rotor inertia: `*_ARMATURE`.
3. Real actuator mass and wheel rim material.

After changing any of them: `python3 sim/scripts/generate_mjcf.py` and `python3 control/lqr_design.py --write`.
