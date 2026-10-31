# Model Notes

## Coordinate Frame

The model uses the same robot frame as the generated URDF:

- `+X`: forward
- `+Y`: left
- `+Z`: up

The source STEP file appeared to use STEP `+Y` as robot forward, STEP `+X` as lateral, and STEP `+Z` as vertical. All MuJoCo dimensions are meters.

## Simplifications

This is not a full mesh import. It is a simulation-focused approximation:

- Chassis: box envelope
- Hip drive modules: side cylinders
- Upper/lower links: capsules
- Wheel carriers: boxes
- Wheels: cylinders
- Battery, compute stack, lidar, and support boards: fixed primitive geoms

The geometry is intentionally simple so the controller and constraints are easy to inspect and tune.

## Closed-Loop Leg Constraint

Each side is built as two branches:

- Rear branch: hip -> upper link -> knee -> lower link -> wheel carrier -> wheel
- Front branch: hip -> upper link -> knee -> lower link

The front lower-link endpoint is connected to the wheel carrier by MuJoCo equality `connect` constraints:

- `left_front_branch_to_carrier`
- `right_front_branch_to_carrier`

The constraints are the main reason this is MJCF instead of URDF.

## Actuators

The current model exposes torque-input motor actuators:

- `left_front_hip_motor`
- `left_rear_hip_motor`
- `right_front_hip_motor`
- `right_rear_hip_motor`
- `left_wheel_motor`
- `right_wheel_motor`

Knees are passive. The leg loop constraints and hip torques determine their motion.

The hip/knee joints also have passive stiffness/damping to represent unmodeled actuator compliance and keep the rough placeholder linkage near its neutral build pose. Reduce those values later if the measured CAD/inertia model becomes stable without them.

## Repo Policy Port

`scripts/repo_control_policy.py` ports the control structure from `Skythinker616/foc-wheel-legged-robot/esp32-controller/software/src/main.cpp`:

- Target speed ramp and target position clamp
- LQR gain fit from `lqr_k.c`
- State vector `[theta, dtheta, x, dx, phi, dphi]`
- Yaw cascade PID
- Roll cascade PID
- Leg-length cascade PID
- Leg-angle cascade PID
- Ground-contact check from MuJoCo contacts
- Virtual leg force/torque mapping to hip torques

### LQR gains are re-derived for THIS robot

The reference `lqr_k(L0)` cubic coefficients encode the *small* robot's dynamics
(wheel radius 0.02 m, body 0.325 kg, leg 0.04–0.14 m). They must **not** be used
for this 13 kg / 0.1075 m-wheel robot — feeding L0≈0.37 m into those cubics is
meaningless extrapolation and was the root cause of the earlier swinging.

`scripts/lqr_design.py` re-derives the gain schedule the same way the reference
`matlab/sys_calc.m` does (linearize the planar model → c2d → discrete LQR → fit
cubics vs L0), but with this robot's masses, inertias, and wheel radius.

```bash
conda run -n bipedal python3 scripts/lqr_design.py --validate   # reproduces reference coeffs to 1e-8
conda run -n bipedal python3 scripts/lqr_design.py --write       # regenerates scripts/lqr_gains.py
```

Edit `BIG_ROBOT` (params) and `Q`/`Ruu` (weights) at the top of `lqr_design.py`,
then `--write` and the controller picks up the new gains automatically.

### VMC uses a constraint-projected Jacobian

The wheel sits on the rear branch, so a plain `mj_jacSite` gives ~0 torque to the
front hip (it reaches the wheel only through the closed-loop constraint). The VMC
in `_side_vmc_torques` builds a constraint-projected wheel Jacobian so **both**
hips actuate — the MuJoCo equivalent of the reference's analytic `leg_conv`.

### Wheel-torque sign

`lqr_t_ratio = -1.0`: this model's wheel joint spins about +Y, the opposite sense
to the reference firmware, so the wheel-torque LQR term takes the opposite sign.

## Simulation Stabilizer (disabled)

The ad-hoc wheel-torque stabilizer is now **off by default** (`sim_stabilizer_enabled
= False`) because the re-derived LQR no longer needs it. It remains in the code as an
optional fallback. After real masses/inertias/motor constants/tire friction are
measured, update `BIG_ROBOT` in `lqr_design.py`, regenerate, and retune `Q`/`Ruu`
and the cascade PIDs.

## Current status

The robot balances in place (pitch within ~0.01 rad, no drift), drives forward and
backward at the commanded speed, adjusts leg height, and yaws in the correct direction —
all while the closed-loop linkage stays intact (loop gap < 0.1 mm over long mixed-command
runs). See `docs/control_overhaul.md` for the full investigation and the bugs that were
fixed (most importantly the `connect`-constraint anchor frame, which caused both the
visible linkage separation and the backward drift). Yaw is intentionally gentle (the
13 kg robot has limited yaw authority before it fights balance); raise `yaw_pid` limits
and `max_yaw_lead` in `repo_control_policy.py` if you want a faster turn rate.

### Anti-swing round (see `docs/control_overhaul.md` §17)

The "still swinging a lot" report was traced to three measured causes and fixed:

- **Coordinated banking** was a ~0.8 Hz pitch limit cycle (drive+turn pitch 9.8°→0.4°) and
  gave no fall protection on this laterally-stable robot → `banking_enabled = False` default.
- **Balance torque switched off on terrain micro-hops** (the old "both tires this step"
  ground test) → per-tire ground **hysteresis** (`air_grace_steps`); rest airborne 12%→0%,
  hip chatter 20→7 Nm RMS.
- **LQR under-damped the leg-swing mode** (`dTheta` weight was 1) → re-balanced
  `Q=[8,50,70,250,800,4]`, `Ruu=0.4`; post-push oscillations 7→4, overshoot −30%.

Measure with `scripts/analyze_balance.py` (use `--flat` on the obstacle-free
`models/pro_wheel_legged_robot_flat.xml`, built by `generate_mjcf.py --flat`, so the demo
arena's ramp at x=2 m doesn't pollute the metrics).

### Capability round (see `docs/control_overhaul.md` §18)

- **Top speed**: was capped at 1.0 m/s by the controller, not dynamics (tracks 5 m/s on flat
  using ~2 of 140 Nm). Xbox `--max-speed` raised to 2.5 m/s.
- **Yaw**: rebuilt as rate feedforward + damped rate loop (was 185% rate overshoot → ~2%,
  exact steady-state). `yaw_pid` outer is now a gentle heading-hold; inner has a small integral.
- **Ride height**: stable/tracking over leg 0.24–0.52 m (base 0.33–0.62 m). LQR `L0_sweep`
  widened to [0.24, 0.52]; Xbox `--max-leg-adjust` raised to ±0.15 m.
- **Hard brake**: `RobotCommand.brake` (Xbox B) snaps the speed target to 0 with widened
  anti-windup (`brake_speed_lead`) for max decel — 1.5 m/s stops in 0.73 s vs 1.38 s coasting.

## Robot Dimensions

| Parameter | Value |
| --- | ---: |
| L0 (hip-to-hip sagittal distance) | 0.140 m |
| L1 (upper link) | 0.220 m |
| L2 (lower link) | 0.350 m |
| Wheel diameter | 0.215 m (radius 0.1075 m) |
| Wheel track (center to center) | 0.368 m |
| Nominal leg length (virtual hip to axle) | 0.370 m |
| Base height at nominal stand | 0.4775 m |

## Placeholder Masses

The current model represents ~13 kg total:

| Part | Mass |
| --- | ---: |
| Chassis | 4.00 kg |
| Each hip drive module | 1.00 kg |
| Battery | 0.80 kg |
| Orin case | 0.50 kg |
| Raspberry Pi 5 | 0.10 kg |
| Power distribution board | 0.10 kg |
| IMU support | 0.03 kg |
| Lidar | 0.20 kg |
| Each upper link | 0.30 kg |
| Each lower link | 0.40 kg |
| Each wheel carrier | 0.70 kg |
| Each wheel | 0.50 kg |

## Xbox Controller

Xbox support uses `pygame` joystick polling. Default mapping:

- Axis 1 inverted: speed (max 2.5 m/s)
- Axis 2: yaw (right-stick horizontal; max 1.8 rad/s)
- Axis 4/5: left/right triggers for leg-height adjustment (also D-pad up/down), latching, ±0.15 m
- Button 0 (A): soft stop while held (coast down)
- Button 1 (B): hard brake while held (snap to 0, max decel)
- Button 7 (Start/Menu): reset targets + ride height

Use `scripts/inspect_gamepad.py` if your SDL mapping differs.
