# PRIMUS — simulation

MuJoCo model of the robot generated from the Fusion assembly **"Robo bipedal - montagem principal"**: CAD meshes,
masses and inertias, closed 5-bar legs, 9:1 actuators. The controller (LQR + VMC + cascaded loops + automatic
landing gear) lives in [`../control/`](../control/) and is shared with the real robot; by default it uses **only
what the real robot measures** (IMU, encoders, motor torques).

All commands run **from `sim/`**. Install once with `python3 -m pip install -r requirements.txt`
(macOS: run the viewers with `mjpython` instead of `python3`).

## Drive the robot

The robot starts **parked** (legs at 0.18 m, resting on its landing gear) and stands up by itself: it raises the
legs, retracts the gear once still, and then follows the commands. "Land" makes it stop, lower the gear and
descend onto it. Leg height goes from 0.18 to 0.52 m.

```bash
python3 scripts/run_viewer.py [--flat]          # viewer, zero command
python3 scripts/run_xbox_viewer.py [--flat]     # Xbox / SDL gamepad
python3 scripts/run_quest_viewer.py [--flat]    # Meta Quest 3 over Wi-Fi
```

| action | Xbox | Meta Quest 3 | keyboard (Quest page in a desktop browser) |
|---|---|---|---|
| speed (up to 1.5 m/s) | left stick Y | left stick | W / S |
| turn | right stick X | right stick | A / D |
| leg height (latches) | triggers or D-pad | triggers | E / Q |
| hard brake | B | A | space |
| soft stop | A | B | – |
| land ↔ stand up | X | X | P |
| restart | Start | Y | R |

If your gamepad maps differently, find the indices with `python3 scripts/inspect_gamepad.py` and pass them to
`run_xbox_viewer.py` (`--speed-axis`, `--brake-button`, ...).

**Meta Quest 3.** `run_quest_viewer.py` prints `https://<computer-ip>:8443`. On the Quest, on the same Wi-Fi, open
it in the browser, accept the self-signed certificate warning (Advanced → proceed) and tap **Enter VR**. You get a
large screen with the robot's first-person camera (chase camera in the corner) and a pilot panel (mode, gear,
speed, turn rate, artificial horizon, leg height, torques, wheel contact, falls, real-time factor, ping). Both
grips re-centre the screen. Without headset input for 0.5 s the robot stops; after a fall the simulation
restarts by itself in 4 s. Needs `aiohttp`, `pillow` and the system `openssl`; on macOS allow incoming
connections for Python if the firewall asks. Lower `--width/--height/--quality` if the Wi-Fi struggles.

## Measure

```bash
python3 scripts/check_model.py --flat                                   # joints, actuators, 5 s balance, loop gap
python3 scripts/analyze_balance.py --flat --speed 0.6 --yaw 1.5         # swing, bounce, contact, torque, drift
python3 scripts/analyze_balance.py --speed 0.5 --yaw 0.3 --height-amplitude 0.03   # arena, leg going up/down
python3 scripts/analyze_balance.py --flat --imu mahony --imu-noise      # realistic IMU
python3 scripts/analyze_balance.py --flat --state sim                   # simulator ground truth instead of sensors
python3 scripts/analyze_balance.py --flat --log 0.25                    # trajectory log for tuning
python3 scripts/record_video.py ../docs/demo_arena.mp4                  # headless mp4 (needs ffmpeg)
```

`--state robot` (default) uses only real-robot signals: IMU (`imu_quat`, `imu_gyro`, `imu_acc`), wheel-encoder
odometry corrected by the shin rotation, and contact estimated from hip torques. `--state sim` uses the true
wheel-centre velocity and tire forces (for comparison, or as the privileged critic in RL). See
[`../docs/model_and_control.md`](../docs/model_and_control.md).

## Layout

```text
models/robo_bipedal.xml        robot (GENERATED): bodies, joints, loops, actuators, sensors, keyframes
models/cena_arena.xml          "arena" scene: terrain, ramp, stairs, floor patches, bumps, rocks
models/cena_plana.xml          flat scene (to measure the controller)
models/meshes/, models/cad/    visual/collision meshes and mass properties from the CAD
scripts/robot_model.py         load a scene, leg pose for a length, LQR model parameters, stand_up()
scripts/pilot.py               gamepad/Quest/keyboard inputs -> RobotCommand (shared by the pilot scripts)
scripts/run_viewer.py, run_xbox_viewer.py, run_quest_viewer.py (+ quest_web/), inspect_gamepad.py
scripts/check_model.py, analyze_balance.py, record_video.py
scripts/generate_mjcf.py       writes the 3 XML files from models/cad + tools/robot_def.py
scripts/generate_urdf.py       writes isaac/urdf/robo_bipedal.urdf + models/cad/loop_closures.json
tools/                         CAD -> meshes/masses pipeline (Fusion scripts + build_assets.py)
warp_rl/                       residual RL on the GPU (MuJoCo Warp + RSL-RL)
isaac/                         URDF -> USD with closed loops, for Isaac Sim (on hold)
```

## Regenerate

After changing an actuator parameter (top of `scripts/generate_mjcf.py`) or masses:

```bash
python3 scripts/generate_mjcf.py
python3 ../control/lqr_design.py --write
```

After a CAD change:

1. In Fusion, run `tools/fusion/fusion_1_classificar_corpos.py` and `fusion_2_exportar_malhas.py`. They export the
   raw STL files to `~/Desktop/robo_sim_export/raw` (temporary; delete after step 2).
2. Then:
   ```bash
   python3 -m pip install trimesh fast-simplification coacd
   RAW=~/Desktop/robo_sim_export/raw python3 tools/build_assets.py       # writes models/meshes and models/cad
   python3 scripts/generate_mjcf.py && python3 scripts/generate_urdf.py && python3 ../control/lqr_design.py --write
   ```
